/**
 * verdict_pipelined.cu — Sprint 11 Task 2: Software Pipelining
 *
 * Adds cp.async double-buffered pipelining to Sprint 9's cooperative kernel.
 * Overlaps GMEM loads of K-tile N+1 with MMA computation of K-tile N.
 *
 * Also tests L2 cache persistence (cudaAccessPolicyWindow) to pin active
 * expert weights in L2.
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a \
 *     --expt-relaxed-constexpr --compiler-options '-fPIC' \
 *     -o verdict_pipelined csrc/verdict_pipelined.cu
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <numeric>

// ============================================================================
// Compile-time Constants (same as Sprint 9)
// ============================================================================
static constexpr int HIDDEN      = 4096;
static constexpr int BM = 16, BN = 64, BK = 64;
static constexpr int SF_BLOCK    = 16;
static constexpr int SF_PER_K    = BK / SF_BLOCK;       // 4
static constexpr int NUM_WARPS   = 8;
static constexpr int WARP_SIZE   = 32;
static constexpr int BLOCK_SIZE  = NUM_WARPS * WARP_SIZE;  // 256
static constexpr int K_PACKED    = HIDDEN / 2;             // 2048
static constexpr int SF_COLS_W1  = HIDDEN / SF_BLOCK;      // 256
static constexpr int SMEM_B      = BN * (BK / 2);          // 2048
static constexpr int SMEM_SFB    = BN * SF_PER_K;          // 256
static constexpr int PARTIALS_PER_CTA = 2 * BN;            // 128

// Double-buffer stage sizes
static constexpr int SF_PER_K_ALIGNED = (SF_PER_K + 3) & ~3;  // 4
static constexpr int STAGE_GEMM1 = 32 + 2*SMEM_B + SF_PER_K_ALIGNED + 2*SMEM_SFB;  // 4644
static constexpr int STAGE_GEMM1_ALIGNED = (STAGE_GEMM1 + 15) & ~15;  // 4656
static constexpr int TOTAL_SMEM_PIPELINED = 2 * STAGE_GEMM1_ALIGNED + 128;  // 9440

static const float E2M1_TABLE[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// ============================================================================
// Host E4M3FN Utilities
// ============================================================================
float h_e4m3fn_decode(uint8_t x) {
    int s = (x >> 7) & 1, e = (x >> 3) & 0xF, m = x & 7;
    if (e == 15 && m == 7) return s ? -NAN : NAN;
    float val = (e == 0) ? ldexpf((float)m, -9) : ldexpf((float)(8 + m), e - 10);
    return s ? -val : val;
}

uint8_t h_e4m3fn_encode(float v) {
    if (isnan(v)) return 0x7F;
    int s = v < 0 ? 1 : 0;
    float av = fabsf(v);
    if (av > 448.0f) av = 448.0f;
    uint8_t best = 0; float best_err = FLT_MAX;
    for (int e = 0; e <= 15; e++)
        for (int m = 0; m <= 7; m++) {
            if (e == 15 && m == 7) continue;
            float repr = (e == 0) ? ldexpf((float)m, -9) : ldexpf((float)(8+m), e-10);
            float err = fabsf(av - repr);
            if (err < best_err) { best_err = err; best = (e << 3) | m; }
        }
    return (s << 7) | best;
}

uint8_t h_e4m3fn_encode_ceil(float val) {
    if (val <= 0) return 0x08;
    if (val >= 448.0f) return 0x7E;
    uint8_t best = 0x7E; float best_repr = 448.0f;
    for (int e = 0; e <= 15; e++)
        for (int m = 0; m <= 7; m++) {
            if (e == 15 && m == 7) continue;
            float repr = (e == 0) ? ldexpf((float)m, -9) : ldexpf((float)(8+m), e-10);
            if (repr >= val && repr < best_repr) { best_repr = repr; best = (e << 3) | m; }
        }
    return best;
}

float host_silu(float x) { return x / (1.0f + expf(-x)); }

// ============================================================================
// Device Helpers
// ============================================================================
__device__ __forceinline__ float d_e4m3fn_decode(uint8_t x) {
    int s = (x >> 7) & 1, e = (x >> 3) & 0xF, m = x & 7;
    float val;
    if (e == 0) val = ldexpf((float)m, -9);
    else if (e == 15 && m == 7) val = 0.0f;
    else val = ldexpf((float)(8 + m), e - 10);
    return s ? -val : val;
}

__device__ __forceinline__ uint8_t d_e4m3fn_encode(float val) {
    uint16_t packed;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(packed) : "f"(val), "f"(0.0f));
    return (uint8_t)((packed >> 8) & 0xFF);
}

__device__ __forceinline__ float d_silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ uint8_t d_quantize_e2m1(float value) {
    float av = fabsf(value); int sign = (value < 0.0f) ? 1 : 0; int idx;
    if      (av < 0.25f) idx = 0; else if (av < 0.75f) idx = 1;
    else if (av < 1.25f) idx = 2; else if (av < 1.75f) idx = 3;
    else if (av < 2.5f)  idx = 4; else if (av < 3.5f)  idx = 5;
    else if (av < 5.0f)  idx = 6; else idx = 7;
    return (uint8_t)((sign << 3) | idx);
}

__device__ __forceinline__ uint32_t swizzle_343(uint32_t off) {
    return off ^ ((off >> 3) & 0x70u);
}

// ============================================================================
// cp.async helpers
// ============================================================================
__device__ __forceinline__ void cp_async_4B(void* smem_dst, const void* gmem_src) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(smem_dst);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem_addr), "l"(gmem_src) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

// ============================================================================
// MMA: scale_vec::4X with native E4M3FN (ue4m3)
// ============================================================================
__device__ __forceinline__ void mma_nvf4_e4m3_m16n8k64(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
    const float (&c)[4], uint32_t sfa, uint32_t sfb)
{
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},"
        "{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
          "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

// ============================================================================
// Atomic Grid Barrier
// ============================================================================
__device__ __forceinline__ void grid_barrier_atomic(
    volatile int* counter, int total_ctas, int gen)
{
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        int target = total_ctas * (gen + 1);
        atomicAdd((int*)counter, 1);
        while (atomicAdd((int*)counter, 0) < target) {}
    }
    __syncthreads();
}

__device__ __forceinline__ uint32_t pack_sf4(const uint8_t* sf) {
    return (uint32_t)sf[0] | ((uint32_t)sf[1]<<8)
         | ((uint32_t)sf[2]<<16) | ((uint32_t)sf[3]<<24);
}

// ============================================================================
// SMEM stage accessor — returns pointers into a specific double-buffer stage
// ============================================================================
struct StagePointers {
    uint8_t* s_A;
    uint8_t* s_B_gate;
    uint8_t* s_B_up;
    uint8_t* s_SFA;
    uint8_t* s_SFB_gate;
    uint8_t* s_SFB_up;
};

__device__ __forceinline__ StagePointers get_stage(char* smem_raw, int stage) {
    StagePointers p;
    char* base = smem_raw + stage * STAGE_GEMM1_ALIGNED;
    p.s_A        = (uint8_t*)base;
    p.s_B_gate   = p.s_A + 32;
    p.s_B_up     = p.s_B_gate + SMEM_B;
    p.s_SFA      = p.s_B_up + SMEM_B;
    p.s_SFB_gate = p.s_SFA + SF_PER_K_ALIGNED;
    p.s_SFB_up   = p.s_SFB_gate + SMEM_SFB;
    return p;
}

// ============================================================================
// Helper: async load a GEMM1 K-tile into a stage buffer
// ============================================================================
__device__ __forceinline__ void async_load_gemm1_tile(
    StagePointers& sp, int tid,
    const uint8_t* input_fp4, int token_id, int k_off,
    const uint8_t* w1_fp4, const uint8_t* w1_sf,
    const uint8_t* input_sf,
    int n_start, int n_half, int k_pk, int k_sf)
{
    // Load A (input) — small, 32 bytes, use cp.async
    for (int i = tid; i < 8; i += BLOCK_SIZE) {
        cp_async_4B(sp.s_A + i * 4,
                    &input_fp4[token_id * K_PACKED + k_pk + i * 4]);
    }

    // Load gate B tile — 2048 bytes = 512 uint32 loads
    for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
        int boff = i * 4;
        int row = boff / (BK / 2), col = boff % (BK / 2);
        cp_async_4B(&sp.s_B_gate[swizzle_343(boff)],
                    &w1_fp4[(long long)(n_start + row) * K_PACKED + k_pk + col]);
    }

    // Load up B tile
    for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
        int boff = i * 4;
        int row = boff / (BK / 2), col = boff % (BK / 2);
        cp_async_4B(&sp.s_B_up[swizzle_343(boff)],
                    &w1_fp4[(long long)(n_half + n_start + row) * K_PACKED + k_pk + col]);
    }

    // Scale factors — 1 byte each, use regular loads (cp.async min is 4B)
    if (tid < SF_PER_K) {
        sp.s_SFA[tid] = input_sf[token_id * SF_COLS_W1 + k_sf + tid];
    }
    for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
        int row = i / SF_PER_K, col = i % SF_PER_K;
        sp.s_SFB_gate[i] = w1_sf[(long long)(n_start + row) * SF_COLS_W1 + k_sf + col];
    }
    for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
        int row = i / SF_PER_K, col = i % SF_PER_K;
        sp.s_SFB_up[i] = w1_sf[(long long)(n_half + n_start + row) * SF_COLS_W1 + k_sf + col];
    }
}

// ============================================================================
// PIPELINED FUSED COOPERATIVE KERNEL
//
// Same grid/work decomposition as Sprint 9, but with:
//   - Double-buffered SMEM for K-tile loops
//   - cp.async for overlapping load N+1 with compute N
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
verdict_fused_pipelined(
    const uint8_t* __restrict__ input_fp4,
    const uint8_t* __restrict__ input_sf,
    const uint8_t* __restrict__ all_w1_fp4,
    const uint8_t* __restrict__ all_w1_sf,
    const uint8_t* __restrict__ all_w2_fp4,
    const uint8_t* __restrict__ all_w2_sf,
    const int*     __restrict__ expert_ids,
    const int*     __restrict__ token_ids,
    const float*   __restrict__ expert_wts,
    float*         __restrict__ output,
    float*         __restrict__ partials,
    uint8_t*       __restrict__ gmem_inter_fp4,
    uint8_t*       __restrict__ gmem_inter_sf,
    volatile int*  __restrict__ barrier_counter,
    int num_pairs,
    int n_half,
    int k_groups)
{
    const int tiles_n       = n_half / BN;
    const int num_tiles     = tiles_n * k_groups;
    const int k_tiles_per_g = (HIDDEN / BK) / k_groups;
    const int k_per_group   = k_tiles_per_g * BK;
    const int n_half_packed = n_half / 2;
    const int sf_cols_w2    = n_half / SF_BLOCK;
    const int n2            = 2 * n_half;

    const int pair_idx = blockIdx.x / num_tiles;
    const int tile     = blockIdx.x % num_tiles;
    const int n_chunk  = tile / k_groups;
    const int k_group  = tile % k_groups;
    const int tid      = threadIdx.x;
    const int warp_id  = tid / WARP_SIZE;
    const int lane_id  = tid % WARP_SIZE;
    const int total_ctas = num_pairs * num_tiles;
    if (pair_idx >= num_pairs) return;

    const int eid      = expert_ids[pair_idx];
    const int token_id = token_ids[pair_idx];
    const float wt     = expert_wts[pair_idx];
    const int n_start  = n_chunk * BN;
    const int k_base   = k_group * k_per_group;

    // Double-buffered SMEM
    extern __shared__ char smem_raw[];

    // Per-warp MMA column mapping
    const int g   = lane_id / 4;
    const int Nl  = 4 * (g & 1) + (g >> 1);
    const int sn  = warp_id * 8 + Nl;
    const int t0  = lane_id % 4;
    const int rbo = sn * (BK / 2);

    const uint8_t* w1_fp4 = all_w1_fp4 + (long long)eid * n2 * K_PACKED;
    const uint8_t* w1_sf  = all_w1_sf  + (long long)eid * n2 * SF_COLS_W1;

    // ================================================================
    // PHASE 1a: GEMM1 — pipelined double-buffered K-tile loop
    // ================================================================
    float gate_acc[4] = {0, 0, 0, 0};
    float up_acc[4]   = {0, 0, 0, 0};

    int stage = 0;

    // Prologue: async load first K-tile into buffer 0
    {
        StagePointers sp = get_stage(smem_raw, 0);
        int k_off = k_base;
        int k_pk  = k_off / 2;
        int k_sf  = k_off / SF_BLOCK;
        async_load_gemm1_tile(sp, tid, input_fp4, token_id, k_off,
                              w1_fp4, w1_sf, input_sf,
                              n_start, n_half, k_pk, k_sf);
        cp_async_commit();
        cp_async_wait_all();
        __syncthreads();
    }

    for (int kt = 0; kt < k_tiles_per_g; kt++) {
        StagePointers cur = get_stage(smem_raw, stage);

        // If not last tile, start async load of next tile into other buffer
        if (kt + 1 < k_tiles_per_g) {
            StagePointers nxt = get_stage(smem_raw, 1 - stage);
            int k_off_next = k_base + (kt + 1) * BK;
            int k_pk_next  = k_off_next / 2;
            int k_sf_next  = k_off_next / SF_BLOCK;
            async_load_gemm1_tile(nxt, tid, input_fp4, token_id, k_off_next,
                                  w1_fp4, w1_sf, input_sf,
                                  n_start, n_half, k_pk_next, k_sf_next);
            cp_async_commit();
        }

        // Compute from current buffer
        uint32_t bg[2], bu[2];
        bg[0] = *(uint32_t*)&cur.s_B_gate[swizzle_343(rbo + t0 * 4)];
        bg[1] = *(uint32_t*)&cur.s_B_gate[swizzle_343(rbo + 16 + t0 * 4)];
        uint32_t sfbg = pack_sf4(&cur.s_SFB_gate[sn * SF_PER_K]);

        bu[0] = *(uint32_t*)&cur.s_B_up[swizzle_343(rbo + t0 * 4)];
        bu[1] = *(uint32_t*)&cur.s_B_up[swizzle_343(rbo + 16 + t0 * 4)];
        uint32_t sfbu = pack_sf4(&cur.s_SFB_up[sn * SF_PER_K]);

        uint32_t a[4] = {0, 0, 0, 0};
        if (lane_id / 4 == 0) {
            a[0] = *(uint32_t*)(cur.s_A + t0 * 4);
            a[2] = *(uint32_t*)(cur.s_A + 16 + t0 * 4);
        }
        uint32_t sfa_pk = pack_sf4(cur.s_SFA);

        mma_nvf4_e4m3_m16n8k64(gate_acc, a, bg, gate_acc, sfa_pk, sfbg);
        mma_nvf4_e4m3_m16n8k64(up_acc, a, bu, up_acc, sfa_pk, sfbu);

        // If not last tile, wait for next tile load and sync
        if (kt + 1 < k_tiles_per_g) {
            cp_async_wait_all();
            __syncthreads();
        }

        stage = 1 - stage;
    }

    // Write partials
    if (lane_id < 4) {
        long long pb = (long long)(pair_idx * num_tiles + tile) * PARTIALS_PER_CTA;
        int c0 = warp_id * 8 + lane_id;
        int c1 = c0 + 4;
        partials[pb + c0]      = gate_acc[0];
        partials[pb + c1]      = gate_acc[1];
        partials[pb + BN + c0] = up_acc[0];
        partials[pb + BN + c1] = up_acc[1];
    }

    grid_barrier_atomic(barrier_counter, total_ctas, 0);

    // ================================================================
    // PHASE 1b: Reduce K-group partials + SwiGLU + FP4 requant
    // ================================================================
    if (k_group == 0 && tid < BN) {
        int col = tid;
        float gs = 0, us = 0, gs_c = 0, us_c = 0;

        for (int kg = 0; kg < k_groups; kg++) {
            int partner_tile = n_chunk * k_groups + kg;
            long long base = (long long)(pair_idx * num_tiles + partner_tile) * PARTIALS_PER_CTA;
            float g_y = partials[base + col] - gs_c;
            float g_t = gs + g_y;
            gs_c = (g_t - gs) - g_y;
            gs = g_t;
            float u_y = partials[base + BN + col] - us_c;
            float u_t = us + u_y;
            us_c = (u_t - us) - u_y;
            us = u_t;
        }

        float sw_val = us * d_silu(gs);

        float abs_sw = fabsf(sw_val);
        float gm = abs_sw;
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 1));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 2));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 4));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 8));

        float st = fmaxf(gm / 6.0f, 1e-30f);
        uint8_t sf_enc = d_e4m3fn_encode(st);
        float as = d_e4m3fn_decode(sf_enc);
        if (as < 1e-30f) as = 1e-30f;

        uint8_t nib = d_quantize_e2m1(sw_val / as);
        uint32_t nib32 = (uint32_t)nib;
        uint32_t neighbor32 = __shfl_down_sync(0xFFFFFFFF, nib32, 1);
        if (col % 2 == 0) {
            gmem_inter_fp4[pair_idx * n_half_packed + (n_start + col) / 2] =
                (uint8_t)(nib32 | (neighbor32 << 4));
        }
        if (col % SF_BLOCK == 0) {
            gmem_inter_sf[pair_idx * sf_cols_w2 + (n_start + col) / SF_BLOCK] = sf_enc;
        }
    }

    grid_barrier_atomic(barrier_counter, total_ctas, 1);

    // ================================================================
    // PHASE 2: GEMM2 — pipelined double-buffered
    // ================================================================
    {
        const int p2_out_tiles = HIDDEN / BN;
        const uint8_t* w2_fp4 = all_w2_fp4 + (long long)eid * HIDDEN * n_half_packed;
        const uint8_t* w2_sf  = all_w2_sf  + (long long)eid * HIDDEN * sf_cols_w2;
        int p2_k_passes = n_half / BK;

        for (int j_tile = tile; j_tile < p2_out_tiles; j_tile += num_tiles) {
            const int j_start = j_tile * BN;
            float p2_acc[4] = {0, 0, 0, 0};

            int p2_stage = 0;

            // Prologue: async load first K-pass into buffer 0
            {
                StagePointers sp = get_stage(smem_raw, 0);
                for (int i = tid; i < 8; i += BLOCK_SIZE)
                    cp_async_4B(sp.s_A + i * 4,
                                &gmem_inter_fp4[pair_idx * n_half_packed + i * 4]);
                for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
                    int boff = i * 4;
                    int row = boff / (BK / 2), col = boff % (BK / 2);
                    int oc = j_start + row;
                    if (oc < HIDDEN)
                        cp_async_4B(&sp.s_B_gate[swizzle_343(boff)],
                                    &w2_fp4[(long long)oc * n_half_packed + col]);
                    else
                        *(uint32_t*)&sp.s_B_gate[swizzle_343(boff)] = 0u;
                }
                if (tid < SF_PER_K)
                    sp.s_SFA[tid] = gmem_inter_sf[pair_idx * sf_cols_w2 + tid];
                for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
                    int row = i / SF_PER_K, col = i % SF_PER_K;
                    int oc = j_start + row;
                    sp.s_SFB_gate[i] = (oc < HIDDEN) ? w2_sf[(long long)oc * sf_cols_w2 + col] : 0;
                }
                cp_async_commit();
                cp_async_wait_all();
                __syncthreads();
            }

            for (int kp = 0; kp < p2_k_passes; kp++) {
                StagePointers cur = get_stage(smem_raw, p2_stage);

                // Async load next K-pass if not last
                if (kp + 1 < p2_k_passes) {
                    StagePointers nxt = get_stage(smem_raw, 1 - p2_stage);
                    int ko_n = (kp + 1) * BK, kpk_n = ko_n / 2, ksf_n = ko_n / SF_BLOCK;

                    for (int i = tid; i < 8; i += BLOCK_SIZE)
                        cp_async_4B(nxt.s_A + i * 4,
                                    &gmem_inter_fp4[pair_idx * n_half_packed + kpk_n + i * 4]);
                    for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
                        int boff = i * 4;
                        int row = boff / (BK / 2), col = boff % (BK / 2);
                        int oc = j_start + row;
                        if (oc < HIDDEN)
                            cp_async_4B(&nxt.s_B_gate[swizzle_343(boff)],
                                        &w2_fp4[(long long)oc * n_half_packed + kpk_n + col]);
                        else
                            *(uint32_t*)&nxt.s_B_gate[swizzle_343(boff)] = 0u;
                    }
                    if (tid < SF_PER_K)
                        nxt.s_SFA[tid] = gmem_inter_sf[pair_idx * sf_cols_w2 + ksf_n + tid];
                    for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
                        int row = i / SF_PER_K, col = i % SF_PER_K;
                        int oc = j_start + row;
                        nxt.s_SFB_gate[i] = (oc < HIDDEN) ? w2_sf[(long long)oc * sf_cols_w2 + ksf_n + col] : 0;
                    }
                    cp_async_commit();
                }

                // Compute from current buffer (reuse B_gate as B2)
                uint32_t br[2];
                br[0] = *(uint32_t*)&cur.s_B_gate[swizzle_343(rbo + t0 * 4)];
                br[1] = *(uint32_t*)&cur.s_B_gate[swizzle_343(rbo + 16 + t0 * 4)];
                uint32_t sfbp = pack_sf4(&cur.s_SFB_gate[sn * SF_PER_K]);

                uint32_t ar[4] = {0, 0, 0, 0};
                if (lane_id / 4 == 0) {
                    ar[0] = *(uint32_t*)(cur.s_A + t0 * 4);
                    ar[2] = *(uint32_t*)(cur.s_A + 16 + t0 * 4);
                }
                uint32_t sfap = pack_sf4(cur.s_SFA);

                mma_nvf4_e4m3_m16n8k64(p2_acc, ar, br, p2_acc, sfap, sfbp);

                if (kp + 1 < p2_k_passes) {
                    cp_async_wait_all();
                    __syncthreads();
                }

                p2_stage = 1 - p2_stage;
            }

            // Scatter output
            if (lane_id < 4) {
                int j0 = j_start + warp_id * 8 + lane_id;
                int j1 = j0 + 4;
                if (j0 < HIDDEN) atomicAdd(&output[token_id * HIDDEN + j0], wt * p2_acc[0]);
                if (j1 < HIDDEN) atomicAdd(&output[token_id * HIDDEN + j1], wt * p2_acc[1]);
            }
        }
    }
}

// ============================================================================
// Sprint 9 baseline kernel (copy for A/B comparison in same binary)
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
verdict_fused_baseline(
    const uint8_t* __restrict__ input_fp4,
    const uint8_t* __restrict__ input_sf,
    const uint8_t* __restrict__ all_w1_fp4,
    const uint8_t* __restrict__ all_w1_sf,
    const uint8_t* __restrict__ all_w2_fp4,
    const uint8_t* __restrict__ all_w2_sf,
    const int*     __restrict__ expert_ids,
    const int*     __restrict__ token_ids,
    const float*   __restrict__ expert_wts,
    float*         __restrict__ output,
    float*         __restrict__ partials,
    uint8_t*       __restrict__ gmem_inter_fp4,
    uint8_t*       __restrict__ gmem_inter_sf,
    volatile int*  __restrict__ barrier_counter,
    int num_pairs,
    int n_half,
    int k_groups)
{
    const int tiles_n       = n_half / BN;
    const int num_tiles     = tiles_n * k_groups;
    const int k_tiles_per_g = (HIDDEN / BK) / k_groups;
    const int k_per_group   = k_tiles_per_g * BK;
    const int n_half_packed = n_half / 2;
    const int sf_cols_w2    = n_half / SF_BLOCK;
    const int n2            = 2 * n_half;

    const int pair_idx = blockIdx.x / num_tiles;
    const int tile     = blockIdx.x % num_tiles;
    const int n_chunk  = tile / k_groups;
    const int k_group  = tile % k_groups;
    const int tid      = threadIdx.x;
    const int warp_id  = tid / WARP_SIZE;
    const int lane_id  = tid % WARP_SIZE;
    const int total_ctas = num_pairs * num_tiles;
    if (pair_idx >= num_pairs) return;

    const int eid      = expert_ids[pair_idx];
    const int token_id = token_ids[pair_idx];
    const float wt     = expert_wts[pair_idx];
    const int n_start  = n_chunk * BN;
    const int k_base   = k_group * k_per_group;

    extern __shared__ char smem_raw[];
    uint8_t* s_A        = (uint8_t*)smem_raw;
    uint8_t* s_B_gate   = s_A + 32;
    uint8_t* s_B_up     = s_B_gate + SMEM_B;
    uint8_t* s_SFA      = s_B_up + SMEM_B;
    uint8_t* s_SFB_gate = s_SFA + SF_PER_K_ALIGNED;
    uint8_t* s_SFB_up   = s_SFB_gate + SMEM_SFB;

    const int g   = lane_id / 4;
    const int Nl  = 4 * (g & 1) + (g >> 1);
    const int sn  = warp_id * 8 + Nl;
    const int t0  = lane_id % 4;
    const int rbo = sn * (BK / 2);

    const uint8_t* w1_fp4 = all_w1_fp4 + (long long)eid * n2 * K_PACKED;
    const uint8_t* w1_sf  = all_w1_sf  + (long long)eid * n2 * SF_COLS_W1;

    float gate_acc[4] = {0, 0, 0, 0};
    float up_acc[4]   = {0, 0, 0, 0};

    for (int kt = 0; kt < k_tiles_per_g; kt++) {
        const int k_off = k_base + kt * BK;
        const int k_pk  = k_off / 2;
        const int k_sf  = k_off / SF_BLOCK;

        for (int i = tid; i < 8; i += BLOCK_SIZE)
            *(uint32_t*)(s_A + i * 4) =
                *(const uint32_t*)&input_fp4[token_id * K_PACKED + k_pk + i * 4];
        for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
            int boff = i * 4;
            int row = boff / (BK / 2), col = boff % (BK / 2);
            *(uint32_t*)&s_B_gate[swizzle_343(boff)] =
                *(const uint32_t*)&w1_fp4[(long long)(n_start + row) * K_PACKED + k_pk + col];
        }
        for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
            int boff = i * 4;
            int row = boff / (BK / 2), col = boff % (BK / 2);
            *(uint32_t*)&s_B_up[swizzle_343(boff)] =
                *(const uint32_t*)&w1_fp4[(long long)(n_half + n_start + row) * K_PACKED + k_pk + col];
        }
        if (tid < SF_PER_K)
            s_SFA[tid] = input_sf[token_id * SF_COLS_W1 + k_sf + tid];
        for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
            int row = i / SF_PER_K, col = i % SF_PER_K;
            s_SFB_gate[i] = w1_sf[(long long)(n_start + row) * SF_COLS_W1 + k_sf + col];
        }
        for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
            int row = i / SF_PER_K, col = i % SF_PER_K;
            s_SFB_up[i] = w1_sf[(long long)(n_half + n_start + row) * SF_COLS_W1 + k_sf + col];
        }

        __syncthreads();

        uint32_t bg[2], bu[2];
        bg[0] = *(uint32_t*)&s_B_gate[swizzle_343(rbo + t0 * 4)];
        bg[1] = *(uint32_t*)&s_B_gate[swizzle_343(rbo + 16 + t0 * 4)];
        uint32_t sfbg = pack_sf4(&s_SFB_gate[sn * SF_PER_K]);
        bu[0] = *(uint32_t*)&s_B_up[swizzle_343(rbo + t0 * 4)];
        bu[1] = *(uint32_t*)&s_B_up[swizzle_343(rbo + 16 + t0 * 4)];
        uint32_t sfbu = pack_sf4(&s_SFB_up[sn * SF_PER_K]);

        uint32_t a[4] = {0, 0, 0, 0};
        if (lane_id / 4 == 0) {
            a[0] = *(uint32_t*)(s_A + t0 * 4);
            a[2] = *(uint32_t*)(s_A + 16 + t0 * 4);
        }
        uint32_t sfa_pk = pack_sf4(s_SFA);

        mma_nvf4_e4m3_m16n8k64(gate_acc, a, bg, gate_acc, sfa_pk, sfbg);
        mma_nvf4_e4m3_m16n8k64(up_acc, a, bu, up_acc, sfa_pk, sfbu);

        __syncthreads();
    }

    if (lane_id < 4) {
        long long pb = (long long)(pair_idx * num_tiles + tile) * PARTIALS_PER_CTA;
        int c0 = warp_id * 8 + lane_id;
        int c1 = c0 + 4;
        partials[pb + c0]      = gate_acc[0];
        partials[pb + c1]      = gate_acc[1];
        partials[pb + BN + c0] = up_acc[0];
        partials[pb + BN + c1] = up_acc[1];
    }

    grid_barrier_atomic(barrier_counter, total_ctas, 0);

    if (k_group == 0 && tid < BN) {
        int col = tid;
        float gs = 0, us = 0, gs_c = 0, us_c = 0;
        for (int kg = 0; kg < k_groups; kg++) {
            int partner_tile = n_chunk * k_groups + kg;
            long long base = (long long)(pair_idx * num_tiles + partner_tile) * PARTIALS_PER_CTA;
            float g_y = partials[base + col] - gs_c;
            float g_t = gs + g_y;
            gs_c = (g_t - gs) - g_y;
            gs = g_t;
            float u_y = partials[base + BN + col] - us_c;
            float u_t = us + u_y;
            us_c = (u_t - us) - u_y;
            us = u_t;
        }
        float sw_val = us * d_silu(gs);
        float abs_sw = fabsf(sw_val);
        float gm = abs_sw;
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 1));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 2));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 4));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 8));
        float st = fmaxf(gm / 6.0f, 1e-30f);
        uint8_t sf_enc = d_e4m3fn_encode(st);
        float as = d_e4m3fn_decode(sf_enc);
        if (as < 1e-30f) as = 1e-30f;
        uint8_t nib = d_quantize_e2m1(sw_val / as);
        uint32_t nib32 = (uint32_t)nib;
        uint32_t neighbor32 = __shfl_down_sync(0xFFFFFFFF, nib32, 1);
        if (col % 2 == 0)
            gmem_inter_fp4[pair_idx * n_half_packed + (n_start + col) / 2] =
                (uint8_t)(nib32 | (neighbor32 << 4));
        if (col % SF_BLOCK == 0)
            gmem_inter_sf[pair_idx * sf_cols_w2 + (n_start + col) / SF_BLOCK] = sf_enc;
    }

    grid_barrier_atomic(barrier_counter, total_ctas, 1);

    {
        const int p2_out_tiles = HIDDEN / BN;
        const uint8_t* w2_fp4 = all_w2_fp4 + (long long)eid * HIDDEN * n_half_packed;
        const uint8_t* w2_sf  = all_w2_sf  + (long long)eid * HIDDEN * sf_cols_w2;
        uint8_t* s_B2   = s_B_gate;
        uint8_t* s_SFB2 = s_SFB_gate;
        int p2_k_passes = n_half / BK;

        for (int j_tile = tile; j_tile < p2_out_tiles; j_tile += num_tiles) {
            const int j_start = j_tile * BN;
            float p2_acc[4] = {0, 0, 0, 0};

            for (int kp = 0; kp < p2_k_passes; kp++) {
                int ko = kp * BK, kpk = ko / 2, ksf = ko / SF_BLOCK;
                for (int i = tid; i < 8; i += BLOCK_SIZE)
                    *(uint32_t*)(s_A + i * 4) =
                        *(const uint32_t*)&gmem_inter_fp4[pair_idx * n_half_packed + kpk + i * 4];
                for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
                    int boff = i * 4;
                    int row = boff / (BK / 2), col = boff % (BK / 2);
                    int oc = j_start + row;
                    *(uint32_t*)&s_B2[swizzle_343(boff)] =
                        (oc < HIDDEN)
                        ? *(const uint32_t*)&w2_fp4[(long long)oc * n_half_packed + kpk + col]
                        : 0u;
                }
                if (tid < SF_PER_K)
                    s_SFA[tid] = gmem_inter_sf[pair_idx * sf_cols_w2 + ksf + tid];
                for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
                    int row = i / SF_PER_K, col = i % SF_PER_K;
                    int oc = j_start + row;
                    s_SFB2[i] = (oc < HIDDEN) ? w2_sf[(long long)oc * sf_cols_w2 + ksf + col] : 0;
                }
                __syncthreads();

                uint32_t br[2];
                br[0] = *(uint32_t*)&s_B2[swizzle_343(rbo + t0 * 4)];
                br[1] = *(uint32_t*)&s_B2[swizzle_343(rbo + 16 + t0 * 4)];
                uint32_t sfbp = pack_sf4(&s_SFB2[sn * SF_PER_K]);
                uint32_t ar[4] = {0, 0, 0, 0};
                if (lane_id / 4 == 0) {
                    ar[0] = *(uint32_t*)(s_A + t0 * 4);
                    ar[2] = *(uint32_t*)(s_A + 16 + t0 * 4);
                }
                uint32_t sfap = pack_sf4(s_SFA);
                mma_nvf4_e4m3_m16n8k64(p2_acc, ar, br, p2_acc, sfap, sfbp);
                __syncthreads();
            }

            if (lane_id < 4) {
                int j0 = j_start + warp_id * 8 + lane_id;
                int j1 = j0 + 4;
                if (j0 < HIDDEN) atomicAdd(&output[token_id * HIDDEN + j0], wt * p2_acc[0]);
                if (j1 < HIDDEN) atomicAdd(&output[token_id * HIDDEN + j1], wt * p2_acc[1]);
            }
        }
    }
}

// ============================================================================
// Host Quantization + Reference (same as Sprint 9)
// ============================================================================
void quantize_to_nvfp4_e4m3(const float* data, int numel,
                             uint8_t* packed, uint8_t* sf_out) {
    int nb = numel / SF_BLOCK;
    memset(packed, 0, numel / 2);
    for (int b = 0; b < nb; b++) {
        int s = b * SF_BLOCK;
        float bmax = 0;
        for (int i = s; i < s+SF_BLOCK; i++) bmax = std::max(bmax, fabsf(data[i]));
        uint8_t sf = h_e4m3fn_encode_ceil(std::max(bmax/6.0f, 1e-30f));
        sf_out[b] = sf;
        float as = h_e4m3fn_decode(sf);
        if (as < 1e-30f) as = 1e-30f;
        for (int i = s; i < s+SF_BLOCK; i++) {
            float sc = data[i]/as, av = fabsf(sc);
            int sign = (sc < 0) ? 1 : 0, idx = 0;
            float bd = av;
            for (int j = 1; j < 8; j++) { float d = fabsf(av-E2M1_TABLE[j]); if (d<bd){bd=d;idx=j;} }
            uint8_t fp4 = (uint8_t)((sign<<3)|idx);
            int bi = i/2;
            if (i%2==0) packed[bi] = fp4; else packed[bi] |= (fp4<<4);
        }
    }
}

float dequant_fp4_e4m3(const uint8_t* pk, const uint8_t* sf, int idx) {
    uint8_t bv = pk[idx/2];
    uint8_t nib = (idx&1) ? (bv>>4) : (bv&0xF);
    int sign = (nib>>3)&1, mag = nib&7;
    float val = E2M1_TABLE[mag] * h_e4m3fn_decode(sf[idx/SF_BLOCK]);
    return sign ? -val : val;
}

void host_quantized_reference_single(
    const uint8_t* ifp4, const uint8_t* isf,
    const uint8_t* w1f, const uint8_t* w1s,
    const uint8_t* w2f, const uint8_t* w2s,
    const int* eids, const float* ewts, float* out,
    int na, int n_half) {
    int n2 = 2 * n_half;
    int k_packed = HIDDEN / 2;
    int sf_cols_w1 = HIDDEN / SF_BLOCK;
    int n_half_packed = n_half / 2;
    int sf_cols_w2 = n_half / SF_BLOCK;

    memset(out, 0, HIDDEN * sizeof(float));
    for (int e = 0; e < na; e++) {
        int eid = eids[e]; float wt = ewts[e];
        const uint8_t* ew1f = w1f + (long long)eid * n2 * k_packed;
        const uint8_t* ew1s = w1s + (long long)eid * n2 * sf_cols_w1;
        const uint8_t* ew2f = w2f + (long long)eid * HIDDEN * n_half_packed;
        const uint8_t* ew2s = w2s + (long long)eid * HIDDEN * sf_cols_w2;
        float* gate = new float[n_half];
        float* up_a = new float[n_half];
        for (int n = 0; n < n_half; n++) {
            float s = 0;
            for (int k = 0; k < HIDDEN; k++)
                s += dequant_fp4_e4m3(ifp4, isf, k) *
                     dequant_fp4_e4m3(ew1f + (long long)n * k_packed,
                                      ew1s + (long long)n * sf_cols_w1, k);
            gate[n] = s;
        }
        for (int n = 0; n < n_half; n++) {
            float s = 0;
            for (int k = 0; k < HIDDEN; k++)
                s += dequant_fp4_e4m3(ifp4, isf, k) *
                     dequant_fp4_e4m3(ew1f + (long long)(n + n_half) * k_packed,
                                      ew1s + (long long)(n + n_half) * sf_cols_w1, k);
            up_a[n] = s;
        }
        float* sw = new float[n_half];
        for (int n = 0; n < n_half; n++) sw[n] = up_a[n] * host_silu(gate[n]);
        uint8_t* ifp = new uint8_t[n_half_packed]();
        uint8_t* isf2 = new uint8_t[sf_cols_w2]();
        quantize_to_nvfp4_e4m3(sw, n_half, ifp, isf2);
        for (int j = 0; j < HIDDEN; j++) {
            float s = 0;
            for (int n = 0; n < n_half; n++)
                s += dequant_fp4_e4m3(ifp, isf2, n) *
                     dequant_fp4_e4m3(ew2f + (long long)j * n_half_packed,
                                      ew2s + (long long)j * sf_cols_w2, n);
            out[j] += wt * s;
        }
        delete[] gate; delete[] up_a; delete[] sw; delete[] ifp; delete[] isf2;
    }
}

struct ErrS { double rmse, rel; int nan_c; };
ErrS compute_error(const float* a, const float* r, int n) {
    ErrS s = {}; double es = 0, rs = 0;
    for (int i = 0; i < n; i++) {
        if (isnan(a[i]) || isinf(a[i])) { s.nan_c++; continue; }
        double d = a[i] - r[i]; es += d * d; rs += (double)r[i] * r[i];
    }
    s.rmse = sqrt(es / n); s.rel = (rs > 0) ? sqrt(es / rs) : 0; return s;
}

#define CHECK_CUDA(c) do { cudaError_t _e = (c); if (_e != cudaSuccess) { \
    printf("CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); } } while(0)

int compute_k_groups(int num_pairs, int tiles_n) {
    int target = 640;
    int k_groups = std::max(1, target / (num_pairs * tiles_n));
    int total_k_tiles = HIDDEN / BK;
    while (total_k_tiles % k_groups != 0 && k_groups > 1) k_groups--;
    return k_groups;
}

// ============================================================================
// Benchmark both kernels for a given config
// ============================================================================
typedef void (*KernelFn)(
    const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*,
    const uint8_t*, const uint8_t*, const int*, const int*, const float*,
    float*, float*, uint8_t*, uint8_t*, volatile int*, int, int, int);

struct BenchResult {
    float median_us, mean_us, p10_us, p90_us;
    float rel_err;
    int nan_count;
    bool correct;
};

BenchResult run_kernel_bench(
    KernelFn kernel, int smem_size, const char* name,
    uint8_t* dif, uint8_t* dis, uint8_t* dw1f, uint8_t* dw1s,
    uint8_t* dw2f, uint8_t* dw2s, int* deids, int* dtids, float* dewts,
    float* dout, float* dpart, uint8_t* dif2, uint8_t* dis2, int* dbar,
    int total_ctas, int num_pairs, int n_half, int k_groups,
    int M, const float* href)
{
    BenchResult res = {};

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    // Correctness check
    CHECK_CUDA(cudaMemset(dout, 0, M * HIDDEN * sizeof(float)));
    CHECK_CUDA(cudaMemset(dbar, 0, sizeof(int)));
    kernel<<<total_ctas, BLOCK_SIZE, smem_size>>>(
        dif, dis, dw1f, dw1s, dw2f, dw2s, deids, dtids, dewts, dout,
        dpart, dif2, dis2, dbar, num_pairs, n_half, k_groups);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  %s: KERNEL ERROR: %s\n", name, cudaGetErrorString(err));
        return res;
    }

    float* hgo = new float[M * HIDDEN];
    CHECK_CUDA(cudaMemcpy(hgo, dout, M * HIDDEN * sizeof(float), cudaMemcpyDeviceToHost));
    ErrS eq = compute_error(hgo, href, M * HIDDEN);
    res.rel_err = eq.rel;
    res.nan_count = eq.nan_c;
    res.correct = (eq.nan_c == 0) && (eq.rel < 0.12);
    delete[] hgo;

    // Warmup
    for (int i = 0; i < 20; i++) {
        CHECK_CUDA(cudaMemset(dout, 0, M * HIDDEN * sizeof(float)));
        CHECK_CUDA(cudaMemset(dbar, 0, sizeof(int)));
        kernel<<<total_ctas, BLOCK_SIZE, smem_size>>>(
            dif, dis, dw1f, dw1s, dw2f, dw2s, deids, dtids, dewts, dout,
            dpart, dif2, dis2, dbar, num_pairs, n_half, k_groups);
    }
    cudaDeviceSynchronize();

    // Benchmark
    std::vector<float> times;
    for (int i = 0; i < 100; i++) {
        cudaEvent_t st, sp;
        cudaEventCreate(&st); cudaEventCreate(&sp);
        CHECK_CUDA(cudaMemset(dout, 0, M * HIDDEN * sizeof(float)));
        CHECK_CUDA(cudaMemset(dbar, 0, sizeof(int)));
        cudaEventRecord(st);
        kernel<<<total_ctas, BLOCK_SIZE, smem_size>>>(
            dif, dis, dw1f, dw1s, dw2f, dw2s, deids, dtids, dewts, dout,
            dpart, dif2, dis2, dbar, num_pairs, n_half, k_groups);
        cudaEventRecord(sp); cudaEventSynchronize(sp);
        float ms; cudaEventElapsedTime(&ms, st, sp);
        times.push_back(ms * 1000.0f);
        cudaEventDestroy(st); cudaEventDestroy(sp);
    }
    std::sort(times.begin(), times.end());
    res.median_us = times[50];
    res.mean_us = std::accumulate(times.begin(), times.end(), 0.0f) / 100;
    res.p10_us = times[10];
    res.p90_us = times[90];

    printf("  %s: median=%.1f μs, mean=%.1f μs, p10=%.1f, p90=%.1f | RelErr=%.2f%% %s\n",
           name, res.median_us, res.mean_us, res.p10_us, res.p90_us,
           res.rel_err * 100, res.correct ? "PASS" : "FAIL");

    return res;
}

// ============================================================================
// L2 Persistence benchmark wrapper
// ============================================================================
BenchResult run_kernel_bench_l2persist(
    KernelFn kernel, int smem_size, const char* name,
    uint8_t* dif, uint8_t* dis, uint8_t* dw1f, uint8_t* dw1s,
    uint8_t* dw2f, uint8_t* dw2s, int* deids, int* dtids, float* dewts,
    float* dout, float* dpart, uint8_t* dif2, uint8_t* dis2, int* dbar,
    int total_ctas, int num_pairs, int n_half, int k_groups,
    int M, const float* href,
    size_t w1_size, size_t w2_size)
{
    // Set L2 persistence for W1 weights
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaAccessPolicyWindow w1_window = {};
    w1_window.base_ptr = (void*)dw1f;
    w1_window.num_bytes = w1_size;
    w1_window.hitRatio = 1.0f;
    w1_window.hitProp = cudaAccessPropertyPersisting;
    w1_window.missProp = cudaAccessPropertyStreaming;
    cudaStreamAttrValue stream_attr;
    stream_attr.accessPolicyWindow = w1_window;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);

    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    // Correctness
    CHECK_CUDA(cudaMemset(dout, 0, M * HIDDEN * sizeof(float)));
    CHECK_CUDA(cudaMemset(dbar, 0, sizeof(int)));
    kernel<<<total_ctas, BLOCK_SIZE, smem_size, stream>>>(
        dif, dis, dw1f, dw1s, dw2f, dw2s, deids, dtids, dewts, dout,
        dpart, dif2, dis2, dbar, num_pairs, n_half, k_groups);
    cudaStreamSynchronize(stream);

    float* hgo = new float[M * HIDDEN];
    CHECK_CUDA(cudaMemcpy(hgo, dout, M * HIDDEN * sizeof(float), cudaMemcpyDeviceToHost));
    ErrS eq = compute_error(hgo, href, M * HIDDEN);
    delete[] hgo;

    BenchResult res = {};
    res.rel_err = eq.rel;
    res.nan_count = eq.nan_c;
    res.correct = (eq.nan_c == 0) && (eq.rel < 0.12);

    // Warmup
    for (int i = 0; i < 20; i++) {
        CHECK_CUDA(cudaMemset(dout, 0, M * HIDDEN * sizeof(float)));
        CHECK_CUDA(cudaMemset(dbar, 0, sizeof(int)));
        kernel<<<total_ctas, BLOCK_SIZE, smem_size, stream>>>(
            dif, dis, dw1f, dw1s, dw2f, dw2s, deids, dtids, dewts, dout,
            dpart, dif2, dis2, dbar, num_pairs, n_half, k_groups);
    }
    cudaStreamSynchronize(stream);

    // Benchmark
    std::vector<float> times;
    for (int i = 0; i < 100; i++) {
        cudaEvent_t st, sp;
        cudaEventCreate(&st); cudaEventCreate(&sp);
        CHECK_CUDA(cudaMemset(dout, 0, M * HIDDEN * sizeof(float)));
        CHECK_CUDA(cudaMemset(dbar, 0, sizeof(int)));
        cudaEventRecord(st, stream);
        kernel<<<total_ctas, BLOCK_SIZE, smem_size, stream>>>(
            dif, dis, dw1f, dw1s, dw2f, dw2s, deids, dtids, dewts, dout,
            dpart, dif2, dis2, dbar, num_pairs, n_half, k_groups);
        cudaEventRecord(sp, stream); cudaEventSynchronize(sp);
        float ms; cudaEventElapsedTime(&ms, st, sp);
        times.push_back(ms * 1000.0f);
        cudaEventDestroy(st); cudaEventDestroy(sp);
    }
    std::sort(times.begin(), times.end());
    res.median_us = times[50];
    res.mean_us = std::accumulate(times.begin(), times.end(), 0.0f) / 100;
    res.p10_us = times[10];
    res.p90_us = times[90];

    printf("  %s: median=%.1f μs, mean=%.1f μs, p10=%.1f, p90=%.1f | RelErr=%.2f%% %s\n",
           name, res.median_us, res.mean_us, res.p10_us, res.p90_us,
           res.rel_err * 100, res.correct ? "PASS" : "FAIL");

    // Reset L2 persistence
    cudaAccessPolicyWindow reset = {};
    reset.num_bytes = 0;
    cudaStreamAttrValue reset_attr;
    reset_attr.accessPolicyWindow = reset;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &reset_attr);
    cudaCtxResetPersistingL2Cache();
    cudaStreamDestroy(stream);

    return res;
}

// ============================================================================
// Test a config: compare baseline vs pipelined vs L2 persist
// ============================================================================
void run_comparison(int M, int topk, int n_half, int NE,
                    const int routing[][10], const char* label) {
    int num_pairs = M * topk;
    int tiles_n = n_half / BN;
    int k_groups = compute_k_groups(num_pairs, tiles_n);
    int num_tiles = tiles_n * k_groups;
    int total_ctas = num_pairs * num_tiles;
    int n2 = 2 * n_half;
    int n_half_packed = n_half / 2;
    int sf_cols_w2 = n_half / SF_BLOCK;

    printf("\n======================================================\n");
    printf("Config: %s\n", label);
    printf("M=%d, topk=%d, N_HALF=%d, K_GROUPS=%d, total_CTAs=%d\n",
           M, topk, n_half, k_groups, total_ctas);
    printf("GEMM1: %d K-tiles/group, GEMM2: %d K-passes\n",
           (HIDDEN / BK) / k_groups, n_half / BK);
    printf("======================================================\n");

    // Check occupancy for both kernels
    int smem_baseline = 32 + 2 * SMEM_B + SF_PER_K_ALIGNED + 2 * SMEM_SFB + 128;
    int smem_pipelined = TOTAL_SMEM_PIPELINED;

    int mb_base = 0, mb_pipe = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb_base, verdict_fused_baseline, BLOCK_SIZE, smem_baseline);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb_pipe, verdict_fused_pipelined, BLOCK_SIZE, smem_pipelined);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int cap_base = mb_base * prop.multiProcessorCount;
    int cap_pipe = mb_pipe * prop.multiProcessorCount;

    printf("Baseline: SMEM=%d, occupancy=%d CTAs/SM, capacity=%d\n",
           smem_baseline, mb_base, cap_base);
    printf("Pipelined: SMEM=%d, occupancy=%d CTAs/SM, capacity=%d\n",
           smem_pipelined, mb_pipe, cap_pipe);

    if (cap_base < total_ctas) {
        printf("ERROR: baseline deadlock risk (%d < %d)\n", cap_base, total_ctas);
        return;
    }
    if (cap_pipe < total_ctas) {
        printf("ERROR: pipelined deadlock risk (%d < %d)\n", cap_pipe, total_ctas);
        return;
    }

    // Build pair tables
    int* h_pair_eids = new int[num_pairs];
    int* h_pair_tids = new int[num_pairs];
    float* h_pair_wts = new float[num_pairs];
    for (int m = 0; m < M; m++)
        for (int j = 0; j < topk; j++) {
            int idx = m * topk + j;
            h_pair_eids[idx] = routing[m][j];
            h_pair_tids[idx] = m;
            h_pair_wts[idx] = 1.0f / topk;
        }

    // Generate + quantize data
    printf("Generating data...\n");
    srand(42);
    auto rf = []() { return ((float)rand() / RAND_MAX - 0.5f) * 2.0f; };

    float* hi = new float[M * HIDDEN];
    for (int i = 0; i < M * HIDDEN; i++) hi[i] = rf();
    float w1s = 1.0f / sqrtf((float)HIDDEN), w2s_f = 1.0f / sqrtf((float)n_half);
    float* hw1 = new float[(long long)NE * n2 * HIDDEN];
    for (long long i = 0; i < (long long)NE * n2 * HIDDEN; i++) hw1[i] = rf() * w1s;
    float* hw2 = new float[(long long)NE * HIDDEN * n_half];
    for (long long i = 0; i < (long long)NE * HIDDEN * n_half; i++) hw2[i] = rf() * w2s_f;

    size_t ifs = (size_t)M * K_PACKED;
    size_t iss = (size_t)M * SF_COLS_W1;
    uint8_t* hif = new uint8_t[ifs]();
    uint8_t* his = new uint8_t[iss]();
    for (int m = 0; m < M; m++)
        quantize_to_nvfp4_e4m3(&hi[m * HIDDEN], HIDDEN,
            &hif[m * K_PACKED], &his[m * SF_COLS_W1]);

    size_t w1fs = (size_t)NE * n2 * K_PACKED;
    size_t w1ss = (size_t)NE * n2 * SF_COLS_W1;
    uint8_t* hw1f = new uint8_t[w1fs]();
    uint8_t* hw1s = new uint8_t[w1ss]();
    for (int e = 0; e < NE; e++)
        for (int n = 0; n < n2; n++)
            quantize_to_nvfp4_e4m3(&hw1[(long long)e * n2 * HIDDEN + (long long)n * HIDDEN], HIDDEN,
                &hw1f[(long long)e * n2 * K_PACKED + (long long)n * K_PACKED],
                &hw1s[(long long)e * n2 * SF_COLS_W1 + (long long)n * SF_COLS_W1]);

    size_t w2fs = (size_t)NE * HIDDEN * n_half_packed;
    size_t w2ss = (size_t)NE * HIDDEN * sf_cols_w2;
    uint8_t* hw2f = new uint8_t[w2fs]();
    uint8_t* hw2s = new uint8_t[w2ss]();
    for (int e = 0; e < NE; e++)
        for (int j = 0; j < HIDDEN; j++)
            quantize_to_nvfp4_e4m3(&hw2[(long long)e * HIDDEN * n_half + (long long)j * n_half], n_half,
                &hw2f[(long long)e * HIDDEN * n_half_packed + (long long)j * n_half_packed],
                &hw2s[(long long)e * HIDDEN * sf_cols_w2 + (long long)j * sf_cols_w2]);

    // Reference
    float* hrq = new float[M * HIDDEN]();
    for (int m = 0; m < M; m++) {
        int tok_eids[10]; float tok_wts[10];
        for (int j = 0; j < topk; j++) {
            tok_eids[j] = routing[m][j];
            tok_wts[j] = 1.0f / topk;
        }
        host_quantized_reference_single(
            &hif[m * K_PACKED], &his[m * SF_COLS_W1],
            hw1f, hw1s, hw2f, hw2s,
            tok_eids, tok_wts, &hrq[m * HIDDEN], topk, n_half);
    }

    // GPU alloc
    uint8_t *dif_d, *dis_d, *dw1f_d, *dw1s_d, *dw2f_d, *dw2s_d, *dif2, *dis2;
    int *deids, *dtids; float *dewts, *dout, *dpart; int *dbar;
    size_t part_sz = (size_t)num_pairs * num_tiles * PARTIALS_PER_CTA * sizeof(float);

    CHECK_CUDA(cudaMalloc(&dif_d, ifs));
    CHECK_CUDA(cudaMalloc(&dis_d, iss));
    CHECK_CUDA(cudaMalloc(&dw1f_d, w1fs));
    CHECK_CUDA(cudaMalloc(&dw1s_d, w1ss));
    CHECK_CUDA(cudaMalloc(&dw2f_d, w2fs));
    CHECK_CUDA(cudaMalloc(&dw2s_d, w2ss));
    CHECK_CUDA(cudaMalloc(&deids, num_pairs * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dtids, num_pairs * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dewts, num_pairs * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dout, M * HIDDEN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dpart, part_sz));
    CHECK_CUDA(cudaMalloc(&dif2, (size_t)num_pairs * n_half_packed));
    CHECK_CUDA(cudaMalloc(&dis2, (size_t)num_pairs * sf_cols_w2));
    CHECK_CUDA(cudaMalloc(&dbar, sizeof(int)));

    CHECK_CUDA(cudaMemcpy(dif_d, hif, ifs, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dis_d, his, iss, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw1f_d, hw1f, w1fs, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw1s_d, hw1s, w1ss, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw2f_d, hw2f, w2fs, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw2s_d, hw2s, w2ss, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deids, h_pair_eids, num_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dtids, h_pair_tids, num_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dewts, h_pair_wts, num_pairs * sizeof(float), cudaMemcpyHostToDevice));

    // Run benchmarks
    printf("\n--- Benchmarks (100 iterations, 20 warmup) ---\n");

    BenchResult base = run_kernel_bench(
        verdict_fused_baseline, smem_baseline, "Baseline (Sprint 9)",
        dif_d, dis_d, dw1f_d, dw1s_d, dw2f_d, dw2s_d, deids, dtids, dewts,
        dout, dpart, dif2, dis2, dbar,
        total_ctas, num_pairs, n_half, k_groups, M, hrq);

    BenchResult pipe = run_kernel_bench(
        verdict_fused_pipelined, smem_pipelined, "Pipelined (cp.async)",
        dif_d, dis_d, dw1f_d, dw1s_d, dw2f_d, dw2s_d, deids, dtids, dewts,
        dout, dpart, dif2, dis2, dbar,
        total_ctas, num_pairs, n_half, k_groups, M, hrq);

    BenchResult l2p = run_kernel_bench_l2persist(
        verdict_fused_baseline, smem_baseline, "Baseline + L2 Persist",
        dif_d, dis_d, dw1f_d, dw1s_d, dw2f_d, dw2s_d, deids, dtids, dewts,
        dout, dpart, dif2, dis2, dbar,
        total_ctas, num_pairs, n_half, k_groups, M, hrq,
        w1fs, w2fs);

    BenchResult l2p_pipe = run_kernel_bench_l2persist(
        verdict_fused_pipelined, smem_pipelined, "Pipelined + L2 Persist",
        dif_d, dis_d, dw1f_d, dw1s_d, dw2f_d, dw2s_d, deids, dtids, dewts,
        dout, dpart, dif2, dis2, dbar,
        total_ctas, num_pairs, n_half, k_groups, M, hrq,
        w1fs, w2fs);

    printf("\n--- Summary ---\n");
    printf("  Baseline:           %.1f μs\n", base.median_us);
    printf("  Pipelined:          %.1f μs (%.1f%% vs baseline)\n",
           pipe.median_us, (pipe.median_us / base.median_us - 1.0f) * 100);
    printf("  Baseline + L2:      %.1f μs (%.1f%% vs baseline)\n",
           l2p.median_us, (l2p.median_us / base.median_us - 1.0f) * 100);
    printf("  Pipelined + L2:     %.1f μs (%.1f%% vs baseline)\n",
           l2p_pipe.median_us, (l2p_pipe.median_us / base.median_us - 1.0f) * 100);

    // Cleanup
    delete[] hi; delete[] hw1; delete[] hw2;
    delete[] h_pair_eids; delete[] h_pair_tids; delete[] h_pair_wts;
    delete[] hif; delete[] his; delete[] hw1f; delete[] hw1s; delete[] hw2f; delete[] hw2s;
    delete[] hrq;
    cudaFree(dif_d); cudaFree(dis_d); cudaFree(dw1f_d); cudaFree(dw1s_d);
    cudaFree(dw2f_d); cudaFree(dw2s_d); cudaFree(deids); cudaFree(dtids); cudaFree(dewts);
    cudaFree(dout); cudaFree(dpart); cudaFree(dif2); cudaFree(dis2); cudaFree(dbar);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM%d%d, %d SMs, %dKB SMEM, %dMB L2)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           (int)(prop.sharedMemPerMultiprocessor / 1024),
           (int)(prop.l2CacheSize / (1024 * 1024)));
    printf("\n=== Sprint 11 Task 2: Software Pipelining + L2 Persistence ===\n");
    printf("Comparing: Baseline (Sprint 9) vs cp.async Pipelined vs L2 Persist\n");

    int topk = 10, NE = 30;

    const int routing_m1[1][10] = {{0,1,2,3,4,5,6,7,8,9}};
    const int routing_m4[4][10] = {
        {0,1,2,3,4,5,6,7,8,9},
        {5,6,7,8,9,10,11,12,13,14},
        {20,21,22,23,24,25,26,27,28,29},
        {0,1,2,3,4,25,26,27,28,29}
    };

    // Config 1: M=1, TP=4
    run_comparison(1, topk, 256, NE, routing_m1, "M=1 TP=4");

    // Config 2: M=4, TP=4
    run_comparison(4, topk, 256, NE, routing_m4, "M=4 TP=4");

    printf("\n=== DONE ===\n");
    return 0;
}
