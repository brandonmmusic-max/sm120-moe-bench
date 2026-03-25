/**
 * verdict_fused_independent.cu — Sprint 9: Independent Per-Token Routing
 *
 * Each CTA handles ONE (token, expert) pair — exactly mirrors CUTLASS grouped GEMM.
 * Grid: num_pairs × num_tiles CTAs, where num_pairs = M × topk.
 *
 * KEY CHANGE from Sprint 7/8:
 *   - No shared routing, no union routing
 *   - Each token's experts are independently routed (same as CUTLASS/vLLM standard)
 *   - Pair table: flat [M*topk] arrays of (expert_id, token_id, weight)
 *   - MTP acceptance rate should match CUTLASS baseline (~75%)
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a \
 *     --expt-relaxed-constexpr --compiler-options '-fPIC' \
 *     -o verdict_fused_independent csrc/verdict_fused_independent.cu
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
// Compile-time Constants
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
// Atomic Grid Barrier (CUDA-graph safe, no cooperative_groups)
// ============================================================================
__device__ __forceinline__ void grid_barrier_atomic(
    volatile int* counter, int total_ctas, int gen)
{
    __syncthreads();
    __threadfence();  // CRITICAL: flush all prior GMEM writes device-wide
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
// INDEPENDENT PER-TOKEN FUSED COOPERATIVE KERNEL
//
// Grid: num_pairs × num_tiles CTAs
// Each CTA: ONE (token, expert) pair, ONE N-tile, ONE K-group.
// Exactly mirrors CUTLASS grouped GEMM: token-expert pairs independently.
//
// Runtime params: n_half, k_groups
//   TP=4 M=1: 10 pairs × 4 tiles × 16 k_groups = 640
//   TP=4 M=4: 40 pairs × 4 tiles × 4 k_groups  = 640
//   EP=4 M=1: 10 pairs × 16 tiles × 4 k_groups  = 640
//   EP=4 M=4: 40 pairs × 16 tiles × 1 k_groups  = 640
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
verdict_fused_independent(
    const uint8_t* __restrict__ input_fp4,        // [M, K_PACKED]
    const uint8_t* __restrict__ input_sf,          // [M, SF_COLS_W1]
    const uint8_t* __restrict__ all_w1_fp4,        // [E, 2*n_half, K_PACKED]
    const uint8_t* __restrict__ all_w1_sf,         // [E, 2*n_half, SF_COLS_W1]
    const uint8_t* __restrict__ all_w2_fp4,        // [E, HIDDEN, n_half/2]
    const uint8_t* __restrict__ all_w2_sf,         // [E, HIDDEN, n_half/SF_BLOCK]
    const int*     __restrict__ expert_ids,        // [num_pairs]
    const int*     __restrict__ token_ids,         // [num_pairs]
    const float*   __restrict__ expert_wts,        // [num_pairs]
    float*         __restrict__ output,            // [M, HIDDEN]
    float*         __restrict__ partials,          // [num_pairs * num_tiles * PARTIALS_PER_CTA]
    uint8_t*       __restrict__ gmem_inter_fp4,    // [num_pairs, n_half/2]
    uint8_t*       __restrict__ gmem_inter_sf,     // [num_pairs, n_half/SF_BLOCK]
    volatile int*  __restrict__ barrier_counter,
    int num_pairs,
    int n_half,
    int k_groups)
{
    // Derive runtime constants
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

    // SMEM layout (single token per CTA):
    //   s_A:        32 bytes (compact, 1 token row 0)
    //   s_B_gate:   2048 bytes
    //   s_B_up:     2048 bytes
    //   s_SFA:      SF_PER_K bytes (4-byte aligned)
    //   s_SFB_gate: 256 bytes
    //   s_SFB_up:   256 bytes
    extern __shared__ char smem_raw[];
    uint8_t* s_A        = (uint8_t*)smem_raw;
    uint8_t* s_B_gate   = s_A + 32;
    uint8_t* s_B_up     = s_B_gate + SMEM_B;
    uint8_t* s_SFA      = s_B_up + SMEM_B;
    uint8_t* s_SFB_gate = s_SFA + ((SF_PER_K + 3) & ~3);
    uint8_t* s_SFB_up   = s_SFB_gate + SMEM_SFB;

    // Per-warp MMA column mapping (CLayout for M=1, scale_vec::4X)
    const int g   = lane_id / 4;
    const int Nl  = 4 * (g & 1) + (g >> 1);
    const int sn  = warp_id * 8 + Nl;
    const int t0  = lane_id % 4;
    const int rbo = sn * (BK / 2);

    // Weight pointers for this expert
    const uint8_t* w1_fp4 = all_w1_fp4 + (long long)eid * n2 * K_PACKED;
    const uint8_t* w1_sf  = all_w1_sf  + (long long)eid * n2 * SF_COLS_W1;

    // ================================================================
    // PHASE 1a: GEMM1 — single token, hybrid K×N
    // ================================================================
    float gate_acc[4] = {0, 0, 0, 0};
    float up_acc[4]   = {0, 0, 0, 0};

    for (int kt = 0; kt < k_tiles_per_g; kt++) {
        const int k_off = k_base + kt * BK;
        const int k_pk  = k_off / 2;
        const int k_sf  = k_off / SF_BLOCK;

        // Load this token's A data (compact: 8 uint32 loads)
        for (int i = tid; i < 8; i += BLOCK_SIZE) {
            *(uint32_t*)(s_A + i * 4) =
                *(const uint32_t*)&input_fp4[token_id * K_PACKED + k_pk + i * 4];
        }

        // Load gate B tile
        for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
            int boff = i * 4;
            int row = boff / (BK / 2), col = boff % (BK / 2);
            *(uint32_t*)&s_B_gate[swizzle_343(boff)] =
                *(const uint32_t*)&w1_fp4[(long long)(n_start + row) * K_PACKED + k_pk + col];
        }

        // Load up B tile
        for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
            int boff = i * 4;
            int row = boff / (BK / 2), col = boff % (BK / 2);
            *(uint32_t*)&s_B_up[swizzle_343(boff)] =
                *(const uint32_t*)&w1_fp4[(long long)(n_half + n_start + row) * K_PACKED + k_pk + col];
        }

        // Load SFA (this token)
        if (tid < SF_PER_K) {
            s_SFA[tid] = input_sf[token_id * SF_COLS_W1 + k_sf + tid];
        }

        // Load SFB gate + up
        for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
            int row = i / SF_PER_K, col = i % SF_PER_K;
            s_SFB_gate[i] = w1_sf[(long long)(n_start + row) * SF_COLS_W1 + k_sf + col];
        }
        for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
            int row = i / SF_PER_K, col = i % SF_PER_K;
            s_SFB_up[i] = w1_sf[(long long)(n_half + n_start + row) * SF_COLS_W1 + k_sf + col];
        }

        __syncthreads();

        // B operands
        uint32_t bg[2], bu[2];
        bg[0] = *(uint32_t*)&s_B_gate[swizzle_343(rbo + t0 * 4)];
        bg[1] = *(uint32_t*)&s_B_gate[swizzle_343(rbo + 16 + t0 * 4)];
        uint32_t sfbg = pack_sf4(&s_SFB_gate[sn * SF_PER_K]);

        bu[0] = *(uint32_t*)&s_B_up[swizzle_343(rbo + t0 * 4)];
        bu[1] = *(uint32_t*)&s_B_up[swizzle_343(rbo + 16 + t0 * 4)];
        uint32_t sfbu = pack_sf4(&s_SFB_up[sn * SF_PER_K]);

        // A operand (single token)
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

    // Write partials (no M dimension — one token per CTA)
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
    //   Leader CTAs (k_group==0) reduce for this pair.
    // ================================================================
    if (k_group == 0 && tid < BN) {
        int col = tid;

        // Kahan compensated summation
        float gs = 0, us = 0;
        float gs_c = 0, us_c = 0;

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

        // Group max across SF_BLOCK=16 columns via warp shuffle
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

        // Pack nibble pairs
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
    // PHASE 2: GEMM2 — single token, output scatter by token_id
    //   When num_tiles < HIDDEN/BN, loop over output tile groups.
    // ================================================================
    {
        const int p2_out_tiles = HIDDEN / BN;  // 64
        const uint8_t* w2_fp4 = all_w2_fp4 + (long long)eid * HIDDEN * n_half_packed;
        const uint8_t* w2_sf  = all_w2_sf  + (long long)eid * HIDDEN * sf_cols_w2;

        // Reuse SMEM
        uint8_t* s_B2   = s_B_gate;
        uint8_t* s_SFB2 = s_SFB_gate;

        int p2_k_passes = n_half / BK;

        for (int j_tile = tile; j_tile < p2_out_tiles; j_tile += num_tiles) {
            const int j_start = j_tile * BN;

            float p2_acc[4] = {0, 0, 0, 0};

            for (int kp = 0; kp < p2_k_passes; kp++) {
                int ko = kp * BK, kpk = ko / 2, ksf = ko / SF_BLOCK;

                // Load intermediate A (this pair)
                for (int i = tid; i < 8; i += BLOCK_SIZE) {
                    *(uint32_t*)(s_A + i * 4) =
                        *(const uint32_t*)&gmem_inter_fp4[pair_idx * n_half_packed + kpk + i * 4];
                }

                // Load W2 B tile
                for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
                    int boff = i * 4;
                    int row = boff / (BK / 2), col = boff % (BK / 2);
                    int oc = j_start + row;
                    *(uint32_t*)&s_B2[swizzle_343(boff)] =
                        (oc < HIDDEN)
                        ? *(const uint32_t*)&w2_fp4[(long long)oc * n_half_packed + kpk + col]
                        : 0u;
                }

                // Load intermediate SFA (this pair)
                if (tid < SF_PER_K) {
                    s_SFA[tid] = gmem_inter_sf[pair_idx * sf_cols_w2 + ksf + tid];
                }

                // Load W2 SFB
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

            // Scatter to output[token_id] with routing weight
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
// Host: Quantization
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

// ============================================================================
// Host Quantized Reference (single token, specific expert set)
// ============================================================================
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

// ============================================================================
// Error
// ============================================================================
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

// ============================================================================
// Compute k_groups to target ~640 CTAs
// ============================================================================
int compute_k_groups(int num_pairs, int tiles_n) {
    int target = 640;
    int k_groups = std::max(1, target / (num_pairs * tiles_n));
    int total_k_tiles = HIDDEN / BK;  // 64
    // k_groups must divide total_k_tiles
    while (total_k_tiles % k_groups != 0 && k_groups > 1) k_groups--;
    return k_groups;
}

// ============================================================================
// Test: Independent per-token routing with different expert sets
// ============================================================================
bool run_independent_test(int M, int topk, int n_half, int NE,
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
    printf("M=%d, topk=%d, num_pairs=%d, N_HALF=%d, K_GROUPS=%d\n",
           M, topk, num_pairs, n_half, k_groups);
    printf("Grid: %d pairs × %d tiles = %d CTAs\n", num_pairs, num_tiles, total_ctas);
    printf("Phase 1: %d N-chunks × %d K-groups, %d K-tiles/group\n",
           tiles_n, k_groups, (HIDDEN / BK) / k_groups);
    printf("Phase 2: %d K-passes, %s\n", n_half / BK,
           num_tiles < HIDDEN/BN ? "LOOPED (multiple output tile groups)" : "single pass");
    printf("======================================================\n");

    // Check occupancy
    int smem_size = 32 + 2 * SMEM_B + ((SF_PER_K + 3) & ~3) + 2 * SMEM_SFB + 128;
    int mb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, verdict_fused_independent, BLOCK_SIZE, smem_size);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int cap = mb * prop.multiProcessorCount;
    printf("Occupancy: %d CTAs/SM × %d SMs = %d (need %d) %s\n",
           mb, prop.multiProcessorCount, cap, total_ctas, cap >= total_ctas ? "OK" : "FAIL");
    if (cap < total_ctas) { printf("ERROR: deadlock risk\n"); return false; }

    // Build pair tables
    int* h_pair_eids = new int[num_pairs];
    int* h_pair_tids = new int[num_pairs];
    float* h_pair_wts = new float[num_pairs];
    for (int m = 0; m < M; m++) {
        for (int j = 0; j < topk; j++) {
            int idx = m * topk + j;
            h_pair_eids[idx] = routing[m][j];
            h_pair_tids[idx] = m;
            h_pair_wts[idx] = 1.0f / topk;
        }
    }

    printf("Routing table:\n");
    for (int m = 0; m < M; m++) {
        printf("  Token %d → experts {", m);
        for (int j = 0; j < topk; j++) printf("%d%s", routing[m][j], j<topk-1?",":"");
        printf("}\n");
    }

    // Generate data
    printf("Generating Xavier-scaled data...\n");
    srand(42);
    auto rf = []() { return ((float)rand() / RAND_MAX - 0.5f) * 2.0f; };

    float* hi = new float[M * HIDDEN];
    for (int i = 0; i < M * HIDDEN; i++) hi[i] = rf();

    float w1s = 1.0f / sqrtf((float)HIDDEN), w2s = 1.0f / sqrtf((float)n_half);
    float* hw1 = new float[(long long)NE * n2 * HIDDEN];
    for (long long i = 0; i < (long long)NE * n2 * HIDDEN; i++) hw1[i] = rf() * w1s;
    float* hw2 = new float[(long long)NE * HIDDEN * n_half];
    for (long long i = 0; i < (long long)NE * HIDDEN * n_half; i++) hw2[i] = rf() * w2s;

    // Quantize inputs
    printf("Quantizing %d tokens + weights (%d experts)...\n", M, NE);
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

    // Compute per-token quantized reference (INDEPENDENT routing per token)
    printf("Computing per-token references...\n");
    float* hrq = new float[M * HIDDEN]();
    for (int m = 0; m < M; m++) {
        int tok_eids[10];
        float tok_wts[10];
        for (int j = 0; j < topk; j++) {
            tok_eids[j] = routing[m][j];
            tok_wts[j] = 1.0f / topk;
        }
        host_quantized_reference_single(
            &hif[m * K_PACKED], &his[m * SF_COLS_W1],
            hw1f, hw1s, hw2f, hw2s,
            tok_eids, tok_wts, &hrq[m * HIDDEN],
            topk, n_half);
    }

    // Upload to GPU
    printf("Uploading to GPU...\n");
    uint8_t *dif, *dis, *dw1f, *dw1s, *dw2f, *dw2s, *dif2, *dis2;
    int *deids, *dtids; float *dewts, *dout, *dpart; int *dbar;

    size_t part_sz = (size_t)num_pairs * num_tiles * PARTIALS_PER_CTA * sizeof(float);
    size_t inter_fp4_sz = (size_t)num_pairs * n_half_packed;
    size_t inter_sf_sz = (size_t)num_pairs * sf_cols_w2;

    CHECK_CUDA(cudaMalloc(&dif, ifs));
    CHECK_CUDA(cudaMalloc(&dis, iss));
    CHECK_CUDA(cudaMalloc(&dw1f, w1fs));
    CHECK_CUDA(cudaMalloc(&dw1s, w1ss));
    CHECK_CUDA(cudaMalloc(&dw2f, w2fs));
    CHECK_CUDA(cudaMalloc(&dw2s, w2ss));
    CHECK_CUDA(cudaMalloc(&deids, num_pairs * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dtids, num_pairs * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dewts, num_pairs * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dout, M * HIDDEN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dpart, part_sz));
    CHECK_CUDA(cudaMalloc(&dif2, inter_fp4_sz));
    CHECK_CUDA(cudaMalloc(&dis2, inter_sf_sz));
    CHECK_CUDA(cudaMalloc(&dbar, sizeof(int)));

    CHECK_CUDA(cudaMemcpy(dif, hif, ifs, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dis, his, iss, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw1f, hw1f, w1fs, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw1s, hw1s, w1ss, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw2f, hw2f, w2fs, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw2s, hw2s, w2ss, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deids, h_pair_eids, num_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dtids, h_pair_tids, num_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dewts, h_pair_wts, num_pairs * sizeof(float), cudaMemcpyHostToDevice));

    cudaFuncSetAttribute(verdict_fused_independent,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    // Launch kernel
    printf("Launching kernel (%d CTAs, smem=%d)...\n", total_ctas, smem_size);
    CHECK_CUDA(cudaMemset(dout, 0, M * HIDDEN * sizeof(float)));
    CHECK_CUDA(cudaMemset(dbar, 0, sizeof(int)));
    verdict_fused_independent<<<total_ctas, BLOCK_SIZE, smem_size>>>(
        dif, dis, dw1f, dw1s, dw2f, dw2s, deids, dtids, dewts, dout,
        dpart, dif2, dis2, dbar, num_pairs, n_half, k_groups);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }

    {
        // Readback
        float* hgo = new float[M * HIDDEN];
        CHECK_CUDA(cudaMemcpy(hgo, dout, M * HIDDEN * sizeof(float), cudaMemcpyDeviceToHost));

        // Per-token correctness
        printf("\n--- Per-Token Correctness (vs independent M=1 QRef) ---\n");
        bool all_pass = true;
        for (int m = 0; m < M; m++) {
            ErrS eq = compute_error(&hgo[m * HIDDEN], &hrq[m * HIDDEN], HIDDEN);
            bool pq = (eq.nan_c == 0) && (eq.rel < 0.12);
            printf("  Token %d: RelErr=%.4f%% %s | NaN=%d\n",
                   m, eq.rel * 100, pq ? "PASS" : "FAIL", eq.nan_c);
            if (!pq) all_pass = false;
        }

        // Aggregate
        ErrS eq_all = compute_error(hgo, hrq, M * HIDDEN);
        float agg_threshold = (k_groups > 4) ? 0.11f : 0.10f;
        bool agg_pass = (eq_all.nan_c == 0) && (eq_all.rel < agg_threshold);
        printf("  Aggregate: RelErr=%.4f%% %s | NaN=%d\n",
               eq_all.rel * 100, agg_pass ? "PASS" : "FAIL", eq_all.nan_c);
        if (!agg_pass) all_pass = false;

        printf("  Sample GPU[tok0, 0:4]: ");
        for (int i = 0; i < 4; i++) printf("%.4f ", hgo[i]); printf("\n");
        printf("  Sample QRef[tok0, 0:4]: ");
        for (int i = 0; i < 4; i++) printf("%.4f ", hrq[i]); printf("\n");

        // Benchmark
        printf("\n--- Benchmark ---\n");
        for (int i = 0; i < 20; i++) {
            CHECK_CUDA(cudaMemset(dout, 0, M * HIDDEN * sizeof(float)));
            CHECK_CUDA(cudaMemset(dbar, 0, sizeof(int)));
            verdict_fused_independent<<<total_ctas, BLOCK_SIZE, smem_size>>>(
                dif, dis, dw1f, dw1s, dw2f, dw2s, deids, dtids, dewts, dout,
                dpart, dif2, dis2, dbar, num_pairs, n_half, k_groups);
        }
        cudaDeviceSynchronize();

        std::vector<float> times;
        for (int i = 0; i < 100; i++) {
            cudaEvent_t st, sp;
            cudaEventCreate(&st); cudaEventCreate(&sp);
            CHECK_CUDA(cudaMemset(dout, 0, M * HIDDEN * sizeof(float)));
            CHECK_CUDA(cudaMemset(dbar, 0, sizeof(int)));
            cudaEventRecord(st);
            verdict_fused_independent<<<total_ctas, BLOCK_SIZE, smem_size>>>(
                dif, dis, dw1f, dw1s, dw2f, dw2s, deids, dtids, dewts, dout,
                dpart, dif2, dis2, dbar, num_pairs, n_half, k_groups);
            cudaEventRecord(sp); cudaEventSynchronize(sp);
            float ms; cudaEventElapsedTime(&ms, st, sp);
            times.push_back(ms * 1000.0f);
            cudaEventDestroy(st); cudaEventDestroy(sp);
        }
        std::sort(times.begin(), times.end());
        float med = times[50], mn = std::accumulate(times.begin(), times.end(), 0.0f) / 100;
        printf("  Latency: median=%.1f μs, mean=%.1f μs, p10=%.1f, p90=%.1f\n",
               med, mn, times[10], times[90]);

        if (M == 1) {
            printf("  vs Sprint 7 TP=4 M=1 (19.8 μs): %.2fx\n", 19.8f / med);
            printf("  vs CUTLASS (98 μs): %.2fx\n", 98.0f / med);
        } else if (M == 4) {
            printf("  vs Sprint 7 TP=4 M=4 shared (26.0 μs): %.2fx\n", 26.0f / med);
            printf("  vs Sprint 7 EP=4 M=4 shared (63.2 μs): %.2fx\n", 63.2f / med);
            printf("  vs CUTLASS M=4 (~120 μs): %.2fx\n", 120.0f / med);
        }

        printf("\n  Overall: %s\n", all_pass ? "PASS" : "FAIL");

        delete[] hgo;

        // Cleanup
        delete[] hi; delete[] hw1; delete[] hw2;
        delete[] h_pair_eids; delete[] h_pair_tids; delete[] h_pair_wts;
        delete[] hif; delete[] his; delete[] hw1f; delete[] hw1s; delete[] hw2f; delete[] hw2s;
        delete[] hrq;
        cudaFree(dif); cudaFree(dis); cudaFree(dw1f); cudaFree(dw1s);
        cudaFree(dw2f); cudaFree(dw2s); cudaFree(deids); cudaFree(dtids); cudaFree(dewts);
        cudaFree(dout); cudaFree(dpart); cudaFree(dif2); cudaFree(dis2); cudaFree(dbar);

        return all_pass;
    }

cleanup:
    delete[] hi; delete[] hw1; delete[] hw2;
    delete[] h_pair_eids; delete[] h_pair_tids; delete[] h_pair_wts;
    delete[] hif; delete[] his; delete[] hw1f; delete[] hw1s; delete[] hw2f; delete[] hw2s;
    delete[] hrq;
    cudaFree(dif); cudaFree(dis); cudaFree(dw1f); cudaFree(dw1s);
    cudaFree(dw2f); cudaFree(dw2s); cudaFree(deids); cudaFree(dtids); cudaFree(dewts);
    cudaFree(dout); cudaFree(dpart); cudaFree(dif2); cudaFree(dis2); cudaFree(dbar);
    return false;
}

// ============================================================================
// Main: All configs
// ============================================================================
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM%d%d, %d SMs, %dKB SMEM)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           (int)(prop.sharedMemPerMultiprocessor / 1024));
    printf("\n=== Sprint 9: Independent Per-Token Routing ===\n");
    printf("MMA: mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.ue4m3\n");

    int topk = 10;
    int NE = 30;  // Need experts 0-29

    // Independent routing: each token has its OWN expert set
    const int routing_m4[4][10] = {
        {0,1,2,3,4,5,6,7,8,9},           // Token 0
        {5,6,7,8,9,10,11,12,13,14},       // Token 1 (50% overlap with token 0)
        {20,21,22,23,24,25,26,27,28,29},  // Token 2 (0% overlap)
        {0,1,2,3,4,25,26,27,28,29}        // Token 3 (50% overlap with token 0)
    };

    // For M=1 test, use token 0's routing
    const int routing_m1[1][10] = {
        {0,1,2,3,4,5,6,7,8,9}
    };

    bool all_pass = true;

    // Config 1: M=1, TP=4 — baseline comparison with Sprint 7
    printf("\n\n*** CONFIG 1: M=1, N_HALF=256, TP=4 ***\n");
    all_pass &= run_independent_test(1, topk, 256, NE, routing_m1,
        "M=1 TP=4 independent (baseline)");

    // Config 2: M=4, TP=4 — independent routing, different expert sets
    printf("\n\n*** CONFIG 2: M=4, N_HALF=256, TP=4, INDEPENDENT ROUTING ***\n");
    all_pass &= run_independent_test(4, topk, 256, NE, routing_m4,
        "M=4 TP=4 independent (per-token correct)");

    // Config 3: M=1, EP=4
    printf("\n\n*** CONFIG 3: M=1, N_HALF=1024, EP=4 ***\n");
    all_pass &= run_independent_test(1, topk, 1024, NE, routing_m1,
        "M=1 EP=4 independent");

    // Config 4: M=4, EP=4, independent routing
    printf("\n\n*** CONFIG 4: M=4, N_HALF=1024, EP=4, INDEPENDENT ROUTING ***\n");
    all_pass &= run_independent_test(4, topk, 1024, NE, routing_m4,
        "M=4 EP=4 independent (per-token correct)");

    printf("\n\n=== FINAL RESULT: %s ===\n", all_pass ? "ALL PASS" : "SOME FAILED");
    return all_pass ? 0 : 1;
}
