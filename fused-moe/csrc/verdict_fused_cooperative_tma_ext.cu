/**
 * VerdictMoE Fused Cooperative TMA Extension for vLLM — SM120 Blackwell
 *
 * Based on verdict_fused_cooperative_ext.cu (Sprint 9 production kernel).
 * Replaces scalar GMEM loads of B weight tiles with TMA bulk loads
 * (cp.async.bulk.tensor.3d) using CUtensorMap descriptors.
 *
 * Changes from cooperative_ext:
 *   - B tile loads (gate, up, w2) via TMA bulk tensor instead of scalar loads
 *   - mbarrier synchronization for TMA completion
 *   - SMEM layout rearranged: B tiles first (128-byte aligned for TMA)
 *   - No swizzle_343 on B operand reads (TMA loads linearly)
 *   - PDL (griddepcontrol.launch_dependents) at kernel exit
 *   - Kernel takes additional CUtensorMap* args for w1 and w2
 *   - Everything else preserved: independent routing, atomic barriers,
 *     scale_vec::4X, consecutive-K packing, Kahan reduction, deterministic
 *     epilogue, pair_output_f32 handling
 *
 * Build: torch JIT with -gencode=arch=compute_120a,code=sm_120a -O2 -lcuda
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda.h>       // Driver API: CUtensorMap
#include <cuda_bf16.h>
#include <stdint.h>

// ============================================================================
// Compile-time Constants
// ============================================================================
constexpr int HIDDEN      = 4096;
constexpr int BN = 64, BK = 64;
constexpr int SF_BLOCK    = 16;
constexpr int SF_PER_K    = BK / SF_BLOCK;       // 4
constexpr int NUM_WARPS   = 8;
constexpr int WARP_SIZE   = 32;
constexpr int BLOCK_SIZE  = NUM_WARPS * WARP_SIZE;  // 256
constexpr int K_PACKED    = HIDDEN / 2;              // 2048
constexpr int SF_COLS_W1  = HIDDEN / SF_BLOCK;       // 256
constexpr int SMEM_B      = BN * (BK / 2);           // 2048
constexpr int SMEM_SFB    = BN * SF_PER_K;            // 256
constexpr int PARTIALS_PER_CTA = 2 * BN;              // 128

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
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
        : "=h"(packed) : "f"(val), "f"(0.0f));
    return (uint8_t)((packed >> 8) & 0xFF);
}

__device__ __forceinline__ float d_silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ uint8_t d_quantize_e2m1(float value) {
    float av = fabsf(value);
    int sign = (value < 0.0f) ? 1 : 0, idx;
    if      (av < 0.25f) idx = 0; else if (av < 0.75f) idx = 1;
    else if (av < 1.25f) idx = 2; else if (av < 1.75f) idx = 3;
    else if (av < 2.5f)  idx = 4; else if (av < 3.5f)  idx = 5;
    else if (av < 5.0f)  idx = 6; else idx = 7;
    return (uint8_t)((sign << 3) | idx);
}

// ============================================================================
// TMA + mbarrier Device Helpers
// ============================================================================

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, uint32_t arrive_count) {
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(smem), "r"(arrive_count));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t tx_bytes) {
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;"
        :: "r"(smem), "r"(tx_bytes));
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* mbar, uint32_t parity) {
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    int done;
    do {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n"
            "selp.b32 %0, 1, 0, p;\n"
            "}\n"
            : "=r"(done) : "r"(smem), "r"(parity));
    } while (!done);
}

// TMA 3D load: copies boxDim bytes from global tensor to SMEM
__device__ __forceinline__ void tma_load_3d(
    void* smem_dst, const CUtensorMap* desc,
    int32_t c0, int32_t c1, int32_t c2,
    uint64_t* mbar)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    uint64_t desc_addr = reinterpret_cast<uint64_t>(desc);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(smem_addr), "l"(desc_addr),
           "r"(c0), "r"(c1), "r"(c2),
           "r"(mbar_addr)
        : "memory");
}

// ============================================================================
// PDL Helper
// ============================================================================
__device__ __forceinline__ void griddepcontrol_launch_dependents() {
    asm volatile("griddepcontrol.launch_dependents;");
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
// Atomic Grid Barrier (CUDA-graph safe)
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
// Independent Per-Token Fused Cooperative TMA Kernel
//
// Grid: num_pairs x num_tiles CTAs
// Each CTA: ONE (token, expert) pair.
// BF16 in -> FP4 quantize -> GEMM1 -> SwiGLU -> requant -> GEMM2 -> BF16 out
//
// B weight tiles loaded via TMA (cp.async.bulk.tensor.3d).
// A, SFA, SFB still loaded cooperatively via scalar loads.
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
verdict_fused_independent_tma_mt(
    const __nv_bfloat16* __restrict__ input_bf16,   // [M, K]
    const uint8_t* __restrict__ all_w1_fp4,         // [E, 2*n_half, K/2] (NOT used for B loads, still needed for prologue compat)
    const uint8_t* __restrict__ all_w1_sf,          // [E, 2*n_half, K/16]
    const uint8_t* __restrict__ all_w2_fp4,         // [E, K, n_half/2] (NOT used for B loads)
    const uint8_t* __restrict__ all_w2_sf,          // [E, K, n_half/16]
    const CUtensorMap* __restrict__ tma_w1,         // TMA descriptor for w1
    const CUtensorMap* __restrict__ tma_w2,         // TMA descriptor for w2
    const int*     __restrict__ expert_ids,         // [num_pairs]
    const int*     __restrict__ token_ids,          // [num_pairs]
    const float*   __restrict__ w1_alpha,           // [num_pairs]
    const float*   __restrict__ w2_alpha,           // [num_pairs]
    const float*   __restrict__ expert_wts,         // [num_pairs]
    __nv_bfloat16* __restrict__ output_bf16,        // [M, K]
    float*         __restrict__ output_f32,         // [M, K]
    float*         __restrict__ pair_output_f32,    // [num_pairs, K] per-pair GEMM2 output
    uint8_t*       __restrict__ input_fp4,          // [M, K/2]
    uint8_t*       __restrict__ input_sf,           // [M, K/16]
    float*         __restrict__ partials,           // [num_pairs * num_tiles * PARTIALS_PER_CTA]
    uint8_t*       __restrict__ gmem_inter_fp4,     // [num_pairs, n_half/2]
    uint8_t*       __restrict__ gmem_inter_sf,      // [num_pairs, n_half/SF_BLOCK]
    volatile int*  __restrict__ barrier_counter,
    int num_pairs,
    int M,
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
    const int eid      = expert_ids[pair_idx];
    const int token_id = token_ids[pair_idx];
    const float alpha1 = w1_alpha[pair_idx];
    const float alpha2 = w2_alpha[pair_idx];
    const float wt     = expert_wts[pair_idx];
    const int n_start  = n_chunk * BN;
    const int k_base   = k_group * k_per_group;

    // ================================================================
    // PROLOGUE: BF16->FP4 quantization + zero output_f32 (ALL M tokens)
    // All CTAs cooperate on this — grid-stride loop.
    // ================================================================
    {
        const int global_tid = blockIdx.x * BLOCK_SIZE + tid;
        const int total_threads = gridDim.x * BLOCK_SIZE;

        // Zero pair_output_f32 (per-pair GEMM2 outputs, summed in epilogue)
        for (int i = global_tid; i < num_pairs * HIDDEN; i += total_threads)
            pair_output_f32[i] = 0.0f;

        // BF16->NVFP4 for all M tokens
        constexpr int num_sf_groups = HIDDEN / SF_BLOCK;  // 256 per token
        const int total_sf_groups = M * num_sf_groups;
        const int half_warp_id = global_tid / 16;
        const int hw_lane = tid % 16;

        if (half_warp_id < total_sf_groups) {
            const int m = half_warp_id / num_sf_groups;
            const int g = half_warp_id % num_sf_groups;
            const int kb = g * SF_BLOCK;

            float val = __bfloat162float(input_bf16[m * HIDDEN + kb + hw_lane]);
            float aval = fabsf(val);

            float wmax = aval;
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1)
                wmax = fmaxf(wmax, __shfl_xor_sync(0xFFFFFFFF, wmax, off));

            float sf_target = fmaxf(wmax / 6.0f, 1e-30f);
            uint8_t sf_byte = d_e4m3fn_encode(sf_target);
            float actual_scale = d_e4m3fn_decode(sf_byte);
            if (actual_scale < 1e-30f) actual_scale = 1e-30f;

            uint8_t nib = d_quantize_e2m1(val / actual_scale);
            uint8_t partner_nib = (uint8_t)__shfl_xor_sync(0xFFFFFFFF, (int)nib, 1);
            if ((hw_lane & 1) == 0)
                input_fp4[m * K_PACKED + kb / 2 + hw_lane / 2] = nib | (partner_nib << 4);
            if (hw_lane == 0)
                input_sf[m * SF_COLS_W1 + g] = sf_byte;
        }
    }

    grid_barrier_atomic(barrier_counter, total_ctas, 0);

    // ================================================================
    // SMEM layout: B tiles first (128-byte aligned for TMA)
    //
    // s_B_gate:  offset 0      (2048 bytes, 128-aligned)
    // s_B_up:    offset 2048   (2048 bytes, 128-aligned)
    // s_A:       offset 4096   (32 bytes)
    // s_SFA:     offset 4128   (8 bytes, 4-aligned padded)
    // s_SFB_gate:offset 4132   (256 bytes)
    // s_SFB_up:  offset 4388   (256 bytes)
    // s_mbar:    8-byte aligned after SFB_up (8 bytes)
    // ================================================================
    extern __shared__ char smem_raw[];
    uint8_t* s_B_gate   = (uint8_t*)smem_raw;                    // offset 0
    uint8_t* s_B_up     = s_B_gate + SMEM_B;                     // offset 2048
    uint8_t* s_A        = s_B_up + SMEM_B;                       // offset 4096
    uint8_t* s_SFA      = s_A + 32;                              // offset 4128
    uint8_t* s_SFB_gate = s_SFA + ((SF_PER_K + 3) & ~3);        // offset 4132
    uint8_t* s_SFB_up   = s_SFB_gate + SMEM_SFB;                // offset 4388
    // mbarrier must be 8-byte aligned
    uint64_t* s_mbar    = (uint64_t*)(((uintptr_t)(s_SFB_up + SMEM_SFB) + 7) & ~7uLL);

    const int g = lane_id / 4;
    const int Nl = 4 * (g & 1) + (g >> 1);
    const int sn = warp_id * 8 + Nl;
    const int t0 = lane_id % 4;
    // B operand byte offset within tile (NO swizzle — linear layout from TMA)
    const int rbo = sn * (BK / 2);

    const uint8_t* w1_sf  = all_w1_sf  + (long long)eid * n2 * SF_COLS_W1;

    // Initialize mbarrier (one thread)
    if (tid == 0) {
        mbarrier_init(s_mbar, 1);
    }
    __syncthreads();

    // ================================================================
    // PHASE 1a: GEMM1 — single token, TMA B loads
    // ================================================================
    float gate_acc[4] = {0, 0, 0, 0};
    float up_acc[4]   = {0, 0, 0, 0};
    uint32_t mbar_phase = 0;

    for (int kt = 0; kt < k_tiles_per_g; kt++) {
        const int k_off = k_base + kt * BK;
        const int k_pk  = k_off / 2;
        const int k_sf  = k_off / SF_BLOCK;

        // TMA: elected thread issues bulk loads for B_gate and B_up
        if (tid == 0) {
            // Expect 2 x SMEM_B bytes (gate + up tiles)
            mbarrier_arrive_expect_tx(s_mbar, 2 * SMEM_B);
            // TMA load gate tile: coord = {k_pk, n_start, eid}
            tma_load_3d(s_B_gate, tma_w1, k_pk, n_start, eid, s_mbar);
            // TMA load up tile: coord = {k_pk, n_half + n_start, eid}
            tma_load_3d(s_B_up, tma_w1, k_pk, n_half + n_start, eid, s_mbar);
        }

        // Load this token's A data (scalar, all threads)
        for (int i = tid; i < 8; i += BLOCK_SIZE) {
            *(uint32_t*)(s_A + i * 4) =
                *(const uint32_t*)&input_fp4[token_id * K_PACKED + k_pk + i * 4];
        }

        // Load SFA (scalar)
        if (tid < SF_PER_K) {
            s_SFA[tid] = input_sf[token_id * SF_COLS_W1 + k_sf + tid];
        }

        // Load SFB (scalar, cooperative)
        for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
            int row = i / SF_PER_K, col = i % SF_PER_K;
            s_SFB_gate[i] = w1_sf[(long long)(n_start + row) * SF_COLS_W1 + k_sf + col];
        }
        for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
            int row = i / SF_PER_K, col = i % SF_PER_K;
            s_SFB_up[i] = w1_sf[(long long)(n_half + n_start + row) * SF_COLS_W1 + k_sf + col];
        }

        // Wait for A/SFA/SFB scalar stores to complete
        __syncthreads();
        // Wait for TMA B tile loads to complete
        mbarrier_wait_parity(s_mbar, mbar_phase & 1);
        mbar_phase++;

        // B operands (NO swizzle — TMA loaded linearly)
        uint32_t bg[2], bu[2];
        bg[0] = *(uint32_t*)&s_B_gate[rbo + t0 * 4];
        bg[1] = *(uint32_t*)&s_B_gate[rbo + 16 + t0 * 4];
        uint32_t sfbg = pack_sf4(&s_SFB_gate[sn * SF_PER_K]);
        bu[0] = *(uint32_t*)&s_B_up[rbo + t0 * 4];
        bu[1] = *(uint32_t*)&s_B_up[rbo + 16 + t0 * 4];
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

    grid_barrier_atomic(barrier_counter, total_ctas, 1);

    // ================================================================
    // PHASE 1b: Reduce + alpha1*SwiGLU + FP4 requant
    // ================================================================
    if (k_group == 0 && tid < BN) {
        int col = tid;

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

        gs *= alpha1;
        us *= alpha1;
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

    grid_barrier_atomic(barrier_counter, total_ctas, 2);

    // ================================================================
    // PHASE 2: GEMM2 — single token, TMA B loads for w2
    // ================================================================
    {
        const int p2_out_tiles = HIDDEN / BN;  // 64
        const uint8_t* w2_sf  = all_w2_sf  + (long long)eid * HIDDEN * sf_cols_w2;
        uint8_t* s_B2   = s_B_gate;   // reuse, still 128-aligned at offset 0
        uint8_t* s_SFB2 = s_SFB_gate;
        int p2_k_passes = n_half / BK;

        // Reinit mbarrier for Phase 2
        if (tid == 0) {
            mbarrier_init(s_mbar, 1);
        }
        __syncthreads();
        uint32_t mbar_phase2 = 0;

        for (int j_tile = tile; j_tile < p2_out_tiles; j_tile += num_tiles) {
            const int j_start = j_tile * BN;

            float p2_acc[4] = {0, 0, 0, 0};

            for (int kp = 0; kp < p2_k_passes; kp++) {
                int ko = kp * BK, kpk = ko / 2, ksf = ko / SF_BLOCK;

                // TMA: load W2 B tile
                if (tid == 0) {
                    mbarrier_arrive_expect_tx(s_mbar, SMEM_B);
                    // W2 TMA: coord = {kpk, j_start, eid}
                    tma_load_3d(s_B2, tma_w2, kpk, j_start, eid, s_mbar);
                }

                // Regular loads: intermediate A, SFA, W2 SFB
                for (int i = tid; i < 8; i += BLOCK_SIZE) {
                    *(uint32_t*)(s_A + i * 4) =
                        *(const uint32_t*)&gmem_inter_fp4[pair_idx * n_half_packed + kpk + i * 4];
                }

                if (tid < SF_PER_K) {
                    s_SFA[tid] = gmem_inter_sf[pair_idx * sf_cols_w2 + ksf + tid];
                }

                for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
                    int row = i / SF_PER_K, col = i % SF_PER_K;
                    int oc = j_start + row;
                    s_SFB2[i] = (oc < HIDDEN) ? w2_sf[(long long)oc * sf_cols_w2 + ksf + col] : 0;
                }

                // Wait for A/SFA/SFB scalar stores
                __syncthreads();
                // Wait for TMA B tile load
                mbarrier_wait_parity(s_mbar, mbar_phase2 & 1);
                mbar_phase2++;

                // B operands (NO swizzle)
                uint32_t br[2];
                br[0] = *(uint32_t*)&s_B2[rbo + t0 * 4];
                br[1] = *(uint32_t*)&s_B2[rbo + 16 + t0 * 4];
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

            // Write per-pair GEMM2 output (deterministic, no atomicAdd)
            if (lane_id < 4) {
                float scale = wt * alpha2;
                int j0 = j_start + warp_id * 8 + lane_id;
                int j1 = j0 + 4;
                if (j0 < HIDDEN) pair_output_f32[pair_idx * HIDDEN + j0] = scale * p2_acc[0];
                if (j1 < HIDDEN) pair_output_f32[pair_idx * HIDDEN + j1] = scale * p2_acc[1];
            }
        }
    }

    grid_barrier_atomic(barrier_counter, total_ctas, 3);

    // ================================================================
    // EPILOGUE: Deterministic reduction + F32 -> BF16 for ALL M tokens
    //
    // Sum per-pair outputs in fixed order (pair 0, 1, ..., topk-1) for
    // each token. Eliminates non-deterministic atomicAdd rounding that
    // caused ~1-2% MTP acceptance loss under strict rejection sampling.
    // ================================================================
    {
        const int global_tid = blockIdx.x * BLOCK_SIZE + tid;
        const int total_threads = gridDim.x * BLOCK_SIZE;
        const int topk = num_pairs / M;
        for (int i = global_tid; i < M * HIDDEN; i += total_threads) {
            int token = i / HIDDEN;
            int col   = i % HIDDEN;
            int pair_start = token * topk;
            float sum = 0.0f;
            for (int p = 0; p < topk; p++) {
                sum += pair_output_f32[(pair_start + p) * HIDDEN + col];
            }
            output_bf16[i] = __float2bfloat16(sum);
        }
    }

    // PDL: signal dependents can launch
    if (threadIdx.x == 0) {
        griddepcontrol_launch_dependents();
    }
}

// ============================================================================
// Host Forward — Independent Per-Token Routing with TMA
// ============================================================================
void verdict_cooperative_tma_forward(
    torch::Tensor input,          // [M, K] BF16
    torch::Tensor w1_fp4,         // [E, 2*N, K//2] uint8
    torch::Tensor w1_sf,          // [E, 2*N, K//16] uint8
    torch::Tensor w1_alpha,       // [num_pairs] float32
    torch::Tensor w2_fp4,         // [E, K, N//2] uint8
    torch::Tensor w2_sf,          // [E, K, N//16] uint8
    torch::Tensor w2_alpha,       // [num_pairs] float32
    torch::Tensor output,         // [M, K] BF16
    torch::Tensor expert_ids,     // [num_pairs] int32
    torch::Tensor token_ids,      // [num_pairs] int32
    torch::Tensor expert_weights, // [num_pairs] float32
    torch::Tensor output_f32,     // [M * K] float32 (unused, kept for compat)
    torch::Tensor pair_output_f32,// [num_pairs * K] float32 — per-pair GEMM2 output
    torch::Tensor input_fp4_buf,  // [M * K/2] uint8
    torch::Tensor input_sf_buf,   // [M * K/16] uint8
    torch::Tensor partials_buf,   // [num_pairs * num_tiles * 128] float32
    torch::Tensor inter_fp4_buf,  // [num_pairs * N_HALF/2] uint8
    torch::Tensor inter_sf_buf,   // [num_pairs * N_HALF/16] uint8
    torch::Tensor barrier_buf,    // [1] int32
    int K, int N_half, int num_pairs,
    int64_t tma_w1_ptr,           // CUtensorMap* for w1 (as int64)
    int64_t tma_w2_ptr)           // CUtensorMap* for w2 (as int64)
{
    auto stream = c10::cuda::getCurrentCUDAStream();
    int M = input.size(0);

    TORCH_CHECK(K == HIDDEN, "K must be ", HIDDEN, " got ", K);
    TORCH_CHECK(M >= 1, "M must be >= 1, got ", M);

    // Compute runtime grid parameters
    int tiles_n = N_half / BN;
    int total_k_tiles = HIDDEN / BK;  // 64
    int k_groups = std::max(1, 640 / (num_pairs * tiles_n));
    while (total_k_tiles % k_groups != 0 && k_groups > 1) k_groups--;
    int num_tiles = tiles_n * k_groups;
    int grid = num_pairs * num_tiles;

    TORCH_CHECK(grid <= 752,
                "Grid size ", grid, " exceeds 752 max concurrent CTAs on SM120");

    // SMEM: B_gate(2048) + B_up(2048) + A(32) + SFA(pad4) + SFB_gate(256) + SFB_up(256) + mbar(8) + pad
    int smem_size = 2 * SMEM_B + 32
                  + ((SF_PER_K + 3) & ~3) + 2 * SMEM_SFB + 16 + 128;

    cudaFuncSetAttribute(verdict_fused_independent_tma_mt,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    // Zero barrier counter
    cudaMemsetAsync(barrier_buf.data_ptr(), 0, sizeof(int), stream);

    // Cast TMA descriptor pointers
    const CUtensorMap* tma_w1 = reinterpret_cast<const CUtensorMap*>(tma_w1_ptr);
    const CUtensorMap* tma_w2 = reinterpret_cast<const CUtensorMap*>(tma_w2_ptr);

    // ONE kernel launch — independent per-token routing with TMA
    verdict_fused_independent_tma_mt<<<grid, BLOCK_SIZE, smem_size, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const uint8_t*>(w1_fp4.data_ptr()),
        reinterpret_cast<const uint8_t*>(w1_sf.data_ptr()),
        reinterpret_cast<const uint8_t*>(w2_fp4.data_ptr()),
        reinterpret_cast<const uint8_t*>(w2_sf.data_ptr()),
        tma_w1,
        tma_w2,
        reinterpret_cast<const int*>(expert_ids.data_ptr()),
        reinterpret_cast<const int*>(token_ids.data_ptr()),
        reinterpret_cast<const float*>(w1_alpha.data_ptr()),
        reinterpret_cast<const float*>(w2_alpha.data_ptr()),
        reinterpret_cast<const float*>(expert_weights.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        reinterpret_cast<float*>(output_f32.data_ptr()),
        reinterpret_cast<float*>(pair_output_f32.data_ptr()),
        reinterpret_cast<uint8_t*>(input_fp4_buf.data_ptr()),
        reinterpret_cast<uint8_t*>(input_sf_buf.data_ptr()),
        reinterpret_cast<float*>(partials_buf.data_ptr()),
        reinterpret_cast<uint8_t*>(inter_fp4_buf.data_ptr()),
        reinterpret_cast<uint8_t*>(inter_sf_buf.data_ptr()),
        reinterpret_cast<volatile int*>(barrier_buf.data_ptr()),
        num_pairs,
        M,
        N_half,
        k_groups);
}

// ============================================================================
// Non-TMA forward (identical to verdict_fused_cooperative_ext.cu)
// Kept here so both paths can be in one extension module.
// ============================================================================

// swizzle_343 needed for the non-TMA path
__device__ __forceinline__ uint32_t swizzle_343(uint32_t off) {
    return off ^ ((off >> 3) & 0x70u);
}

__global__ void __launch_bounds__(BLOCK_SIZE, 4)
verdict_fused_independent_mt(
    const __nv_bfloat16* __restrict__ input_bf16,
    const uint8_t* __restrict__ all_w1_fp4,
    const uint8_t* __restrict__ all_w1_sf,
    const uint8_t* __restrict__ all_w2_fp4,
    const uint8_t* __restrict__ all_w2_sf,
    const int*     __restrict__ expert_ids,
    const int*     __restrict__ token_ids,
    const float*   __restrict__ w1_alpha,
    const float*   __restrict__ w2_alpha,
    const float*   __restrict__ expert_wts,
    __nv_bfloat16* __restrict__ output_bf16,
    float*         __restrict__ output_f32,
    float*         __restrict__ pair_output_f32,
    uint8_t*       __restrict__ input_fp4,
    uint8_t*       __restrict__ input_sf,
    float*         __restrict__ partials,
    uint8_t*       __restrict__ gmem_inter_fp4,
    uint8_t*       __restrict__ gmem_inter_sf,
    volatile int*  __restrict__ barrier_counter,
    int num_pairs,
    int M,
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
    const int eid      = expert_ids[pair_idx];
    const int token_id = token_ids[pair_idx];
    const float alpha1 = w1_alpha[pair_idx];
    const float alpha2 = w2_alpha[pair_idx];
    const float wt     = expert_wts[pair_idx];
    const int n_start  = n_chunk * BN;
    const int k_base   = k_group * k_per_group;

    // PROLOGUE
    {
        const int global_tid = blockIdx.x * BLOCK_SIZE + tid;
        const int total_threads = gridDim.x * BLOCK_SIZE;
        for (int i = global_tid; i < num_pairs * HIDDEN; i += total_threads)
            pair_output_f32[i] = 0.0f;
        constexpr int num_sf_groups = HIDDEN / SF_BLOCK;
        const int total_sf_groups = M * num_sf_groups;
        const int half_warp_id = global_tid / 16;
        const int hw_lane = tid % 16;
        if (half_warp_id < total_sf_groups) {
            const int m = half_warp_id / num_sf_groups;
            const int g = half_warp_id % num_sf_groups;
            const int kb = g * SF_BLOCK;
            float val = __bfloat162float(input_bf16[m * HIDDEN + kb + hw_lane]);
            float aval = fabsf(val);
            float wmax = aval;
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1)
                wmax = fmaxf(wmax, __shfl_xor_sync(0xFFFFFFFF, wmax, off));
            float sf_target = fmaxf(wmax / 6.0f, 1e-30f);
            uint8_t sf_byte = d_e4m3fn_encode(sf_target);
            float actual_scale = d_e4m3fn_decode(sf_byte);
            if (actual_scale < 1e-30f) actual_scale = 1e-30f;
            uint8_t nib = d_quantize_e2m1(val / actual_scale);
            uint8_t partner_nib = (uint8_t)__shfl_xor_sync(0xFFFFFFFF, (int)nib, 1);
            if ((hw_lane & 1) == 0)
                input_fp4[m * K_PACKED + kb / 2 + hw_lane / 2] = nib | (partner_nib << 4);
            if (hw_lane == 0)
                input_sf[m * SF_COLS_W1 + g] = sf_byte;
        }
    }
    grid_barrier_atomic(barrier_counter, total_ctas, 0);

    extern __shared__ char smem_raw2[];
    uint8_t* s_A        = (uint8_t*)smem_raw2;
    uint8_t* s_B_gate   = s_A + 32;
    uint8_t* s_B_up     = s_B_gate + SMEM_B;
    uint8_t* s_SFA      = s_B_up + SMEM_B;
    uint8_t* s_SFB_gate = s_SFA + ((SF_PER_K + 3) & ~3);
    uint8_t* s_SFB_up   = s_SFB_gate + SMEM_SFB;

    const int g2 = lane_id / 4;
    const int Nl2 = 4 * (g2 & 1) + (g2 >> 1);
    const int sn2 = warp_id * 8 + Nl2;
    const int t02 = lane_id % 4;
    const int rbo2 = sn2 * (BK / 2);

    const uint8_t* w1_fp4 = all_w1_fp4 + (long long)eid * n2 * K_PACKED;
    const uint8_t* w1_sf  = all_w1_sf  + (long long)eid * n2 * SF_COLS_W1;

    float gate_acc[4] = {0, 0, 0, 0};
    float up_acc[4]   = {0, 0, 0, 0};
    for (int kt = 0; kt < k_tiles_per_g; kt++) {
        const int k_off = k_base + kt * BK;
        const int k_pk  = k_off / 2;
        const int k_sf  = k_off / SF_BLOCK;
        for (int i = tid; i < 8; i += BLOCK_SIZE)
            *(uint32_t*)(s_A + i * 4) = *(const uint32_t*)&input_fp4[token_id * K_PACKED + k_pk + i * 4];
        for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
            int boff = i * 4; int row = boff / (BK / 2), col = boff % (BK / 2);
            *(uint32_t*)&s_B_gate[swizzle_343(boff)] = *(const uint32_t*)&w1_fp4[(long long)(n_start + row) * K_PACKED + k_pk + col];
        }
        for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
            int boff = i * 4; int row = boff / (BK / 2), col = boff % (BK / 2);
            *(uint32_t*)&s_B_up[swizzle_343(boff)] = *(const uint32_t*)&w1_fp4[(long long)(n_half + n_start + row) * K_PACKED + k_pk + col];
        }
        if (tid < SF_PER_K) s_SFA[tid] = input_sf[token_id * SF_COLS_W1 + k_sf + tid];
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
        bg[0] = *(uint32_t*)&s_B_gate[swizzle_343(rbo2 + t02 * 4)];
        bg[1] = *(uint32_t*)&s_B_gate[swizzle_343(rbo2 + 16 + t02 * 4)];
        uint32_t sfbg = pack_sf4(&s_SFB_gate[sn2 * SF_PER_K]);
        bu[0] = *(uint32_t*)&s_B_up[swizzle_343(rbo2 + t02 * 4)];
        bu[1] = *(uint32_t*)&s_B_up[swizzle_343(rbo2 + 16 + t02 * 4)];
        uint32_t sfbu = pack_sf4(&s_SFB_up[sn2 * SF_PER_K]);
        uint32_t a[4] = {0, 0, 0, 0};
        if (lane_id / 4 == 0) { a[0] = *(uint32_t*)(s_A + t02 * 4); a[2] = *(uint32_t*)(s_A + 16 + t02 * 4); }
        uint32_t sfa_pk = pack_sf4(s_SFA);
        mma_nvf4_e4m3_m16n8k64(gate_acc, a, bg, gate_acc, sfa_pk, sfbg);
        mma_nvf4_e4m3_m16n8k64(up_acc, a, bu, up_acc, sfa_pk, sfbu);
        __syncthreads();
    }
    if (lane_id < 4) {
        long long pb = (long long)(pair_idx * num_tiles + tile) * PARTIALS_PER_CTA;
        int c0 = warp_id * 8 + lane_id, c1 = c0 + 4;
        partials[pb + c0] = gate_acc[0]; partials[pb + c1] = gate_acc[1];
        partials[pb + BN + c0] = up_acc[0]; partials[pb + BN + c1] = up_acc[1];
    }
    grid_barrier_atomic(barrier_counter, total_ctas, 1);

    if (k_group == 0 && tid < BN) {
        int col = tid;
        float gs = 0, us = 0, gs_c = 0, us_c = 0;
        for (int kg = 0; kg < k_groups; kg++) {
            int partner_tile = n_chunk * k_groups + kg;
            long long base = (long long)(pair_idx * num_tiles + partner_tile) * PARTIALS_PER_CTA;
            float g_y = partials[base + col] - gs_c; float g_t = gs + g_y; gs_c = (g_t - gs) - g_y; gs = g_t;
            float u_y = partials[base + BN + col] - us_c; float u_t = us + u_y; us_c = (u_t - us) - u_y; us = u_t;
        }
        gs *= alpha1; us *= alpha1;
        float sw_val = us * d_silu(gs);
        float abs_sw = fabsf(sw_val); float gm = abs_sw;
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 1));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 2));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 4));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 8));
        float st = fmaxf(gm / 6.0f, 1e-30f);
        uint8_t sf_enc = d_e4m3fn_encode(st);
        float as2 = d_e4m3fn_decode(sf_enc); if (as2 < 1e-30f) as2 = 1e-30f;
        uint8_t nib = d_quantize_e2m1(sw_val / as2);
        uint32_t nib32 = (uint32_t)nib;
        uint32_t neighbor32 = __shfl_down_sync(0xFFFFFFFF, nib32, 1);
        if (col % 2 == 0) gmem_inter_fp4[pair_idx * n_half_packed + (n_start + col) / 2] = (uint8_t)(nib32 | (neighbor32 << 4));
        if (col % SF_BLOCK == 0) gmem_inter_sf[pair_idx * sf_cols_w2 + (n_start + col) / SF_BLOCK] = sf_enc;
    }
    grid_barrier_atomic(barrier_counter, total_ctas, 2);

    {
        const int p2_out_tiles = HIDDEN / BN;
        const uint8_t* w2_fp4_p = all_w2_fp4 + (long long)eid * HIDDEN * n_half_packed;
        const uint8_t* w2_sf_p  = all_w2_sf  + (long long)eid * HIDDEN * sf_cols_w2;
        uint8_t* s_B22 = s_B_gate; uint8_t* s_SFB22 = s_SFB_gate;
        int p2_k_passes = n_half / BK;
        for (int j_tile = tile; j_tile < p2_out_tiles; j_tile += num_tiles) {
            const int j_start = j_tile * BN;
            float p2_acc[4] = {0, 0, 0, 0};
            for (int kp = 0; kp < p2_k_passes; kp++) {
                int ko = kp * BK, kpk = ko / 2, ksf = ko / SF_BLOCK;
                for (int i = tid; i < 8; i += BLOCK_SIZE)
                    *(uint32_t*)(s_A + i * 4) = *(const uint32_t*)&gmem_inter_fp4[pair_idx * n_half_packed + kpk + i * 4];
                for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
                    int boff = i * 4; int row = boff / (BK / 2), col = boff % (BK / 2); int oc = j_start + row;
                    *(uint32_t*)&s_B22[swizzle_343(boff)] = (oc < HIDDEN) ? *(const uint32_t*)&w2_fp4_p[(long long)oc * n_half_packed + kpk + col] : 0u;
                }
                if (tid < SF_PER_K) s_SFA[tid] = gmem_inter_sf[pair_idx * sf_cols_w2 + ksf + tid];
                for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
                    int row = i / SF_PER_K, col = i % SF_PER_K; int oc = j_start + row;
                    s_SFB22[i] = (oc < HIDDEN) ? w2_sf_p[(long long)oc * sf_cols_w2 + ksf + col] : 0;
                }
                __syncthreads();
                uint32_t br[2];
                br[0] = *(uint32_t*)&s_B22[swizzle_343(rbo2 + t02 * 4)];
                br[1] = *(uint32_t*)&s_B22[swizzle_343(rbo2 + 16 + t02 * 4)];
                uint32_t sfbp = pack_sf4(&s_SFB22[sn2 * SF_PER_K]);
                uint32_t ar[4] = {0, 0, 0, 0};
                if (lane_id / 4 == 0) { ar[0] = *(uint32_t*)(s_A + t02 * 4); ar[2] = *(uint32_t*)(s_A + 16 + t02 * 4); }
                uint32_t sfap = pack_sf4(s_SFA);
                mma_nvf4_e4m3_m16n8k64(p2_acc, ar, br, p2_acc, sfap, sfbp);
                __syncthreads();
            }
            if (lane_id < 4) {
                float scale = wt * alpha2;
                int j0 = j_start + warp_id * 8 + lane_id, j1 = j0 + 4;
                if (j0 < HIDDEN) pair_output_f32[pair_idx * HIDDEN + j0] = scale * p2_acc[0];
                if (j1 < HIDDEN) pair_output_f32[pair_idx * HIDDEN + j1] = scale * p2_acc[1];
            }
        }
    }
    grid_barrier_atomic(barrier_counter, total_ctas, 3);

    {
        const int global_tid = blockIdx.x * BLOCK_SIZE + tid;
        const int total_threads = gridDim.x * BLOCK_SIZE;
        const int topk = num_pairs / M;
        for (int i = global_tid; i < M * HIDDEN; i += total_threads) {
            int token = i / HIDDEN, col = i % HIDDEN;
            int pair_start = token * topk;
            float sum = 0.0f;
            for (int p = 0; p < topk; p++) sum += pair_output_f32[(pair_start + p) * HIDDEN + col];
            output_bf16[i] = __float2bfloat16(sum);
        }
    }
}

void verdict_cooperative_forward(
    torch::Tensor input,
    torch::Tensor w1_fp4,
    torch::Tensor w1_sf,
    torch::Tensor w1_alpha,
    torch::Tensor w2_fp4,
    torch::Tensor w2_sf,
    torch::Tensor w2_alpha,
    torch::Tensor output,
    torch::Tensor expert_ids,
    torch::Tensor token_ids,
    torch::Tensor expert_weights,
    torch::Tensor output_f32,
    torch::Tensor pair_output_f32,
    torch::Tensor input_fp4_buf,
    torch::Tensor input_sf_buf,
    torch::Tensor partials_buf,
    torch::Tensor inter_fp4_buf,
    torch::Tensor inter_sf_buf,
    torch::Tensor barrier_buf,
    int K, int N_half, int num_pairs)
{
    auto stream = c10::cuda::getCurrentCUDAStream();
    int M = input.size(0);

    TORCH_CHECK(K == HIDDEN, "K must be ", HIDDEN, " got ", K);
    TORCH_CHECK(M >= 1, "M must be >= 1, got ", M);

    int tiles_n = N_half / BN;
    int total_k_tiles = HIDDEN / BK;
    int k_groups = std::max(1, 640 / (num_pairs * tiles_n));
    while (total_k_tiles % k_groups != 0 && k_groups > 1) k_groups--;
    int num_tiles = tiles_n * k_groups;
    int grid = num_pairs * num_tiles;

    TORCH_CHECK(grid <= 752, "Grid size ", grid, " exceeds 752 max concurrent CTAs on SM120");

    int smem_size = 32 + 2 * SMEM_B + ((SF_PER_K + 3) & ~3) + 2 * SMEM_SFB + 128;

    cudaFuncSetAttribute(verdict_fused_independent_mt,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    cudaMemsetAsync(barrier_buf.data_ptr(), 0, sizeof(int), stream);

    verdict_fused_independent_mt<<<grid, BLOCK_SIZE, smem_size, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const uint8_t*>(w1_fp4.data_ptr()),
        reinterpret_cast<const uint8_t*>(w1_sf.data_ptr()),
        reinterpret_cast<const uint8_t*>(w2_fp4.data_ptr()),
        reinterpret_cast<const uint8_t*>(w2_sf.data_ptr()),
        reinterpret_cast<const int*>(expert_ids.data_ptr()),
        reinterpret_cast<const int*>(token_ids.data_ptr()),
        reinterpret_cast<const float*>(w1_alpha.data_ptr()),
        reinterpret_cast<const float*>(w2_alpha.data_ptr()),
        reinterpret_cast<const float*>(expert_weights.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        reinterpret_cast<float*>(output_f32.data_ptr()),
        reinterpret_cast<float*>(pair_output_f32.data_ptr()),
        reinterpret_cast<uint8_t*>(input_fp4_buf.data_ptr()),
        reinterpret_cast<uint8_t*>(input_sf_buf.data_ptr()),
        reinterpret_cast<float*>(partials_buf.data_ptr()),
        reinterpret_cast<uint8_t*>(inter_fp4_buf.data_ptr()),
        reinterpret_cast<uint8_t*>(inter_sf_buf.data_ptr()),
        reinterpret_cast<volatile int*>(barrier_buf.data_ptr()),
        num_pairs,
        M,
        N_half,
        k_groups);
}

// ============================================================================
// Host: create TMA descriptor for weight tensor
// Layout: [experts, rows, K_bytes] (3D, row-major)
// Returns device pointer to CUtensorMap (caller must cudaFree)
// ============================================================================
int64_t create_tma_weight_descriptor(
    torch::Tensor weight_fp4,  // [E, N, K/2] or [E, 2*N, K/2] uint8
    int box_k_bytes,           // BK/2 = 32
    int box_n)                 // BN = 64
{
    TORCH_CHECK(weight_fp4.is_cuda(), "weight must be on GPU");
    TORCH_CHECK(weight_fp4.dim() == 3, "weight must be 3D [E, rows, K_bytes]");

    void* dev_ptr = weight_fp4.data_ptr();
    uint64_t K_bytes = weight_fp4.size(2);  // innermost dim (K/2)
    uint64_t rows    = weight_fp4.size(1);  // rows per expert
    uint64_t experts = weight_fp4.size(0);  // number of experts

    CUtensorMap map __attribute__((aligned(64)));
    cuuint64_t globalDim[3]     = {K_bytes, rows, experts};
    cuuint64_t globalStrides[2] = {K_bytes, K_bytes * rows};
    cuuint32_t boxDim[3]        = {(uint32_t)box_k_bytes, (uint32_t)box_n, 1};
    cuuint32_t elemStrides[3]   = {1, 1, 1};

    CUresult err = cuTensorMapEncodeTiled(
        &map,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        3,
        dev_ptr,
        globalDim,
        globalStrides,
        boxDim,
        elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    TORCH_CHECK(err == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", err);

    CUtensorMap* d_map;
    cudaMalloc(&d_map, sizeof(CUtensorMap));
    cudaMemcpy(d_map, &map, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

    return reinterpret_cast<int64_t>(d_map);
}

void free_tma_descriptor(int64_t ptr) {
    cudaFree(reinterpret_cast<void*>(ptr));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &verdict_cooperative_forward,
          "VerdictMoE fused independent per-token routing forward (SM120)");
    m.def("forward_tma", &verdict_cooperative_tma_forward,
          "VerdictMoE fused independent per-token routing forward with TMA (SM120)");
    m.def("create_tma_weight_descriptor", &create_tma_weight_descriptor,
          "Create TMA descriptor for weight tensor (returns device ptr as int64)");
    m.def("free_tma_descriptor", &free_tma_descriptor,
          "Free TMA descriptor device memory");
}
