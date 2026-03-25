/**
 * verdict_fused_cooperative_e4m3.cu — Sprint 6 Task 0
 *
 * Fused cooperative MMA kernel: GEMM1 → SwiGLU → E4M3 requant → GEMM2
 *
 * KEY OPTIMIZATIONS (vs Sprint 5 baseline @ 116 μs):
 *
 *  1. VECTORIZED PACKING: 2 uint32_t SMEM loads per operand replaces
 *     16 scalar get_nibble_swz calls. ~40 μs savings.
 *
 *  2. HYBRID K×N-DISTRIBUTED Phase 1: Grid = 640 CTAs (same occupancy),
 *     but each CTA covers 1 N-chunk (BN=64) × 1 K-group (16 K-tiles).
 *     Phase 1b reduces 4 partials instead of 64, using 160 leader CTAs
 *     instead of 10. Phase 1b drops from ~21 μs to ~0.2 μs.
 *
 *  3. VECTORIZED GMEM→SMEM: uint32_t coalesced loads.
 *
 *  4. Gate + Up B tiles loaded simultaneously (2 syncs/K-tile vs 4).
 *
 * Consecutive-K packing, scale_vec::4X with native E4M3FN, atomic barriers.
 * CUDA-graph safe: no cooperative_groups, no -rdc=true.
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a \
 *     --expt-relaxed-constexpr --compiler-options '-fPIC' \
 *     -o verdict_fused_cooperative_e4m3 verdict_fused_cooperative_e4m3.cu
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
// Constants (Qwen3.5-397B-A17B, EP=4)
// ============================================================================
static constexpr int HIDDEN      = 4096;
static constexpr int N_HALF      = 1024;
static constexpr int N2          = 2 * N_HALF;
static constexpr int NUM_ACTIVE  = 10;

static constexpr int BM = 16;
static constexpr int BN = 64;
static constexpr int BK = 64;

static constexpr int SF_BLOCK   = 16;
static constexpr int SF_PER_K   = BK / SF_BLOCK;       // 4

static constexpr int NUM_WARPS  = 8;
static constexpr int WARP_SIZE  = 32;
static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;  // 256

static constexpr int K_PACKED      = HIDDEN / 2;
static constexpr int N_HALF_PACKED = N_HALF / 2;
static constexpr int SF_COLS_W1    = HIDDEN / SF_BLOCK;   // 256
static constexpr int SF_COLS_W2    = N_HALF / SF_BLOCK;   // 64

// Hybrid K×N distribution for Phase 1
static constexpr int TILES_N       = N_HALF / BN;          // 16 N-chunks
static constexpr int K_GROUPS      = 4;                    // K-partitions per N-chunk
static constexpr int NUM_TILES     = TILES_N * K_GROUPS;   // 64 (same grid size!)
static constexpr int K_TILES_PER_G = (HIDDEN / BK) / K_GROUPS;  // 16 K-tiles per group
static constexpr int K_PER_GROUP   = K_TILES_PER_G * BK;        // 1024

// Phase 2: same as before — each CTA handles 1 output tile of BN=64
static constexpr int TILES_P2     = HIDDEN / BN;           // 64

// SMEM layout: Phase 1 needs gate+up B tiles simultaneously
static constexpr int SMEM_A       = BM * (BK / 2);         // 512
static constexpr int SMEM_B       = BN * (BK / 2);         // 2048
static constexpr int SMEM_SFA_PAD = 16;
static constexpr int SMEM_SFB     = BN * SF_PER_K;         // 256
// Phase 1: A + B_gate + B_up + SFA + SFB_gate + SFB_up
static constexpr int SMEM_TOTAL   = SMEM_A + 2*SMEM_B + SMEM_SFA_PAD + 2*SMEM_SFB + 128;

// Partials buffer: much smaller than Sprint 5 (320 KB vs 5.2 MB)
static constexpr int PARTIALS_PER_CTA = 2 * BN;            // 128 floats (gate + up)

static const float E2M1_TABLE[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// ============================================================================
// E4M3FN Host Utilities
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

// ============================================================================
// Scale factor pack
// ============================================================================
__device__ __forceinline__ uint32_t pack_sf4(const uint8_t* sf) {
    return (uint32_t)sf[0] | ((uint32_t)sf[1]<<8) | ((uint32_t)sf[2]<<16) | ((uint32_t)sf[3]<<24);
}

// ============================================================================
// HYBRID K×N-DISTRIBUTED FUSED COOPERATIVE KERNEL
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
verdict_consec_k_fused(
    const uint8_t* __restrict__ input_fp4,
    const uint8_t* __restrict__ input_sf,
    const uint8_t* __restrict__ all_w1_fp4,
    const uint8_t* __restrict__ all_w1_sf,
    const uint8_t* __restrict__ all_w2_fp4,
    const uint8_t* __restrict__ all_w2_sf,
    const int*     __restrict__ expert_ids,
    const float*   __restrict__ expert_wts,
    float*         __restrict__ output,
    float*         __restrict__ partials,
    uint8_t*       __restrict__ gmem_inter_fp4,
    uint8_t*       __restrict__ gmem_inter_sf,
    volatile int*  __restrict__ barrier_counter,
    int num_active)
{
    const int eidx = blockIdx.x / NUM_TILES;
    const int tile = blockIdx.x % NUM_TILES;
    const int n_chunk = tile / K_GROUPS;    // 0..15: which N-chunk
    const int k_group = tile % K_GROUPS;    // 0..3: which K-partition
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int total_ctas = num_active * NUM_TILES;
    if (eidx >= num_active) return;

    const int eid = expert_ids[eidx];
    const float wt = expert_wts[eidx];
    const int n_start = n_chunk * BN;
    const int k_base = k_group * K_PER_GROUP;  // K offset for this group

    extern __shared__ char smem_raw[];
    uint8_t* s_A        = (uint8_t*)smem_raw;
    uint8_t* s_B_gate   = s_A + SMEM_A;
    uint8_t* s_B_up     = s_B_gate + SMEM_B;
    uint8_t* s_SFA      = s_B_up + SMEM_B;
    uint8_t* s_SFB_gate = s_SFA + SMEM_SFA_PAD;
    uint8_t* s_SFB_up   = s_SFB_gate + SMEM_SFB;

    // Per-warp MMA column mapping
    const int g = lane_id / 4;
    const int Nl = 4 * (g & 1) + (g >> 1);
    const int sn = warp_id * 8 + Nl;
    const int t0 = lane_id % 4;
    const int rbo = sn * (BK / 2);

    const uint8_t* w1_fp4 = all_w1_fp4 + (long long)eid * N2 * K_PACKED;
    const uint8_t* w1_sf  = all_w1_sf  + (long long)eid * N2 * SF_COLS_W1;

    // ================================================================
    // PHASE 1a: GEMM1 — Hybrid K×N-distributed
    //   Each CTA covers BN=64 N-columns × 16 K-tiles (1024 K-elements)
    //   Gate and Up B loaded simultaneously. 2 syncs per K-tile.
    // ================================================================
    float gate_acc[4] = {0,0,0,0};
    float up_acc[4] = {0,0,0,0};

    for (int kt = 0; kt < K_TILES_PER_G; kt++) {
        const int k_off = k_base + kt * BK;
        const int k_pk = k_off / 2;
        const int k_sf = k_off / SF_BLOCK;

        // --- Cooperative load: A + gate_B + up_B + scales ---
        // A: vectorized uint32_t
        for (int i = tid; i < SMEM_A / 4; i += BLOCK_SIZE) {
            int boff = i * 4;
            int row = boff / (BK / 2), col = boff % (BK / 2);
            *(uint32_t*)&s_A[swizzle_343(boff)] =
                (row == 0) ? *(const uint32_t*)&input_fp4[k_pk + col] : 0u;
        }

        // Gate B: 2048 bytes
        for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
            int boff = i * 4;
            int row = boff / (BK / 2), col = boff % (BK / 2);
            *(uint32_t*)&s_B_gate[swizzle_343(boff)] =
                *(const uint32_t*)&w1_fp4[(long long)(n_start + row) * K_PACKED + k_pk + col];
        }

        // Up B: 2048 bytes
        for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
            int boff = i * 4;
            int row = boff / (BK / 2), col = boff % (BK / 2);
            *(uint32_t*)&s_B_up[swizzle_343(boff)] =
                *(const uint32_t*)&w1_fp4[(long long)(N_HALF + n_start + row) * K_PACKED + k_pk + col];
        }

        // SFA
        if (tid < SF_PER_K) s_SFA[tid] = input_sf[k_sf + tid];

        // SFB gate + up
        for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
            int row = i / SF_PER_K, col = i % SF_PER_K;
            s_SFB_gate[i] = w1_sf[(long long)(n_start + row) * SF_COLS_W1 + k_sf + col];
        }
        for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
            int row = i / SF_PER_K, col = i % SF_PER_K;
            s_SFB_up[i] = w1_sf[(long long)(N_HALF + n_start + row) * SF_COLS_W1 + k_sf + col];
        }

        __syncthreads();

        // --- Vectorized pack + MMA ---
        uint32_t a[4] = {0, 0, 0, 0};
        if (lane_id / 4 == 0) {
            a[0] = *(uint32_t*)&s_A[swizzle_343(t0 * 4)];
            a[2] = *(uint32_t*)&s_A[swizzle_343(16 + t0 * 4)];
        }
        uint32_t sfa_pk = pack_sf4(s_SFA);

        // Gate MMA
        uint32_t bg[2];
        bg[0] = *(uint32_t*)&s_B_gate[swizzle_343(rbo + t0 * 4)];
        bg[1] = *(uint32_t*)&s_B_gate[swizzle_343(rbo + 16 + t0 * 4)];
        uint32_t sfbg = pack_sf4(&s_SFB_gate[sn * SF_PER_K]);
        mma_nvf4_e4m3_m16n8k64(gate_acc, a, bg, gate_acc, sfa_pk, sfbg);

        // Up MMA (reuse a and sfa_pk)
        uint32_t bu[2];
        bu[0] = *(uint32_t*)&s_B_up[swizzle_343(rbo + t0 * 4)];
        bu[1] = *(uint32_t*)&s_B_up[swizzle_343(rbo + 16 + t0 * 4)];
        uint32_t sfbu = pack_sf4(&s_SFB_up[sn * SF_PER_K]);
        mma_nvf4_e4m3_m16n8k64(up_acc, a, bu, up_acc, sfa_pk, sfbu);

        __syncthreads();
    }

    // Write partial gate[64] + up[64] to partials buffer
    // Layout: partials[eidx * NUM_TILES * 128 + tile * 128 + col]
    if (lane_id < 4) {
        long long pb = (long long)eidx * NUM_TILES * PARTIALS_PER_CTA + (long long)tile * PARTIALS_PER_CTA;
        int c0 = warp_id * 8 + lane_id;
        int c1 = c0 + 4;
        partials[pb + c0]      = gate_acc[0];
        partials[pb + c1]      = gate_acc[1];
        partials[pb + BN + c0] = up_acc[0];
        partials[pb + BN + c1] = up_acc[1];
    }

    grid_barrier_atomic(barrier_counter, total_ctas, 0);

    // ================================================================
    // PHASE 1b: Reduce 4 K-group partials + SwiGLU + FP4 requant
    //   160 leader CTAs (k_group==0), each handles 64 columns.
    //   4-way reduction instead of 64-way. Register-only with shfl.
    // ================================================================
    if (k_group == 0 && tid < BN) {
        int col = tid;
        float gs = 0, us = 0;

        // Sum 4 K-group partials
        #pragma unroll
        for (int kg = 0; kg < K_GROUPS; kg++) {
            int partner_tile = n_chunk * K_GROUPS + kg;
            long long base = (long long)eidx * NUM_TILES * PARTIALS_PER_CTA
                           + (long long)partner_tile * PARTIALS_PER_CTA;
            gs += partials[base + col];
            us += partials[base + BN + col];
        }

        float sw_val = us * d_silu(gs);

        // Group max across SF_BLOCK=16 columns via warp shuffle
        // Thread layout: tid 0-31 in warp 0, tid 32-63 in warp 1
        // Groups: {0-15}, {16-31}, {32-47}, {48-63} — each within one warp half
        float abs_sw = fabsf(sw_val);
        float gm = abs_sw;
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 1));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 2));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 4));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 8));
        // gm now has max of 16 consecutive threads (within each warp's 16-thread halves)

        float st = fmaxf(gm / 6.0f, 1e-30f);
        uint8_t sf_enc = d_e4m3fn_encode(st);
        float as = d_e4m3fn_decode(sf_enc);
        if (as < 1e-30f) as = 1e-30f;

        uint8_t nib = d_quantize_e2m1(sw_val / as);

        // Pack nibble pairs: even threads write byte (low=self, high=neighbor)
        uint32_t nib32 = (uint32_t)nib;
        uint32_t neighbor32 = __shfl_down_sync(0xFFFFFFFF, nib32, 1);
        if (col % 2 == 0) {
            gmem_inter_fp4[eidx * N_HALF_PACKED + (n_start + col) / 2] =
                (uint8_t)(nib32 | (neighbor32 << 4));
        }

        // Write scale factor (one per 16-column group)
        if (col % SF_BLOCK == 0) {
            gmem_inter_sf[eidx * (N_HALF / SF_BLOCK) + (n_start + col) / SF_BLOCK] = sf_enc;
        }
    }

    grid_barrier_atomic(barrier_counter, total_ctas, 1);

    // ================================================================
    // PHASE 2: GEMM2 — N-distributed, 16 K-passes
    //   All 640 CTAs, each handles 1 output tile of BN=64.
    //   Reuse SMEM: s_A, s_B_gate (as s_B), s_SFA, s_SFB_gate (as s_SFB).
    // ================================================================
    {
        const int j_start = tile * BN;  // tile 0..63, maps to output columns
        const uint8_t* w2_fp4 = all_w2_fp4 + (long long)eid * HIDDEN * N_HALF_PACKED;
        const uint8_t* w2_sf  = all_w2_sf  + (long long)eid * HIDDEN * SF_COLS_W2;

        // Reuse SMEM sections
        uint8_t* s_B2   = s_B_gate;
        uint8_t* s_SFB2 = s_SFB_gate;

        float acc[4] = {0,0,0,0};

        for (int kp = 0; kp < N_HALF / BK; kp++) {
            int ko = kp * BK, kpk = ko / 2, ksf = ko / SF_BLOCK;

            // Load intermediate A — vectorized
            for (int i = tid; i < SMEM_A / 4; i += BLOCK_SIZE) {
                int boff = i * 4;
                int row = boff / (BK / 2), col = boff % (BK / 2);
                *(uint32_t*)&s_A[swizzle_343(boff)] =
                    (row == 0) ? *(const uint32_t*)&gmem_inter_fp4[eidx * N_HALF_PACKED + kpk + col] : 0u;
            }

            // Load W2 B — vectorized
            for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
                int boff = i * 4;
                int row = boff / (BK / 2), col = boff % (BK / 2);
                int oc = j_start + row;
                *(uint32_t*)&s_B2[swizzle_343(boff)] =
                    (oc < HIDDEN)
                    ? *(const uint32_t*)&w2_fp4[(long long)oc * N_HALF_PACKED + kpk + col]
                    : 0u;
            }

            // Load scales
            if (tid < SF_PER_K)
                s_SFA[tid] = gmem_inter_sf[eidx * (N_HALF / SF_BLOCK) + ksf + tid];
            for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
                int row = i / SF_PER_K, col = i % SF_PER_K;
                int oc = j_start + row;
                s_SFB2[i] = (oc < HIDDEN) ? w2_sf[(long long)oc * SF_COLS_W2 + ksf + col] : 0;
            }

            __syncthreads();

            // Vectorized pack + MMA
            uint32_t ar[4] = {0, 0, 0, 0};
            if (lane_id / 4 == 0) {
                ar[0] = *(uint32_t*)&s_A[swizzle_343(t0 * 4)];
                ar[2] = *(uint32_t*)&s_A[swizzle_343(16 + t0 * 4)];
            }
            uint32_t sfap = pack_sf4(s_SFA);

            uint32_t br[2];
            br[0] = *(uint32_t*)&s_B2[swizzle_343(rbo + t0 * 4)];
            br[1] = *(uint32_t*)&s_B2[swizzle_343(rbo + 16 + t0 * 4)];
            uint32_t sfbp = pack_sf4(&s_SFB2[sn * SF_PER_K]);

            mma_nvf4_e4m3_m16n8k64(acc, ar, br, acc, sfap, sfbp);

            __syncthreads();
        }

        // Scatter output
        if (lane_id < 4) {
            int j0 = j_start + warp_id * 8 + lane_id;
            int j1 = j0 + 4;
            if (j0 < HIDDEN) atomicAdd(&output[j0], wt * acc[0]);
            if (j1 < HIDDEN) atomicAdd(&output[j1], wt * acc[1]);
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
// Host References
// ============================================================================
void host_reference(const float* input, const float* w1, const float* w2,
                    const int* eids, const float* ewts, float* out, int na) {
    memset(out, 0, HIDDEN*sizeof(float));
    for (int e = 0; e < na; e++) {
        int eid = eids[e]; float wt = ewts[e];
        const float* ew1 = w1+(long long)eid*N2*HIDDEN;
        const float* ew2 = w2+(long long)eid*HIDDEN*N_HALF;
        float gate[N_HALF], up[N_HALF];
        for (int n = 0; n < N_HALF; n++) {
            float s = 0; for (int k = 0; k < HIDDEN; k++) s += input[k]*ew1[(long long)n*HIDDEN+k];
            gate[n] = s;
        }
        for (int n = 0; n < N_HALF; n++) {
            float s = 0; for (int k = 0; k < HIDDEN; k++) s += input[k]*ew1[(long long)(n+N_HALF)*HIDDEN+k];
            up[n] = s;
        }
        float inter[N_HALF];
        for (int n = 0; n < N_HALF; n++) inter[n] = up[n]*host_silu(gate[n]);
        for (int j = 0; j < HIDDEN; j++) {
            float s = 0; for (int n = 0; n < N_HALF; n++) s += inter[n]*ew2[(long long)j*N_HALF+n];
            out[j] += wt*s;
        }
    }
}

void host_quantized_reference(
    const uint8_t* ifp4, const uint8_t* isf,
    const uint8_t* w1f, const uint8_t* w1s,
    const uint8_t* w2f, const uint8_t* w2s,
    const int* eids, const float* ewts, float* out, int na) {
    memset(out, 0, HIDDEN*sizeof(float));
    for (int e = 0; e < na; e++) {
        int eid = eids[e]; float wt = ewts[e];
        const uint8_t* ew1f = w1f+(long long)eid*N2*K_PACKED;
        const uint8_t* ew1s = w1s+(long long)eid*N2*SF_COLS_W1;
        const uint8_t* ew2f = w2f+(long long)eid*HIDDEN*N_HALF_PACKED;
        const uint8_t* ew2s = w2s+(long long)eid*HIDDEN*SF_COLS_W2;
        float gate[N_HALF], up_a[N_HALF];
        for (int n = 0; n < N_HALF; n++) {
            float s = 0;
            for (int k = 0; k < HIDDEN; k++)
                s += dequant_fp4_e4m3(ifp4,isf,k) * dequant_fp4_e4m3(ew1f+(long long)n*K_PACKED, ew1s+(long long)n*SF_COLS_W1, k);
            gate[n] = s;
        }
        for (int n = 0; n < N_HALF; n++) {
            float s = 0;
            for (int k = 0; k < HIDDEN; k++)
                s += dequant_fp4_e4m3(ifp4,isf,k) * dequant_fp4_e4m3(ew1f+(long long)(n+N_HALF)*K_PACKED, ew1s+(long long)(n+N_HALF)*SF_COLS_W1, k);
            up_a[n] = s;
        }
        float sw[N_HALF];
        for (int n = 0; n < N_HALF; n++) sw[n] = up_a[n]*host_silu(gate[n]);
        uint8_t ifp[N_HALF_PACKED], isf2[N_HALF/SF_BLOCK];
        memset(ifp, 0, sizeof(ifp));
        quantize_to_nvfp4_e4m3(sw, N_HALF, ifp, isf2);
        for (int j = 0; j < HIDDEN; j++) {
            float s = 0;
            for (int n = 0; n < N_HALF; n++)
                s += dequant_fp4_e4m3(ifp,isf2,n) * dequant_fp4_e4m3(ew2f+(long long)j*N_HALF_PACKED, ew2s+(long long)j*SF_COLS_W2, n);
            out[j] += wt*s;
        }
    }
}

// ============================================================================
// Error + Main
// ============================================================================
struct ErrS { double rmse, rel; int nan_c; };
ErrS compute_error(const float* a, const float* r, int n) {
    ErrS s={}; double es=0,rs=0;
    for (int i=0;i<n;i++) { if(isnan(a[i])||isinf(a[i])){s.nan_c++;continue;}
        double d=a[i]-r[i]; es+=d*d; rs+=(double)r[i]*r[i]; }
    s.rmse=sqrt(es/n); s.rel=(rs>0)?sqrt(es/rs):0; return s;
}
#define CHECK_CUDA(c) do{cudaError_t _e=(c);if(_e!=cudaSuccess){printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);}}while(0)

int main() {
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM%d%d, %d SMs, %dKB SMEM)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           (int)(prop.sharedMemPerMultiprocessor/1024));
    printf("\n=== Sprint 6: Hybrid K×N-Distributed + Vectorized Packing ===\n");
    printf("Phase 1: %d N-chunks × %d K-groups = %d CTAs/expert (640 total)\n",
           TILES_N, K_GROUPS, NUM_TILES);
    printf("Phase 1b: 4-way reduction (160 leaders) instead of 64-way (10 leaders)\n");
    printf("Packing: uint32_t vectorized (2 loads vs 16 scalar)\n");
    printf("MMA: mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.ue4m3\n\n");

    const int NE=16, TC=NUM_ACTIVE*NUM_TILES;
    printf("Config: K=%d, N_half=%d, %d active/%d experts, %d CTAs\n",
           HIDDEN, N_HALF, NUM_ACTIVE, NE, TC);
    printf("Phase 1: %d K-tiles/group, Phase 2: %d K-passes\n",
           K_TILES_PER_G, N_HALF/BK);
    printf("Partials: %d KB (was %.1f MB)\n\n",
           (int)(NUM_ACTIVE*NUM_TILES*PARTIALS_PER_CTA*sizeof(float)/1024),
           NUM_ACTIVE*64.0f*N2*sizeof(float)/1048576);

    int mb=0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, verdict_consec_k_fused, BLOCK_SIZE, SMEM_TOTAL);
    int cap = mb*prop.multiProcessorCount;
    printf("Occupancy: %d CTAs/SM x %d SMs = %d (need %d) %s\n\n",
           mb, prop.multiProcessorCount, cap, TC, cap>=TC?"OK":"FAIL");
    if (cap < TC) { printf("ERROR: deadlock risk\n"); return 1; }

    printf("Generating Xavier-scaled data...\n");
    srand(42);
    auto rf = [](){return((float)rand()/RAND_MAX-0.5f)*2.0f;};
    float* hi = new float[HIDDEN]; for(int i=0;i<HIDDEN;i++) hi[i]=rf();
    float w1s = 1.0f/sqrtf((float)HIDDEN), w2s = 1.0f/sqrtf((float)N_HALF);
    float* hw1 = new float[(long long)NE*N2*HIDDEN];
    for(long long i=0;i<(long long)NE*N2*HIDDEN;i++) hw1[i]=rf()*w1s;
    float* hw2 = new float[(long long)NE*HIDDEN*N_HALF];
    for(long long i=0;i<(long long)NE*HIDDEN*N_HALF;i++) hw2[i]=rf()*w2s;
    int heids[NUM_ACTIVE]; float hewts[NUM_ACTIVE];
    for(int i=0;i<NUM_ACTIVE;i++){heids[i]=i%NE; hewts[i]=1.0f/NUM_ACTIVE;}

    printf("Quantizing to NVFP4...\n");
    size_t ifs=HIDDEN/2, iss=HIDDEN/SF_BLOCK;
    uint8_t *hif=new uint8_t[ifs](), *his=new uint8_t[iss]();
    quantize_to_nvfp4_e4m3(hi, HIDDEN, hif, his);
    size_t w1fs=(size_t)NE*N2*K_PACKED, w1ss_sz=(size_t)NE*N2*SF_COLS_W1;
    uint8_t *hw1f=new uint8_t[w1fs](), *hw1s_d=new uint8_t[w1ss_sz]();
    for(int e=0;e<NE;e++) for(int n=0;n<N2;n++)
        quantize_to_nvfp4_e4m3(&hw1[(long long)e*N2*HIDDEN+(long long)n*HIDDEN], HIDDEN,
            &hw1f[(long long)e*N2*K_PACKED+(long long)n*K_PACKED],
            &hw1s_d[(long long)e*N2*SF_COLS_W1+(long long)n*SF_COLS_W1]);
    size_t w2fs=(size_t)NE*HIDDEN*N_HALF_PACKED, w2ss_sz=(size_t)NE*HIDDEN*SF_COLS_W2;
    uint8_t *hw2f=new uint8_t[w2fs](), *hw2s_d=new uint8_t[w2ss_sz]();
    for(int e=0;e<NE;e++) for(int j=0;j<HIDDEN;j++)
        quantize_to_nvfp4_e4m3(&hw2[(long long)e*HIDDEN*N_HALF+(long long)j*N_HALF], N_HALF,
            &hw2f[(long long)e*HIDDEN*N_HALF_PACKED+(long long)j*N_HALF_PACKED],
            &hw2s_d[(long long)e*HIDDEN*SF_COLS_W2+(long long)j*SF_COLS_W2]);

    printf("Computing references...\n");
    float* hrf=new float[HIDDEN](); host_reference(hi,hw1,hw2,heids,hewts,hrf,NUM_ACTIVE);
    float* hrq=new float[HIDDEN]();
    host_quantized_reference(hif,his,hw1f,hw1s_d,hw2f,hw2s_d,heids,hewts,hrq,NUM_ACTIVE);
    ErrS qf0=compute_error(hrq,hrf,HIDDEN);
    printf("  QRef vs FP32 baseline: %.4f%%\n", qf0.rel*100);

    printf("Uploading to GPU...\n");
    uint8_t *dif,*dis,*dw1f,*dw1s,*dw2f,*dw2s,*dif2,*dis2;
    int *deids; float *dewts,*dout,*dpart; int *dbar;
    size_t part_sz = (size_t)NUM_ACTIVE * NUM_TILES * PARTIALS_PER_CTA * sizeof(float);
    CHECK_CUDA(cudaMalloc(&dif,ifs)); CHECK_CUDA(cudaMalloc(&dis,iss));
    CHECK_CUDA(cudaMalloc(&dw1f,w1fs)); CHECK_CUDA(cudaMalloc(&dw1s,w1ss_sz));
    CHECK_CUDA(cudaMalloc(&dw2f,w2fs)); CHECK_CUDA(cudaMalloc(&dw2s,w2ss_sz));
    CHECK_CUDA(cudaMalloc(&deids,NUM_ACTIVE*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dewts,NUM_ACTIVE*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dout,HIDDEN*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dpart,part_sz));
    CHECK_CUDA(cudaMalloc(&dif2,NUM_ACTIVE*N_HALF_PACKED));
    CHECK_CUDA(cudaMalloc(&dis2,NUM_ACTIVE*(N_HALF/SF_BLOCK)));
    CHECK_CUDA(cudaMalloc(&dbar,sizeof(int)));
    CHECK_CUDA(cudaMemcpy(dif,hif,ifs,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dis,his,iss,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw1f,hw1f,w1fs,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw1s,hw1s_d,w1ss_sz,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw2f,hw2f,w2fs,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw2s,hw2s_d,w2ss_sz,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deids,heids,NUM_ACTIVE*sizeof(int),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dewts,hewts,NUM_ACTIVE*sizeof(float),cudaMemcpyHostToDevice));

    cudaFuncSetAttribute(verdict_consec_k_fused, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);

    // === Run kernel ===
    printf("\nLaunching kernel (%d CTAs)...\n", TC);
    CHECK_CUDA(cudaMemset(dout,0,HIDDEN*sizeof(float)));
    CHECK_CUDA(cudaMemset(dbar,0,sizeof(int)));
    verdict_consec_k_fused<<<TC,BLOCK_SIZE,SMEM_TOTAL>>>(
        dif,dis,dw1f,dw1s,dw2f,dw2s,deids,dewts,dout,dpart,dif2,dis2,dbar,NUM_ACTIVE);
    cudaError_t err=cudaDeviceSynchronize();
    if(err!=cudaSuccess){printf("Kernel error: %s\n",cudaGetErrorString(err));return 1;}

    float* hgo=new float[HIDDEN];
    CHECK_CUDA(cudaMemcpy(hgo,dout,HIDDEN*sizeof(float),cudaMemcpyDeviceToHost));

    printf("\n=== Correctness ===\n");
    printf("GPU[0:8]:  "); for(int i=0;i<8;i++) printf("%10.4f ",hgo[i]); printf("\n");
    printf("QRef[0:8]: "); for(int i=0;i<8;i++) printf("%10.4f ",hrq[i]); printf("\n");
    printf("FP32[0:8]: "); for(int i=0;i<8;i++) printf("%10.4f ",hrf[i]); printf("\n");

    ErrS eq=compute_error(hgo,hrq,HIDDEN), ef=compute_error(hgo,hrf,HIDDEN);
    ErrS qf=compute_error(hrq,hrf,HIDDEN);
    printf("\nGPU vs QRef:  RMSE=%.6f RelErr=%.4f%% NaN=%d\n", eq.rmse, eq.rel*100, eq.nan_c);
    printf("GPU vs FP32:  RMSE=%.6f RelErr=%.4f%% NaN=%d\n", ef.rmse, ef.rel*100, ef.nan_c);
    printf("QRef vs FP32: RMSE=%.6f RelErr=%.4f%% (baseline)\n", qf.rmse, qf.rel*100);

    bool pq = (eq.nan_c==0)&&(eq.rel<0.10), pf = (ef.nan_c==0)&&(ef.rel<0.50);
    printf("\nvs QRef (<10%%): %s\nvs FP32 (<50%%): %s\n", pq?"PASS":"FAIL", pf?"PASS":"FAIL");

    // === Benchmark ===
    printf("\n=== Benchmark ===\n");
    for(int i=0;i<20;i++){
        CHECK_CUDA(cudaMemset(dout,0,HIDDEN*sizeof(float)));
        CHECK_CUDA(cudaMemset(dbar,0,sizeof(int)));
        verdict_consec_k_fused<<<TC,BLOCK_SIZE,SMEM_TOTAL>>>(
            dif,dis,dw1f,dw1s,dw2f,dw2s,deids,dewts,dout,dpart,dif2,dis2,dbar,NUM_ACTIVE);
    }
    cudaDeviceSynchronize();

    std::vector<float> times;
    for(int i=0;i<100;i++){
        cudaEvent_t st,sp; cudaEventCreate(&st); cudaEventCreate(&sp);
        CHECK_CUDA(cudaMemset(dout,0,HIDDEN*sizeof(float)));
        CHECK_CUDA(cudaMemset(dbar,0,sizeof(int)));
        cudaEventRecord(st);
        verdict_consec_k_fused<<<TC,BLOCK_SIZE,SMEM_TOTAL>>>(
            dif,dis,dw1f,dw1s,dw2f,dw2s,deids,dewts,dout,dpart,dif2,dis2,dbar,NUM_ACTIVE);
        cudaEventRecord(sp); cudaEventSynchronize(sp);
        float ms; cudaEventElapsedTime(&ms,st,sp); times.push_back(ms*1000.0f);
        cudaEventDestroy(st); cudaEventDestroy(sp);
    }
    std::sort(times.begin(),times.end());
    float med=times[50], mn=std::accumulate(times.begin(),times.end(),0.0f)/100;
    printf("Latency: median=%.1f us, mean=%.1f us, p10=%.1f, p90=%.1f\n",
           med, mn, times[10], times[90]);
    printf("\nComparison:\n");
    printf("  Sprint 5 (scalar packing, 64-way reduce):   116.1 us\n");
    printf("  Sprint 6a (vec packing, 64-way reduce):      75.4 us\n");
    printf("  VLLM_CUTLASS baseline:                       98.0 us\n");
    printf("  Sprint 4 cooperative (FP32):                 38.9 us\n");
    printf("  THIS (vec packing, 4-way reduce, K×N):       %.1f us\n", med);
    if(med>0) {
        printf("  Speedup vs Sprint 5: %.2fx\n", 116.1f/med);
        printf("  Speedup vs Sprint 6a (vec only): %.2fx\n", 75.4f/med);
        printf("  Speedup vs VLLM_CUTLASS: %.2fx\n", 98.0f/med);
    }

    delete[] hi; delete[] hw1; delete[] hw2;
    delete[] hif; delete[] his; delete[] hw1f; delete[] hw1s_d; delete[] hw2f; delete[] hw2s_d;
    delete[] hrf; delete[] hrq; delete[] hgo;
    cudaFree(dif); cudaFree(dis); cudaFree(dw1f); cudaFree(dw1s);
    cudaFree(dw2f); cudaFree(dw2s); cudaFree(deids); cudaFree(dewts);
    cudaFree(dout); cudaFree(dpart); cudaFree(dif2); cudaFree(dis2); cudaFree(dbar);
    printf("\nDone.\n");
    return (pq&&pf) ? 0 : 1;
}
