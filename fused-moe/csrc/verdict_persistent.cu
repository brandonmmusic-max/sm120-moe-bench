/**
 * verdict_persistent.cu — Sprint 11 Task 1: Persistent Kernel
 *
 * Converts the cooperative kernel (640 CTAs, 4 atomic barriers) into a
 * persistent kernel (752 CTAs, 0 barriers). Each CTA loops over work items
 * from a queue, processing the full GEMM1→SwiGLU→requant→GEMM2 pipeline
 * independently.
 *
 * Key differences from Sprint 9:
 *   - Grid: min(752, total_work_items) CTAs — CUDA graph safe at fixed 752
 *   - NO atomic barriers — each work item is fully independent
 *   - Each CTA does full K-reduction (all 64 K-tiles) — no K-group splitting
 *   - No partials buffer, no intermediate gmem buffers
 *   - SwiGLU + FP4 requant happens in SMEM (no gmem round-trip)
 *   - GEMM2 A operand (intermediate) hoisted into registers
 *   - Multiple work items per CTA via persistent loop (grid stride)
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a \
 *     --expt-relaxed-constexpr --compiler-options '-fPIC' \
 *     -o verdict_persistent csrc/verdict_persistent.cu
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
static constexpr int TOTAL_K_TILES = HIDDEN / BK;          // 64
static constexpr int P2_OUT_TILES  = HIDDEN / BN;          // 64

static_assert(HIDDEN % BN == 0, "HIDDEN must be a multiple of BN");
static_assert(HIDDEN % BK == 0, "HIDDEN must be a multiple of BK");

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

__device__ __forceinline__ uint32_t pack_sf4(const uint8_t* sf) {
    return (uint32_t)sf[0] | ((uint32_t)sf[1]<<8)
         | ((uint32_t)sf[2]<<16) | ((uint32_t)sf[3]<<24);
}

// ============================================================================
// PERSISTENT FUSED MOE KERNEL — 0 BARRIERS
//
// Each CTA independently processes work items via grid-stride loop.
// Work item = (token-expert pair, N-tile, output-group).
// Full GEMM1→SwiGLU→requant→GEMM2(subset) pipeline per work item.
// No K-group splitting, no partials, no intermediate gmem, no barriers.
//
// GEMM1 is duplicated across out_groups, but weights hit L2 after first load.
// GEMM2 output tiles are split across out_groups for parallelism.
//
// Grid: 752 CTAs (fixed for CUDA graphs), persistent loop
// total_work_items = num_pairs × tiles_n × out_groups
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
verdict_persistent_fused_moe(
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
    int num_pairs,
    int n_half,
    int out_groups,                                // GEMM2 output tile groups
    int total_work_items)                          // num_pairs × tiles_n × out_groups
{
    // Runtime constants
    const int tiles_n       = n_half / BN;
    const int n_half_packed = n_half / 2;
    const int sf_cols_w2    = n_half / SF_BLOCK;
    const int n2            = 2 * n_half;

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // MMA column mapping (CLayout for M=1, scale_vec::4X)
    const int g   = lane_id / 4;
    const int Nl  = 4 * (g & 1) + (g >> 1);
    const int sn  = warp_id * 8 + Nl;
    const int t0  = lane_id % 4;
    const int rbo = sn * (BK / 2);

    // SMEM layout (same as Sprint 9, ~4.7 KB):
    //   s_A:        32 bytes   — input FP4 (GEMM1) / intermediate FP4 (GEMM2)
    //   s_B_gate:   2048 bytes — gate weights (GEMM1) / W2 weights (GEMM2)
    //   s_B_up:     2048 bytes — up weights (GEMM1) / SwiGLU buffer (post-GEMM1)
    //   s_SFA:      4 bytes    — input SF (GEMM1) / intermediate SF (GEMM2)
    //   s_SFB_gate: 256 bytes  — gate weight SF (GEMM1) / W2 SF (GEMM2)
    //   s_SFB_up:   256 bytes  — up weight SF (GEMM1 only)
    extern __shared__ char smem_raw[];
    uint8_t* s_A        = (uint8_t*)smem_raw;
    uint8_t* s_B_gate   = s_A + 32;
    uint8_t* s_B_up     = s_B_gate + SMEM_B;
    uint8_t* s_SFA      = s_B_up + SMEM_B;
    uint8_t* s_SFB_gate = s_SFA + ((SF_PER_K + 3) & ~3);
    uint8_t* s_SFB_up   = s_SFB_gate + SMEM_SFB;

    // SwiGLU buffer reuses s_B_up area (only needed after GEMM1 is done)
    float* s_swiglu = (float*)s_B_up;  // 64 floats = 256 bytes, fits in 2048

    // Derived: output tiles per GEMM2 group
    const int tiles_per_ogrp = P2_OUT_TILES / out_groups;  // e.g. 64/4=16

    // ================================================================
    // PERSISTENT LOOP — grid-stride over independent work items
    // ================================================================
    int work_idx = blockIdx.x;
    while (work_idx < total_work_items) {
        // Decode work item: (pair_idx, n_chunk, out_group)
        const int items_per_pair = tiles_n * out_groups;
        const int pair_idx  = work_idx / items_per_pair;
        const int rem       = work_idx % items_per_pair;
        const int n_chunk   = rem / out_groups;
        const int out_group = rem % out_groups;

        const int eid      = expert_ids[pair_idx];
        const int token_id = token_ids[pair_idx];
        const float wt     = expert_wts[pair_idx];
        const int n_start  = n_chunk * BN;

        // Weight pointers for this expert
        const uint8_t* w1_fp4 = all_w1_fp4 + (long long)eid * n2 * K_PACKED;
        const uint8_t* w1_sf  = all_w1_sf  + (long long)eid * n2 * SF_COLS_W1;

        // ============================================================
        // PHASE 1: GEMM1 — Full K reduction (ALL 64 K-tiles)
        // No K-group splitting → no partials → no barrier needed.
        // ============================================================
        float gate_acc[4] = {0, 0, 0, 0};
        float up_acc[4]   = {0, 0, 0, 0};

        for (int kt = 0; kt < TOTAL_K_TILES; kt++) {
            const int k_off = kt * BK;
            const int k_pk  = k_off / 2;
            const int k_sf  = k_off / SF_BLOCK;

            // Load A: input token FP4 tile (32 bytes)
            for (int i = tid; i < 8; i += BLOCK_SIZE) {
                *(uint32_t*)(s_A + i * 4) =
                    *(const uint32_t*)&input_fp4[token_id * K_PACKED + k_pk + i * 4];
            }

            // Load gate B tile (BN×BK/2 = 2048 bytes)
            for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
                int boff = i * 4;
                int row = boff / (BK / 2), col = boff % (BK / 2);
                *(uint32_t*)&s_B_gate[swizzle_343(boff)] =
                    *(const uint32_t*)&w1_fp4[(long long)(n_start + row) * K_PACKED + k_pk + col];
            }

            // Load up B tile (BN×BK/2 = 2048 bytes)
            for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
                int boff = i * 4;
                int row = boff / (BK / 2), col = boff % (BK / 2);
                *(uint32_t*)&s_B_up[swizzle_343(boff)] =
                    *(const uint32_t*)&w1_fp4[(long long)(n_half + n_start + row) * K_PACKED + k_pk + col];
            }

            // Load SFA (input scale factors, 4 bytes)
            if (tid < SF_PER_K) {
                s_SFA[tid] = input_sf[token_id * SF_COLS_W1 + k_sf + tid];
            }

            // Load SFB gate + up (256 bytes each)
            for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
                int row = i / SF_PER_K, col = i % SF_PER_K;
                s_SFB_gate[i] = w1_sf[(long long)(n_start + row) * SF_COLS_W1 + k_sf + col];
            }
            for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
                int row = i / SF_PER_K, col = i % SF_PER_K;
                s_SFB_up[i] = w1_sf[(long long)(n_half + n_start + row) * SF_COLS_W1 + k_sf + col];
            }

            __syncthreads();

            // Read B operands from SMEM
            uint32_t bg[2], bu[2];
            bg[0] = *(uint32_t*)&s_B_gate[swizzle_343(rbo + t0 * 4)];
            bg[1] = *(uint32_t*)&s_B_gate[swizzle_343(rbo + 16 + t0 * 4)];
            uint32_t sfbg = pack_sf4(&s_SFB_gate[sn * SF_PER_K]);

            bu[0] = *(uint32_t*)&s_B_up[swizzle_343(rbo + t0 * 4)];
            bu[1] = *(uint32_t*)&s_B_up[swizzle_343(rbo + 16 + t0 * 4)];
            uint32_t sfbu = pack_sf4(&s_SFB_up[sn * SF_PER_K]);

            // Read A operand (single token)
            uint32_t a[4] = {0, 0, 0, 0};
            if (lane_id / 4 == 0) {
                a[0] = *(uint32_t*)(s_A + t0 * 4);
                a[2] = *(uint32_t*)(s_A + 16 + t0 * 4);
            }
            uint32_t sfa_pk = pack_sf4(s_SFA);

            // MMA: gate and up accumulation
            mma_nvf4_e4m3_m16n8k64(gate_acc, a, bg, gate_acc, sfa_pk, sfbg);
            mma_nvf4_e4m3_m16n8k64(up_acc, a, bu, up_acc, sfa_pk, sfbu);

            __syncthreads();
        }

        // ============================================================
        // PHASE 2: SwiGLU + FP4 requantization (entirely in SMEM)
        // No gmem intermediate — saves bandwidth and memory.
        // ============================================================

        // Collect GEMM1 results from MMA layout → SwiGLU → SMEM
        // Lanes 0-3 in each warp hold the M=0 row results.
        if (lane_id < 4) {
            int c0 = warp_id * 8 + lane_id;
            int c1 = c0 + 4;
            s_swiglu[c0] = up_acc[0] * d_silu(gate_acc[0]);
            s_swiglu[c1] = up_acc[1] * d_silu(gate_acc[1]);
        }
        __syncthreads();

        // Quantize SwiGLU output to FP4 → store in s_A (for GEMM2 A operand)
        // 64 threads for BN=64 columns, warp shuffles for SF_BLOCK=16 group max.
        // Thread layout: tid 0-31 = warp 0, tid 32-63 = warp 1.
        // Groups of 16 lanes within each warp naturally isolate via XOR shuffles.
        if (tid < BN) {
            int col = tid;
            float sw_val = s_swiglu[col];

            // Group max across SF_BLOCK=16 via butterfly warp shuffle
            float gm = fabsf(sw_val);
            gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 1));
            gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 2));
            gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 4));
            gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 8));

            float st = fmaxf(gm / 6.0f, 1e-30f);
            uint8_t sf_enc = d_e4m3fn_encode(st);
            float as = d_e4m3fn_decode(sf_enc);
            if (as < 1e-30f) as = 1e-30f;

            uint8_t nib = d_quantize_e2m1(sw_val / as);

            // Pack nibble pairs into s_A (consecutive-K packing)
            uint32_t nib32 = (uint32_t)nib;
            uint32_t neighbor32 = __shfl_down_sync(0xFFFFFFFF, nib32, 1);
            if (col % 2 == 0) {
                s_A[col / 2] = (uint8_t)(nib32 | (neighbor32 << 4));
            }

            // Scale factors into s_SFA
            if (col % SF_BLOCK == 0) {
                s_SFA[col / SF_BLOCK] = sf_enc;
            }
        }
        __syncthreads();

        // ============================================================
        // PHASE 3: GEMM2 — iterate over ALL HIDDEN/BN output tiles
        //
        // A = intermediate FP4 [1, BK=64] from s_A (hoisted to registers)
        // B = W2[output_row, n_start:n_start+BK] loaded each tile
        // One K-pass per output tile (BK=BN=64 intermediate columns)
        // atomicAdd weighted partial to output[token_id]
        // ============================================================
        {
            const uint8_t* w2_fp4 = all_w2_fp4 + (long long)eid * HIDDEN * n_half_packed;
            const uint8_t* w2_sf  = all_w2_sf  + (long long)eid * HIDDEN * sf_cols_w2;
            uint8_t* s_B2   = s_B_gate;   // reuse SMEM
            uint8_t* s_SFB2 = s_SFB_gate;  // reuse SMEM

            // K-offset into W2 for this work item's intermediate columns
            const int kpk = n_start / 2;
            const int ksf = n_start / SF_BLOCK;

            // Hoist A operand into registers (same for all output tiles)
            uint32_t ar_fixed[4] = {0, 0, 0, 0};
            if (lane_id / 4 == 0) {
                ar_fixed[0] = *(uint32_t*)(s_A + t0 * 4);
                ar_fixed[2] = *(uint32_t*)(s_A + 16 + t0 * 4);
            }
            uint32_t sfap_fixed = pack_sf4(s_SFA);

            const int j_tile_start = out_group * tiles_per_ogrp;
            const int j_tile_end   = j_tile_start + tiles_per_ogrp;
            for (int j_tile = j_tile_start; j_tile < j_tile_end; j_tile++) {
                const int j_start = j_tile * BN;

                float p2_acc[4] = {0, 0, 0, 0};

                // Load W2 B tile: rows [j_start, j_start+BN), cols [n_start, n_start+BK)
                for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
                    int boff = i * 4;
                    int row = boff / (BK / 2), col = boff % (BK / 2);
                    *(uint32_t*)&s_B2[swizzle_343(boff)] =
                        *(const uint32_t*)&w2_fp4[(long long)(j_start + row) * n_half_packed + kpk + col];
                }

                // Load W2 SFB
                for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
                    int row = i / SF_PER_K, col = i % SF_PER_K;
                    s_SFB2[i] = w2_sf[(long long)(j_start + row) * sf_cols_w2 + ksf + col];
                }

                __syncthreads();

                // Read B operands
                uint32_t br[2];
                br[0] = *(uint32_t*)&s_B2[swizzle_343(rbo + t0 * 4)];
                br[1] = *(uint32_t*)&s_B2[swizzle_343(rbo + 16 + t0 * 4)];
                uint32_t sfbp = pack_sf4(&s_SFB2[sn * SF_PER_K]);

                // MMA: one K-pass (BK=64 intermediate columns)
                mma_nvf4_e4m3_m16n8k64(p2_acc, ar_fixed, br, p2_acc, sfap_fixed, sfbp);

                // Scatter weighted partial to output[token_id]
                if (lane_id < 4) {
                    int j0 = j_start + warp_id * 8 + lane_id;
                    int j1 = j0 + 4;
                    atomicAdd(&output[token_id * HIDDEN + j0], wt * p2_acc[0]);
                    atomicAdd(&output[token_id * HIDDEN + j1], wt * p2_acc[1]);
                }

                __syncthreads();
            }
        }

        work_idx += gridDim.x;  // stride to next work item
    }
}

// ============================================================================
// Host: Quantization (same as Sprint 9)
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
// Host Quantized Reference (same as Sprint 9)
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
// Test: Persistent kernel with independent per-token routing
// ============================================================================
bool run_persistent_test(int M, int topk, int n_half, int NE,
                          const int routing[][10], const char* label) {
    int num_pairs = M * topk;
    int tiles_n = n_half / BN;
    int n2 = 2 * n_half;
    int n_half_packed = n_half / 2;
    int sf_cols_w2 = n_half / SF_BLOCK;

    // Query GPU for max CTAs
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int max_ctas = prop.multiProcessorCount * 4;  // 188 × 4 = 752

    // Compute out_groups to target ~max_ctas work items
    int base_items = num_pairs * tiles_n;
    int out_groups = std::max(1, max_ctas / base_items);
    // Must divide P2_OUT_TILES=64 evenly
    while (P2_OUT_TILES % out_groups != 0 && out_groups > 1) out_groups--;
    int total_work_items = base_items * out_groups;
    int grid_size = std::min(max_ctas, total_work_items);
    int tiles_per_ogrp = P2_OUT_TILES / out_groups;

    printf("\n======================================================\n");
    printf("PERSISTENT KERNEL: %s\n", label);
    printf("M=%d, topk=%d, num_pairs=%d, N_HALF=%d\n",
           M, topk, num_pairs, n_half);
    printf("Work items: %d pairs × %d tiles × %d out_groups = %d\n",
           num_pairs, tiles_n, out_groups, total_work_items);
    printf("Grid: %d CTAs (max %d), 0 barriers\n", grid_size, max_ctas);
    printf("GEMM1: %d K-tiles per CTA (full K, duplicated across out_groups)\n",
           TOTAL_K_TILES);
    printf("GEMM2: %d output tiles per CTA (out of %d)\n",
           tiles_per_ogrp, P2_OUT_TILES);
    printf("======================================================\n");

    // Check occupancy
    int smem_size = 32 + 2 * SMEM_B + ((SF_PER_K + 3) & ~3) + 2 * SMEM_SFB + 128;
    int mb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, verdict_persistent_fused_moe, BLOCK_SIZE, smem_size);
    int cap = mb * prop.multiProcessorCount;
    printf("Occupancy: %d CTAs/SM × %d SMs = %d max concurrent\n",
           mb, prop.multiProcessorCount, cap);

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

    // Quantize
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

    // Compute per-token quantized reference
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
    uint8_t *dif, *dis, *dw1f, *dw1s, *dw2f, *dw2s;
    int *deids, *dtids; float *dewts, *dout;

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

    CHECK_CUDA(cudaMemcpy(dif, hif, ifs, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dis, his, iss, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw1f, hw1f, w1fs, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw1s, hw1s, w1ss, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw2f, hw2f, w2fs, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw2s, hw2s, w2ss, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deids, h_pair_eids, num_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dtids, h_pair_tids, num_pairs * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dewts, h_pair_wts, num_pairs * sizeof(float), cudaMemcpyHostToDevice));

    cudaFuncSetAttribute(verdict_persistent_fused_moe,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    // Launch kernel (correctness)
    printf("Launching persistent kernel (%d CTAs, smem=%d)...\n", grid_size, smem_size);
    CHECK_CUDA(cudaMemset(dout, 0, M * HIDDEN * sizeof(float)));
    verdict_persistent_fused_moe<<<grid_size, BLOCK_SIZE, smem_size>>>(
        dif, dis, dw1f, dw1s, dw2f, dw2s, deids, dtids, dewts, dout,
        num_pairs, n_half, out_groups, total_work_items);
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
        // Persistent kernel accumulates 64 K-tiles in one pass (no Kahan reduction).
        // Error from FP4 quantization is ~9-10%, same as Sprint 9 with k_groups>4.
        bool agg_pass = (eq_all.nan_c == 0) && (eq_all.rel < 0.11f);
        printf("  Aggregate: RelErr=%.4f%% %s | NaN=%d\n",
               eq_all.rel * 100, agg_pass ? "PASS" : "FAIL", eq_all.nan_c);
        if (!agg_pass) all_pass = false;

        printf("  Sample GPU[tok0, 0:4]: ");
        for (int i = 0; i < 4; i++) printf("%.4f ", hgo[i]); printf("\n");
        printf("  Sample QRef[tok0, 0:4]: ");
        for (int i = 0; i < 4; i++) printf("%.4f ", hrq[i]); printf("\n");

        // Benchmark — also test with fixed 752 grid (CUDA-graph mode)
        printf("\n--- Benchmark ---\n");

        // Warmup
        for (int i = 0; i < 20; i++) {
            CHECK_CUDA(cudaMemset(dout, 0, M * HIDDEN * sizeof(float)));
            verdict_persistent_fused_moe<<<grid_size, BLOCK_SIZE, smem_size>>>(
                dif, dis, dw1f, dw1s, dw2f, dw2s, deids, dtids, dewts, dout,
                num_pairs, n_half, out_groups, total_work_items);
        }
        cudaDeviceSynchronize();

        // Bench: optimal grid (min of max_ctas, work_items)
        std::vector<float> times;
        for (int i = 0; i < 100; i++) {
            cudaEvent_t st, sp;
            cudaEventCreate(&st); cudaEventCreate(&sp);
            CHECK_CUDA(cudaMemset(dout, 0, M * HIDDEN * sizeof(float)));
            cudaEventRecord(st);
            verdict_persistent_fused_moe<<<grid_size, BLOCK_SIZE, smem_size>>>(
                dif, dis, dw1f, dw1s, dw2f, dw2s, deids, dtids, dewts, dout,
                num_pairs, n_half, out_groups, total_work_items);
            cudaEventRecord(sp); cudaEventSynchronize(sp);
            float ms; cudaEventElapsedTime(&ms, st, sp);
            times.push_back(ms * 1000.0f);
            cudaEventDestroy(st); cudaEventDestroy(sp);
        }
        std::sort(times.begin(), times.end());
        float med = times[50], mn = std::accumulate(times.begin(), times.end(), 0.0f) / 100;
        printf("  [grid=%d] Latency: median=%.1f μs, mean=%.1f μs, p10=%.1f, p90=%.1f\n",
               grid_size, med, mn, times[10], times[90]);

        // Also bench with fixed 752 grid (CUDA-graph safe)
        if (grid_size != max_ctas) {
            times.clear();
            for (int i = 0; i < 20; i++) {
                CHECK_CUDA(cudaMemset(dout, 0, M * HIDDEN * sizeof(float)));
                verdict_persistent_fused_moe<<<max_ctas, BLOCK_SIZE, smem_size>>>(
                    dif, dis, dw1f, dw1s, dw2f, dw2s, deids, dtids, dewts, dout,
                    num_pairs, n_half, out_groups, total_work_items);
            }
            cudaDeviceSynchronize();
            for (int i = 0; i < 100; i++) {
                cudaEvent_t st, sp;
                cudaEventCreate(&st); cudaEventCreate(&sp);
                CHECK_CUDA(cudaMemset(dout, 0, M * HIDDEN * sizeof(float)));
                cudaEventRecord(st);
                verdict_persistent_fused_moe<<<max_ctas, BLOCK_SIZE, smem_size>>>(
                    dif, dis, dw1f, dw1s, dw2f, dw2s, deids, dtids, dewts, dout,
                    num_pairs, n_half, out_groups, total_work_items);
                cudaEventRecord(sp); cudaEventSynchronize(sp);
                float ms; cudaEventElapsedTime(&ms, st, sp);
                times.push_back(ms * 1000.0f);
                cudaEventDestroy(st); cudaEventDestroy(sp);
            }
            std::sort(times.begin(), times.end());
            float med752 = times[50];
            printf("  [grid=%d fixed] Latency: median=%.1f μs, mean=%.1f μs\n",
                   max_ctas, med752,
                   std::accumulate(times.begin(), times.end(), 0.0f) / 100);
        }

        // Comparison
        if (M == 1 && n_half == 256) {
            printf("  vs Sprint 9 TP=4 M=1 (17.9 μs): %.2fx\n", 17.9f / med);
            printf("  vs CUTLASS (98 μs): %.2fx\n", 98.0f / med);
        } else if (M == 4 && n_half == 256) {
            printf("  vs Sprint 9 TP=4 M=4 (44.4 μs): %.2fx\n", 44.4f / med);
            printf("  vs CUTLASS M=4 (~120 μs): %.2fx\n", 120.0f / med);
        }

        printf("\n  Overall: %s\n", all_pass ? "PASS" : "FAIL");

        delete[] hgo;
        delete[] hi; delete[] hw1; delete[] hw2;
        delete[] h_pair_eids; delete[] h_pair_tids; delete[] h_pair_wts;
        delete[] hif; delete[] his; delete[] hw1f; delete[] hw1s; delete[] hw2f; delete[] hw2s;
        delete[] hrq;
        cudaFree(dif); cudaFree(dis); cudaFree(dw1f); cudaFree(dw1s);
        cudaFree(dw2f); cudaFree(dw2s); cudaFree(deids); cudaFree(dtids); cudaFree(dewts);
        cudaFree(dout);

        return all_pass;
    }

cleanup:
    delete[] hi; delete[] hw1; delete[] hw2;
    delete[] h_pair_eids; delete[] h_pair_tids; delete[] h_pair_wts;
    delete[] hif; delete[] his; delete[] hw1f; delete[] hw1s; delete[] hw2f; delete[] hw2s;
    delete[] hrq;
    cudaFree(dif); cudaFree(dis); cudaFree(dw1f); cudaFree(dw1s);
    cudaFree(dw2f); cudaFree(dw2s); cudaFree(deids); cudaFree(dtids); cudaFree(dewts);
    cudaFree(dout);
    return false;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM%d%d, %d SMs, %dKB SMEM)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           (int)(prop.sharedMemPerMultiprocessor / 1024));
    printf("\n=== Sprint 11 Task 1: Persistent Kernel (0 Barriers) ===\n");
    printf("MMA: mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.ue4m3\n");
    printf("Grid: persistent loop, full K per CTA, no K-group splitting\n");
    printf("Eliminated: partials buffer, gmem intermediate, barrier counter\n");

    int topk = 10;
    int NE = 30;

    const int routing_m4[4][10] = {
        {0,1,2,3,4,5,6,7,8,9},
        {5,6,7,8,9,10,11,12,13,14},
        {20,21,22,23,24,25,26,27,28,29},
        {0,1,2,3,4,25,26,27,28,29}
    };

    const int routing_m1[1][10] = {
        {0,1,2,3,4,5,6,7,8,9}
    };

    bool all_pass = true;

    printf("\n\n*** CONFIG 1: M=1, N_HALF=256, TP=4 ***\n");
    all_pass &= run_persistent_test(1, topk, 256, NE, routing_m1,
        "M=1 TP=4 persistent");

    printf("\n\n*** CONFIG 2: M=4, N_HALF=256, TP=4, INDEPENDENT ROUTING ***\n");
    all_pass &= run_persistent_test(4, topk, 256, NE, routing_m4,
        "M=4 TP=4 persistent (independent routing)");

    printf("\n\n*** CONFIG 3: M=1, N_HALF=1024, EP=4 ***\n");
    all_pass &= run_persistent_test(1, topk, 1024, NE, routing_m1,
        "M=1 EP=4 persistent");

    printf("\n\n*** CONFIG 4: M=4, N_HALF=1024, EP=4, INDEPENDENT ROUTING ***\n");
    all_pass &= run_persistent_test(4, topk, 1024, NE, routing_m4,
        "M=4 EP=4 persistent (independent routing)");

    printf("\n\n=== FINAL RESULT: %s ===\n", all_pass ? "ALL PASS" : "SOME FAILED");
    return all_pass ? 0 : 1;
}
