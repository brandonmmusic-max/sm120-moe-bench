/**
 * VerdictMoE Phase 3: Cooperative MMA Kernel for SM120
 *
 * Single cooperative kernel launch for all 10 active experts (M=1 decode).
 * Uses NVF4 MMA (m16n8k64) for both GEMM1 and GEMM2 with FP4 block-scaled weights.
 * TMA-style async loads via ldmatrix.b4x16_p64.
 * Fused GEMM1 -> SwiGLU -> GEMM2 with grid.sync() between stages.
 *
 * Dimensions (Qwen3.5-397B-A17B, EP=4):
 *   K=4096 (hidden), N_half=1024 (intermediate), 10 active experts, M=1
 *
 * Grid: 640 CTAs = 10 experts x 64 tiles, 256 threads/CTA
 * Phase 1: GEMM1 (K-distributed, 64 K-tiles), NVF4 MMA
 * Phase 2: Reduce + SwiGLU + FP4 requant (10 leader CTAs)
 * Phase 3: GEMM2 (N-distributed, 64 output-col tiles), NVF4 MMA
 *
 * Build (inside vLLM container):
 *   nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a \
 *     -rdc=true -o verdict_mma_cooperative verdict_mma_cooperative.cu
 */

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <numeric>

namespace cg = cooperative_groups;

// ============================================================================
// Constants
// ============================================================================
static constexpr int HIDDEN      = 4096;
static constexpr int N_HALF      = 1024;
static constexpr int N2          = 2 * N_HALF;   // 2048
static constexpr int NUM_ACTIVE  = 10;
static constexpr int TILES       = 64;
static constexpr int K_PER_TILE  = HIDDEN / TILES;  // 64

static constexpr int BM = 16;
static constexpr int BN = 64;    // warps * 8 cols/warp
static constexpr int BK = 64;    // MMA K dimension for NVF4
static constexpr int MMA_N = 8;
static constexpr int SF_BLOCK = 32;

static constexpr int NUM_WARPS  = 8;
static constexpr int WARP_SIZE  = 32;
static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;  // 256

static constexpr int K_PACKED      = HIDDEN / 2;
static constexpr int N_HALF_PACKED = N_HALF / 2;
static constexpr int SF_COLS_W1    = HIDDEN / SF_BLOCK;   // 128
static constexpr int SF_COLS_W2    = N_HALF / SF_BLOCK;   // 32

// SMEM: A[512] + B[2048] + SFA[16] + SFB[128] + padding
static constexpr int SMEM_A   = BM * (BK / 2);           // 512
static constexpr int SMEM_B   = BN * (BK / 2);           // 2048
static constexpr int SMEM_SFA = 16;                       // aligned
static constexpr int SMEM_SFB = BN * (BK / SF_BLOCK);    // 128
static constexpr int SMEM_TOTAL = SMEM_A + SMEM_B + SMEM_SFA + SMEM_SFB + 256;

// E2M1 value table
static const float E2M1_TABLE[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// ============================================================================
// PTX Inline Helpers
// ============================================================================

// ============================================================================
// Register packing per cute ALayout/BLayout traits for NVF4 m16n8k64
// ============================================================================
// ALayout: (T32,V32) → (M16,K64) with strides ((128,1),(16,8,512))
//   Thread t: t0=t%4, t1=t/4. Only t0=0 threads have M=0 data.
//   Reg 0 nibbles 0-3: K={t1, t1+16, t1+32, t1+48} at M=0
//   Reg 0 nibbles 4-7: M=1 (zero for M=1 input)
//   Reg 1 nibbles 0-3: K={t1+8, t1+24, t1+40, t1+56} at M=0
//   Reg 1 nibbles 4-7: M=1 (zero)
//   Regs 2-3: M≥8 (zero)
//
// BLayout: (T32,V16) → (N8,K64) with strides ((64,1),(8,256))
//   Thread t: t0=t%4, t1=t/4
//   Reg 0 (8 nibbles): N=t0, K={t1, t1+8, t1+16, ..., t1+56}
//   Reg 1 (8 nibbles): N=t0+4, K={t1, t1+8, ..., t1+56}
//
// CLayout: SM80_16x8_Row: (T32,V4) → (M16,N8) strides ((32,1),(16,8))
//   M=0 output: only threads with t%4==0, d[0] only, N=t/4

__device__ __forceinline__ uint32_t get_nibble(const uint8_t* smem, int k) {
    uint8_t byte = smem[k / 2];
    return (k & 1) ? ((byte >> 4) & 0xFu) : (byte & 0xFu);
}

// Pack A registers for M=1 input (only row 0 of A tile has data)
// s_A: [BM=16, BK/2=32] row-major, row 0 = input FP4 data
__device__ __forceinline__ void pack_a_m1(
    uint32_t (&a)[4], const uint8_t* s_A, int lane_id)
{
    a[0] = a[1] = a[2] = a[3] = 0;
    if (lane_id % 4 == 0) {
        int t1 = lane_id / 4;  // 0..7
        // Row 0 data starts at s_A[0], 32 bytes for K=0..63
        a[0] = get_nibble(s_A, t1)
             | (get_nibble(s_A, t1 + 16) << 4)
             | (get_nibble(s_A, t1 + 32) << 8)
             | (get_nibble(s_A, t1 + 48) << 12);
        a[1] = get_nibble(s_A, t1 + 8)
             | (get_nibble(s_A, t1 + 24) << 4)
             | (get_nibble(s_A, t1 + 40) << 8)
             | (get_nibble(s_A, t1 + 56) << 12);
    }
}

// Pack B registers for weight tile
// s_B: [BN rows, BK/2 bytes] row-major. Each row = one N-column's K-data.
// warp_id selects which 8 rows (N-columns) this warp processes.
__device__ __forceinline__ void pack_b(
    uint32_t (&b)[2], const uint8_t* s_B_warp, int lane_id)
{
    int t0 = lane_id % 4;   // maps to N column 0..3 (and +4 for reg 1)
    int t1 = lane_id / 4;   // maps to K offset 0..7

    const uint8_t* row_n0 = s_B_warp + t0 * (BK / 2);       // N=t0
    const uint8_t* row_n4 = s_B_warp + (t0 + 4) * (BK / 2); // N=t0+4

    b[0] = b[1] = 0;
    #pragma unroll
    for (int vi = 0; vi < 8; vi++) {
        int k = t1 + vi * 8;
        b[0] |= get_nibble(row_n0, k) << (vi * 4);
        b[1] |= get_nibble(row_n4, k) << (vi * 4);
    }
}

// Block-scaled NVF4 MMA: m16n8k64, scale_vec::2X, UE8M0 scales
__device__ __forceinline__ void mma_nvf4_m16n8k64(
    float (&d)[4],
    const uint32_t (&a)[4], const uint32_t (&b)[2],
    const float (&c)[4],
    uint32_t sfa, uint32_t sfb)
{
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},"
        "{%4,%5,%6,%7},"
        "{%8,%9},"
        "{%10,%11,%12,%13},"
        "{%14},{%15,%16},"
        "{%17},{%18,%19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
          "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

__device__ __forceinline__ float d_silu(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Device E2M1 quantization (nearest)
__device__ __forceinline__ uint8_t quantize_e2m1_dev(float value) {
    float av = fabsf(value);
    int sign = (value < 0.0f) ? 1 : 0;
    int idx;
    if      (av < 0.25f)  idx = 0;
    else if (av < 0.75f)  idx = 1;
    else if (av < 1.25f)  idx = 2;
    else if (av < 1.75f)  idx = 3;
    else if (av < 2.5f)   idx = 4;
    else if (av < 3.5f)   idx = 5;
    else if (av < 5.0f)   idx = 6;
    else                   idx = 7;
    return (uint8_t)((sign << 3) | idx);
}

// ============================================================================
// COOPERATIVE KERNEL: Fused GEMM1 -> SwiGLU -> GEMM2 with MMA
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
verdict_mma_cooperative(
    const uint8_t* __restrict__ input_fp4,      // [1, K/2]
    const uint8_t* __restrict__ input_sf,        // [1, K/SF_BLOCK]
    const uint8_t* __restrict__ all_w1_fp4,      // [E, N2, K/2]
    const uint8_t* __restrict__ all_w1_sf,       // [E, N2, K/SF_BLOCK]
    const uint8_t* __restrict__ all_w2_fp4,      // [E, HIDDEN, N_HALF/2]
    const uint8_t* __restrict__ all_w2_sf,       // [E, HIDDEN, N_HALF/SF_BLOCK]
    const int*     __restrict__ expert_ids,      // [NUM_ACTIVE]
    const float*   __restrict__ expert_wts,      // [NUM_ACTIVE]
    float*         __restrict__ output,          // [HIDDEN] zeroed
    float*         __restrict__ partials,        // [NUM_ACTIVE * TILES * N2]
    uint8_t*       __restrict__ gmem_inter_fp4,  // [NUM_ACTIVE * N_HALF/2]
    uint8_t*       __restrict__ gmem_inter_sf)   // [NUM_ACTIVE * N_HALF/SF_BLOCK]
{
    cg::grid_group grid = cg::this_grid();

    const int eidx    = blockIdx.x / TILES;
    const int tile    = blockIdx.x % TILES;
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    if (eidx >= NUM_ACTIVE) return;

    const int eid = expert_ids[eidx];
    const float wt = expert_wts[eidx];

    // SMEM layout
    extern __shared__ char smem_raw[];
    uint8_t* s_A   = (uint8_t*)smem_raw;
    uint8_t* s_B   = s_A + SMEM_A;
    uint8_t* s_SFA = s_B + SMEM_B;
    uint8_t* s_SFB = s_SFA + SMEM_SFA;

    // ================================================================
    // PHASE 1: GEMM1 with NVF4 MMA — K-distributed
    // ================================================================
    {
        const int k_start     = tile * K_PER_TILE;
        const int k_start_pk  = k_start / 2;
        const int k_start_sf  = k_start / SF_BLOCK;

        const uint8_t* w1_fp4 = all_w1_fp4 + (long long)eid * N2 * K_PACKED;
        const uint8_t* w1_sf  = all_w1_sf  + (long long)eid * N2 * SF_COLS_W1;

        const int N_PASSES = N2 / BN;  // 32

        for (int np = 0; np < N_PASSES; np++) {
            const int n_off = np * BN;

            // Load A: input FP4 [16, 32 bytes] — only row 0 valid (M=1)
            for (int i = tid; i < SMEM_A; i += BLOCK_SIZE) {
                int row = i / (BK / 2);
                int col = i % (BK / 2);
                s_A[i] = (row == 0) ? input_fp4[k_start_pk + col] : 0;
            }

            // Load B: weight FP4 [64, 32 bytes]
            for (int i = tid; i < SMEM_B; i += BLOCK_SIZE) {
                int row = i / (BK / 2);
                int col = i % (BK / 2);
                s_B[i] = w1_fp4[(long long)(n_off + row) * K_PACKED + k_start_pk + col];
            }

            // Load SFA: 2 UE8M0 scale bytes for K-tile
            if (tid < (BK / SF_BLOCK))
                s_SFA[tid] = input_sf[k_start_sf + tid];

            // Load SFB: [64, 2] weight scale bytes
            for (int i = tid; i < BN * (BK / SF_BLOCK); i += BLOCK_SIZE) {
                int row = i / (BK / SF_BLOCK);
                int col = i % (BK / SF_BLOCK);
                s_SFB[i] = w1_sf[(long long)(n_off + row) * SF_COLS_W1 + k_start_sf + col];
            }

            __syncthreads();

            // MMA: each warp handles 8 output columns
            {
                uint32_t a_regs[4];
                pack_a_m1(a_regs, s_A, lane_id);

                uint32_t b_regs[2];
                int b_base = warp_id * MMA_N * (BK / 2);
                pack_b(b_regs, s_B + b_base, lane_id);

                // SFA: same for all threads (input K-block scales)
                uint16_t sfa_pk = (uint16_t)s_SFA[0] |
                                  ((uint16_t)s_SFA[1] << 8);

                // SFB: MMA applies same scale to all N cols within tile
                // Use first column's scales as representative
                int g_sf_n   = warp_id * 8;  // first N-col in this warp
                int sfb_idx  = g_sf_n * (BK / SF_BLOCK);
                uint16_t sfb_pk = (uint16_t)s_SFB[sfb_idx] |
                                  ((uint16_t)s_SFB[sfb_idx + 1] << 8);

                float acc[4] = {0, 0, 0, 0};
                mma_nvf4_m16n8k64(acc, a_regs, b_regs, acc,
                                  (uint32_t)sfa_pk, (uint32_t)sfb_pk);

                // CLayout: t%4==0 threads hold M=0 result in d[0], N=t/4
                if (lane_id % 4 == 0) {
                    int n_col = lane_id / 4;  // 0..7
                    long long pb = (long long)eidx * TILES * N2 +
                                   (long long)tile * N2;
                    partials[pb + n_off + warp_id * 8 + n_col] = acc[0];
                }
            }
            __syncthreads();
        }
    }

    // ================================================================
    // GRID SYNC — all GEMM1 partials written
    // ================================================================
    grid.sync();

    // ================================================================
    // PHASE 2: Reduce + SwiGLU + FP4 requant (leader CTAs, tile==0)
    // ================================================================
    if (tile == 0) {
        constexpr int COLS_PER_THR = N_HALF / BLOCK_SIZE;  // 4

        float sw_vals[COLS_PER_THR];
        long long p_base = (long long)eidx * TILES * N2;

        for (int ci = 0; ci < COLS_PER_THR; ci++) {
            int col = tid * COLS_PER_THR + ci;
            float gate_sum = 0.0f, up_sum = 0.0f;
            for (int t = 0; t < TILES; t++) {
                long long tb = p_base + (long long)t * N2;
                gate_sum += partials[tb + col];
                up_sum   += partials[tb + N_HALF + col];
            }
            sw_vals[ci] = up_sum * d_silu(gate_sum);
        }

        // Block max for FP4 quantization (8 threads per SF group)
        float local_max = 0.0f;
        for (int ci = 0; ci < COLS_PER_THR; ci++)
            local_max = fmaxf(local_max, fabsf(sw_vals[ci]));

        // Reduce across 8-lane group (SF_BLOCK=32 / 4 cols per thread = 8)
        float gmax = local_max;
        gmax = fmaxf(gmax, __shfl_xor_sync(0xFFFFFFFF, gmax, 4));
        gmax = fmaxf(gmax, __shfl_xor_sync(0xFFFFFFFF, gmax, 2));
        gmax = fmaxf(gmax, __shfl_xor_sync(0xFFFFFFFF, gmax, 1));

        // UE8M0 scale (bias=127)
        float scale = fmaxf(gmax / 6.0f, 1e-30f);
        int exp_val = 127 + (int)ceilf(log2f(scale));
        exp_val = max(1, min(254, exp_val));
        float actual_scale = exp2f((float)(exp_val - 127));

        // Quantize & pack FP4
        int base_col = tid * COLS_PER_THR;
        int byte0 = base_col / 2;

        uint8_t fp4_0 = quantize_e2m1_dev(sw_vals[0] / actual_scale);
        uint8_t fp4_1 = quantize_e2m1_dev(sw_vals[1] / actual_scale);
        uint8_t fp4_2 = quantize_e2m1_dev(sw_vals[2] / actual_scale);
        uint8_t fp4_3 = quantize_e2m1_dev(sw_vals[3] / actual_scale);

        gmem_inter_fp4[eidx * N_HALF_PACKED + byte0]     = fp4_0 | (fp4_1 << 4);
        gmem_inter_fp4[eidx * N_HALF_PACKED + byte0 + 1] = fp4_2 | (fp4_3 << 4);

        // Write scale (first thread in each 8-lane group)
        if ((lane_id % 8) == 0) {
            int group = tid / 8;
            gmem_inter_sf[eidx * (N_HALF / SF_BLOCK) + group] = (uint8_t)exp_val;
        }
    }

    // ================================================================
    // GRID SYNC — intermediate FP4 ready
    // ================================================================
    grid.sync();

    // ================================================================
    // PHASE 3: GEMM2 with NVF4 MMA — N-distributed
    // ================================================================
    {
        const int j_start = tile * 64;  // 64 output cols per CTA

        const uint8_t* w2_fp4 = all_w2_fp4 + (long long)eid * HIDDEN * N_HALF_PACKED;
        const uint8_t* w2_sf  = all_w2_sf  + (long long)eid * HIDDEN * SF_COLS_W2;

        const int K_PASSES = N_HALF / BK;  // 16

        float acc[4] = {0, 0, 0, 0};

        for (int kp = 0; kp < K_PASSES; kp++) {
            const int k_off    = kp * BK;
            const int k_off_pk = k_off / 2;
            const int k_off_sf = k_off / SF_BLOCK;

            // Load A: intermediate FP4 [16, 32 bytes] — row 0 only
            for (int i = tid; i < SMEM_A; i += BLOCK_SIZE) {
                int row = i / (BK / 2);
                int col = i % (BK / 2);
                s_A[i] = (row == 0) ?
                    gmem_inter_fp4[eidx * N_HALF_PACKED + k_off_pk + col] : 0;
            }

            // Load B: W2 weight FP4 [64, 32 bytes]
            // W2 layout: [HIDDEN, N_HALF/2], row = output col, col = packed intermediate
            for (int i = tid; i < SMEM_B; i += BLOCK_SIZE) {
                int row = i / (BK / 2);
                int col = i % (BK / 2);
                int out_col = j_start + row;
                s_B[i] = (out_col < HIDDEN) ?
                    w2_fp4[(long long)out_col * N_HALF_PACKED + k_off_pk + col] : 0;
            }

            // Load SFA: intermediate scales
            if (tid < (BK / SF_BLOCK))
                s_SFA[tid] = gmem_inter_sf[eidx * (N_HALF / SF_BLOCK) + k_off_sf + tid];

            // Load SFB: W2 weight scales [64, 2]
            for (int i = tid; i < BN * (BK / SF_BLOCK); i += BLOCK_SIZE) {
                int row = i / (BK / SF_BLOCK);
                int col = i % (BK / SF_BLOCK);
                int out_col = j_start + row;
                s_SFB[i] = (out_col < HIDDEN) ?
                    w2_sf[(long long)out_col * SF_COLS_W2 + k_off_sf + col] : 0;
            }

            __syncthreads();

            // MMA
            {
                uint32_t a_regs[4];
                pack_a_m1(a_regs, s_A, lane_id);

                uint32_t b_regs[2];
                int b_base = warp_id * MMA_N * (BK / 2);
                pack_b(b_regs, s_B + b_base, lane_id);

                uint16_t sfa_pk = (uint16_t)s_SFA[0] |
                                  ((uint16_t)s_SFA[1] << 8);

                // SFB: uniform across N cols in tile
                int g_sf_n  = warp_id * 8;
                int sfb_idx = g_sf_n * (BK / SF_BLOCK);
                uint16_t sfb_pk = (uint16_t)s_SFB[sfb_idx] |
                                  ((uint16_t)s_SFB[sfb_idx + 1] << 8);

                mma_nvf4_m16n8k64(acc, a_regs, b_regs, acc,
                                  (uint32_t)sfa_pk, (uint32_t)sfb_pk);
            }
            __syncthreads();
        }

        // CLayout: t%4==0 threads hold M=0 in d[0], N=t/4
        if (lane_id % 4 == 0) {
            int n_col = lane_id / 4;
            int out_j = j_start + warp_id * 8 + n_col;
            if (out_j < HIDDEN)
                atomicAdd(&output[out_j], wt * acc[0]);
        }
    }
}

// ============================================================================
// Host: FP4 Quantization
// ============================================================================
void quantize_to_nvfp4(
    const float* data, int numel,
    uint8_t* packed_out, uint8_t* sf_out, int sf_block = 32)
{
    int num_blocks = (numel + sf_block - 1) / sf_block;
    memset(packed_out, 0, numel / 2);

    for (int b = 0; b < num_blocks; b++) {
        int start = b * sf_block;
        int end = std::min(start + sf_block, numel);

        float bmax = 0;
        for (int i = start; i < end; i++)
            bmax = std::max(bmax, fabsf(data[i]));

        float scale = bmax / 6.0f;
        if (scale < 1e-30f) scale = 1e-30f;

        int exp_val = 127 + (int)ceilf(log2f(scale));
        exp_val = std::max(1, std::min(254, exp_val));
        sf_out[b] = (uint8_t)exp_val;

        float actual_scale = powf(2.0f, (float)(exp_val - 127));

        for (int i = start; i < end; i++) {
            float scaled = data[i] / actual_scale;
            float av = fabsf(scaled);
            int sign = (scaled < 0) ? 1 : 0;
            int idx = 0;
            float best_diff = av;
            for (int j = 1; j < 8; j++) {
                float diff = fabsf(av - E2M1_TABLE[j]);
                if (diff < best_diff) { best_diff = diff; idx = j; }
            }
            uint8_t fp4 = (uint8_t)((sign << 3) | idx);
            int byte_idx = i / 2;
            if (i % 2 == 0)
                packed_out[byte_idx] = fp4;
            else
                packed_out[byte_idx] |= (fp4 << 4);
        }
    }
}

// Dequantize single FP4 element
float dequant_fp4(const uint8_t* packed, const uint8_t* sf,
                  int idx, int sf_block = 32)
{
    int byte_idx = idx / 2;
    uint8_t byte = packed[byte_idx];
    uint8_t nib = (idx & 1) ? (byte >> 4) : (byte & 0xF);
    int sign = (nib >> 3) & 1;
    int mag = nib & 7;
    float val = E2M1_TABLE[mag];
    float scale = powf(2.0f, (float)((int)sf[idx / sf_block] - 127));
    return sign ? -val * scale : val * scale;
}

// ============================================================================
// Host: FP32 Reference
// ============================================================================
float host_silu(float x) { return x / (1.0f + expf(-x)); }

void host_reference(
    const float* input,        // [HIDDEN]
    const float* all_w1,       // [E, N2, HIDDEN] row-major
    const float* all_w2,       // [E, HIDDEN, N_HALF] row-major
    const int* expert_ids,
    const float* expert_wts,
    float* output,             // [HIDDEN]
    int num_active)
{
    memset(output, 0, HIDDEN * sizeof(float));

    for (int e = 0; e < num_active; e++) {
        int eid = expert_ids[e];
        float wt = expert_wts[e];

        const float* ew1 = all_w1 + (long long)eid * N2 * HIDDEN;
        const float* ew2 = all_w2 + (long long)eid * HIDDEN * N_HALF;

        float gate[N_HALF], up[N_HALF];
        for (int n = 0; n < N_HALF; n++) {
            float sum = 0;
            for (int k = 0; k < HIDDEN; k++)
                sum += input[k] * ew1[(long long)n * HIDDEN + k];
            gate[n] = sum;
        }
        for (int n = 0; n < N_HALF; n++) {
            float sum = 0;
            for (int k = 0; k < HIDDEN; k++)
                sum += input[k] * ew1[(long long)(n + N_HALF) * HIDDEN + k];
            up[n] = sum;
        }

        float inter[N_HALF];
        for (int n = 0; n < N_HALF; n++)
            inter[n] = up[n] * host_silu(gate[n]);

        for (int j = 0; j < HIDDEN; j++) {
            float sum = 0;
            for (int n = 0; n < N_HALF; n++)
                sum += inter[n] * ew2[(long long)j * N_HALF + n];
            output[j] += wt * sum;
        }
    }
}

// Quantized reference: same FP4 data as kernel
void host_quantized_reference(
    const uint8_t* input_fp4, const uint8_t* input_sf,
    const uint8_t* all_w1_fp4, const uint8_t* all_w1_sf,
    const uint8_t* all_w2_fp4, const uint8_t* all_w2_sf,
    const int* expert_ids, const float* expert_wts,
    float* output, int num_active)
{
    memset(output, 0, HIDDEN * sizeof(float));

    for (int e = 0; e < num_active; e++) {
        int eid = expert_ids[e];
        float wt = expert_wts[e];

        const uint8_t* w1f = all_w1_fp4 + (long long)eid * N2 * K_PACKED;
        const uint8_t* w1s = all_w1_sf  + (long long)eid * N2 * SF_COLS_W1;
        const uint8_t* w2f = all_w2_fp4 + (long long)eid * HIDDEN * N_HALF_PACKED;
        const uint8_t* w2s = all_w2_sf  + (long long)eid * HIDDEN * SF_COLS_W2;

        // GEMM1 using dequantized FP4
        float gate[N_HALF], up_arr[N_HALF];
        for (int n = 0; n < N_HALF; n++) {
            float sum = 0;
            for (int k = 0; k < HIDDEN; k++) {
                float a = dequant_fp4(input_fp4, input_sf, k);
                float b = dequant_fp4(w1f + (long long)n * K_PACKED,
                                      w1s + (long long)n * SF_COLS_W1, k);
                sum += a * b;
            }
            gate[n] = sum;
        }
        for (int n = 0; n < N_HALF; n++) {
            float sum = 0;
            for (int k = 0; k < HIDDEN; k++) {
                float a = dequant_fp4(input_fp4, input_sf, k);
                float b = dequant_fp4(w1f + (long long)(n + N_HALF) * K_PACKED,
                                      w1s + (long long)(n + N_HALF) * SF_COLS_W1, k);
                sum += a * b;
            }
            up_arr[n] = sum;
        }

        // SwiGLU + FP4 requant
        float sw[N_HALF];
        for (int n = 0; n < N_HALF; n++)
            sw[n] = up_arr[n] * host_silu(gate[n]);

        // Requant to FP4
        uint8_t inter_fp4[N_HALF_PACKED];
        uint8_t inter_sf[N_HALF / SF_BLOCK];
        memset(inter_fp4, 0, sizeof(inter_fp4));
        quantize_to_nvfp4(sw, N_HALF, inter_fp4, inter_sf);

        // GEMM2 using dequantized FP4
        for (int j = 0; j < HIDDEN; j++) {
            float sum = 0;
            for (int n = 0; n < N_HALF; n++) {
                float a = dequant_fp4(inter_fp4, inter_sf, n);
                float b = dequant_fp4(w2f + (long long)j * N_HALF_PACKED,
                                      w2s + (long long)j * SF_COLS_W2, n);
                sum += a * b;
            }
            output[j] += wt * sum;
        }
    }
}

// ============================================================================
// Main: Correctness Test + Benchmark
// ============================================================================
int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM%d%d, %d SMs)\n", prop.name,
           prop.major, prop.minor, prop.multiProcessorCount);

    // Check cooperative launch support
    int supports_coop = 0;
    cudaDeviceGetAttribute(&supports_coop,
        cudaDevAttrCooperativeLaunch, 0);
    if (!supports_coop) {
        printf("ERROR: Device does not support cooperative launch\n");
        return 1;
    }

    const int NUM_EXPERTS = 16;  // local experts for test
    const int TOTAL_CTAS = NUM_ACTIVE * TILES;  // 640

    printf("Config: K=%d, N_half=%d, experts=%d, tiles=%d, CTAs=%d\n",
           HIDDEN, N_HALF, NUM_ACTIVE, TILES, TOTAL_CTAS);

    // Check max CTAs for cooperative launch
    int max_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks, verdict_mma_cooperative, BLOCK_SIZE, SMEM_TOTAL);
    printf("Max CTAs/SM: %d, Total capacity: %d, Need: %d\n",
           max_blocks, max_blocks * prop.multiProcessorCount, TOTAL_CTAS);
    if (max_blocks * prop.multiProcessorCount < TOTAL_CTAS) {
        printf("WARNING: Not enough capacity for cooperative launch\n");
    }

    // ====== Generate random data ======
    srand(42);
    auto randf = []() { return ((float)rand() / RAND_MAX - 0.5f) * 2.0f; };

    float* h_input = new float[HIDDEN];
    for (int i = 0; i < HIDDEN; i++) h_input[i] = randf();

    float* h_w1 = new float[(long long)NUM_EXPERTS * N2 * HIDDEN];
    for (long long i = 0; i < (long long)NUM_EXPERTS * N2 * HIDDEN; i++)
        h_w1[i] = randf() * 0.1f;

    float* h_w2 = new float[(long long)NUM_EXPERTS * HIDDEN * N_HALF];
    for (long long i = 0; i < (long long)NUM_EXPERTS * HIDDEN * N_HALF; i++)
        h_w2[i] = randf() * 0.1f;

    int h_expert_ids[NUM_ACTIVE];
    float h_expert_wts[NUM_ACTIVE];
    for (int i = 0; i < NUM_ACTIVE; i++) {
        h_expert_ids[i] = i % NUM_EXPERTS;
        h_expert_wts[i] = 1.0f / NUM_ACTIVE;
    }

    // ====== Quantize to FP4 ======
    printf("\nQuantizing to NVFP4...\n");

    size_t inp_fp4_sz = HIDDEN / 2;
    size_t inp_sf_sz  = HIDDEN / SF_BLOCK;
    uint8_t* h_inp_fp4 = new uint8_t[inp_fp4_sz]();
    uint8_t* h_inp_sf  = new uint8_t[inp_sf_sz]();
    quantize_to_nvfp4(h_input, HIDDEN, h_inp_fp4, h_inp_sf);

    size_t w1_fp4_sz = (long long)NUM_EXPERTS * N2 * K_PACKED;
    size_t w1_sf_sz  = (long long)NUM_EXPERTS * N2 * SF_COLS_W1;
    uint8_t* h_w1_fp4 = new uint8_t[w1_fp4_sz]();
    uint8_t* h_w1_sf  = new uint8_t[w1_sf_sz]();
    for (int e = 0; e < NUM_EXPERTS; e++) {
        for (int n = 0; n < N2; n++) {
            quantize_to_nvfp4(
                &h_w1[(long long)e * N2 * HIDDEN + (long long)n * HIDDEN],
                HIDDEN,
                &h_w1_fp4[(long long)e * N2 * K_PACKED + (long long)n * K_PACKED],
                &h_w1_sf[(long long)e * N2 * SF_COLS_W1 + (long long)n * SF_COLS_W1]);
        }
    }

    size_t w2_fp4_sz = (long long)NUM_EXPERTS * HIDDEN * N_HALF_PACKED;
    size_t w2_sf_sz  = (long long)NUM_EXPERTS * HIDDEN * SF_COLS_W2;
    uint8_t* h_w2_fp4 = new uint8_t[w2_fp4_sz]();
    uint8_t* h_w2_sf  = new uint8_t[w2_sf_sz]();
    for (int e = 0; e < NUM_EXPERTS; e++) {
        for (int j = 0; j < HIDDEN; j++) {
            quantize_to_nvfp4(
                &h_w2[(long long)e * HIDDEN * N_HALF + (long long)j * N_HALF],
                N_HALF,
                &h_w2_fp4[(long long)e * HIDDEN * N_HALF_PACKED + (long long)j * N_HALF_PACKED],
                &h_w2_sf[(long long)e * HIDDEN * SF_COLS_W2 + (long long)j * SF_COLS_W2]);
        }
    }

    // ====== FP32 Reference ======
    printf("Computing FP32 reference...\n");
    float* h_ref_output = new float[HIDDEN]();
    host_reference(h_input, h_w1, h_w2,
                   h_expert_ids, h_expert_wts, h_ref_output, NUM_ACTIVE);

    // ====== Quantized Reference ======
    printf("Computing quantized reference...\n");
    float* h_qref_output = new float[HIDDEN]();
    host_quantized_reference(h_inp_fp4, h_inp_sf,
                             h_w1_fp4, h_w1_sf,
                             h_w2_fp4, h_w2_sf,
                             h_expert_ids, h_expert_wts,
                             h_qref_output, NUM_ACTIVE);

    // ====== Upload to GPU ======
    printf("Uploading to GPU...\n");
    uint8_t *d_inp_fp4, *d_inp_sf;
    uint8_t *d_w1_fp4, *d_w1_sf;
    uint8_t *d_w2_fp4, *d_w2_sf;
    int *d_expert_ids;
    float *d_expert_wts;
    float *d_output, *d_partials;
    uint8_t *d_inter_fp4, *d_inter_sf;

    cudaMalloc(&d_inp_fp4, inp_fp4_sz);
    cudaMalloc(&d_inp_sf, inp_sf_sz);
    cudaMalloc(&d_w1_fp4, w1_fp4_sz);
    cudaMalloc(&d_w1_sf, w1_sf_sz);
    cudaMalloc(&d_w2_fp4, w2_fp4_sz);
    cudaMalloc(&d_w2_sf, w2_sf_sz);
    cudaMalloc(&d_expert_ids, NUM_ACTIVE * sizeof(int));
    cudaMalloc(&d_expert_wts, NUM_ACTIVE * sizeof(float));
    cudaMalloc(&d_output, HIDDEN * sizeof(float));
    cudaMalloc(&d_partials, (long long)NUM_ACTIVE * TILES * N2 * sizeof(float));
    cudaMalloc(&d_inter_fp4, NUM_ACTIVE * N_HALF_PACKED);
    cudaMalloc(&d_inter_sf, NUM_ACTIVE * (N_HALF / SF_BLOCK));

    cudaMemcpy(d_inp_fp4, h_inp_fp4, inp_fp4_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inp_sf, h_inp_sf, inp_sf_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w1_fp4, h_w1_fp4, w1_fp4_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w1_sf, h_w1_sf, w1_sf_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2_fp4, h_w2_fp4, w2_fp4_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2_sf, h_w2_sf, w2_sf_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_expert_ids, h_expert_ids, NUM_ACTIVE * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_expert_wts, h_expert_wts, NUM_ACTIVE * sizeof(float),
               cudaMemcpyHostToDevice);

    // ====== Run MMA Cooperative Kernel ======
    printf("\nLaunching cooperative MMA kernel (%d CTAs)...\n", TOTAL_CTAS);

    cudaFuncSetAttribute(verdict_mma_cooperative,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);

    cudaMemset(d_output, 0, HIDDEN * sizeof(float));

    void* args[] = {
        &d_inp_fp4, &d_inp_sf,
        &d_w1_fp4, &d_w1_sf,
        &d_w2_fp4, &d_w2_sf,
        &d_expert_ids, &d_expert_wts,
        &d_output, &d_partials,
        &d_inter_fp4, &d_inter_sf
    };

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)verdict_mma_cooperative,
        dim3(TOTAL_CTAS), dim3(BLOCK_SIZE),
        args, SMEM_TOTAL, 0);

    if (err != cudaSuccess) {
        printf("Launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // ====== Correctness Check ======
    float* h_gpu_output = new float[HIDDEN];
    cudaMemcpy(h_gpu_output, d_output, HIDDEN * sizeof(float),
               cudaMemcpyDeviceToHost);

    printf("\n=== Correctness ===\n");
    printf("Output[0:8] (GPU MMA):   ");
    for (int i = 0; i < 8; i++) printf("%10.6f ", h_gpu_output[i]);
    printf("\nOutput[0:8] (Quant ref): ");
    for (int i = 0; i < 8; i++) printf("%10.6f ", h_qref_output[i]);
    printf("\nOutput[0:8] (FP32 ref):  ");
    for (int i = 0; i < 8; i++) printf("%10.6f ", h_ref_output[i]);
    printf("\n");

    // Compute errors
    double mma_vs_qref_sum = 0, mma_vs_fp32_sum = 0;
    double qref_norm = 0, fp32_norm = 0;
    int nan_count = 0;

    for (int i = 0; i < HIDDEN; i++) {
        if (isnan(h_gpu_output[i]) || isinf(h_gpu_output[i])) { nan_count++; continue; }
        double d1 = h_gpu_output[i] - h_qref_output[i];
        double d2 = h_gpu_output[i] - h_ref_output[i];
        mma_vs_qref_sum += d1 * d1;
        mma_vs_fp32_sum += d2 * d2;
        qref_norm += (double)h_qref_output[i] * h_qref_output[i];
        fp32_norm += (double)h_ref_output[i] * h_ref_output[i];
    }

    double rmse_qref = sqrt(mma_vs_qref_sum / HIDDEN);
    double rmse_fp32 = sqrt(mma_vs_fp32_sum / HIDDEN);
    double rel_err_qref = (qref_norm > 0) ? sqrt(mma_vs_qref_sum / qref_norm) : 0;
    double rel_err_fp32 = (fp32_norm > 0) ? sqrt(mma_vs_fp32_sum / fp32_norm) : 0;

    printf("\nMMA vs Quantized Ref: RMSE=%.6f, RelErr=%.4f%%\n",
           rmse_qref, rel_err_qref * 100);
    printf("MMA vs FP32 Ref:      RMSE=%.6f, RelErr=%.4f%%\n",
           rmse_fp32, rel_err_fp32 * 100);
    printf("NaN/Inf count: %d / %d\n", nan_count, HIDDEN);

    bool passed = (nan_count == 0) && (rel_err_fp32 < 0.50);  // 50% tolerance for FP4
    printf("Correctness: %s\n", passed ? "PASS" : "FAIL");

    // ====== Benchmark ======
    printf("\n=== Benchmark ===\n");
    const int WARMUP = 20;
    const int ITERS  = 100;

    // Warmup
    for (int i = 0; i < WARMUP; i++) {
        cudaMemset(d_output, 0, HIDDEN * sizeof(float));
        cudaLaunchCooperativeKernel((void*)verdict_mma_cooperative,
            dim3(TOTAL_CTAS), dim3(BLOCK_SIZE), args, SMEM_TOTAL, 0);
    }
    cudaDeviceSynchronize();

    // Timed iterations
    std::vector<float> times;
    for (int i = 0; i < ITERS; i++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaMemset(d_output, 0, HIDDEN * sizeof(float));
        cudaEventRecord(start);
        cudaLaunchCooperativeKernel((void*)verdict_mma_cooperative,
            dim3(TOTAL_CTAS), dim3(BLOCK_SIZE), args, SMEM_TOTAL, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        times.push_back(ms * 1000.0f);  // μs

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    std::sort(times.begin(), times.end());
    float median = times[ITERS / 2];
    float p10    = times[ITERS / 10];
    float p90    = times[ITERS * 9 / 10];
    float mean   = std::accumulate(times.begin(), times.end(), 0.0f) / ITERS;

    printf("Latency (μs/layer): median=%.1f, mean=%.1f, p10=%.1f, p90=%.1f\n",
           median, mean, p10, p90);
    printf("Comparison: Scalar=~280μs, VLLM_CUTLASS=98μs, "
           "Cooperative FP32=38.9μs\n");

    // ====== Cleanup ======
    delete[] h_input; delete[] h_w1; delete[] h_w2;
    delete[] h_inp_fp4; delete[] h_inp_sf;
    delete[] h_w1_fp4; delete[] h_w1_sf;
    delete[] h_w2_fp4; delete[] h_w2_sf;
    delete[] h_ref_output; delete[] h_qref_output; delete[] h_gpu_output;

    cudaFree(d_inp_fp4); cudaFree(d_inp_sf);
    cudaFree(d_w1_fp4); cudaFree(d_w1_sf);
    cudaFree(d_w2_fp4); cudaFree(d_w2_sf);
    cudaFree(d_expert_ids); cudaFree(d_expert_wts);
    cudaFree(d_output); cudaFree(d_partials);
    cudaFree(d_inter_fp4); cudaFree(d_inter_sf);

    printf("\nDone.\n");
    return passed ? 0 : 1;
}
