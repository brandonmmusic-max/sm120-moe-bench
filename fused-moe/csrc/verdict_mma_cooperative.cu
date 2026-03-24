/**
 * VerdictMoE Sprint 4 Task 2: Cooperative MMA Kernel with Swizzle Pipeline
 *
 * Fixes from Task 1 validation (empirically determined hardware layout):
 * - pack_a_m1_v2: group g=tid/4 -> M=2*g; a[0]/a[2]=M-even, a[1]/a[3]=M-odd
 * - pack_b_v2: group g=tid/4 -> N=4*(g%2)+g/2; both b[0]/b[1] same N
 * - C output: d[0]=C[2*(tid/4), tid%4], d[1]=C[2*(tid/4), tid%4+4]
 * - SFB per-thread-group (each group's N column gets its own scale)
 * - SMEM writes/reads use Swizzle<3,4,3> for ldmatrix-compatible layout
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

// E2M1 value table (host only - device code uses inline function)
static const float E2M1_TABLE[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// Device-side E2M1 lookup (avoids __constant__ issues with -rdc=true)
__device__ __forceinline__ float d_e2m1_val(int idx) {
    // Computed inline to avoid constant memory with relocatable device code
    switch (idx & 7) {
        case 0: return 0.0f;
        case 1: return 0.5f;
        case 2: return 1.0f;
        case 3: return 1.5f;
        case 4: return 2.0f;
        case 5: return 3.0f;
        case 6: return 4.0f;
        case 7: return 6.0f;
    }
    return 0.0f;
}

// ============================================================================
// Swizzle<3,4,3> SMEM Layout (validated Sprint 4 Task 1)
// ============================================================================
__device__ __forceinline__ uint32_t swizzle_343(uint32_t byte_offset) {
    return byte_offset ^ ((byte_offset >> 3) & 0x70u);
}

// Get FP4 nibble from swizzled SMEM
__device__ __forceinline__ uint32_t get_nibble_swz(
    const uint8_t* smem, int row_byte_off, int k)
{
    int linear_addr = row_byte_off + k / 2;
    uint8_t byte_val = smem[swizzle_343(linear_addr)];
    return (k & 1) ? ((byte_val >> 4) & 0xFu) : (byte_val & 0xFu);
}

// ============================================================================
// CRITICAL DISCOVERY: MMA scale_vec::2X byte-to-register mapping
// ============================================================================
//
// The mxf4nvf4 block_scale MMA with scale_vec::2X applies scales PER REGISTER
// GROUP, NOT per contiguous K block:
//   SFA byte 0 → scales ALL nibbles in a[0] and a[1]
//   SFA byte 1 → scales ALL nibbles in a[2] and a[3]
//   SFA bytes 2,3 → IGNORED
//   (Same pattern for SFB: byte 0 → b[0], byte 1 → b[1])
//
// Since a[0] contains K values {t0, t0+8, t0+16, ..., t0+56} which span
// BOTH SF_BLOCK=32 blocks, we must RESCALE nibbles during packing so that
// each register uses a uniform scale. We pick the MAX of the two block
// scales as the unified scale, and rescale nibbles from the smaller block.
//
// HARDWARE REGISTER LAYOUT (empirically determined, Sprint 4 Task 1):
//   A: group g=tid/4: a[0]=M=2g K=t0+p*8, a[2]=M=2g K=t0+4+p*8
//   B: group g=tid/4: N=4*(g%2)+g/2, b[0] K=t0+p*8, b[1] K=t0+4+p*8
//   C: d[0]=C[2*g, tid%4], d[1]=C[2*g, tid%4+4]

// Rescale an FP4 nibble from old_sf to new_sf (new_sf >= old_sf)
// Uses integer bit manipulation to avoid exp2f (which causes Heisenbugs with -rdc=true -O2)
__device__ __forceinline__ uint32_t rescale_nib(uint32_t nib, int sf_old, int sf_new) {
    if (sf_old == sf_new) return nib;
    int mag = nib & 7;
    if (mag == 0) return nib;  // 0.0 stays 0.0 regardless of scale
    int sign = (nib >> 3) & 1;
    float val = d_e2m1_val(mag);
    // scale ratio = 2^(sf_old - sf_new), always <= 1.0
    // Construct the float directly via IEEE 754 bit manipulation
    int diff = sf_new - sf_old;  // diff >= 0
    if (diff >= 24) return (sign << 3);  // rounds to 0
    // 2^(-diff) = IEEE float with exponent = 127 - diff
    float scale_ratio = __int_as_float((uint32_t)(127 - diff) << 23);
    val *= scale_ratio;
    // Re-quantize to nearest E2M1
    int idx;
    if      (val < 0.25f)  idx = 0;
    else if (val < 0.75f)  idx = 1;
    else if (val < 1.25f)  idx = 2;
    else if (val < 1.75f)  idx = 3;
    else if (val < 2.5f)   idx = 4;
    else if (val < 3.5f)   idx = 5;
    else if (val < 5.0f)   idx = 6;
    else                    idx = 7;
    return (sign << 3) | idx;
}

// Pack A for M=1 decode with per-register rescaling
// sf_a: [2] UE8M0 scale bytes for K blocks 0 and 1
// Returns the unified scale byte (max of the two)
__device__ __forceinline__ uint8_t pack_a_m1_rescaled(
    uint32_t (&a)[4], const uint8_t* s_A,
    const uint8_t* sf_a, int lane_id)
{
    a[0] = a[1] = a[2] = a[3] = 0;
    uint8_t sf_max = (sf_a[0] > sf_a[1]) ? sf_a[0] : sf_a[1];
    if (lane_id / 4 != 0) return sf_max;  // Only group 0 -> M=0
    int t0 = lane_id % 4;
    int sf0 = (int)sf_a[0], sf1 = (int)sf_a[1], sfm = (int)sf_max;
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        int k0 = t0 + p * 8;
        int k2 = t0 + 4 + p * 8;
        uint32_t nib0 = get_nibble_swz(s_A, 0, k0);
        uint32_t nib2 = get_nibble_swz(s_A, 0, k2);
        nib0 = rescale_nib(nib0, (k0 < 32) ? sf0 : sf1, sfm);
        nib2 = rescale_nib(nib2, (k2 < 32) ? sf0 : sf1, sfm);
        a[0] |= nib0 << (p * 4);
        a[2] |= nib2 << (p * 4);
    }
    return sf_max;
}

// Pack B with per-register rescaling
// sf_b: pointer to 2 UE8M0 bytes for this N column's K blocks
// Returns the unified scale byte
__device__ __forceinline__ uint8_t pack_b_rescaled(
    uint32_t (&b)[2], const uint8_t* s_B, int warp_n_base,
    const uint8_t* s_SFB, int lane_id)
{
    int g = lane_id / 4;
    int t0 = lane_id % 4;
    int N_local = 4 * (g & 1) + (g >> 1);
    int row_byte_off = (warp_n_base + N_local) * (BK / 2);

    // Get scale bytes for this N column
    int sfb_n = warp_n_base + N_local;
    int sf0 = (int)s_SFB[sfb_n * 2];
    int sf1 = (int)s_SFB[sfb_n * 2 + 1];
    int sfm = (sf0 > sf1) ? sf0 : sf1;

    b[0] = b[1] = 0;
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        int k0 = t0 + p * 8;
        int k2 = t0 + 4 + p * 8;
        uint32_t nib0 = get_nibble_swz(s_B, row_byte_off, k0);
        uint32_t nib2 = get_nibble_swz(s_B, row_byte_off, k2);
        nib0 = rescale_nib(nib0, (k0 < 32) ? sf0 : sf1, sfm);
        nib2 = rescale_nib(nib2, (k2 < 32) ? sf0 : sf1, sfm);
        b[0] |= nib0 << (p * 4);
        b[1] |= nib2 << (p * 4);
    }
    return (uint8_t)sfm;
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
            // Apply Swizzle<3,4,3> on write
            for (int i = tid; i < SMEM_A; i += BLOCK_SIZE) {
                int row = i / (BK / 2);
                int col = i % (BK / 2);
                uint8_t val = (row == 0) ? input_fp4[k_start_pk + col] : 0;
                s_A[swizzle_343(i)] = val;
            }

            // Load B: weight FP4 [64, 32 bytes]
            // Apply Swizzle<3,4,3> on write
            for (int i = tid; i < SMEM_B; i += BLOCK_SIZE) {
                int row = i / (BK / 2);
                int col = i % (BK / 2);
                s_B[swizzle_343(i)] = w1_fp4[(long long)(n_off + row) * K_PACKED + k_start_pk + col];
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

            // === RESCALE SMEM DATA IN-PLACE ===
            // Compute unified scale (max of 2 block scales) for A and B
            uint8_t sfa_max = (s_SFA[0] > s_SFA[1]) ? s_SFA[0] : s_SFA[1];

            // Rescale A row 0 in SMEM (32 bytes = 64 nibbles)
            if (s_SFA[0] != s_SFA[1]) {
                for (int i = tid; i < 32; i += BLOCK_SIZE) {
                    int swz_i = swizzle_343(i);
                    uint8_t byte_val = s_A[swz_i];
                    int k_lo = i * 2, k_hi = k_lo + 1;
                    uint8_t nib_lo = byte_val & 0xF;
                    uint8_t nib_hi = (byte_val >> 4) & 0xF;
                    int sf_lo = (k_lo < 32) ? (int)s_SFA[0] : (int)s_SFA[1];
                    int sf_hi = (k_hi < 32) ? (int)s_SFA[0] : (int)s_SFA[1];
                    nib_lo = (uint8_t)rescale_nib(nib_lo, sf_lo, (int)sfa_max);
                    nib_hi = (uint8_t)rescale_nib(nib_hi, sf_hi, (int)sfa_max);
                    s_A[swz_i] = (nib_hi << 4) | nib_lo;
                }
            }

            // Rescale B rows in SMEM (2048 bytes = 64 rows x 32 bytes)
            for (int i = tid; i < SMEM_B; i += BLOCK_SIZE) {
                int row = i / (BK / 2);   // N column (0..63)
                int col = i % (BK / 2);   // packed K byte
                int sfb_row = row;
                uint8_t sf0 = s_SFB[sfb_row * 2];
                uint8_t sf1 = s_SFB[sfb_row * 2 + 1];
                if (sf0 == sf1) continue;
                uint8_t sfm = (sf0 > sf1) ? sf0 : sf1;
                int swz_i = swizzle_343(i);
                uint8_t byte_val = s_B[swz_i];
                int k_lo = col * 2, k_hi = k_lo + 1;
                uint8_t nib_lo = byte_val & 0xF;
                uint8_t nib_hi = (byte_val >> 4) & 0xF;
                int sf_lo_v = (k_lo < 32) ? (int)sf0 : (int)sf1;
                int sf_hi_v = (k_hi < 32) ? (int)sf0 : (int)sf1;
                nib_lo = (uint8_t)rescale_nib(nib_lo, sf_lo_v, (int)sfm);
                nib_hi = (uint8_t)rescale_nib(nib_hi, sf_hi_v, (int)sfm);
                s_B[swz_i] = (nib_hi << 4) | nib_lo;
            }

            __syncthreads();

            // MMA: simple packing (data already rescaled in SMEM)
            {
                // Pack A (simple, no rescaling needed)
                uint32_t a_regs[4];
                a_regs[0] = a_regs[1] = a_regs[2] = a_regs[3] = 0;
                if (lane_id / 4 == 0) {
                    int t0 = lane_id % 4;
                    #pragma unroll
                    for (int p = 0; p < 8; p++) {
                        a_regs[0] |= get_nibble_swz(s_A, 0, t0 + p*8) << (p*4);
                        a_regs[2] |= get_nibble_swz(s_A, 0, t0 + 4 + p*8) << (p*4);
                    }
                }

                // Pack B (simple, no rescaling needed)
                uint32_t b_regs[2];
                {
                    int g = lane_id / 4;
                    int t0 = lane_id % 4;
                    int N_local = 4*(g&1) + (g>>1);
                    int rbo = (warp_id*8 + N_local) * (BK/2);
                    b_regs[0] = b_regs[1] = 0;
                    #pragma unroll
                    for (int p = 0; p < 8; p++) {
                        b_regs[0] |= get_nibble_swz(s_B, rbo, t0+p*8) << (p*4);
                        b_regs[1] |= get_nibble_swz(s_B, rbo, t0+4+p*8) << (p*4);
                    }
                }

                // Unified scales (same byte in both positions)
                uint32_t sfa_pk = (uint32_t)sfa_max | ((uint32_t)sfa_max << 8);
                int g = lane_id / 4;
                int N_local = 4*(g&1) + (g>>1);
                int sfb_n = warp_id*8 + N_local;
                uint8_t sfbm = (s_SFB[sfb_n*2] > s_SFB[sfb_n*2+1]) ?
                               s_SFB[sfb_n*2] : s_SFB[sfb_n*2+1];
                uint32_t sfb_pk = (uint32_t)sfbm | ((uint32_t)sfbm << 8);

                float acc[4] = {0, 0, 0, 0};
                mma_nvf4_m16n8k64(acc, a_regs, b_regs, acc,
                                  sfa_pk, sfb_pk);

                // C output: d[0]=C[0, lane_id], d[1]=C[0, lane_id+4]
                if (lane_id < 4) {
                    long long pb = (long long)eidx * TILES * N2 +
                                   (long long)tile * N2;
                    partials[pb + n_off + warp_id * 8 + lane_id]     = acc[0];
                    partials[pb + n_off + warp_id * 8 + lane_id + 4] = acc[1];
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
            // Apply Swizzle<3,4,3> on write
            for (int i = tid; i < SMEM_A; i += BLOCK_SIZE) {
                int row = i / (BK / 2);
                int col = i % (BK / 2);
                uint8_t val = (row == 0) ?
                    gmem_inter_fp4[eidx * N_HALF_PACKED + k_off_pk + col] : 0;
                s_A[swizzle_343(i)] = val;
            }

            // Load B: W2 weight FP4 [64, 32 bytes]
            // W2 layout: [HIDDEN, N_HALF/2], row = output col, col = packed intermediate
            // Apply Swizzle<3,4,3> on write
            for (int i = tid; i < SMEM_B; i += BLOCK_SIZE) {
                int row = i / (BK / 2);
                int col = i % (BK / 2);
                int out_col = j_start + row;
                uint8_t val = (out_col < HIDDEN) ?
                    w2_fp4[(long long)out_col * N_HALF_PACKED + k_off_pk + col] : 0;
                s_B[swizzle_343(i)] = val;
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

            // Rescale SMEM data in-place + simple MMA
            {
                uint8_t sfa_max2 = (s_SFA[0] > s_SFA[1]) ? s_SFA[0] : s_SFA[1];
                // Rescale A row 0
                if (s_SFA[0] != s_SFA[1]) {
                    for (int i = tid; i < 32; i += BLOCK_SIZE) {
                        int swz_i = swizzle_343(i);
                        uint8_t bv = s_A[swz_i];
                        int k_lo = i*2, k_hi = k_lo+1;
                        uint8_t nl = bv & 0xF, nh = (bv>>4) & 0xF;
                        nl = (uint8_t)rescale_nib(nl, (k_lo<32)?(int)s_SFA[0]:(int)s_SFA[1], (int)sfa_max2);
                        nh = (uint8_t)rescale_nib(nh, (k_hi<32)?(int)s_SFA[0]:(int)s_SFA[1], (int)sfa_max2);
                        s_A[swz_i] = (nh<<4)|nl;
                    }
                }
                // Rescale B rows
                for (int i = tid; i < SMEM_B; i += BLOCK_SIZE) {
                    int row = i/(BK/2), col = i%(BK/2);
                    uint8_t sf0 = s_SFB[row*2], sf1 = s_SFB[row*2+1];
                    if (sf0 == sf1) continue;
                    uint8_t sfm = (sf0>sf1)?sf0:sf1;
                    int swz_i = swizzle_343(i);
                    uint8_t bv = s_B[swz_i];
                    int k_lo = col*2, k_hi = k_lo+1;
                    uint8_t nl = bv&0xF, nh = (bv>>4)&0xF;
                    nl = (uint8_t)rescale_nib(nl, (k_lo<32)?(int)sf0:(int)sf1, (int)sfm);
                    nh = (uint8_t)rescale_nib(nh, (k_hi<32)?(int)sf0:(int)sf1, (int)sfm);
                    s_B[swz_i] = (nh<<4)|nl;
                }
                __syncthreads();

                // Simple packing (data already rescaled)
                uint32_t a_regs[4] = {0,0,0,0};
                if (lane_id/4 == 0) {
                    int t0 = lane_id%4;
                    for (int p=0;p<8;p++) {
                        a_regs[0] |= get_nibble_swz(s_A,0,t0+p*8)<<(p*4);
                        a_regs[2] |= get_nibble_swz(s_A,0,t0+4+p*8)<<(p*4);
                    }
                }
                uint32_t b_regs[2] = {0,0};
                {
                    int g=lane_id/4, t0=lane_id%4;
                    int N_loc = 4*(g&1)+(g>>1);
                    int rbo = (warp_id*8+N_loc)*(BK/2);
                    for (int p=0;p<8;p++) {
                        b_regs[0] |= get_nibble_swz(s_B,rbo,t0+p*8)<<(p*4);
                        b_regs[1] |= get_nibble_swz(s_B,rbo,t0+4+p*8)<<(p*4);
                    }
                }

                uint32_t sfa_pk2 = (uint32_t)sfa_max2 | ((uint32_t)sfa_max2<<8);
                int g2=lane_id/4, N_loc2=4*(g2&1)+(g2>>1);
                int sfb_n2 = warp_id*8+N_loc2;
                uint8_t sfbm2 = (s_SFB[sfb_n2*2]>s_SFB[sfb_n2*2+1]) ?
                                s_SFB[sfb_n2*2]:s_SFB[sfb_n2*2+1];
                uint32_t sfb_pk2 = (uint32_t)sfbm2 | ((uint32_t)sfbm2<<8);

                mma_nvf4_m16n8k64(acc, a_regs, b_regs, acc, sfa_pk2, sfb_pk2);
            }
            __syncthreads();
        }

        // C output: d[0]=C[0, lane_id], d[1]=C[0, lane_id+4]
        // Only group 0 (lanes 0-3) has M=0 results
        if (lane_id < 4) {
            int out_j0 = j_start + warp_id * 8 + lane_id;
            int out_j1 = j_start + warp_id * 8 + lane_id + 4;
            if (out_j0 < HIDDEN)
                atomicAdd(&output[out_j0], wt * acc[0]);
            if (out_j1 < HIDDEN)
                atomicAdd(&output[out_j1], wt * acc[1]);
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
        times.push_back(ms * 1000.0f);  // us

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    std::sort(times.begin(), times.end());
    float median = times[ITERS / 2];
    float p10    = times[ITERS / 10];
    float p90    = times[ITERS * 9 / 10];
    float mean   = std::accumulate(times.begin(), times.end(), 0.0f) / ITERS;

    printf("Latency (us/layer): median=%.1f, mean=%.1f, p10=%.1f, p90=%.1f\n",
           median, mean, p10, p90);
    printf("Comparison: Scalar=~280us, VLLM_CUTLASS=98us, "
           "Cooperative FP32=38.9us\n");

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
