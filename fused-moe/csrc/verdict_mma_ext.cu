/**
 * VerdictMoE MMA Extension for vLLM — SM120 Blackwell
 *
 * MMA-based fused MoE 5-kernel pipeline (CUDA-graph safe, no cooperative groups):
 *   0. bf16_to_nvfp4:       Quantize BF16 input to NVFP4 (E2M1 + UE8M0 scales)
 *   1. gemm1_mma:           MMA GEMM1 with FP4 input × FP4 weights → partials
 *   2. swiglu_reduce_fp4:   Reduce partials + SwiGLU + FP4 requant
 *   3. gemm2_mma_scatter:   MMA GEMM2 with FP4 intermediate × FP4 weights → output
 *   4. convert_f32_to_bf16: Final output conversion
 *
 * Ported from validated cooperative kernel (Sprint 4 Task 2).
 * Scale format bridge: vLLM weights use E4M3FN scales (SF_BLOCK=16),
 * MMA instruction uses UE8M0. Conversion done on-the-fly in SMEM.
 *
 * Build: torch JIT with -gencode=arch=compute_120a,code=sm_120a -O2
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// ============================================================================
// Constants
// ============================================================================
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE  = 32;
constexpr int NUM_WARPS  = BLOCK_SIZE / WARP_SIZE;  // 8
constexpr int BM = 16;    // MMA M dimension
constexpr int BK = 64;    // MMA K dimension (NVF4 native)
constexpr int BN = 64;    // N columns per MMA pass (8 warps × 8)
constexpr int MMA_N = 8;  // MMA N dimension

constexpr int SF_BLOCK_ACT = 32;  // Our activation/intermediate quantization block size
constexpr int SF_BLOCK_WT  = 16;  // vLLM NVFP4 weight scale block size
constexpr int SF_PER_TILE_ACT = BK / SF_BLOCK_ACT;  // 2 scales per K=64 tile
constexpr int SF_PER_TILE_WT  = BK / SF_BLOCK_WT;   // 4 scales per K=64 tile

// SMEM sizes for MMA tile data
constexpr int SMEM_A = BM * (BK / 2);      // 512 bytes
constexpr int SMEM_B = BN * (BK / 2);      // 2048 bytes

// Total SMEM: A + B + SFA(16) + SFB_raw(256) + SFB_ratio(1024) + SFB_ue8m0(128) + pad
constexpr int SMEM_MMA = 4096;

// ============================================================================
// Device Helpers
// ============================================================================

// E2M1 lookup (inline, avoids __constant__ issues)
__device__ __forceinline__ float d_e2m1_val(int idx) {
    switch (idx & 7) {
        case 0: return 0.0f;  case 1: return 0.5f;
        case 2: return 1.0f;  case 3: return 1.5f;
        case 4: return 2.0f;  case 5: return 3.0f;
        case 6: return 4.0f;  case 7: return 6.0f;
    }
    return 0.0f;
}

// E4M3FN decode (unsigned, for scale factors)
__device__ __forceinline__ float decode_e4m3fn_u(uint8_t x) {
    int e = (x >> 3) & 0xF;
    int m = x & 7;
    if (e == 15 && m == 7) return 0.0f;  // NaN → 0
    if (e == 0) return __int2float_rn(m) * 0.001953125f;  // subnormal: m * 2^-9
    return ldexpf(1.0f + __int2float_rn(m) * 0.125f, e - 7);
}

// Swizzle<3,4,3> — validated Sprint 4 Task 1
__device__ __forceinline__ uint32_t swizzle_343(uint32_t byte_offset) {
    return byte_offset ^ ((byte_offset >> 3) & 0x70u);
}

// Read FP4 nibble from swizzled SMEM
__device__ __forceinline__ uint32_t get_nibble_swz(
    const uint8_t* smem, int row_byte_off, int k)
{
    int addr = row_byte_off + k / 2;
    uint8_t bv = smem[swizzle_343(addr)];
    return (k & 1) ? ((bv >> 4) & 0xFu) : (bv & 0xFu);
}

// Rescale FP4 nibble using float ratio (ratio = old_scale / unified_scale, <= 1.0)
__device__ __forceinline__ uint32_t rescale_nib_f(uint32_t nib, float ratio) {
    int mag = nib & 7;
    if (mag == 0) return nib;  // 0.0 stays 0.0
    int sign = (nib >> 3) & 1;
    float val = d_e2m1_val(mag) * ratio;
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

// E2M1 quantization (nearest)
__device__ __forceinline__ uint8_t quantize_e2m1(float value) {
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

__device__ __forceinline__ float d_silu(float x) {
    return x / (1.0f + __expf(-x));
}

// Block-scaled NVF4 MMA: m16n8k64, scale_vec::2X, UE8M0 scales
// Validated Sprint 4 Task 2
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

// ============================================================================
// Shared helper: Precompute B-operand scale ratios in SMEM
// Converts E4M3FN weight scales → unified UE8M0 per row + float ratios
// ============================================================================
__device__ __forceinline__ void precompute_sfb(
    const uint8_t* s_SFB_raw,   // [BN * SF_PER_TILE_WT] E4M3FN bytes
    float*         s_SFB_ratio,  // [BN * SF_PER_TILE_WT] output ratios
    uint8_t*       s_SFB_ue8m0,  // [BN * 2] unified UE8M0 per row per K-half
    int tid)
{
    // Compute 2 UE8M0 values per row (one per K-half), matching scale_vec::2X.
    // K-half 0: scale groups 0,1 (K=0..31)
    // K-half 1: scale groups 2,3 (K=32..63)
    if (tid < BN) {
        float sf[SF_PER_TILE_WT];
        for (int j = 0; j < SF_PER_TILE_WT; j++)
            sf[j] = decode_e4m3fn_u(s_SFB_raw[tid * SF_PER_TILE_WT + j]);

        // K-half 0: merge groups 0,1
        float smax0 = fmaxf(sf[0], sf[1]);
        float sf_norm0 = fmaxf(smax0, 1e-30f);
        int ue0 = 127 + (int)ceilf(log2f(sf_norm0));
        ue0 = max(1, min(254, ue0));
        float unified0 = exp2f((float)(ue0 - 127));

        // K-half 1: merge groups 2,3
        float smax1 = fmaxf(sf[2], sf[3]);
        float sf_norm1 = fmaxf(smax1, 1e-30f);
        int ue1 = 127 + (int)ceilf(log2f(sf_norm1));
        ue1 = max(1, min(254, ue1));
        float unified1 = exp2f((float)(ue1 - 127));

        s_SFB_ue8m0[tid * 2]     = (uint8_t)ue0;
        s_SFB_ue8m0[tid * 2 + 1] = (uint8_t)ue1;

        // Ratios: each group divided by its K-half's unified scale
        s_SFB_ratio[tid * SF_PER_TILE_WT + 0] = sf[0] / unified0;
        s_SFB_ratio[tid * SF_PER_TILE_WT + 1] = sf[1] / unified0;
        s_SFB_ratio[tid * SF_PER_TILE_WT + 2] = sf[2] / unified1;
        s_SFB_ratio[tid * SF_PER_TILE_WT + 3] = sf[3] / unified1;
    }
}

// ============================================================================
// Shared helper: Rescale B-operand in SMEM using precomputed ratios
// ============================================================================
__device__ __forceinline__ void rescale_B_smem(
    uint8_t*       s_B,
    const float*   s_SFB_ratio,
    int tid)
{
    for (int i = tid; i < SMEM_B; i += BLOCK_SIZE) {
        int row = i / (BK / 2);
        int col = i % (BK / 2);
        int sg = (col * 2) / SF_BLOCK_WT;  // scale group (0..3)
        float ratio = s_SFB_ratio[row * SF_PER_TILE_WT + sg];
        if (ratio >= 0.999f) continue;  // no rescaling needed

        int swz_i = swizzle_343(i);
        uint8_t bv = s_B[swz_i];
        uint8_t nlo = (uint8_t)rescale_nib_f(bv & 0xF, ratio);
        uint8_t nhi = (uint8_t)rescale_nib_f((bv >> 4) & 0xF, ratio);
        s_B[swz_i] = (nhi << 4) | nlo;
    }
}

// ============================================================================
// Shared helper: Rescale A-operand in SMEM (UE8M0 scales, row 0 only for M=1)
// Returns unified UE8M0 byte
// ============================================================================
__device__ __forceinline__ uint8_t rescale_A_smem(
    uint8_t* s_A, const uint8_t* s_SFA, int tid)
{
    uint8_t sfa_max = (s_SFA[0] > s_SFA[1]) ? s_SFA[0] : s_SFA[1];
    if (s_SFA[0] != s_SFA[1]) {
        float r0 = exp2f((float)((int)s_SFA[0] - (int)sfa_max));
        float r1 = exp2f((float)((int)s_SFA[1] - (int)sfa_max));
        for (int i = tid; i < (BK / 2); i += BLOCK_SIZE) {
            int swz_i = swizzle_343(i);
            uint8_t bv = s_A[swz_i];
            int k_lo = i * 2, k_hi = k_lo + 1;
            float ratio_lo = (k_lo < 32) ? r0 : r1;
            float ratio_hi = (k_hi < 32) ? r0 : r1;
            uint8_t nlo = (uint8_t)rescale_nib_f(bv & 0xF, ratio_lo);
            uint8_t nhi = (uint8_t)rescale_nib_f((bv >> 4) & 0xF, ratio_hi);
            s_A[swz_i] = (nhi << 4) | nlo;
        }
    }
    return sfa_max;
}

// ============================================================================
// Shared helper: Pack A registers + Pack B registers + MMA + write output
// For M=1 decode. Writes 2 partials per warp (lanes 0-3 active).
// ============================================================================
__device__ __forceinline__ void do_mma_pass(
    const uint8_t* s_A, const uint8_t* s_B,
    uint8_t sfa_ue8m0, const uint8_t* s_SFB_ue8m0,
    int warp_id, int lane_id,
    float& out0, float& out1,
    bool accumulate)
{
    // Pack A (M=1, only group 0 = lanes 0-3)
    uint32_t a_regs[4] = {0, 0, 0, 0};
    if (lane_id / 4 == 0) {
        int t0 = lane_id % 4;
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            a_regs[0] |= get_nibble_swz(s_A, 0, t0 + p * 8) << (p * 4);
            a_regs[2] |= get_nibble_swz(s_A, 0, t0 + 4 + p * 8) << (p * 4);
        }
    }

    // Pack B
    uint32_t b_regs[2] = {0, 0};
    {
        int g = lane_id / 4;
        int t0 = lane_id % 4;
        int N_local = 4 * (g & 1) + (g >> 1);
        int rbo = (warp_id * 8 + N_local) * (BK / 2);
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            b_regs[0] |= get_nibble_swz(s_B, rbo, t0 + p * 8) << (p * 4);
            b_regs[1] |= get_nibble_swz(s_B, rbo, t0 + 4 + p * 8) << (p * 4);
        }
    }

    // Per-K-half scales: scale_vec::2X packs byte0=K_half0, byte1=K_half1
    uint32_t sfa_pk = (uint32_t)sfa_ue8m0 | ((uint32_t)sfa_ue8m0 << 8);
    int g = lane_id / 4;
    int N_local = 4 * (g & 1) + (g >> 1);
    int sfb_row = warp_id * 8 + N_local;
    uint8_t sfb_lo = s_SFB_ue8m0[sfb_row * 2];      // K-half 0
    uint8_t sfb_hi = s_SFB_ue8m0[sfb_row * 2 + 1];  // K-half 1
    uint32_t sfb_pk = (uint32_t)sfb_lo | ((uint32_t)sfb_hi << 8);

    float c_in[4] = {0, 0, 0, 0};
    if (accumulate) { c_in[0] = out0; c_in[1] = out1; }
    float acc[4];
    mma_nvf4_m16n8k64(acc, a_regs, b_regs, c_in, sfa_pk, sfb_pk);
    out0 = acc[0];
    out1 = acc[1];
}

// ============================================================================
// Kernel 0: BF16 → NVFP4 (UE8M0 scales, SF_BLOCK=32)
// ============================================================================
__global__ void bf16_to_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ input,  // [M, K]
    uint8_t* __restrict__ out_fp4,            // [M, K/2]
    uint8_t* __restrict__ out_sf,             // [M, K/32]
    int M, int K)
{
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    int gid = blockIdx.x * warps_per_block + (threadIdx.x / WARP_SIZE);
    int lane = threadIdx.x % WARP_SIZE;
    int total_groups = M * (K / SF_BLOCK_ACT);
    if (gid >= total_groups) return;

    int m = gid / (K / SF_BLOCK_ACT);
    int g = gid % (K / SF_BLOCK_ACT);
    int k_base = g * SF_BLOCK_ACT;

    float val = __bfloat162float(input[m * K + k_base + lane]);
    float aval = fabsf(val);

    // Warp reduce max
    float wmax = aval;
    for (int off = 16; off > 0; off >>= 1)
        wmax = fmaxf(wmax, __shfl_xor_sync(0xFFFFFFFF, wmax, off));

    // UE8M0 scale
    float sf = fmaxf(wmax / 6.0f, 1e-30f);
    int exp_val = 127 + (int)ceilf(log2f(sf));
    exp_val = max(1, min(254, exp_val));
    float actual_scale = exp2f((float)(exp_val - 127));

    // Quantize
    uint8_t nib = quantize_e2m1(val / actual_scale);

    // Pack: even lanes write low nibble, odd lanes provide high nibble
    uint8_t partner = __shfl_xor_sync(0xFFFFFFFF, nib, 1);
    if ((lane & 1) == 0)
        out_fp4[m * (K / 2) + k_base / 2 + lane / 2] = nib | (partner << 4);

    if (lane == 0)
        out_sf[m * (K / SF_BLOCK_ACT) + g] = (uint8_t)exp_val;
}

// ============================================================================
// Kernel 1: GEMM1 MMA — K-distributed, FP4 input × FP4 weights
// ============================================================================
// Grid: (num_active * tiles_per_expert), Block: 256
// Each CTA: one K-tile, loops over N_PASSES for all N2 output columns
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
verdict_gemm1_mma(
    const uint8_t* __restrict__ input_fp4,   // [M, K/2]
    const uint8_t* __restrict__ input_sf,    // [M, K/32]
    const uint8_t* __restrict__ all_w1_fp4,  // [E, N2, K/2]
    const uint8_t* __restrict__ all_w1_sf,   // [E, N2, K/16] E4M3FN
    const float*   __restrict__ w1_alpha,    // [num_active]
    float*         __restrict__ partials,    // [num_active, tiles, N2]
    const int*     __restrict__ expert_ids,
    const int*     __restrict__ token_ids,
    int K, int N_half, int num_active, int tiles_per_expert)
{
    const int eidx = blockIdx.x / tiles_per_expert;
    const int tile = blockIdx.x % tiles_per_expert;
    if (eidx >= num_active) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int eid   = expert_ids[eidx];
    const int tok   = token_ids[eidx];
    const float alpha = w1_alpha[eidx];

    const int N2 = 2 * N_half;
    const int K_packed = K / 2;
    const int sf_cols_wt = K / SF_BLOCK_WT;
    const int sf_cols_act = K / SF_BLOCK_ACT;
    const int k_start = tile * BK;
    const int k_start_pk = k_start / 2;

    const uint8_t* w1_fp4 = all_w1_fp4 + (long long)eid * N2 * K_packed;
    const uint8_t* w1_sf  = all_w1_sf  + (long long)eid * N2 * sf_cols_wt;

    // SMEM layout
    extern __shared__ char smem_raw[];
    uint8_t* s_A         = (uint8_t*)smem_raw;
    uint8_t* s_B         = s_A + SMEM_A;
    uint8_t* s_SFA       = s_B + SMEM_B;
    uint8_t* s_SFB_raw   = s_SFA + 16;
    float*   s_SFB_ratio = (float*)(s_SFB_raw + BN * SF_PER_TILE_WT);
    uint8_t* s_SFB_ue8m0 = (uint8_t*)(s_SFB_ratio + BN * SF_PER_TILE_WT);

    const int N_PASSES = N2 / BN;

    for (int np = 0; np < N_PASSES; np++) {
        const int n_off = np * BN;

        // Load A: input FP4 row [BM, BK/2] — only row 0 valid (M=1)
        for (int i = tid; i < SMEM_A; i += BLOCK_SIZE) {
            int row = i / (BK / 2);
            int col = i % (BK / 2);
            uint8_t val = (row == 0) ? input_fp4[tok * K_packed + k_start_pk + col] : 0;
            s_A[swizzle_343(i)] = val;
        }

        // Load B: weight FP4 [BN, BK/2]
        for (int i = tid; i < SMEM_B; i += BLOCK_SIZE) {
            int row = i / (BK / 2);
            int col = i % (BK / 2);
            s_B[swizzle_343(i)] = w1_fp4[(long long)(n_off + row) * K_packed + k_start_pk + col];
        }

        // Load SFA: 2 UE8M0 input scale bytes
        if (tid < SF_PER_TILE_ACT)
            s_SFA[tid] = input_sf[tok * sf_cols_act + k_start / SF_BLOCK_ACT + tid];

        // Load SFB: [BN, 4] E4M3FN weight scale bytes
        for (int i = tid; i < BN * SF_PER_TILE_WT; i += BLOCK_SIZE) {
            int row = i / SF_PER_TILE_WT;
            int col = i % SF_PER_TILE_WT;
            s_SFB_raw[i] = w1_sf[(long long)(n_off + row) * sf_cols_wt + k_start / SF_BLOCK_WT + col];
        }

        __syncthreads();

        // Precompute B scale ratios
        precompute_sfb(s_SFB_raw, s_SFB_ratio, s_SFB_ue8m0, tid);
        __syncthreads();

        // DISABLED rescaling for debugging — use raw FP4 data + computed scales
        // uint8_t sfa_ue8m0 = rescale_A_smem(s_A, s_SFA, tid);
        // rescale_B_smem(s_B, s_SFB_ratio, tid);
        uint8_t sfa_ue8m0 = (s_SFA[0] > s_SFA[1]) ? s_SFA[0] : s_SFA[1];
        __syncthreads();

        // MMA + write partials
        float out0 = 0.0f, out1 = 0.0f;

        // BYPASS: skip rescaling, use fixed scale 127 (1.0) for debugging
        // do_mma_pass(s_A, s_B, sfa_ue8m0, s_SFB_ue8m0,
        //             warp_id, lane_id, out0, out1, false);
        {
            // Pack A
            uint32_t a_regs[4] = {0, 0, 0, 0};
            int g = lane_id / 4;
            int t0 = lane_id % 4;
            if (g == 0) {
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    a_regs[0] |= get_nibble_swz(s_A, 0, t0 + p*8) << (p*4);
                    a_regs[2] |= get_nibble_swz(s_A, 0, t0+4 + p*8) << (p*4);
                }
            }
            // Pack B
            uint32_t b_regs[2] = {0, 0};
            {
                int N_local = 4*(g&1) + (g>>1);
                int rbo = (warp_id*8 + N_local) * (BK/2);
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    b_regs[0] |= get_nibble_swz(s_B, rbo, t0+p*8) << (p*4);
                    b_regs[1] |= get_nibble_swz(s_B, rbo, t0+4+p*8) << (p*4);
                }
            }
            // DEBUG: use fixed scale 127 (=1.0) for both A and B
            uint32_t sfa_pk = 127u | (127u << 8);
            uint32_t sfb_pk = 127u | (127u << 8);
            float c_zero[4] = {0,0,0,0};
            float acc[4];
            mma_nvf4_m16n8k64(acc, a_regs, b_regs, c_zero, sfa_pk, sfb_pk);
            out0 = acc[0];
            out1 = acc[1];
        }

        if (lane_id < 4) {
            long long pb = (long long)eidx * tiles_per_expert * N2
                         + (long long)tile * N2;
            partials[pb + n_off + warp_id * 8 + lane_id]     = alpha * out0;
            partials[pb + n_off + warp_id * 8 + lane_id + 4] = alpha * out1;
        }
        __syncthreads();
    }
}

// ============================================================================
// Kernel 2: Reduce partials + SwiGLU + FP4 requant
// ============================================================================
// Grid: (num_active), Block: 256
// Outputs FP4-packed intermediate with UE8M0 scales (SF_BLOCK=32)
__global__ void verdict_swiglu_reduce_fp4(
    const float* __restrict__ partials,    // [num_active, tiles, N2]
    uint8_t*     __restrict__ inter_fp4,   // [num_active, N_half/2]
    uint8_t*     __restrict__ inter_sf,    // [num_active, N_half/32]
    int N_half, int num_active, int tiles_per_expert)
{
    const int eidx = blockIdx.x;
    if (eidx >= num_active) return;
    const int tid     = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;

    const int N2 = 2 * N_half;
    const int COLS_PER_THR = N_half / BLOCK_SIZE;
    const int THREADS_PER_GROUP = SF_BLOCK_ACT / COLS_PER_THR;

    long long p_base = (long long)eidx * tiles_per_expert * N2;

    // Compute SwiGLU values in registers
    float sw_vals[8];  // max COLS_PER_THR=8 for N_half=2048
    float local_max = 0.0f;

    for (int ci = 0; ci < COLS_PER_THR; ci++) {
        int col = tid * COLS_PER_THR + ci;
        float gate_sum = 0.0f, up_sum = 0.0f;
        for (int t = 0; t < tiles_per_expert; t++) {
            long long tb = p_base + (long long)t * N2;
            gate_sum += partials[tb + col];
            up_sum   += partials[tb + N_half + col];
        }
        sw_vals[ci] = up_sum * d_silu(gate_sum);
        local_max = fmaxf(local_max, fabsf(sw_vals[ci]));
    }

    // Reduce max within SF group (THREADS_PER_GROUP threads)
    float gmax = local_max;
    for (int off = THREADS_PER_GROUP / 2; off > 0; off >>= 1)
        gmax = fmaxf(gmax, __shfl_xor_sync(0xFFFFFFFF, gmax, off));

    // UE8M0 scale
    float scale = fmaxf(gmax / 6.0f, 1e-30f);
    int exp_val = 127 + (int)ceilf(log2f(scale));
    exp_val = max(1, min(254, exp_val));
    float actual_scale = exp2f((float)(exp_val - 127));

    // Quantize & pack FP4 (COLS_PER_THR values → COLS_PER_THR/2 bytes)
    int base_col = tid * COLS_PER_THR;
    int base_byte = base_col / 2;

    for (int ci = 0; ci < COLS_PER_THR; ci += 2) {
        uint8_t nib0 = quantize_e2m1(sw_vals[ci]     / actual_scale);
        uint8_t nib1 = quantize_e2m1(sw_vals[ci + 1] / actual_scale);
        inter_fp4[eidx * (N_half / 2) + base_byte + ci / 2] = nib0 | (nib1 << 4);
    }

    // Write scale (one thread per SF group)
    if ((lane_id % THREADS_PER_GROUP) == 0) {
        int group = tid / THREADS_PER_GROUP;
        inter_sf[eidx * (N_half / SF_BLOCK_ACT) + group] = (uint8_t)exp_val;
    }
}

// ============================================================================
// Kernel 3: GEMM2 MMA + weighted scatter
// ============================================================================
// Grid: (num_active * tiles_per_expert), Block: 256
// Each CTA: 64 output columns, accumulates across K_PASSES of intermediate
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
verdict_gemm2_mma_scatter(
    const uint8_t* __restrict__ inter_fp4,   // [num_active, N_half/2]
    const uint8_t* __restrict__ inter_sf,    // [num_active, N_half/32]
    const uint8_t* __restrict__ all_w2_fp4,  // [E, K, N_half/2]
    const uint8_t* __restrict__ all_w2_sf,   // [E, K, N_half/16] E4M3FN
    const float*   __restrict__ w2_alpha,
    const int*     __restrict__ expert_ids,
    const float*   __restrict__ expert_wts,
    const int*     __restrict__ token_ids,
    float*         __restrict__ output_f32,  // [M, K]
    int K, int N_half, int num_active, int tiles_per_expert)
{
    const int eidx = blockIdx.x / tiles_per_expert;
    const int tile = blockIdx.x % tiles_per_expert;
    if (eidx >= num_active) return;

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int eid   = expert_ids[eidx];
    const int tok   = token_ids[eidx];
    const float wt  = expert_wts[eidx];
    const float alpha = w2_alpha[eidx];

    const int N_half_packed = N_half / 2;
    const int sf_cols_wt = N_half / SF_BLOCK_WT;
    const int sf_cols_act = N_half / SF_BLOCK_ACT;
    const int j_start = tile * BN;  // output column base

    const uint8_t* w2_fp4 = all_w2_fp4 + (long long)eid * K * N_half_packed;
    const uint8_t* w2_sf  = all_w2_sf  + (long long)eid * K * sf_cols_wt;

    // SMEM layout (same as GEMM1)
    extern __shared__ char smem_raw[];
    uint8_t* s_A         = (uint8_t*)smem_raw;
    uint8_t* s_B         = s_A + SMEM_A;
    uint8_t* s_SFA       = s_B + SMEM_B;
    uint8_t* s_SFB_raw   = s_SFA + 16;
    float*   s_SFB_ratio = (float*)(s_SFB_raw + BN * SF_PER_TILE_WT);
    uint8_t* s_SFB_ue8m0 = (uint8_t*)(s_SFB_ratio + BN * SF_PER_TILE_WT);

    const int K_PASSES = N_half / BK;
    float acc0 = 0.0f, acc1 = 0.0f;

    for (int kp = 0; kp < K_PASSES; kp++) {
        const int k_off = kp * BK;
        const int k_off_pk = k_off / 2;

        // Load A: intermediate FP4 row [BM, BK/2] — row 0 only
        for (int i = tid; i < SMEM_A; i += BLOCK_SIZE) {
            int row = i / (BK / 2);
            int col = i % (BK / 2);
            uint8_t val = (row == 0) ?
                inter_fp4[eidx * N_half_packed + k_off_pk + col] : 0;
            s_A[swizzle_343(i)] = val;
        }

        // Load B: W2 weight FP4 [BN, BK/2]
        // W2 layout: [K, N_half/2], row = output col, col = packed intermediate
        for (int i = tid; i < SMEM_B; i += BLOCK_SIZE) {
            int row = i / (BK / 2);
            int col = i % (BK / 2);
            int out_col = j_start + row;
            uint8_t val = (out_col < K) ?
                w2_fp4[(long long)out_col * N_half_packed + k_off_pk + col] : 0;
            s_B[swizzle_343(i)] = val;
        }

        // Load SFA: 2 UE8M0 intermediate scale bytes
        if (tid < SF_PER_TILE_ACT)
            s_SFA[tid] = inter_sf[eidx * sf_cols_act + k_off / SF_BLOCK_ACT + tid];

        // Load SFB: [BN, 4] E4M3FN W2 scale bytes
        for (int i = tid; i < BN * SF_PER_TILE_WT; i += BLOCK_SIZE) {
            int row = i / SF_PER_TILE_WT;
            int col = i % SF_PER_TILE_WT;
            int out_col = j_start + row;
            s_SFB_raw[i] = (out_col < K) ?
                w2_sf[(long long)out_col * sf_cols_wt + k_off / SF_BLOCK_WT + col] : 0;
        }

        __syncthreads();

        // Precompute B scale ratios
        precompute_sfb(s_SFB_raw, s_SFB_ratio, s_SFB_ue8m0, tid);
        __syncthreads();

        // Rescale A and B in SMEM
        uint8_t sfa_ue8m0 = rescale_A_smem(s_A, s_SFA, tid);
        rescale_B_smem(s_B, s_SFB_ratio, tid);
        __syncthreads();

        // MMA (accumulate across K-passes)
        do_mma_pass(s_A, s_B, sfa_ue8m0, s_SFB_ue8m0,
                    warp_id, lane_id, acc0, acc1, kp > 0);

        __syncthreads();
    }

    // Write output: weighted atomicAdd
    if (lane_id < 4) {
        int j0 = j_start + warp_id * 8 + lane_id;
        int j1 = j0 + 4;
        float scale = wt * alpha;
        if (j0 < K) atomicAdd(&output_f32[tok * K + j0], scale * acc0);
        if (j1 < K) atomicAdd(&output_f32[tok * K + j1], scale * acc1);
    }
}

// ============================================================================
// Kernel 4: F32 → BF16
// ============================================================================
__global__ void convert_f32_to_bf16(
    const float* __restrict__ f32_buf,
    __nv_bfloat16* __restrict__ bf16_out,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) bf16_out[i] = __float2bfloat16(f32_buf[i]);
}

// ============================================================================
// Host Orchestrator
// ============================================================================
void verdict_mma_forward(
    torch::Tensor input,         // [M, K] BF16
    torch::Tensor w1_fp4,        // [E, 2*N, K//2] uint8
    torch::Tensor w1_sf,         // [E, 2*N, K//16] uint8 (E4M3FN)
    torch::Tensor w1_alpha,      // [num_active] float32
    torch::Tensor w2_fp4,        // [E, K, N//2] uint8
    torch::Tensor w2_sf,         // [E, K, N//16] uint8 (E4M3FN)
    torch::Tensor w2_alpha,      // [num_active] float32
    torch::Tensor output,        // [M, K] BF16
    torch::Tensor expert_ids,    // [num_active] int32
    torch::Tensor expert_wts,    // [num_active] float32
    torch::Tensor token_ids,     // [num_active] int32
    torch::Tensor partials,      // [num_active * tiles * 2 * N_half] float32
    torch::Tensor output_f32,    // [M * K] float32
    torch::Tensor input_fp4_buf, // [M * K/2] uint8
    torch::Tensor input_sf_buf,  // [M * K/32] uint8
    torch::Tensor inter_fp4_buf, // [num_active * N_half/2] uint8
    torch::Tensor inter_sf_buf,  // [num_active * N_half/32] uint8
    int K, int N_half, int num_active, int tiles_per_expert)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    int M = input.size(0);
    int total_blocks = num_active * tiles_per_expert;
    int N2 = 2 * N_half;

    // Zero output scratch
    cudaMemsetAsync(output_f32.data_ptr(), 0,
                    output_f32.numel() * sizeof(float), stream);

    // Kernel 0: BF16 → NVFP4
    {
        int num_groups = M * (K / SF_BLOCK_ACT);
        int k0_blocks = (num_groups + (BLOCK_SIZE / WARP_SIZE) - 1)
                      / (BLOCK_SIZE / WARP_SIZE);
        bf16_to_nvfp4_kernel<<<k0_blocks, BLOCK_SIZE, 0, stream>>>(
            (const __nv_bfloat16*)input.data_ptr(),
            (uint8_t*)input_fp4_buf.data_ptr(),
            (uint8_t*)input_sf_buf.data_ptr(),
            M, K);
    }

    // Kernel 1: GEMM1 MMA
    verdict_gemm1_mma<<<total_blocks, BLOCK_SIZE, SMEM_MMA, stream>>>(
        (const uint8_t*)input_fp4_buf.data_ptr(),
        (const uint8_t*)input_sf_buf.data_ptr(),
        (const uint8_t*)w1_fp4.data_ptr(),
        (const uint8_t*)w1_sf.data_ptr(),
        (const float*)w1_alpha.data_ptr(),
        (float*)partials.data_ptr(),
        (const int*)expert_ids.data_ptr(),
        (const int*)token_ids.data_ptr(),
        K, N_half, num_active, tiles_per_expert);

    // Kernel 2: SwiGLU + FP4 requant
    verdict_swiglu_reduce_fp4<<<num_active, BLOCK_SIZE, 0, stream>>>(
        (const float*)partials.data_ptr(),
        (uint8_t*)inter_fp4_buf.data_ptr(),
        (uint8_t*)inter_sf_buf.data_ptr(),
        N_half, num_active, tiles_per_expert);

    // Kernel 3: GEMM2 MMA + scatter
    verdict_gemm2_mma_scatter<<<total_blocks, BLOCK_SIZE, SMEM_MMA, stream>>>(
        (const uint8_t*)inter_fp4_buf.data_ptr(),
        (const uint8_t*)inter_sf_buf.data_ptr(),
        (const uint8_t*)w2_fp4.data_ptr(),
        (const uint8_t*)w2_sf.data_ptr(),
        (const float*)w2_alpha.data_ptr(),
        (const int*)expert_ids.data_ptr(),
        (const float*)expert_wts.data_ptr(),
        (const int*)token_ids.data_ptr(),
        (float*)output_f32.data_ptr(),
        K, N_half, num_active, tiles_per_expert);

    // Kernel 4: F32 → BF16
    int total_elems = M * K;
    int conv_blocks = (total_elems + 255) / 256;
    convert_f32_to_bf16<<<conv_blocks, 256, 0, stream>>>(
        (const float*)output_f32.data_ptr(),
        (__nv_bfloat16*)output.data_ptr(),
        total_elems);
}

// ============================================================================
// PyBind11
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &verdict_mma_forward, "VerdictMoE MMA fused forward (NVFP4)");
}
