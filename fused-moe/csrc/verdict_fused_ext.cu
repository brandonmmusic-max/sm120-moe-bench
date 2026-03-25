/**
 * VerdictMoE Split MMA Extension for vLLM — SM120 Blackwell
 * Sprint 6: Non-cooperative split kernel (CUDA-graph compatible)
 *
 * Two MMA kernels replacing the cooperative fused kernel:
 *   K1A: GEMM1 MMA — N-distributed, K-accumulated in registers
 *        + alpha × SwiGLU + E4M3 requant (per tile)
 *   K1B: GEMM2 MMA — output-distributed, N_half/BK K-passes
 *        + weighted atomicAdd scatter
 *
 * CUDA stream ordering guarantees K1A completes before K1B reads intermediate.
 * NO cooperative barriers. NO occupancy constraints. Works with any grid size.
 *
 * Wrapper kernels:
 *   K0: BF16 → NVFP4 (E4M3FN scales, SF_BLOCK=16)
 *   K2: F32 → BF16
 *
 * Key innovation: Consecutive-K packing aligns each MMA register with a
 * single SF_BLOCK=16 scale block, enabling raw E4M3FN checkpoint scales
 * to pass directly to the MMA instruction. Zero rescaling overhead.
 *
 * Build: torch JIT with -gencode=arch=compute_120a,code=sm_120a -O2
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// ============================================================================
// MMA Tile Constants (hardware-fixed)
// ============================================================================
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE  = 32;
constexpr int NUM_WARPS  = BLOCK_SIZE / WARP_SIZE;  // 8
constexpr int BM = 16;    // MMA M dimension
constexpr int BK = 64;    // MMA K dimension (NVF4 native)
constexpr int BN = 64;    // N columns per tile (8 warps × 8)

constexpr int SF_BLOCK   = 16;   // NVFP4 scale block size
constexpr int SF_PER_K   = BK / SF_BLOCK;  // 4

// SMEM layout for MMA tiles + SwiGLU scratch
constexpr int SMEM_A       = BM * (BK / 2);       // 512
constexpr int SMEM_B       = BN * (BK / 2);       // 2048
constexpr int SMEM_SFA_PAD = 16;
constexpr int SMEM_SFB     = BN * SF_PER_K;       // 256
constexpr int SMEM_SW      = BN * sizeof(float);   // 256 (SwiGLU scratch)
constexpr int SMEM_K1A     = SMEM_A + SMEM_B + SMEM_SFA_PAD + SMEM_SFB + SMEM_SW + 128;
constexpr int SMEM_K1B     = SMEM_A + SMEM_B + SMEM_SFA_PAD + SMEM_SFB + 128;

// ============================================================================
// Device Helpers
// ============================================================================
__device__ __forceinline__ float d_e4m3fn_decode_u(uint8_t x) {
    int e = (x >> 3) & 0xF, m = x & 7;
    if (e == 15 && m == 7) return 0.0f;
    if (e == 0) return __int2float_rn(m) * 0.001953125f;
    return ldexpf(1.0f + __int2float_rn(m) * 0.125f, e - 7);
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
    int sign = (value < 0.0f) ? 1 : 0;
    int idx;
    if      (av < 0.25f) idx = 0;
    else if (av < 0.75f) idx = 1;
    else if (av < 1.25f) idx = 2;
    else if (av < 1.75f) idx = 3;
    else if (av < 2.5f)  idx = 4;
    else if (av < 3.5f)  idx = 5;
    else if (av < 5.0f)  idx = 6;
    else                  idx = 7;
    return (uint8_t)((sign << 3) | idx);
}

__device__ __forceinline__ uint32_t swizzle_343(uint32_t off) {
    return off ^ ((off >> 3) & 0x70u);
}

__device__ __forceinline__ uint32_t get_nibble_swz(
    const uint8_t* smem, int rbo, int k)
{
    int addr = rbo + k / 2;
    uint8_t bv = smem[swizzle_343(addr)];
    return (k & 1) ? ((bv >> 4) & 0xFu) : (bv & 0xFu);
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
// Fast consecutive-K packing — direct uint32 SMEM loads
// ============================================================================
// With consecutive-K storage, 4 bytes at s_B[swizzle_343(rbo + t0*4)]
// contain exactly the 8 FP4 nibbles K[t0*8..t0*8+7] in MMA register layout.
// swizzle_343 preserves 4-byte alignment (XOR only touches bits [6:4]).
// Replaces 16 scalar nibble reads + 16 shift-ORs with 2 uint32 loads.
__device__ __forceinline__ void pack_a_consec(
    uint32_t (&a)[4], const uint8_t* s_A, int lane_id)
{
    a[0] = a[1] = a[2] = a[3] = 0;
    if (lane_id / 4 == 0) {
        int t0 = lane_id % 4;
        a[0] = *(const uint32_t*)&s_A[swizzle_343(t0 * 4)];
        a[2] = *(const uint32_t*)&s_A[swizzle_343(16 + t0 * 4)];
    }
}

__device__ __forceinline__ void pack_b_consec(
    uint32_t (&b)[2], const uint8_t* s_B, int rbo, int lane_id)
{
    int t0 = lane_id % 4;
    b[0] = *(const uint32_t*)&s_B[swizzle_343(rbo + t0 * 4)];
    b[1] = *(const uint32_t*)&s_B[swizzle_343(rbo + 16 + t0 * 4)];
}

__device__ __forceinline__ uint32_t pack_sf4(const uint8_t* sf) {
    return *(const uint32_t*)sf;
}

// ============================================================================
// Kernel 0: BF16 → NVFP4 with E4M3FN scales (SF_BLOCK=16)
// ============================================================================
__global__ void bf16_to_nvfp4_e4m3_kernel(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_sf,
    int M, int K)
{
    const int half_warps_per_block = BLOCK_SIZE / 16;
    int gid = blockIdx.x * half_warps_per_block + (threadIdx.x / 16);
    int lane = threadIdx.x % 16;
    int sf_groups_per_row = K / SF_BLOCK;
    int total_groups = M * sf_groups_per_row;
    if (gid >= total_groups) return;

    int m = gid / sf_groups_per_row;
    int g = gid % sf_groups_per_row;
    int k_base = g * SF_BLOCK;

    float val = __bfloat162float(input[m * K + k_base + lane]);
    float aval = fabsf(val);

    float wmax = aval;
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1)
        wmax = fmaxf(wmax, __shfl_xor_sync(0xFFFFFFFF, wmax, off));

    float sf_target = fmaxf(wmax / 6.0f, 1e-30f);
    uint8_t sf_byte = d_e4m3fn_encode(sf_target);
    float actual_scale = d_e4m3fn_decode_u(sf_byte);
    if (actual_scale < 1e-30f) actual_scale = 1e-30f;

    uint8_t nib = d_quantize_e2m1(val / actual_scale);

    uint8_t partner_nib = __shfl_xor_sync(0xFFFFFFFF, nib, 1);
    if ((lane & 1) == 0)
        out_fp4[m * (K / 2) + k_base / 2 + lane / 2] = nib | (partner_nib << 4);

    if (lane == 0)
        out_sf[m * sf_groups_per_row + g] = sf_byte;
}

// ============================================================================
// Kernel 1A: GEMM1 + SwiGLU + FP4 requant (N-distributed, non-cooperative)
// ============================================================================
// Grid: num_active * tiles_1a CTAs, where tiles_1a = N_half / BN
// Each tile accumulates K in registers (no partials needed).
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
verdict_gemm1_swiglu(
    const uint8_t* __restrict__ input_fp4,
    const uint8_t* __restrict__ input_sf,
    const uint8_t* __restrict__ all_w1_fp4,
    const uint8_t* __restrict__ all_w1_sf,
    const float*   __restrict__ w1_alpha,
    const int*     __restrict__ expert_ids,
    const int*     __restrict__ token_ids,
    uint8_t*       __restrict__ gmem_inter_fp4,
    uint8_t*       __restrict__ gmem_inter_sf,
    int K, int N_half, int num_active, int tiles_1a)
{
    const int eidx = blockIdx.x / tiles_1a;
    const int tile = blockIdx.x % tiles_1a;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    if (eidx >= num_active) return;

    const int eid = expert_ids[eidx];
    const int tok = token_ids[eidx];
    const float alpha1 = w1_alpha[eidx];

    const int N2 = 2 * N_half;
    const int K_packed = K / 2;
    const int N_half_packed = N_half / 2;
    const int sf_cols_w1 = K / SF_BLOCK;
    const int sf_cols_w2 = N_half / SF_BLOCK;
    const int sf_cols_in = K / SF_BLOCK;
    const int K_BLOCKS = K / BK;

    extern __shared__ char smem_raw[];
    uint8_t* s_A   = (uint8_t*)smem_raw;
    uint8_t* s_B   = s_A + SMEM_A;
    uint8_t* s_SFA = s_B + SMEM_B;
    uint8_t* s_SFB = s_SFA + SMEM_SFA_PAD;
    float*   s_sw  = (float*)(s_SFB + SMEM_SFB);

    // Warp/lane column assignment (MMA CLayout for M=1, scale_vec::4X)
    int g = lane_id / 4;
    int Nl = 4 * (g & 1) + (g >> 1);
    int sn = warp_id * 8 + Nl;
    int rbo = sn * (BK / 2);

    // ---- GEMM1: N-distributed, K-accumulated in registers ----
    float gate_acc[4] = {0, 0, 0, 0};
    float up_acc[4]   = {0, 0, 0, 0};

    const int n_base = tile * BN;
    const uint8_t* w1_fp4 = all_w1_fp4 + (long long)eid * N2 * K_packed;
    const uint8_t* w1_sf  = all_w1_sf  + (long long)eid * N2 * sf_cols_w1;

    for (int kb = 0; kb < K_BLOCKS; kb++) {
        int k_off = kb * BK;
        int k_pk  = k_off / 2;
        int k_sf  = k_off / SF_BLOCK;

        // Load A (input)
        for (int i = tid; i < SMEM_A; i += BLOCK_SIZE) {
            int row = i / (BK / 2), col = i % (BK / 2);
            s_A[swizzle_343(i)] = (row == 0) ?
                input_fp4[tok * K_packed + k_pk + col] : 0;
        }
        if (tid < SF_PER_K)
            s_SFA[tid] = input_sf[tok * sf_cols_in + k_sf + tid];

        // Load B (gate)
        for (int i = tid; i < SMEM_B; i += BLOCK_SIZE) {
            int row = i / (BK / 2), col = i % (BK / 2);
            s_B[swizzle_343(i)] =
                w1_fp4[(long long)(n_base + row) * K_packed + k_pk + col];
        }
        for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
            int row = i / SF_PER_K, col = i % SF_PER_K;
            s_SFB[i] = w1_sf[(long long)(n_base + row) * sf_cols_w1 + k_sf + col];
        }
        __syncthreads();

        uint32_t a_regs[4]; pack_a_consec(a_regs, s_A, lane_id);
        uint32_t sfa_pk = pack_sf4(s_SFA);
        uint32_t b_regs[2]; pack_b_consec(b_regs, s_B, rbo, lane_id);
        uint32_t sfb_pk = pack_sf4(&s_SFB[sn * SF_PER_K]);
        mma_nvf4_e4m3_m16n8k64(gate_acc, a_regs, b_regs, gate_acc, sfa_pk, sfb_pk);
        __syncthreads();

        // Load B (up)
        for (int i = tid; i < SMEM_B; i += BLOCK_SIZE) {
            int row = i / (BK / 2), col = i % (BK / 2);
            s_B[swizzle_343(i)] =
                w1_fp4[(long long)(N_half + n_base + row) * K_packed + k_pk + col];
        }
        for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
            int row = i / SF_PER_K, col = i % SF_PER_K;
            s_SFB[i] = w1_sf[(long long)(N_half + n_base + row) * sf_cols_w1 + k_sf + col];
        }
        __syncthreads();

        pack_b_consec(b_regs, s_B, rbo, lane_id);
        sfb_pk = pack_sf4(&s_SFB[sn * SF_PER_K]);
        mma_nvf4_e4m3_m16n8k64(up_acc, a_regs, b_regs, up_acc, sfa_pk, sfb_pk);
        __syncthreads();
    }

    // ---- Alpha × SwiGLU + E4M3 requant ----
    if (lane_id < 4) {
        float g0 = gate_acc[0] * alpha1, g1 = gate_acc[1] * alpha1;
        float u0 = up_acc[0] * alpha1, u1 = up_acc[1] * alpha1;
        s_sw[warp_id * 8 + lane_id]     = u0 * d_silu(g0);
        s_sw[warp_id * 8 + lane_id + 4] = u1 * d_silu(g1);
    }
    __syncthreads();

    // Cooperative FP4 quantization: first BN threads
    if (tid < BN) {
        int col = tid;
        float val = s_sw[col];
        float aval = fabsf(val);

        float wmax = aval;
        #pragma unroll
        for (int off = 8; off > 0; off >>= 1)
            wmax = fmaxf(wmax, __shfl_xor_sync(0xFFFFFFFF, wmax, off));

        float st = fmaxf(wmax / 6.0f, 1e-30f);
        uint8_t sfb = d_e4m3fn_encode(st);
        float as_val = d_e4m3fn_decode_u(sfb);
        if (as_val < 1e-30f) as_val = 1e-30f;

        uint8_t nib = d_quantize_e2m1(val / as_val);
        uint8_t partner = __shfl_xor_sync(0xFFFFFFFF, nib, 1);

        if ((col & 1) == 0)
            gmem_inter_fp4[eidx * N_half_packed + tile * (BN / 2) + col / 2] =
                nib | (partner << 4);

        if ((col & (SF_BLOCK - 1)) == 0)
            gmem_inter_sf[eidx * sf_cols_w2 + tile * (BN / SF_BLOCK) + col / SF_BLOCK] =
                sfb;
    }
}

// ============================================================================
// Kernel 1B: GEMM2 + scatter (output-distributed, non-cooperative)
// ============================================================================
// Grid: num_active * tiles_1b CTAs, where tiles_1b = K / BN
// Each tile handles BN=64 output columns, loops over N_half/BK K-passes.
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
verdict_gemm2_scatter(
    const uint8_t* __restrict__ gmem_inter_fp4,
    const uint8_t* __restrict__ gmem_inter_sf,
    const uint8_t* __restrict__ all_w2_fp4,
    const uint8_t* __restrict__ all_w2_sf,
    const float*   __restrict__ w2_alpha,
    const int*     __restrict__ expert_ids,
    const float*   __restrict__ expert_wts,
    const int*     __restrict__ token_ids,
    float*         __restrict__ output_f32,
    int K, int N_half, int num_active, int tiles_1b)
{
    const int eidx = blockIdx.x / tiles_1b;
    const int tile = blockIdx.x % tiles_1b;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    if (eidx >= num_active) return;

    const int eid = expert_ids[eidx];
    const int tok = token_ids[eidx];
    const float wt = expert_wts[eidx];
    const float alpha2 = w2_alpha[eidx];

    const int N_half_packed = N_half / 2;
    const int sf_cols_w2 = N_half / SF_BLOCK;
    const int K_PASSES = N_half / BK;

    extern __shared__ char smem_raw[];
    uint8_t* s_A   = (uint8_t*)smem_raw;
    uint8_t* s_B   = s_A + SMEM_A;
    uint8_t* s_SFA = s_B + SMEM_B;
    uint8_t* s_SFB = s_SFA + SMEM_SFA_PAD;

    int g2 = lane_id / 4;
    int Nl2 = 4 * (g2 & 1) + (g2 >> 1);
    int sn2 = warp_id * 8 + Nl2;

    const int j_start = tile * BN;
    const uint8_t* w2_fp4 = all_w2_fp4 + (long long)eid * K * N_half_packed;
    const uint8_t* w2_sf  = all_w2_sf  + (long long)eid * K * sf_cols_w2;
    float acc[4] = {0, 0, 0, 0};

    for (int kp = 0; kp < K_PASSES; kp++) {
        int ko = kp * BK, kpk = ko / 2, ksf = ko / SF_BLOCK;

        // Load A: intermediate FP4
        for (int i = tid; i < SMEM_A; i += BLOCK_SIZE) {
            int row = i / (BK / 2), col = i % (BK / 2);
            s_A[swizzle_343(i)] = (row == 0) ?
                gmem_inter_fp4[eidx * N_half_packed + kpk + col] : 0;
        }
        // Load B: W2 weights
        for (int i = tid; i < SMEM_B; i += BLOCK_SIZE) {
            int row = i / (BK / 2), col = i % (BK / 2);
            int oc = j_start + row;
            s_B[swizzle_343(i)] = (oc < K) ?
                w2_fp4[(long long)oc * N_half_packed + kpk + col] : 0;
        }
        if (tid < SF_PER_K)
            s_SFA[tid] = gmem_inter_sf[eidx * sf_cols_w2 + ksf + tid];
        for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
            int row = i / SF_PER_K, col = i % SF_PER_K;
            int oc = j_start + row;
            s_SFB[i] = (oc < K) ?
                w2_sf[(long long)oc * sf_cols_w2 + ksf + col] : 0;
        }
        __syncthreads();

        uint32_t ar[4]; pack_a_consec(ar, s_A, lane_id);
        int rbo2 = sn2 * (BK / 2);
        uint32_t br[2]; pack_b_consec(br, s_B, rbo2, lane_id);
        uint32_t sfap = pack_sf4(s_SFA);
        uint32_t sfbp = pack_sf4(&s_SFB[sn2 * SF_PER_K]);
        mma_nvf4_e4m3_m16n8k64(acc, ar, br, acc, sfap, sfbp);
        __syncthreads();
    }

    // Scatter with atomicAdd
    if (lane_id < 4) {
        float scale = wt * alpha2;
        int j0 = j_start + warp_id * 8 + lane_id;
        int j1 = j0 + 4;
        if (j0 < K) atomicAdd(&output_f32[tok * K + j0], scale * acc[0]);
        if (j1 < K) atomicAdd(&output_f32[tok * K + j1], scale * acc[1]);
    }
}

// ============================================================================
// Kernel 2: F32 → BF16
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
void verdict_fused_forward(
    torch::Tensor input,         // [M, K] BF16
    torch::Tensor w1_fp4,        // [E, 2*N, K//2] uint8
    torch::Tensor w1_sf,         // [E, 2*N, K//16] uint8
    torch::Tensor w1_alpha,      // [num_active] float32
    torch::Tensor w2_fp4,        // [E, K, N//2] uint8
    torch::Tensor w2_sf,         // [E, K, N//16] uint8
    torch::Tensor w2_alpha,      // [num_active] float32
    torch::Tensor output,        // [M, K] BF16
    torch::Tensor expert_ids,    // [num_active] int32
    torch::Tensor expert_wts,    // [num_active] float32
    torch::Tensor token_ids,     // [num_active] int32
    torch::Tensor output_f32,    // [M * K] float32
    torch::Tensor input_fp4_buf, // [M * K/2] uint8
    torch::Tensor input_sf_buf,  // [M * K/16] uint8
    torch::Tensor inter_fp4_buf, // [num_active * N_half/2] uint8
    torch::Tensor inter_sf_buf,  // [num_active * N_half/16] uint8
    torch::Tensor barrier_buf,   // [1] int32 (unused, kept for API compat)
    int K, int N_half, int num_active, int tiles_1a)
{
    auto stream = c10::cuda::getCurrentCUDAStream();
    int M = input.size(0);
    int tiles_1b = K / BN;  // K/64 = 64 for K=4096

    // Zero output accumulator
    cudaMemsetAsync(output_f32.data_ptr(), 0,
                    output_f32.numel() * sizeof(float), stream);

    // K0: BF16 → NVFP4
    {
        int sf_groups = M * (K / SF_BLOCK);
        int half_warps_per_block = BLOCK_SIZE / 16;
        int k0_blocks = (sf_groups + half_warps_per_block - 1) / half_warps_per_block;
        bf16_to_nvfp4_e4m3_kernel<<<k0_blocks, BLOCK_SIZE, 0, stream>>>(
            (const __nv_bfloat16*)input.data_ptr(),
            (uint8_t*)input_fp4_buf.data_ptr(),
            (uint8_t*)input_sf_buf.data_ptr(),
            M, K);
    }

    // K1A: GEMM1 + SwiGLU + requant (N-distributed)
    {
        int grid_1a = num_active * tiles_1a;
        cudaFuncSetAttribute(verdict_gemm1_swiglu,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_K1A);
        verdict_gemm1_swiglu<<<grid_1a, BLOCK_SIZE, SMEM_K1A, stream>>>(
            (const uint8_t*)input_fp4_buf.data_ptr(),
            (const uint8_t*)input_sf_buf.data_ptr(),
            (const uint8_t*)w1_fp4.data_ptr(),
            (const uint8_t*)w1_sf.data_ptr(),
            (const float*)w1_alpha.data_ptr(),
            (const int*)expert_ids.data_ptr(),
            (const int*)token_ids.data_ptr(),
            (uint8_t*)inter_fp4_buf.data_ptr(),
            (uint8_t*)inter_sf_buf.data_ptr(),
            K, N_half, num_active, tiles_1a);
    }

    // K1B: GEMM2 + scatter (output-distributed)
    // Stream ordering guarantees K1A completes before K1B reads intermediate
    {
        int grid_1b = num_active * tiles_1b;
        cudaFuncSetAttribute(verdict_gemm2_scatter,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_K1B);
        verdict_gemm2_scatter<<<grid_1b, BLOCK_SIZE, SMEM_K1B, stream>>>(
            (const uint8_t*)inter_fp4_buf.data_ptr(),
            (const uint8_t*)inter_sf_buf.data_ptr(),
            (const uint8_t*)w2_fp4.data_ptr(),
            (const uint8_t*)w2_sf.data_ptr(),
            (const float*)w2_alpha.data_ptr(),
            (const int*)expert_ids.data_ptr(),
            (const float*)expert_wts.data_ptr(),
            (const int*)token_ids.data_ptr(),
            (float*)output_f32.data_ptr(),
            K, N_half, num_active, tiles_1b);
    }

    // K2: F32 → BF16
    {
        int total_elems = M * K;
        int conv_blocks = (total_elems + 255) / 256;
        convert_f32_to_bf16<<<conv_blocks, 256, 0, stream>>>(
            (const float*)output_f32.data_ptr(),
            (__nv_bfloat16*)output.data_ptr(),
            total_elems);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &verdict_fused_forward,
          "VerdictMoE split MMA forward (N-distributed GEMM1 + output-distributed GEMM2)");
}
