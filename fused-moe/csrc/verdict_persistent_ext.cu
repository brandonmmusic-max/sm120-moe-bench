/**
 * VerdictMoE Persistent Extension for vLLM — SM120 Blackwell
 * Sprint 11 Task 1: 0-barrier persistent kernel
 *
 * THREE kernel launches (vs Sprint 9's single launch with 4 barriers):
 *   1. BF16→FP4 quantization (tiny prologue kernel)
 *   2. Persistent fused MoE (GEMM1→SwiGLU→requant→GEMM2, 0 barriers)
 *   3. F32→BF16 conversion (tiny epilogue kernel)
 *
 * Each CTA handles the FULL K dimension — no K-group splitting, no partials,
 * no intermediate gmem buffers. GEMM2 output tiles split via out_groups for
 * parallelism (GEMM1 duplicated across groups, weights cached in L2).
 *
 * Build: torch JIT with -gencode=arch=compute_120a,code=sm_120a -O2
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
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
constexpr int TOTAL_K_TILES = HIDDEN / BK;            // 64
constexpr int P2_OUT_TILES  = HIDDEN / BN;            // 64

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

__device__ __forceinline__ uint32_t swizzle_343(uint32_t off) {
    return off ^ ((off >> 3) & 0x70u);
}

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
// Kernel 1: BF16→FP4 Quantization (tiny prologue)
// ============================================================================
__global__ void verdict_bf16_to_fp4(
    const __nv_bfloat16* __restrict__ input_bf16,  // [M, K]
    uint8_t* __restrict__ input_fp4,               // [M, K/2]
    uint8_t* __restrict__ input_sf,                // [M, K/16]
    float*   __restrict__ output_f32,              // [M, K] — zeroed here
    int M)
{
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    // Zero output_f32
    for (int i = global_tid; i < M * HIDDEN; i += total_threads)
        output_f32[i] = 0.0f;

    // BF16→NVFP4 quantization (half-warp per SF_BLOCK=16 group)
    constexpr int num_sf_groups = HIDDEN / SF_BLOCK;
    const int total_sf_groups = M * num_sf_groups;
    const int half_warp_id = global_tid / 16;
    const int hw_lane = threadIdx.x % 16;

    if (half_warp_id < total_sf_groups) {
        const int m = half_warp_id / num_sf_groups;
        const int g = half_warp_id % num_sf_groups;
        const int kb = g * SF_BLOCK;

        float val = __bfloat162float(input_bf16[m * HIDDEN + kb + hw_lane]);

        float wmax = fabsf(val);
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

// ============================================================================
// Kernel 2: Persistent Fused MoE (0 barriers)
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
verdict_persistent_fused_moe_mt(
    const uint8_t* __restrict__ input_fp4,
    const uint8_t* __restrict__ input_sf,
    const uint8_t* __restrict__ all_w1_fp4,
    const uint8_t* __restrict__ all_w1_sf,
    const uint8_t* __restrict__ all_w2_fp4,
    const uint8_t* __restrict__ all_w2_sf,
    const int*     __restrict__ expert_ids,
    const int*     __restrict__ token_ids,
    const float*   __restrict__ w1_alpha,
    const float*   __restrict__ w2_alpha,
    const float*   __restrict__ expert_wts,
    float*         __restrict__ output_f32,
    int num_pairs,
    int n_half,
    int out_groups,
    int total_work_items)
{
    const int tiles_n       = n_half / BN;
    const int n_half_packed = n_half / 2;
    const int sf_cols_w2    = n_half / SF_BLOCK;
    const int n2            = 2 * n_half;
    const int tiles_per_ogrp = P2_OUT_TILES / out_groups;

    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int g   = lane_id / 4;
    const int Nl  = 4 * (g & 1) + (g >> 1);
    const int sn  = warp_id * 8 + Nl;
    const int t0  = lane_id % 4;
    const int rbo = sn * (BK / 2);

    extern __shared__ char smem_raw[];
    uint8_t* s_A        = (uint8_t*)smem_raw;
    uint8_t* s_B_gate   = s_A + 32;
    uint8_t* s_B_up     = s_B_gate + SMEM_B;
    uint8_t* s_SFA      = s_B_up + SMEM_B;
    uint8_t* s_SFB_gate = s_SFA + ((SF_PER_K + 3) & ~3);
    uint8_t* s_SFB_up   = s_SFB_gate + SMEM_SFB;

    float* s_swiglu = (float*)s_B_up;

    int work_idx = blockIdx.x;
    while (work_idx < total_work_items) {
        const int items_per_pair = tiles_n * out_groups;
        const int pair_idx  = work_idx / items_per_pair;
        const int rem       = work_idx % items_per_pair;
        const int n_chunk   = rem / out_groups;
        const int out_group = rem % out_groups;

        const int eid      = expert_ids[pair_idx];
        const int token_id = token_ids[pair_idx];
        const float alpha1 = w1_alpha[pair_idx];
        const float alpha2 = w2_alpha[pair_idx];
        const float wt     = expert_wts[pair_idx];
        const int n_start  = n_chunk * BN;

        const uint8_t* w1_fp4 = all_w1_fp4 + (long long)eid * n2 * K_PACKED;
        const uint8_t* w1_sf  = all_w1_sf  + (long long)eid * n2 * SF_COLS_W1;

        // GEMM1: full K reduction
        float gate_acc[4] = {0, 0, 0, 0};
        float up_acc[4]   = {0, 0, 0, 0};

        for (int kt = 0; kt < TOTAL_K_TILES; kt++) {
            const int k_off = kt * BK;
            const int k_pk  = k_off / 2;
            const int k_sf  = k_off / SF_BLOCK;

            for (int i = tid; i < 8; i += BLOCK_SIZE)
                *(uint32_t*)(s_A + i * 4) =
                    *(const uint32_t*)&input_fp4[token_id * K_PACKED + k_pk + i * 4];

            for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
                int boff = i * 4, row = boff / (BK / 2), col = boff % (BK / 2);
                *(uint32_t*)&s_B_gate[swizzle_343(boff)] =
                    *(const uint32_t*)&w1_fp4[(long long)(n_start + row) * K_PACKED + k_pk + col];
            }

            for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
                int boff = i * 4, row = boff / (BK / 2), col = boff % (BK / 2);
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

        // SwiGLU with alpha1
        if (lane_id < 4) {
            int c0 = warp_id * 8 + lane_id;
            int c1 = c0 + 4;
            float g0 = gate_acc[0] * alpha1, g1 = gate_acc[1] * alpha1;
            float u0 = up_acc[0] * alpha1,   u1 = up_acc[1] * alpha1;
            s_swiglu[c0] = u0 * d_silu(g0);
            s_swiglu[c1] = u1 * d_silu(g1);
        }
        __syncthreads();

        // Quantize to FP4
        if (tid < BN) {
            int col = tid;
            float sw_val = s_swiglu[col];

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
            uint32_t nib32 = (uint32_t)nib;
            uint32_t neighbor32 = __shfl_down_sync(0xFFFFFFFF, nib32, 1);
            if (col % 2 == 0)
                s_A[col / 2] = (uint8_t)(nib32 | (neighbor32 << 4));
            if (col % SF_BLOCK == 0)
                s_SFA[col / SF_BLOCK] = sf_enc;
        }
        __syncthreads();

        // GEMM2 with alpha2
        {
            const uint8_t* w2_fp4 = all_w2_fp4 + (long long)eid * HIDDEN * n_half_packed;
            const uint8_t* w2_sf  = all_w2_sf  + (long long)eid * HIDDEN * sf_cols_w2;
            uint8_t* s_B2   = s_B_gate;
            uint8_t* s_SFB2 = s_SFB_gate;
            const int kpk = n_start / 2;
            const int ksf = n_start / SF_BLOCK;

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

                for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
                    int boff = i * 4, row = boff / (BK / 2), col = boff % (BK / 2);
                    *(uint32_t*)&s_B2[swizzle_343(boff)] =
                        *(const uint32_t*)&w2_fp4[(long long)(j_start + row) * n_half_packed + kpk + col];
                }
                for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
                    int row = i / SF_PER_K, col = i % SF_PER_K;
                    s_SFB2[i] = w2_sf[(long long)(j_start + row) * sf_cols_w2 + ksf + col];
                }

                __syncthreads();

                uint32_t br[2];
                br[0] = *(uint32_t*)&s_B2[swizzle_343(rbo + t0 * 4)];
                br[1] = *(uint32_t*)&s_B2[swizzle_343(rbo + 16 + t0 * 4)];
                uint32_t sfbp = pack_sf4(&s_SFB2[sn * SF_PER_K]);

                mma_nvf4_e4m3_m16n8k64(p2_acc, ar_fixed, br, p2_acc, sfap_fixed, sfbp);

                if (lane_id < 4) {
                    float scale = wt * alpha2;
                    int j0 = j_start + warp_id * 8 + lane_id;
                    int j1 = j0 + 4;
                    atomicAdd(&output_f32[token_id * HIDDEN + j0], scale * p2_acc[0]);
                    atomicAdd(&output_f32[token_id * HIDDEN + j1], scale * p2_acc[1]);
                }

                __syncthreads();
            }
        }

        work_idx += gridDim.x;
    }
}

// ============================================================================
// Kernel 3: F32→BF16 (tiny epilogue)
// ============================================================================
__global__ void verdict_f32_to_bf16(
    const float* __restrict__ input_f32,
    __nv_bfloat16* __restrict__ output_bf16,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel)
        output_bf16[idx] = __float2bfloat16(input_f32[idx]);
}

// ============================================================================
// Host Forward — Persistent MoE (3 launches, 0 barriers)
// ============================================================================
void verdict_persistent_forward(
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
    torch::Tensor output_f32,     // [M * K] float32
    torch::Tensor input_fp4_buf,  // [M * K/2] uint8
    torch::Tensor input_sf_buf,   // [M * K/16] uint8
    int K, int N_half, int num_pairs)
{
    auto stream = c10::cuda::getCurrentCUDAStream();
    int M = input.size(0);

    TORCH_CHECK(K == HIDDEN, "K must be ", HIDDEN, " got ", K);
    TORCH_CHECK(M >= 1, "M must be >= 1, got ", M);

    // Get max CTAs
    static int max_ctas = -1;
    if (max_ctas < 0) {
        int dev; cudaGetDevice(&dev);
        cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
        max_ctas = prop.multiProcessorCount * 4;
    }

    // Compute out_groups for parallelism
    int tiles_n = N_half / BN;
    int base_items = num_pairs * tiles_n;
    int out_groups = std::max(1, max_ctas / base_items);
    while (P2_OUT_TILES % out_groups != 0 && out_groups > 1) out_groups--;
    int total_work_items = base_items * out_groups;
    int grid = std::min(max_ctas, total_work_items);

    int smem_size = 32 + 2 * SMEM_B
                  + ((SF_PER_K + 3) & ~3) + 2 * SMEM_SFB + 128;

    // --- Launch 1: BF16→FP4 + zero output ---
    {
        int threads = 256;
        int blocks = std::min(max_ctas, (M * HIDDEN + threads - 1) / threads);
        verdict_bf16_to_fp4<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
            reinterpret_cast<uint8_t*>(input_fp4_buf.data_ptr()),
            reinterpret_cast<uint8_t*>(input_sf_buf.data_ptr()),
            reinterpret_cast<float*>(output_f32.data_ptr()),
            M);
    }

    // --- Launch 2: Persistent fused MoE (0 barriers) ---
    cudaFuncSetAttribute(verdict_persistent_fused_moe_mt,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    verdict_persistent_fused_moe_mt<<<grid, BLOCK_SIZE, smem_size, stream>>>(
        reinterpret_cast<const uint8_t*>(input_fp4_buf.data_ptr()),
        reinterpret_cast<const uint8_t*>(input_sf_buf.data_ptr()),
        reinterpret_cast<const uint8_t*>(w1_fp4.data_ptr()),
        reinterpret_cast<const uint8_t*>(w1_sf.data_ptr()),
        reinterpret_cast<const uint8_t*>(w2_fp4.data_ptr()),
        reinterpret_cast<const uint8_t*>(w2_sf.data_ptr()),
        reinterpret_cast<const int*>(expert_ids.data_ptr()),
        reinterpret_cast<const int*>(token_ids.data_ptr()),
        reinterpret_cast<const float*>(w1_alpha.data_ptr()),
        reinterpret_cast<const float*>(w2_alpha.data_ptr()),
        reinterpret_cast<const float*>(expert_weights.data_ptr()),
        reinterpret_cast<float*>(output_f32.data_ptr()),
        num_pairs,
        N_half,
        out_groups,
        total_work_items);

    // --- Launch 3: F32→BF16 ---
    {
        int numel = M * HIDDEN;
        int threads = 256;
        int blocks = (numel + threads - 1) / threads;
        verdict_f32_to_bf16<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float*>(output_f32.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
            numel);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &verdict_persistent_forward,
          "VerdictMoE persistent 0-barrier forward (SM120)");
}
