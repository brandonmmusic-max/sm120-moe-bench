/**
 * VerdictMoE CUDA Extension for vLLM
 *
 * Fused MoE kernel with on-the-fly NVFP4 dequantization.
 * 3-kernel pipeline (avoids cooperative groups / -rdc=true):
 *   1. verdict_gemm1_distributed: 640 CTAs, FP4 weight dequant, partial GEMM1
 *   2. verdict_swiglu_reduce: 10 CTAs, reduce partials → SwiGLU → intermediate
 *   3. verdict_gemm2_scatter: 640 CTAs, FP4 weight dequant, GEMM2 + weighted atomicAdd
 *
 * SM120: compute_120a, 99KB SMEM, no BF16 MMA.
 * Weight format: NVFP4 (E2M1 packed uint8 + E4M3FN block scales + per-expert alpha)
 * Input: BF16, Output: BF16
 *
 * Build:
 *   torch.utils.cpp_extension.load(..., extra_cuda_cflags=['-gencode=arch=compute_120a,code=sm_120a'])
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
constexpr int TILES_PER_EXPERT = 64;
constexpr int COLS_PER_TILE = 64;
constexpr int FP4_BLOCK_SIZE = 16;  // NVFP4 block scale group size

// ============================================================================
// FP4 E2M1 LUT (16 entries, unsigned)
// sign bit handled separately
// ============================================================================
__constant__ float c_fp4_lut[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

// E4M3FN LUT (256 entries)
__constant__ float c_e4m3fn_lut[256];

static bool luts_initialized = false;

void init_luts() {
    if (luts_initialized) return;
    float h_e4m3fn_lut[256];
    for (int i = 0; i < 256; i++) {
        int s = (i >> 7) & 1;
        int e = (i >> 3) & 0xF;
        int m = i & 7;
        float val;
        if (e == 15 && m == 7) {
            val = 0.0f;  // NaN → 0 for safety
        } else if (e == 0) {
            val = ldexpf((float)m / 8.0f, -6);  // subnormal
        } else {
            val = ldexpf(1.0f + (float)m / 8.0f, e - 7);  // normal
        }
        h_e4m3fn_lut[i] = s ? -val : val;
    }
    cudaMemcpyToSymbol(c_e4m3fn_lut, h_e4m3fn_lut, sizeof(h_e4m3fn_lut));
    luts_initialized = true;
}

// ============================================================================
// Device Helpers
// ============================================================================

// Decode FP4 E2M1 nibble to float — register-only, no memory access
__device__ __forceinline__ float decode_fp4(uint8_t nibble) {
    int mag = nibble & 0x7;
    int e = mag >> 1;
    int m = mag & 1;
    float val;
    if (e == 0) {
        val = m ? 0.5f : 0.0f;
    } else {
        // (1.0 + m*0.5) * 2^(e-1) = (2+m) * 2^(e-2)
        val = (float)(2 + m) * (float)(1 << (e - 1)) * 0.5f;
    }
    return (nibble & 0x8) ? -val : val;
}

// Decode E4M3FN byte to float — inline computation (avoids constant memory serialization)
__device__ __forceinline__ float decode_e4m3fn(uint8_t x) {
    int s = (x >> 7) & 1;
    int e = (x >> 3) & 0xF;
    int m = x & 7;
    float val;
    if (e == 15 && m == 7) {
        val = 0.0f;  // NaN → 0
    } else if (e == 0) {
        val = __int2float_rn(m) * 0.001953125f;  // m * 2^(-9) = m/512
    } else {
        // Use ldexpf to avoid integer overflow for e>=14
        // (1 + m/8) * 2^(e-7) — correct for all valid exponents
        val = ldexpf(1.0f + __int2float_rn(m) * 0.125f, e - 7);
    }
    return s ? -val : val;
}

// Read and dequant one element from NVFP4 packed weight tensor
// w_fp4: [rows, cols_packed] uint8 (2 FP4 per byte)
// w_sf:  [rows, cols/16] uint8 (E4M3FN block scales)
// Returns: fp4_val * blockscale
__device__ __forceinline__ float read_nvfp4(
    const uint8_t* w_fp4, const uint8_t* w_sf,
    int row, int col, int cols_packed, int sf_cols)
{
    uint8_t byte = w_fp4[row * cols_packed + col / 2];
    uint8_t nibble = (col & 1) ? (byte >> 4) : (byte & 0xF);
    float fp4_val = decode_fp4(nibble);
    float scale = decode_e4m3fn(w_sf[row * sf_cols + col / FP4_BLOCK_SIZE]);
    return fp4_val * scale;
}

__device__ __forceinline__ float d_silu(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ uint8_t float_to_e4m3(float v) {
    uint16_t packed;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(packed) : "f"(0.0f), "f"(v));
    return (uint8_t)(packed & 0xFF);
}

// ============================================================================
// Kernel 1: Distributed GEMM1 with SMEM-tiled FP4 weight loading
// ============================================================================
// Grid: (num_active * tiles_per_expert, 1, 1) = 640 blocks
// Each block: k_per = K / tiles_per_expert elements of GEMM1 K-reduction
//
// SMEM tiling: cooperatively load [N2, k_per_packed] FP4 tile + scales into SMEM,
// then each thread reads from SMEM (fast, no L1 thrashing from strided GMEM).
//
// SMEM layout: [input: k_per floats] [w1_fp4: N2*k_per_packed bytes] [w1_sf: N2*k_per_sf bytes]
__global__ void verdict_gemm1_distributed(
    const __nv_bfloat16* __restrict__ input,  // [M, K]
    const uint8_t* __restrict__ all_w1_fp4,   // [E, 2*N_half, K//2]
    const uint8_t* __restrict__ all_w1_sf,    // [E, 2*N_half, K//16]
    const float* __restrict__ w1_alpha,       // [num_active] per-expert weight scale
    float* __restrict__ partials,             // [num_active, tiles, 2, N_half]
    const int* __restrict__ expert_ids,       // [num_active]
    const int* __restrict__ token_ids,        // [num_active]
    int K, int N_half, int num_active, int tiles_per_expert)
{
    const int eidx = blockIdx.x / tiles_per_expert;
    const int tile = blockIdx.x % tiles_per_expert;
    const int tid = threadIdx.x;
    if (eidx >= num_active) return;

    const int eid = expert_ids[eidx];
    const int tok = token_ids[eidx];
    const float alpha = w1_alpha[eidx];
    const int N2 = 2 * N_half;
    const int K_packed = K / 2;
    const int sf_cols = K / FP4_BLOCK_SIZE;

    const int k_per = K / tiles_per_expert;
    const int k_per_packed = k_per / 2;
    const int k_per_sf = k_per / FP4_BLOCK_SIZE;
    const int k_start = tile * k_per;
    const int k_start_packed = k_start / 2;
    const int k_start_sf = k_start / FP4_BLOCK_SIZE;

    // Weight pointers for this expert
    const uint8_t* w1_fp4 = all_w1_fp4 + (long long)eid * N2 * K_packed;
    const uint8_t* w1_sf = all_w1_sf + (long long)eid * N2 * sf_cols;

    // Input pointer
    const __nv_bfloat16* inp = input + tok * K;

    // SMEM layout
    extern __shared__ char smem_raw[];
    float* s_input = (float*)smem_raw;
    uint8_t* s_fp4 = (uint8_t*)(s_input + k_per);
    uint8_t* s_sf = s_fp4 + N2 * k_per_packed;

    // 1. Load input slice to SMEM
    for (int i = tid; i < k_per; i += BLOCK_SIZE)
        s_input[i] = __bfloat162float(inp[k_start + i]);

    // 2. Cooperative load of FP4 weight tile [N2, k_per_packed] into SMEM
    int fp4_tile_size = N2 * k_per_packed;
    for (int i = tid; i < fp4_tile_size; i += BLOCK_SIZE) {
        int row = i / k_per_packed;
        int col = i % k_per_packed;
        s_fp4[i] = w1_fp4[row * K_packed + k_start_packed + col];
    }

    // 3. Cooperative load of scale tile [N2, k_per_sf] into SMEM
    int sf_tile_size = N2 * k_per_sf;
    for (int i = tid; i < sf_tile_size; i += BLOCK_SIZE) {
        int row = i / k_per_sf;
        int col = i % k_per_sf;
        s_sf[i] = w1_sf[row * sf_cols + k_start_sf + col];
    }

    __syncthreads();

    // 4. Compute partial gate/up from SMEM (no GMEM access in inner loop)
    // Loop to handle N_half > BLOCK_SIZE (e.g., N_half=1024 with 256 threads)
    long long part_base = (long long)eidx * tiles_per_expert * 2 * N_half
                        + (long long)tile * 2 * N_half;

    for (int row = tid; row < N_half; row += BLOCK_SIZE) {
        float gate_p = 0.0f, up_p = 0.0f;

        // Precompute row offsets in SMEM
        int gate_fp4_base = row * k_per_packed;
        int gate_sf_base = row * k_per_sf;
        int up_fp4_base = (row + N_half) * k_per_packed;
        int up_sf_base = (row + N_half) * k_per_sf;

        for (int ki = 0; ki < k_per; ki++) {
            float inp_k = s_input[ki];
            int ki_packed = ki / 2;
            int ki_nibble = ki & 1;
            int ki_sf = ki / FP4_BLOCK_SIZE;

            // Gate weight from SMEM
            uint8_t byte_g = s_fp4[gate_fp4_base + ki_packed];
            uint8_t nib_g = ki_nibble ? (byte_g >> 4) : (byte_g & 0xF);
            float sf_g = decode_e4m3fn(s_sf[gate_sf_base + ki_sf]);
            gate_p += inp_k * decode_fp4(nib_g) * sf_g;

            // Up weight from SMEM
            uint8_t byte_u = s_fp4[up_fp4_base + ki_packed];
            uint8_t nib_u = ki_nibble ? (byte_u >> 4) : (byte_u & 0xF);
            float sf_u = decode_e4m3fn(s_sf[up_sf_base + ki_sf]);
            up_p += inp_k * decode_fp4(nib_u) * sf_u;
        }

        gate_p *= alpha;
        up_p *= alpha;

        partials[part_base + row] = gate_p;
        partials[part_base + N_half + row] = up_p;
    }
}

// ============================================================================
// Kernel 2: Reduce partials + SwiGLU + E4M3 requant
// ============================================================================
// Grid: (num_active, 1, 1) = 10 blocks, one per expert
// Each block: 256 threads, tid < N_half active
__global__ void verdict_swiglu_reduce(
    float* __restrict__ partials,      // [num_active, tiles, 2, N_half]
    float* __restrict__ gmem_inter,    // [num_active, N_half] dequanted E4M3
    int N_half, int num_active, int tiles_per_expert)
{
    const int eidx = blockIdx.x;
    const int tid = threadIdx.x;
    if (eidx >= num_active) return;

    long long part_base = (long long)eidx * tiles_per_expert * 2 * N_half;

    // Loop to handle N_half > BLOCK_SIZE
    for (int col = tid; col < N_half; col += BLOCK_SIZE) {
        float gate_sum = 0.0f, up_sum = 0.0f;
        for (int t = 0; t < tiles_per_expert; t++) {
            gate_sum += partials[part_base + (long long)t * 2 * N_half + col];
            up_sum   += partials[part_base + (long long)t * 2 * N_half + N_half + col];
        }

        float sw = up_sum * d_silu(gate_sum);
        // E4M3 requant (lossy but matches validated pipeline)
        uint8_t e4m3 = float_to_e4m3(sw);
        gmem_inter[eidx * N_half + col] = decode_e4m3fn(e4m3);
    }
}

// ============================================================================
// Kernel 3: GEMM2 with FP4 dequant + weighted atomicAdd scatter
// ============================================================================
// Grid: (num_active * tiles_per_expert, 1, 1) = 640 blocks
// Each block handles 64 output columns of GEMM2
// 256 threads / 64 cols = 4 threads per col doing K-reduction
//
// W2 layout [E, K, N_half//2] has N_packed=128 bytes per row = 1 cache line.
// Access is already well-coalesced (4 threads/col read sequential bytes in same row).
__global__ void verdict_gemm2_scatter(
    const float* __restrict__ gmem_inter,    // [num_active, N_half]
    const uint8_t* __restrict__ all_w2_fp4,  // [E, K, N_half//2]
    const uint8_t* __restrict__ all_w2_sf,   // [E, K, N_half//16]
    const float* __restrict__ w2_alpha,      // [num_active] per-expert weight scale
    __nv_bfloat16* __restrict__ output,      // [M, K]
    const int* __restrict__ expert_ids,      // [num_active]
    const float* __restrict__ expert_wts,    // [num_active]
    const int* __restrict__ token_ids,       // [num_active]
    int K, int N_half, int num_active, int tiles_per_expert)
{
    const int eidx = blockIdx.x / tiles_per_expert;
    const int tile = blockIdx.x % tiles_per_expert;
    const int tid = threadIdx.x;
    if (eidx >= num_active) return;

    const int eid = expert_ids[eidx];
    const float wt = expert_wts[eidx];
    const int tok = token_ids[eidx];
    const float alpha = w2_alpha[eidx];
    const int N_packed = N_half / 2;
    const int sf_cols = N_half / FP4_BLOCK_SIZE;

    const uint8_t* w2_fp4 = all_w2_fp4 + (long long)eid * K * N_packed;
    const uint8_t* w2_sf = all_w2_sf + (long long)eid * K * sf_cols;
    const float* inter = gmem_inter + eidx * N_half;

    // Load intermediate to SMEM
    extern __shared__ float smem[];
    for (int i = tid; i < N_half; i += BLOCK_SIZE)
        smem[i] = inter[i];
    __syncthreads();

    // 256 threads / 64 cols = 4 threads per col
    int col_local = tid >> 2;
    int k_quarter = tid & 3;
    int j = tile * COLS_PER_TILE + col_local;

    if (j < K) {
        int k_per = N_half >> 2;
        int k_start = k_quarter * k_per;
        float acc = 0.0f;

        #pragma unroll 8
        for (int k = k_start; k < k_start + k_per; k++) {
            float w = read_nvfp4(w2_fp4, w2_sf, j, k, N_packed, sf_cols);
            acc += smem[k] * w;
        }
        acc *= alpha;

        acc += __shfl_xor_sync(0xFFFFFFFF, acc, 1);
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, 2);

        if (k_quarter == 0) {
            float* out_f32 = (float*)output;
            atomicAdd(&out_f32[tok * K + j], wt * acc);
        }
    }
}

// ============================================================================
// Kernel 4: Convert float output to BF16 (final step)
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
// Host orchestrator: launches 3 kernels + conversion
// ============================================================================
void verdict_fused_moe_forward(
    torch::Tensor input,           // [M, K] BF16
    torch::Tensor w1_fp4,          // [E, 2*N, K//2] uint8
    torch::Tensor w1_sf,           // [E, 2*N, K//16] uint8 (E4M3FN)
    torch::Tensor w1_alpha,        // [num_active] float32 — per-active-expert weight scale
    torch::Tensor w2_fp4,          // [E, K, N//2] uint8
    torch::Tensor w2_sf,           // [E, K, N//16] uint8 (E4M3FN)
    torch::Tensor w2_alpha,        // [num_active] float32
    torch::Tensor output,          // [M, K] BF16
    torch::Tensor expert_ids,      // [num_active] int32
    torch::Tensor expert_wts,      // [num_active] float32
    torch::Tensor token_ids,       // [num_active] int32
    torch::Tensor partials,        // [num_active * tiles * 2 * N_half] float32
    torch::Tensor gmem_inter,      // [num_active * N_half] float32
    torch::Tensor output_f32,      // [M, K] float32 — scratch for atomicAdd
    int K, int N_half, int num_active, int tiles_per_expert)
{
    init_luts();

    int total_blocks = num_active * tiles_per_expert;
    int k_per = K / tiles_per_expert;
    int N2 = 2 * N_half;
    // SMEM for GEMM1: input[k_per floats] + fp4_tile[N2*k_per/2 bytes] + sf_tile[N2*k_per/16 bytes]
    int smem_k1 = k_per * (int)sizeof(float) + N2 * (k_per / 2) + N2 * (k_per / FP4_BLOCK_SIZE);
    int smem_k3 = N_half * (int)sizeof(float);

    auto stream = at::cuda::getCurrentCUDAStream();

    // Raise dynamic SMEM limit for GEMM1 (may exceed default 48KB with large N_half)
    if (smem_k1 > 48 * 1024) {
        cudaFuncSetAttribute(verdict_gemm1_distributed,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_k1);
    }

    // Zero output scratch
    cudaMemsetAsync(output_f32.data_ptr(), 0, output_f32.numel() * sizeof(float), stream);

    // Kernel 1: Distributed GEMM1
    verdict_gemm1_distributed<<<total_blocks, BLOCK_SIZE, smem_k1, stream>>>(
        (const __nv_bfloat16*)input.data_ptr(),
        (const uint8_t*)w1_fp4.data_ptr(),
        (const uint8_t*)w1_sf.data_ptr(),
        (const float*)w1_alpha.data_ptr(),
        (float*)partials.data_ptr(),
        (const int*)expert_ids.data_ptr(),
        (const int*)token_ids.data_ptr(),
        K, N_half, num_active, tiles_per_expert);

    // Kernel 2: Reduce + SwiGLU
    verdict_swiglu_reduce<<<num_active, BLOCK_SIZE, 0, stream>>>(
        (float*)partials.data_ptr(),
        (float*)gmem_inter.data_ptr(),
        N_half, num_active, tiles_per_expert);

    // Kernel 3: GEMM2 + scatter
    verdict_gemm2_scatter<<<total_blocks, BLOCK_SIZE, smem_k3, stream>>>(
        (const float*)gmem_inter.data_ptr(),
        (const uint8_t*)w2_fp4.data_ptr(),
        (const uint8_t*)w2_sf.data_ptr(),
        (const float*)w2_alpha.data_ptr(),
        (__nv_bfloat16*)output_f32.data_ptr(),  // actually float*
        (const int*)expert_ids.data_ptr(),
        (const float*)expert_wts.data_ptr(),
        (const int*)token_ids.data_ptr(),
        K, N_half, num_active, tiles_per_expert);

    // Kernel 4: Convert f32 → bf16
    int M = input.size(0);
    int total_elems = M * K;
    int conv_blocks = (total_elems + 255) / 256;
    convert_f32_to_bf16<<<conv_blocks, 256, 0, stream>>>(
        (const float*)output_f32.data_ptr(),
        (__nv_bfloat16*)output.data_ptr(),
        total_elems);
}

// ============================================================================
// Benchmark-only: launch just the 3 MoE kernels, return timing
// ============================================================================
std::vector<float> verdict_fused_moe_benchmark(
    torch::Tensor input,
    torch::Tensor w1_fp4, torch::Tensor w1_sf, torch::Tensor w1_alpha,
    torch::Tensor w2_fp4, torch::Tensor w2_sf, torch::Tensor w2_alpha,
    torch::Tensor output,
    torch::Tensor expert_ids, torch::Tensor expert_wts, torch::Tensor token_ids,
    torch::Tensor partials, torch::Tensor gmem_inter, torch::Tensor output_f32,
    int K, int N_half, int num_active, int tiles_per_expert,
    int warmup, int iters)
{
    init_luts();

    // Warmup
    for (int i = 0; i < warmup; i++) {
        verdict_fused_moe_forward(input, w1_fp4, w1_sf, w1_alpha,
            w2_fp4, w2_sf, w2_alpha, output, expert_ids, expert_wts, token_ids,
            partials, gmem_inter, output_f32, K, N_half, num_active, tiles_per_expert);
    }
    cudaDeviceSynchronize();

    // Benchmark
    std::vector<float> times;
    for (int i = 0; i < iters; i++) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        verdict_fused_moe_forward(input, w1_fp4, w1_sf, w1_alpha,
            w2_fp4, w2_sf, w2_alpha, output, expert_ids, expert_wts, token_ids,
            partials, gmem_inter, output_f32, K, N_half, num_active, tiles_per_expert);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        times.push_back(ms * 1000.0f);  // convert to μs
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    return times;
}

// ============================================================================
// Per-kernel timing (diagnostic)
// ============================================================================
std::vector<float> verdict_per_kernel_timing(
    torch::Tensor input,
    torch::Tensor w1_fp4, torch::Tensor w1_sf, torch::Tensor w1_alpha,
    torch::Tensor w2_fp4, torch::Tensor w2_sf, torch::Tensor w2_alpha,
    torch::Tensor output,
    torch::Tensor expert_ids, torch::Tensor expert_wts, torch::Tensor token_ids,
    torch::Tensor partials, torch::Tensor gmem_inter, torch::Tensor output_f32,
    int K, int N_half, int num_active, int tiles_per_expert)
{
    init_luts();
    int total_blocks = num_active * tiles_per_expert;
    int k_per = K / tiles_per_expert;
    int N2 = 2 * N_half;
    // SMEM for GEMM1: input[k_per floats] + fp4_tile[N2*k_per/2 bytes] + sf_tile[N2*k_per/16 bytes]
    int smem_k1 = k_per * (int)sizeof(float) + N2 * (k_per / 2) + N2 * (k_per / FP4_BLOCK_SIZE);
    int smem_k3 = N_half * (int)sizeof(float);
    auto stream = at::cuda::getCurrentCUDAStream();

    // Raise dynamic SMEM limit for GEMM1 if needed
    if (smem_k1 > 48 * 1024) {
        cudaFuncSetAttribute(verdict_gemm1_distributed,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_k1);
    }

    // Warmup
    for (int i = 0; i < 20; i++) {
        verdict_fused_moe_forward(input, w1_fp4, w1_sf, w1_alpha,
            w2_fp4, w2_sf, w2_alpha, output, expert_ids, expert_wts, token_ids,
            partials, gmem_inter, output_f32, K, N_half, num_active, tiles_per_expert);
    }
    cudaDeviceSynchronize();

    // Time each kernel separately (100 iters, take median)
    auto time_kernel = [&](auto launch_fn) -> float {
        std::vector<float> ts;
        for (int i = 0; i < 100; i++) {
            cudaEvent_t s, e;
            cudaEventCreate(&s); cudaEventCreate(&e);
            cudaEventRecord(s, stream);
            launch_fn();
            cudaEventRecord(e, stream);
            cudaEventSynchronize(e);
            float ms; cudaEventElapsedTime(&ms, s, e);
            ts.push_back(ms * 1000.0f);
            cudaEventDestroy(s); cudaEventDestroy(e);
        }
        std::sort(ts.begin(), ts.end());
        return ts[ts.size()/2];
    };

    float t_memset = time_kernel([&]() {
        cudaMemsetAsync(output_f32.data_ptr(), 0, output_f32.numel() * sizeof(float), stream);
    });

    float t_gemm1 = time_kernel([&]() {
        verdict_gemm1_distributed<<<total_blocks, BLOCK_SIZE, smem_k1, stream>>>(
            (const __nv_bfloat16*)input.data_ptr(),
            (const uint8_t*)w1_fp4.data_ptr(), (const uint8_t*)w1_sf.data_ptr(),
            (const float*)w1_alpha.data_ptr(), (float*)partials.data_ptr(),
            (const int*)expert_ids.data_ptr(), (const int*)token_ids.data_ptr(),
            K, N_half, num_active, tiles_per_expert);
    });

    float t_swiglu = time_kernel([&]() {
        verdict_swiglu_reduce<<<num_active, BLOCK_SIZE, 0, stream>>>(
            (float*)partials.data_ptr(), (float*)gmem_inter.data_ptr(),
            N_half, num_active, tiles_per_expert);
    });

    float t_gemm2 = time_kernel([&]() {
        verdict_gemm2_scatter<<<total_blocks, BLOCK_SIZE, smem_k3, stream>>>(
            (const float*)gmem_inter.data_ptr(),
            (const uint8_t*)w2_fp4.data_ptr(), (const uint8_t*)w2_sf.data_ptr(),
            (const float*)w2_alpha.data_ptr(),
            (__nv_bfloat16*)output_f32.data_ptr(),
            (const int*)expert_ids.data_ptr(), (const float*)expert_wts.data_ptr(),
            (const int*)token_ids.data_ptr(),
            K, N_half, num_active, tiles_per_expert);
    });

    int M = input.size(0);
    int total_elems = M * K;
    int conv_blocks = (total_elems + 255) / 256;
    float t_conv = time_kernel([&]() {
        convert_f32_to_bf16<<<conv_blocks, 256, 0, stream>>>(
            (const float*)output_f32.data_ptr(),
            (__nv_bfloat16*)output.data_ptr(), total_elems);
    });

    return {t_memset, t_gemm1, t_swiglu, t_gemm2, t_conv};
}

// ============================================================================
// PyBind11 module
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &verdict_fused_moe_forward, "VerdictMoE fused forward (NVFP4)");
    m.def("benchmark", &verdict_fused_moe_benchmark, "VerdictMoE fused benchmark");
    m.def("per_kernel_timing", &verdict_per_kernel_timing, "Per-kernel timing breakdown");
}
