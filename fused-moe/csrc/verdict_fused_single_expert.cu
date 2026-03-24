/**
 * Task 3: Fused Single Expert Kernel for SM120
 *
 * verdict_fused_moe_single_expert:
 *   GEMM1 [M,4096]×[4096,512] → SwiGLU → requant FP32→E4M3 → SMEM handoff → GEMM2 [M,256]×[256,4096] → output
 *
 * Two kernel variants:
 *   v1: Single-block (256 threads), SMEM handoff, bit-exact correctness baseline
 *   v2: Multi-block cooperative (16×256 threads), distributed GEMM1+GEMM2, for performance
 *
 * 256 threads/block (8 cooperative warps, no warp specialization).
 * SM120 constraints: compute_120a, 99KB SMEM, no tcgen05/TMEM, no BF16 MMA, 1×1×1 cluster.
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a \
 *     -o verdict_fused_single_expert verdict_fused_single_expert.cu
 */

#include <cooperative_groups.h>
#include <cuda_runtime.h>
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
constexpr int BLOCK_SIZE = 256;

// ============================================================================
// E4M3 Lookup Table (constant memory)
// ============================================================================
__constant__ float c_e4m3_lut[256];
static float h_e4m3_lut[256];

void init_e4m3_lut() {
    for (int i = 0; i < 256; i++) {
        int s = (i >> 7) & 1;
        int e = (i >> 3) & 0xF;
        int m = i & 0x7;
        float val;
        if (e == 15 && m == 7) val = 0.0f;
        else if (e == 0) val = ldexpf((float)m / 8.0f, -6);
        else val = ldexpf(1.0f + (float)m / 8.0f, e - 7);
        h_e4m3_lut[i] = s ? -val : val;
    }
    cudaMemcpyToSymbol(c_e4m3_lut, h_e4m3_lut, sizeof(h_e4m3_lut));
}

float host_e4m3_decode(uint8_t x) { return h_e4m3_lut[x]; }

uint8_t host_e4m3_encode(float v) {
    if (isnan(v)) return 0x7F;
    int s = v < 0 ? 1 : 0;
    float av = fabsf(v);
    if (av > 448.0f) av = 448.0f;
    uint8_t best = 0;
    float best_err = FLT_MAX;
    for (int e = 0; e <= 15; e++) {
        for (int m = 0; m <= 7; m++) {
            if (e == 15 && m == 7) continue;
            float repr = (e == 0) ? ldexpf((float)m / 8.0f, -6)
                                  : ldexpf(1.0f + (float)m / 8.0f, e - 7);
            float err = fabsf(av - repr);
            if (err < best_err) { best_err = err; best = (e << 3) | m; }
        }
    }
    return (s << 7) | best;
}

float host_silu(float x) { return x / (1.0f + expf(-x)); }

// ============================================================================
// Device Helpers
// ============================================================================

__device__ __forceinline__ uint8_t float_to_e4m3(float v) {
    uint16_t packed;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(packed) : "f"(0.0f), "f"(v));
    return (uint8_t)(packed & 0xFF);
}

__device__ __forceinline__ float d_silu(float x) {
    return x / (1.0f + __expf(-x));
}

// ============================================================================
// V1: Single-block fused kernel (SMEM handoff, correctness baseline)
// ============================================================================
__global__ void verdict_fused_v1(
    const float* __restrict__ input,
    const float* __restrict__ w1,      // [K, 2*N_half]
    const float* __restrict__ w2,      // [N_half, K]
    float* __restrict__ output,        // [M, K]
    int K, int N_half)
{
    const int m = blockIdx.x;
    const int tid = threadIdx.x;
    const int N2 = 2 * N_half;

    extern __shared__ char smem[];
    float*   s_input = (float*)smem;
    uint8_t* s_inter = (uint8_t*)(smem + K * sizeof(float));

    // Load input row to SMEM
    for (int i = tid; i < K; i += BLOCK_SIZE)
        s_input[i] = input[m * K + i];
    __syncthreads();

    // GEMM1 + SwiGLU + E4M3 requant → SMEM
    if (tid < N_half) {
        float gate_acc = 0.0f, up_acc = 0.0f;
        for (int k = 0; k < K; k++) {
            float inp = s_input[k];
            gate_acc += inp * w1[k * N2 + tid];
            up_acc   += inp * w1[k * N2 + tid + N_half];
        }
        float swiglu = up_acc * d_silu(gate_acc);
        uint8_t e4m3_byte = float_to_e4m3(swiglu);
        // SMEM handoff with Swizzle<3,4,3>
        int swiz_col = tid ^ ((0 & 7) << 3);  // row=0 in SMEM
        s_inter[swiz_col] = e4m3_byte;
    }
    __syncthreads();

    // GEMM2: E4M3 (SMEM) × W2 (GMEM) → output
    float acc2[16] = {};
    for (int k = 0; k < N_half; k++) {
        int swiz_k = k ^ ((0 & 7) << 3);
        float a = c_e4m3_lut[s_inter[swiz_k]];
        #pragma unroll
        for (int jj = 0; jj < 16; jj++)
            acc2[jj] += a * w2[k * K + tid + jj * BLOCK_SIZE];
    }
    for (int jj = 0; jj < 16; jj++)
        output[m * K + tid + jj * BLOCK_SIZE] = acc2[jj];
}

// ============================================================================
// V2: Multi-block cooperative fused kernel (distributed GEMM1 + GEMM2)
// ============================================================================
//
// Architecture (for M=1, num_blocks=16):
//   Phase 1a: Each block computes partial GEMM1 over K/num_blocks elements
//             Writes partial gate/up sums to GMEM buffer
//   Grid sync
//   Phase 1b: Block 0 reduces partials → SwiGLU → E4M3 → GMEM intermediate
//   Grid sync
//   Phase 2:  All blocks cooperate on GEMM2 (4096 output cols / 16 = 256/block)
//
// SMEM per block: input_slice[K/num_blocks] floats
// GMEM extra: partials[num_blocks * 2 * N_half] + intermediate[N_half]
//
__global__ void verdict_fused_v2(
    const float* __restrict__ input,
    const float* __restrict__ w1,      // [K, 2*N_half]
    const float* __restrict__ w2,      // [N_half, K]
    float* __restrict__ output,
    float* __restrict__ partials,      // [num_blocks, 2, N_half]
    float* __restrict__ gmem_inter,    // [N_half] dequanted intermediate
    int K, int N_half)
{
    cg::grid_group grid = cg::this_grid();
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_blocks = gridDim.x;
    const int N2 = 2 * N_half;

    extern __shared__ char smem[];
    float* s_input_slice = (float*)smem;

    // ---- Phase 1a: Distributed GEMM1 (partial dot products) ----
    int k_per_block = K / num_blocks;
    int k_start = bid * k_per_block;

    // Load input slice to SMEM
    for (int i = tid; i < k_per_block; i += BLOCK_SIZE)
        s_input_slice[i] = input[k_start + i];
    __syncthreads();

    // Each thread: partial sum for 1 gate col + 1 up col
    if (tid < N_half) {
        float gate_partial = 0.0f, up_partial = 0.0f;
        for (int ki = 0; ki < k_per_block; ki++) {
            float inp = s_input_slice[ki];
            int k = k_start + ki;
            gate_partial += inp * w1[k * N2 + tid];
            up_partial   += inp * w1[k * N2 + tid + N_half];
        }
        partials[bid * 2 * N_half + tid] = gate_partial;
        partials[bid * 2 * N_half + N_half + tid] = up_partial;
    }

    grid.sync();

    // ---- Phase 1b: Reduce + SwiGLU + E4M3 requant (block 0 only) ----
    if (bid == 0 && tid < N_half) {
        float gate_sum = 0.0f, up_sum = 0.0f;
        for (int b = 0; b < num_blocks; b++) {
            gate_sum += partials[b * 2 * N_half + tid];
            up_sum   += partials[b * 2 * N_half + N_half + tid];
        }
        float swiglu = up_sum * d_silu(gate_sum);
        uint8_t e4m3_byte = float_to_e4m3(swiglu);
        gmem_inter[tid] = c_e4m3_lut[e4m3_byte];
    }

    grid.sync();

    // ---- Phase 2: Distributed GEMM2 ----
    // Each thread across all blocks handles a unique output column
    int global_tid = bid * BLOCK_SIZE + tid;
    int total_threads = num_blocks * BLOCK_SIZE;

    for (int j = global_tid; j < K; j += total_threads) {
        float acc = 0.0f;
        for (int k = 0; k < N_half; k++)
            acc += gmem_inter[k] * w2[k * K + j];
        output[j] = acc;
    }
}

// ============================================================================
// 5 SEPARATE BASELINE KERNELS
// ============================================================================

__global__ void kern_gemm1_gate(
    const float* __restrict__ input, const float* __restrict__ w1,
    float* __restrict__ gate, int K, int N_half)
{
    int m = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N_half) return;
    int N2 = 2 * N_half;
    float acc = 0.0f;
    for (int k = 0; k < K; k++)
        acc += input[m * K + k] * w1[k * N2 + j];
    gate[m * N_half + j] = acc;
}

__global__ void kern_gemm1_up(
    const float* __restrict__ input, const float* __restrict__ w1,
    float* __restrict__ up, int K, int N_half)
{
    int m = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N_half) return;
    int N2 = 2 * N_half;
    float acc = 0.0f;
    for (int k = 0; k < K; k++)
        acc += input[m * K + k] * w1[k * N2 + j + N_half];
    up[m * N_half + j] = acc;
}

__global__ void kern_swiglu(
    const float* __restrict__ gate, const float* __restrict__ up,
    float* __restrict__ inter, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float g = gate[i];
    inter[i] = up[i] * g / (1.0f + __expf(-g));
}

__global__ void kern_requant(float* __restrict__ data, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    uint8_t e4m3 = float_to_e4m3(data[i]);
    data[i] = c_e4m3_lut[e4m3];
}

__global__ void kern_gemm2(
    const float* __restrict__ inter, const float* __restrict__ w2,
    float* __restrict__ output, int K, int N_half)
{
    int m = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= K) return;
    float acc = 0.0f;
    for (int k = 0; k < N_half; k++)
        acc += inter[m * N_half + k] * w2[k * K + j];
    output[m * K + j] = acc;
}

// ============================================================================
// Host Reference
// ============================================================================

void host_reference(
    const float* input, const float* w1, const float* w2,
    float* ref_fp32, float* ref_e4m3,
    int M, int K, int N_half)
{
    int N2 = 2 * N_half;
    for (int m = 0; m < M; m++) {
        std::vector<float> gate(N_half), up(N_half);
        for (int j = 0; j < N_half; j++) {
            float ga = 0, ua = 0;
            for (int k = 0; k < K; k++) {
                ga += input[m * K + k] * w1[k * N2 + j];
                ua += input[m * K + k] * w1[k * N2 + j + N_half];
            }
            gate[j] = ga; up[j] = ua;
        }
        std::vector<float> inter(N_half), inter_q(N_half);
        for (int j = 0; j < N_half; j++) {
            inter[j] = up[j] * host_silu(gate[j]);
            inter_q[j] = host_e4m3_decode(host_e4m3_encode(inter[j]));
        }
        for (int j = 0; j < K; j++) {
            float acc = 0, acc_q = 0;
            for (int k = 0; k < N_half; k++) {
                acc   += inter[k]   * w2[k * K + j];
                acc_q += inter_q[k] * w2[k * K + j];
            }
            ref_fp32[m * K + j] = acc;
            ref_e4m3[m * K + j] = acc_q;
        }
    }
}

// ============================================================================
// Error Stats
// ============================================================================

struct ErrorStats {
    float max_abs, avg_abs, rms_ref, norm_pct;
    int nan_count, zero_count, close_count, total;
};

ErrorStats compute_error(const float* actual, const float* ref, int n) {
    ErrorStats s = {};
    s.total = n;
    double sum_err = 0, sum_ref2 = 0;
    for (int i = 0; i < n; i++) {
        if (isnan(actual[i])) { s.nan_count++; continue; }
        if (actual[i] == 0.0f) s.zero_count++;
        float e = fabsf(actual[i] - ref[i]);
        s.max_abs = fmaxf(s.max_abs, e);
        sum_err += e;
        sum_ref2 += (double)ref[i] * ref[i];
        if ((fabsf(ref[i]) > 0.001f && e / fabsf(ref[i]) < 0.05f) ||
            (fabsf(ref[i]) <= 0.001f && fabsf(actual[i]) < 0.05f))
            s.close_count++;
    }
    s.avg_abs = (float)(sum_err / n);
    s.rms_ref = sqrtf((float)(sum_ref2 / n));
    s.norm_pct = s.rms_ref > 0 ? 100.0f * s.avg_abs / s.rms_ref : 0;
    return s;
}

void print_error(const char* label, ErrorStats s) {
    printf("  %s:\n", label);
    printf("    Normalized error: %.4f%%, Max abs: %.6f, Within 5%%: %d/%d (%.1f%%)\n",
           s.norm_pct, s.max_abs, s.close_count, s.total,
           100.0f * s.close_count / s.total);
}

// ============================================================================
// Benchmark
// ============================================================================

struct BenchResult { float avg_us, med_us, p5_us, p95_us; };

BenchResult bench_stats(std::vector<float>& t) {
    std::sort(t.begin(), t.end());
    BenchResult r;
    r.avg_us = std::accumulate(t.begin(), t.end(), 0.0f) / t.size();
    r.med_us = t[t.size() / 2];
    r.p5_us  = t[(int)(t.size() * 0.05f)];
    r.p95_us = t[(int)(t.size() * 0.95f)];
    return r;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM%d%d, %d SMs, %dKB SMEM)\n\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           (int)(prop.sharedMemPerMultiprocessor / 1024));

    int M = 1, K = 4096, N_half = 256;
    if (argc > 1) M = atoi(argv[1]);
    int N2 = 2 * N_half;

    printf("=== Task 3: Fused Single Expert ===\n");
    printf("  M=%d, K=%d, N_half=%d\n", M, K, N_half);
    printf("  GEMM1: [%d,%d]×[%d,%d]→[%d,%d] → SwiGLU → E4M3 → GEMM2: [%d,%d]×[%d,%d]→[%d,%d]\n",
           M, K, K, N2, M, N2, M, N_half, N_half, K, M, K);

    init_e4m3_lut();

    // V2 config
    int num_blocks_v2 = 16;
    int smem_v2 = (K / num_blocks_v2) * sizeof(float);  // input slice
    printf("  V2: %d blocks × %d threads, SMEM/block=%d bytes\n\n", num_blocks_v2, BLOCK_SIZE, smem_v2);

    // Allocate
    srand(42);
    auto randf = []() { return (float)rand() / RAND_MAX * 2.0f - 1.0f; };
    float sw1 = 1.0f / sqrtf((float)K), sw2 = 1.0f / sqrtf((float)N_half);

    std::vector<float> h_input(M * K), h_w1(K * N2), h_w2(N_half * K);
    std::vector<float> h_out_v1(M * K), h_out_v2(M * K), h_out_sep(M * K);
    std::vector<float> h_ref_fp32(M * K), h_ref_e4m3(M * K);

    for (auto& v : h_input) v = randf();
    for (auto& v : h_w1)    v = randf() * sw1;
    for (auto& v : h_w2)    v = randf() * sw2;

    float *d_input, *d_w1, *d_w2;
    float *d_out_v1, *d_out_v2, *d_out_sep;
    float *d_gate, *d_up, *d_inter;
    float *d_partials, *d_gmem_inter;

    cudaMalloc(&d_input, M * K * sizeof(float));
    cudaMalloc(&d_w1, K * N2 * sizeof(float));
    cudaMalloc(&d_w2, N_half * K * sizeof(float));
    cudaMalloc(&d_out_v1, M * K * sizeof(float));
    cudaMalloc(&d_out_v2, M * K * sizeof(float));
    cudaMalloc(&d_out_sep, M * K * sizeof(float));
    cudaMalloc(&d_gate, M * N_half * sizeof(float));
    cudaMalloc(&d_up, M * N_half * sizeof(float));
    cudaMalloc(&d_inter, M * N_half * sizeof(float));
    cudaMalloc(&d_partials, num_blocks_v2 * 2 * N_half * sizeof(float));
    cudaMalloc(&d_gmem_inter, N_half * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w1, h_w1.data(), K * N2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, h_w2.data(), N_half * K * sizeof(float), cudaMemcpyHostToDevice);

    // ================================================================
    // Correctness Test
    // ================================================================
    printf("--- Correctness Test ---\n");

    // V1: single-block (correctness baseline)
    int smem_v1 = K * sizeof(float) + N_half;
    verdict_fused_v1<<<M, BLOCK_SIZE, smem_v1>>>(d_input, d_w1, d_w2, d_out_v1, K, N_half);
    cudaError_t err = cudaDeviceSynchronize();
    printf("  V1 (single-block): %s\n", err == cudaSuccess ? "OK" : cudaGetErrorString(err));

    // V2: multi-block cooperative
    {
        void* args[] = { &d_input, &d_w1, &d_w2, &d_out_v2, &d_partials, &d_gmem_inter, &K, &N_half };
        cudaMemset(d_partials, 0, num_blocks_v2 * 2 * N_half * sizeof(float));
        err = cudaLaunchCooperativeKernel(
            (void*)verdict_fused_v2, num_blocks_v2, BLOCK_SIZE, args, smem_v2);
        if (err != cudaSuccess) {
            printf("  V2 launch: %s\n", cudaGetErrorString(err));
        } else {
            err = cudaDeviceSynchronize();
            printf("  V2 (cooperative %d blocks): %s\n", num_blocks_v2,
                   err == cudaSuccess ? "OK" : cudaGetErrorString(err));
        }
    }

    // 5 separate kernels
    dim3 grid1((N_half + BLOCK_SIZE - 1) / BLOCK_SIZE, M);
    dim3 grid2((K + BLOCK_SIZE - 1) / BLOCK_SIZE, M);
    int swiglu_n = M * N_half;

    kern_gemm1_gate<<<grid1, BLOCK_SIZE>>>(d_input, d_w1, d_gate, K, N_half);
    kern_gemm1_up<<<grid1, BLOCK_SIZE>>>(d_input, d_w1, d_up, K, N_half);
    kern_swiglu<<<(swiglu_n + 255) / 256, 256>>>(d_gate, d_up, d_inter, swiglu_n);
    kern_requant<<<(swiglu_n + 255) / 256, 256>>>(d_inter, swiglu_n);
    kern_gemm2<<<grid2, BLOCK_SIZE>>>(d_inter, d_w2, d_out_sep, K, N_half);
    err = cudaDeviceSynchronize();
    printf("  Separate (5 kernels): %s\n", err == cudaSuccess ? "OK" : cudaGetErrorString(err));

    // Copy results
    cudaMemcpy(h_out_v1.data(), d_out_v1, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_v2.data(), d_out_v2, M * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_sep.data(), d_out_sep, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    // Host reference
    host_reference(h_input.data(), h_w1.data(), h_w2.data(),
                   h_ref_fp32.data(), h_ref_e4m3.data(), M, K, N_half);

    printf("\n  First 4 values:\n");
    printf("    V1:        [%.6f, %.6f, %.6f, %.6f]\n",
           h_out_v1[0], h_out_v1[1], h_out_v1[2], h_out_v1[3]);
    printf("    V2:        [%.6f, %.6f, %.6f, %.6f]\n",
           h_out_v2[0], h_out_v2[1], h_out_v2[2], h_out_v2[3]);
    printf("    Separate:  [%.6f, %.6f, %.6f, %.6f]\n",
           h_out_sep[0], h_out_sep[1], h_out_sep[2], h_out_sep[3]);
    printf("    Ref(FP32): [%.6f, %.6f, %.6f, %.6f]\n",
           h_ref_fp32[0], h_ref_fp32[1], h_ref_fp32[2], h_ref_fp32[3]);
    printf("    Ref(E4M3): [%.6f, %.6f, %.6f, %.6f]\n\n",
           h_ref_e4m3[0], h_ref_e4m3[1], h_ref_e4m3[2], h_ref_e4m3[3]);

    auto ev1_fp32 = compute_error(h_out_v1.data(), h_ref_fp32.data(), M * K);
    auto ev1_e4m3 = compute_error(h_out_v1.data(), h_ref_e4m3.data(), M * K);
    auto ev2_fp32 = compute_error(h_out_v2.data(), h_ref_fp32.data(), M * K);
    auto ev2_e4m3 = compute_error(h_out_v2.data(), h_ref_e4m3.data(), M * K);
    auto ev1_sep  = compute_error(h_out_v1.data(), h_out_sep.data(), M * K);
    auto ev2_v1   = compute_error(h_out_v2.data(), h_out_v1.data(), M * K);

    print_error("V1 vs FP32 ref (E4M3 quant error)", ev1_fp32);
    print_error("V1 vs E4M3 ref (FP32 rounding)", ev1_e4m3);
    print_error("V2 vs FP32 ref (E4M3 quant error)", ev2_fp32);
    print_error("V2 vs E4M3 ref (FP32 rounding)", ev2_e4m3);
    print_error("V1 vs Separate (should be ~0)", ev1_sep);
    print_error("V2 vs V1 (should be ~0)", ev2_v1);

    // Pass criteria
    bool pass_v1 = (ev1_e4m3.nan_count == 0) && (ev1_e4m3.norm_pct < 0.01f);
    bool pass_v2 = (ev2_e4m3.nan_count == 0) && (ev2_e4m3.norm_pct < 0.5f);
    bool pass_quant = (ev1_fp32.nan_count == 0) && (ev1_fp32.norm_pct < 10.0f);
    bool pass_match = (ev1_sep.nan_count == 0) && (ev1_sep.norm_pct < 0.01f);

    printf("\n  CORRECTNESS:\n");
    printf("    V1 bit-exact vs E4M3 ref: %s (%.4f%%)\n", pass_v1 ? "PASS" : "FAIL", ev1_e4m3.norm_pct);
    printf("    V2 vs E4M3 ref:           %s (%.4f%%)\n", pass_v2 ? "PASS" : "FAIL", ev2_e4m3.norm_pct);
    printf("    E4M3 quant error:         %s (%.2f%%)\n", pass_quant ? "PASS" : "FAIL", ev1_fp32.norm_pct);
    printf("    V1 ≈ Separate:            %s\n", pass_match ? "PASS" : "FAIL");

    // ================================================================
    // Benchmark
    // ================================================================
    printf("\n--- Benchmark (M=%d) ---\n", M);

    const int warmup = 50, iters = 200;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // V1 benchmark
    for (int i = 0; i < warmup; i++)
        verdict_fused_v1<<<M, BLOCK_SIZE, smem_v1>>>(d_input, d_w1, d_w2, d_out_v1, K, N_half);
    cudaDeviceSynchronize();
    std::vector<float> v1_times;
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);
        verdict_fused_v1<<<M, BLOCK_SIZE, smem_v1>>>(d_input, d_w1, d_w2, d_out_v1, K, N_half);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        v1_times.push_back(ms * 1000.0f);
    }

    // V2 benchmark
    {
        void* args[] = { &d_input, &d_w1, &d_w2, &d_out_v2, &d_partials, &d_gmem_inter, &K, &N_half };
        for (int i = 0; i < warmup; i++) {
            cudaMemset(d_partials, 0, num_blocks_v2 * 2 * N_half * sizeof(float));
            cudaLaunchCooperativeKernel((void*)verdict_fused_v2, num_blocks_v2, BLOCK_SIZE, args, smem_v2);
        }
        cudaDeviceSynchronize();
    }
    std::vector<float> v2_times;
    {
        void* args[] = { &d_input, &d_w1, &d_w2, &d_out_v2, &d_partials, &d_gmem_inter, &K, &N_half };
        for (int i = 0; i < iters; i++) {
            cudaEventRecord(start);
            cudaLaunchCooperativeKernel((void*)verdict_fused_v2, num_blocks_v2, BLOCK_SIZE, args, smem_v2);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms; cudaEventElapsedTime(&ms, start, stop);
            v2_times.push_back(ms * 1000.0f);
        }
    }

    // Separate benchmark
    for (int i = 0; i < warmup; i++) {
        kern_gemm1_gate<<<grid1, BLOCK_SIZE>>>(d_input, d_w1, d_gate, K, N_half);
        kern_gemm1_up<<<grid1, BLOCK_SIZE>>>(d_input, d_w1, d_up, K, N_half);
        kern_swiglu<<<(swiglu_n + 255) / 256, 256>>>(d_gate, d_up, d_inter, swiglu_n);
        kern_requant<<<(swiglu_n + 255) / 256, 256>>>(d_inter, swiglu_n);
        kern_gemm2<<<grid2, BLOCK_SIZE>>>(d_inter, d_w2, d_out_sep, K, N_half);
    }
    cudaDeviceSynchronize();
    std::vector<float> sep_times;
    for (int i = 0; i < iters; i++) {
        cudaEventRecord(start);
        kern_gemm1_gate<<<grid1, BLOCK_SIZE>>>(d_input, d_w1, d_gate, K, N_half);
        kern_gemm1_up<<<grid1, BLOCK_SIZE>>>(d_input, d_w1, d_up, K, N_half);
        kern_swiglu<<<(swiglu_n + 255) / 256, 256>>>(d_gate, d_up, d_inter, swiglu_n);
        kern_requant<<<(swiglu_n + 255) / 256, 256>>>(d_inter, swiglu_n);
        kern_gemm2<<<grid2, BLOCK_SIZE>>>(d_inter, d_w2, d_out_sep, K, N_half);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        sep_times.push_back(ms * 1000.0f);
    }

    auto r_v1 = bench_stats(v1_times);
    auto r_v2 = bench_stats(v2_times);
    auto r_sep = bench_stats(sep_times);

    printf("\n  %-30s  avg      med      p5       p95\n", "");
    printf("  %-30s  %.1f μs  %.1f μs  %.1f μs  %.1f μs\n",
           "V1 (1 block, SMEM handoff)", r_v1.avg_us, r_v1.med_us, r_v1.p5_us, r_v1.p95_us);
    printf("  %-30s  %.1f μs  %.1f μs  %.1f μs  %.1f μs\n",
           "V2 (cooperative, 16 blocks)", r_v2.avg_us, r_v2.med_us, r_v2.p5_us, r_v2.p95_us);
    printf("  %-30s  %.1f μs  %.1f μs  %.1f μs  %.1f μs\n",
           "Separate (5 launches)", r_sep.avg_us, r_sep.med_us, r_sep.p5_us, r_sep.p95_us);

    float speedup_v2 = r_sep.med_us / r_v2.med_us;
    float speedup_v1 = r_sep.med_us / r_v1.med_us;
    printf("\n  Speedup V2 vs Separate: %.2fx\n", speedup_v2);
    printf("  Speedup V1 vs Separate: %.2fx\n", speedup_v1);
    printf("  V2 vs V1: %.2fx\n", r_v1.med_us / r_v2.med_us);

    // ================================================================
    // Summary
    // ================================================================
    printf("\n=== TASK 3 SUMMARY ===\n");
    printf("  Pipeline: GEMM1[%d,%d]×[%d,%d] → SwiGLU → E4M3 → GEMM2[%d,%d]×[%d,%d]\n",
           M, K, K, N2, M, N_half, N_half, K);
    printf("  E4M3 quant error: %.2f%% (vs FP32 PyTorch reference)\n", ev1_fp32.norm_pct);
    printf("  V1: %.1f μs (1 block, SMEM handoff, bit-exact)\n", r_v1.med_us);
    printf("  V2: %.1f μs (%d blocks, cooperative, grid sync)\n", r_v2.med_us, num_blocks_v2);
    printf("  5-Separate: %.1f μs\n", r_sep.med_us);
    printf("  Best fused speedup: %.2fx\n", fmaxf(speedup_v1, speedup_v2));
    printf("  VERDICT: %s\n",
           (pass_v1 && pass_v2 && pass_quant && pass_match) ? "PASSED" : "FAILED");

    // Cleanup
    cudaFree(d_input); cudaFree(d_w1); cudaFree(d_w2);
    cudaFree(d_out_v1); cudaFree(d_out_v2); cudaFree(d_out_sep);
    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_inter);
    cudaFree(d_partials); cudaFree(d_gmem_inter);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return (pass_v1 && pass_v2 && pass_quant && pass_match) ? 0 : 1;
}
