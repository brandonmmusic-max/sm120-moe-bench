/**
 * SM120 Multi-GPU Sequence-Parallel Flash Attention
 *
 * Architecture:
 * 1. KV pre-distributed across GPUs via P2P cudaMemcpyPeer (one-time setup)
 * 2. Each GPU runs sm120_fa_v4 kernel on its local KV chunk → (O_partial, LSE)
 * 3. Custom CUDA combine kernel merges partial results via online softmax:
 *    m = max(lse_0, ..., lse_N-1)
 *    O = Σ exp(lse_i - m) * O_i / Σ exp(lse_i - m)
 *
 * Communication: only the partial O + LSE are gathered (~33KB per tile)
 * P2P enabled via iommu=pt on Threadripper
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// ============================================================================
// Online softmax combine kernel
// Merges N partial attention outputs using their log-sum-exp values
//
// Inputs (all on the combine GPU):
//   partial_O: [N, B*Hq, Sq, D] bf16  — partial outputs from each GPU
//   partial_lse: [N, B*Hq, Sq] f32    — log-sum-exp from each GPU
//   N: number of partials to combine
//
// Output:
//   O: [B*Hq, Sq, D] bf16
// ============================================================================
__global__ void online_softmax_combine(
    const __nv_bfloat16* __restrict__ partial_O,  // [N, total_rows, D]
    const float* __restrict__ partial_lse,         // [N, total_rows]
    __nv_bfloat16* __restrict__ O,                 // [total_rows, D]
    int N, int total_rows, int D
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;

    // Find global max LSE across partials
    float m = -FLT_MAX;
    for (int i = 0; i < N; i++) {
        float lse_i = partial_lse[i * total_rows + row];
        m = fmaxf(m, lse_i);
    }

    // Weighted sum: O = Σ exp(lse_i - m) * O_i / Σ exp(lse_i - m)
    for (int d = 0; d < D; d++) {
        float acc = 0.0f;
        float total_weight = 0.0f;
        for (int i = 0; i < N; i++) {
            float w = __expf(partial_lse[i * total_rows + row] - m);
            float val = __bfloat162float(partial_O[(i * total_rows + row) * D + d]);
            acc += w * val;
            if (d == 0) total_weight += w;  // only compute once
        }
        if (d > 0) {
            // Recompute total_weight (avoid storing it)
            total_weight = 0.0f;
            for (int i = 0; i < N; i++)
                total_weight += __expf(partial_lse[i * total_rows + row] - m);
        }
        O[row * D + d] = __float2bfloat16(acc / total_weight);
    }
}

// Vectorized version: each thread handles one row, processes D in a loop
// More efficient for D=128
__global__ void online_softmax_combine_vec(
    const __nv_bfloat16* __restrict__ partial_O,
    const float* __restrict__ partial_lse,
    __nv_bfloat16* __restrict__ O,
    int N, int total_rows, int D
) {
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int d_base = threadIdx.x * 2;  // each thread handles 2 elements
    if (row >= total_rows || d_base >= D) return;

    // Compute weights (shared across D elements in same row)
    float m = -FLT_MAX;
    for (int i = 0; i < N; i++)
        m = fmaxf(m, partial_lse[i * total_rows + row]);

    float weights[8];  // max 8 GPUs
    float total_w = 0.0f;
    for (int i = 0; i < N; i++) {
        weights[i] = __expf(partial_lse[i * total_rows + row] - m);
        total_w += weights[i];
    }
    float inv_w = 1.0f / total_w;

    // Weighted combine for this thread's 2 elements
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int i = 0; i < N; i++) {
        float w = weights[i] * inv_w;
        acc0 += w * __bfloat162float(partial_O[(i * total_rows + row) * D + d_base]);
        acc1 += w * __bfloat162float(partial_O[(i * total_rows + row) * D + d_base + 1]);
    }

    // Vectorized store
    __nv_bfloat16 v0 = __float2bfloat16(acc0);
    __nv_bfloat16 v1 = __float2bfloat16(acc1);
    uint32_t packed;
    asm("mov.b32 %0, {%1, %2};" : "=r"(packed) : "h"(*(uint16_t*)&v0), "h"(*(uint16_t*)&v1));
    *reinterpret_cast<uint32_t*>(&O[row * D + d_base]) = packed;
}

// ============================================================================
// Host API for multi-GPU attention
// ============================================================================

// Forward declaration of single-GPU kernel
extern "C" void sm120_flash_attn_forward(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    int batch, int Hq, int Hkv, int Sq, int Skv, int hd, cudaStream_t stream
);

extern "C" void sm120_multi_gpu_attn_forward(
    // Q on each GPU (pre-broadcast)
    const __nv_bfloat16** Q_devs,      // [num_gpus] pointers to Q on each device
    // KV chunks on each GPU (pre-distributed)
    const __nv_bfloat16** K_devs,      // [num_gpus] pointers to K chunks
    const __nv_bfloat16** V_devs,      // [num_gpus] pointers to V chunks
    int* Skv_per_gpu,                   // [num_gpus] KV sequence length per GPU
    // Output on combine_gpu
    __nv_bfloat16* O_out,              // [B*Hq, Sq, D] on combine_gpu
    // Params
    int num_gpus, int combine_gpu,
    int batch, int Hq, int Hkv, int Sq, int hd,
    cudaStream_t* streams               // [num_gpus] one stream per GPU
) {
    float sc = 1.0f / sqrtf((float)hd);
    int total_rows = batch * Hq * Sq;

    // Allocate partial outputs + LSE on each GPU
    __nv_bfloat16* partial_O[8];
    float* partial_lse[8];
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaMalloc(&partial_O[g], total_rows * hd * sizeof(__nv_bfloat16));
        cudaMalloc(&partial_lse[g], total_rows * sizeof(float));
    }

    // Launch attention kernel on each GPU (parallel across GPUs)
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        sm120_flash_attn_forward(
            Q_devs[g], K_devs[g], V_devs[g],
            partial_O[g], partial_lse[g],
            batch, Hq, Hkv, Sq, Skv_per_gpu[g], hd, streams[g]
        );
    }

    // Gather partial results to combine_gpu via P2P
    cudaSetDevice(combine_gpu);
    __nv_bfloat16* gathered_O;
    float* gathered_lse;
    cudaMalloc(&gathered_O, num_gpus * total_rows * hd * sizeof(__nv_bfloat16));
    cudaMalloc(&gathered_lse, num_gpus * total_rows * sizeof(float));

    for (int g = 0; g < num_gpus; g++) {
        // Sync source GPU
        cudaStreamSynchronize(streams[g]);

        // P2P copy to combine GPU
        cudaMemcpyPeer(
            gathered_O + g * total_rows * hd, combine_gpu,
            partial_O[g], g,
            total_rows * hd * sizeof(__nv_bfloat16));
        cudaMemcpyPeer(
            gathered_lse + g * total_rows, combine_gpu,
            partial_lse[g], g,
            total_rows * sizeof(float));
    }

    // Launch combine kernel
    cudaSetDevice(combine_gpu);
    // Use vectorized combine: 64 threads per row (D/2 = 64 pairs), 4 rows per block
    dim3 block(64, 4);
    dim3 grid((total_rows + 3) / 4);
    online_softmax_combine_vec<<<grid, block, 0, streams[combine_gpu]>>>(
        gathered_O, gathered_lse, O_out, num_gpus, total_rows, hd);

    // Cleanup
    cudaStreamSynchronize(streams[combine_gpu]);
    for (int g = 0; g < num_gpus; g++) {
        cudaSetDevice(g);
        cudaFree(partial_O[g]);
        cudaFree(partial_lse[g]);
    }
    cudaSetDevice(combine_gpu);
    cudaFree(gathered_O);
    cudaFree(gathered_lse);
}
