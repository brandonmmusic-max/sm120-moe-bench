/**
 * SM120 Flash Decode — Split-KV kernel for single-token decode
 *
 * When Q is very small (1-4 tokens) and KV cache is long (1K-128K),
 * the main kernel underutilizes the GPU because only 1-4 CTAs launch.
 * Split-KV partitions the KV dimension across multiple CTAs, each
 * computing partial attention, then a reduction kernel combines them.
 *
 * Phase 1: Partial attention per KV split
 *   Each CTA processes a KV chunk and outputs:
 *   - partial_O[split, head, Sq, D] in FP32
 *   - partial_lse[split, head, Sq] (log-sum-exp for rescaling)
 *
 * Phase 2: Reduce across splits
 *   Combine partial results using the log-sum-exp trick:
 *   O = sum_i (exp(lse_i - lse_max) * O_i) / sum_i exp(lse_i - lse_max)
 *
 * This is mathematically exact — produces identical results to non-split.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define WARP_SIZE 32
#define HEAD_DIM 128
#define BLOCK_KV 64     // KV elements per CTA
#define MMA_K 16

// ============================================================================
// Phase 2: Reduction kernel — combine split results
// Grid: (num_q_rows, batch * num_heads)
// Each thread reduces across num_splits for one (row, head_dim_subset)
// ============================================================================
__global__ void split_kv_reduce(
    const float* __restrict__ partial_O,    // [num_splits, B*H, Sq, D]
    const float* __restrict__ partial_lse,  // [num_splits, B*H, Sq]
    __nv_bfloat16* __restrict__ O,          // [B*H, Sq, D]
    int num_splits,
    int Sq
) {
    const int row = blockIdx.x;       // query row
    const int head = blockIdx.y;      // batch * head index
    const int tid = threadIdx.x;      // 0..127 → covers HEAD_DIM=128

    if (row >= Sq || tid >= HEAD_DIM) return;

    const int head_stride = Sq * HEAD_DIM;
    const int split_stride = gridDim.y * Sq * HEAD_DIM;  // B*H * Sq * D
    const int lse_split_stride = gridDim.y * Sq;

    // Find global max LSE across splits
    float max_lse = -FLT_MAX;
    for (int s = 0; s < num_splits; s++) {
        float lse = partial_lse[s * lse_split_stride + head * Sq + row];
        max_lse = fmaxf(max_lse, lse);
    }

    // Combine: O = sum(exp(lse_i - max_lse) * O_i) / sum(exp(lse_i - max_lse))
    float sum_exp = 0.0f;
    float sum_val = 0.0f;

    for (int s = 0; s < num_splits; s++) {
        float lse = partial_lse[s * lse_split_stride + head * Sq + row];
        float w = __expf(lse - max_lse);
        sum_exp += w;
        sum_val += w * partial_O[s * split_stride + head * head_stride + row * HEAD_DIM + tid];
    }

    float result = (sum_exp > 0.0f) ? sum_val / sum_exp : 0.0f;
    O[head * Sq * HEAD_DIM + row * HEAD_DIM + tid] = __float2bfloat16(result);
}

// ============================================================================
// Phase 1: Partial attention — each CTA processes one KV chunk
// Uses the validated MMA fragment layout from the main kernel.
// Simplified: single warp, no multi-warp tiling for Q (decode = small Q)
// ============================================================================

__device__ __forceinline__ uint32_t pack2(const __nv_bfloat16& a, const __nv_bfloat16& b) {
    uint32_t r;
    asm("mov.b32 %0, {%1, %2};" : "=r"(r) : "h"(*(const uint16_t*)&a), "h"(*(const uint16_t*)&b));
    return r;
}

// For decode (Q=1..few), use scalar Q@K^T + scalar P@V
// MMA is wasteful when Sq=1 because only 1 of 16 MMA rows is used
__global__ void split_kv_partial(
    const __nv_bfloat16* __restrict__ Q,   // [B*H, Sq, D]
    const __nv_bfloat16* __restrict__ K,   // [B*Hkv, Skv, D]
    const __nv_bfloat16* __restrict__ V,   // [B*Hkv, Skv, D]
    float* __restrict__ partial_O,          // [num_splits, B*H, Sq, D]
    float* __restrict__ partial_lse,        // [num_splits, B*H, Sq]
    int Sq, int Skv, int Hq, int Hkv,
    float scale, int kv_per_split
) {
    const int split_idx = blockIdx.x;
    const int q_row = blockIdx.y;    // which query row
    const int head = blockIdx.z;     // batch * head
    const int kv_head = head / (Hq / Hkv);
    const int tid = threadIdx.x;     // 0..127

    if (q_row >= Sq) return;

    const int kv_start = split_idx * kv_per_split;
    const int kv_end = min(kv_start + kv_per_split, Skv);
    if (kv_start >= Skv) return;

    const __nv_bfloat16* q_row_ptr = Q + head * Sq * HEAD_DIM + q_row * HEAD_DIM;
    const __nv_bfloat16* k_ptr = K + kv_head * Skv * HEAD_DIM;
    const __nv_bfloat16* v_ptr = V + kv_head * Skv * HEAD_DIM;

    // Each thread handles a subset of head_dim columns for output
    // 128 threads, HEAD_DIM=128 → 1 output column per thread
    const int d = tid;

    // Load Q row into registers (each thread loads 1 element)
    // Actually for dot product we need ALL d values per thread
    // Use shared memory for Q
    __shared__ __nv_bfloat16 q_smem[HEAD_DIM];
    if (tid < HEAD_DIM) q_smem[tid] = q_row_ptr[tid];
    __syncthreads();

    // Online softmax state
    float rowmax = -FLT_MAX;
    float rowsum = 0.0f;
    float o_val = 0.0f;  // output for column d

    // Iterate over KV positions in this split
    for (int kv = kv_start; kv < kv_end; kv++) {
        // Compute score: Q[row, :] dot K[kv, :]
        // Each thread computes partial dot product, then reduce
        float partial_dot = 0.0f;
        if (tid < HEAD_DIM) {
            partial_dot = __bfloat162float(q_smem[tid]) *
                          __bfloat162float(k_ptr[kv * HEAD_DIM + tid]);
        }

        // Warp-level reduction for dot product
        // First reduce within warp
        for (int mask = 16; mask > 0; mask >>= 1)
            partial_dot += __shfl_xor_sync(0xffffffff, partial_dot, mask);

        // Cross-warp reduction via shared memory
        __shared__ float warp_sums[4];
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        if (lane_id == 0) warp_sums[warp_id] = partial_dot;
        __syncthreads();

        float score;
        if (tid < 4) {
            score = warp_sums[tid];
            for (int mask = 2; mask > 0; mask >>= 1)
                score += __shfl_xor_sync(0xf, score, mask);
            warp_sums[0] = score;
        }
        __syncthreads();
        score = warp_sums[0] * scale;

        // Online softmax update
        float new_max = fmaxf(rowmax, score);
        float rescale = __expf(rowmax - new_max);
        float p = __expf(score - new_max);

        rowsum = rowsum * rescale + p;
        o_val = o_val * rescale + p * __bfloat162float(v_ptr[kv * HEAD_DIM + d]);
        rowmax = new_max;
    }

    // Write partial results
    int num_heads = gridDim.z;
    int split_offset = split_idx * num_heads * Sq;
    int head_offset = head * Sq;

    if (d < HEAD_DIM) {
        partial_O[split_idx * num_heads * Sq * HEAD_DIM +
                  head * Sq * HEAD_DIM + q_row * HEAD_DIM + d] = o_val;
    }

    // Write LSE (only one thread per row)
    if (tid == 0) {
        float lse = rowmax + logf(fmaxf(rowsum, 1e-10f));
        partial_lse[split_idx * num_heads * Sq + head * Sq + q_row] = lse;
    }
}

// ============================================================================
// Host interface
// ============================================================================
extern "C" void sm120_flash_decode(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O,
    int batch, int Hq, int Hkv, int Sq, int Skv, int head_dim,
    cudaStream_t stream
) {
    float scale = 1.0f / sqrtf((float)head_dim);

    // Choose number of splits based on KV length
    // More splits = more parallelism but more reduction overhead
    int num_splits;
    if (Skv <= 256) num_splits = 1;
    else if (Skv <= 1024) num_splits = 4;
    else if (Skv <= 4096) num_splits = 8;
    else if (Skv <= 16384) num_splits = 16;
    else num_splits = 32;

    int kv_per_split = (Skv + num_splits - 1) / num_splits;
    int total_heads = batch * Hq;

    // Allocate temporary buffers
    float *partial_O, *partial_lse;
    cudaMalloc(&partial_O, num_splits * total_heads * Sq * HEAD_DIM * sizeof(float));
    cudaMalloc(&partial_lse, num_splits * total_heads * Sq * sizeof(float));

    // Phase 1: partial attention
    dim3 grid1(num_splits, Sq, total_heads);
    dim3 block1(128);  // 128 threads = HEAD_DIM
    split_kv_partial<<<grid1, block1, 0, stream>>>(
        Q, K, V, partial_O, partial_lse,
        Sq, Skv, Hq, Hkv, scale, kv_per_split
    );

    // Phase 2: reduce
    dim3 grid2(Sq, total_heads);
    dim3 block2(HEAD_DIM);
    split_kv_reduce<<<grid2, block2, 0, stream>>>(
        partial_O, partial_lse, O,
        num_splits, Sq
    );

    cudaFree(partial_O);
    cudaFree(partial_lse);
}
