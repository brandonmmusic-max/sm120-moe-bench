/**
 * SM120 Flash Decode — Paged KV Cache, Split-KV for vLLM integration
 *
 * Designed for decode workloads: Sq=1 per sequence, large Skv (1K-128K+).
 * Supports:
 *   - Paged KV cache (block_table lookup)
 *   - GQA (grouped query attention, arbitrary q:kv head ratio)
 *   - BF16 Q/output, BF16 KV cache (FP8 support planned)
 *   - Variable sequence lengths per request
 *   - Pre-allocated temporaries (CUDA graph compatible)
 *   - Parameterized HEAD_DIM (128 or 256)
 *
 * KV cache layout (vLLM): [num_blocks, block_size, num_kv_heads, head_dim]
 * block_table: [num_seqs, max_blocks_per_seq]
 * query: [batch * num_q_heads, 1, head_dim]  (reshaped before launch)
 * output: [batch * num_q_heads, 1, head_dim]
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define WARP_SIZE 32

// ==========================================================================
// Phase 1: Partial attention — each CTA processes one KV chunk for one head
// Grid: (num_splits, batch_size, num_q_heads)
// Block: (HEAD_DIM) threads
// ==========================================================================
template <int HEAD_DIM>
__global__ void split_kv_partial_paged(
    const __nv_bfloat16* __restrict__ Q,         // [batch, num_q_heads, HEAD_DIM]
    const __nv_bfloat16* __restrict__ key_cache,  // [num_blocks, block_size, num_kv_heads, HEAD_DIM]
    const __nv_bfloat16* __restrict__ val_cache,  // [num_blocks, block_size, num_kv_heads, HEAD_DIM]
    const int* __restrict__ block_table,           // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,              // [num_seqs]
    float* __restrict__ partial_O,                 // [num_splits, batch, num_q_heads, HEAD_DIM]
    float* __restrict__ partial_lse,               // [num_splits, batch, num_q_heads]
    int num_q_heads,
    int num_kv_heads,
    int block_size,
    int max_blocks_per_seq,
    float scale,
    int kv_per_split
) {
    const int split_idx = blockIdx.x;
    const int seq_idx   = blockIdx.y;
    const int q_head    = blockIdx.z;
    const int tid       = threadIdx.x;  // 0..HEAD_DIM-1

    // GQA: map query head to KV head
    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int kv_head = q_head / gqa_ratio;

    const int Skv = seq_lens[seq_idx];
    const int kv_start = split_idx * kv_per_split;
    const int kv_end_raw = kv_start + kv_per_split;
    const int kv_end = kv_end_raw < Skv ? kv_end_raw : Skv;

    // Early exit if this split has no work
    if (kv_start >= Skv) {
        // Write sentinel values
        const int total_heads = gridDim.y * num_q_heads;  // batch * num_q_heads
        const int head_linear = seq_idx * num_q_heads + q_head;
        if (tid < HEAD_DIM) {
            partial_O[split_idx * total_heads * HEAD_DIM + head_linear * HEAD_DIM + tid] = 0.0f;
        }
        if (tid == 0) {
            partial_lse[split_idx * total_heads + head_linear] = -FLT_MAX;
        }
        return;
    }

    // Load Q vector into shared memory (all threads participate)
    __shared__ float q_smem[HEAD_DIM];
    const int q_offset = seq_idx * num_q_heads * HEAD_DIM + q_head * HEAD_DIM;
    if (tid < HEAD_DIM) {
        q_smem[tid] = __bfloat162float(Q[q_offset + tid]);
    }
    __syncthreads();

    // Online softmax state — each thread tracks one output dimension
    float rowmax = -FLT_MAX;
    float rowsum = 0.0f;
    float o_val = 0.0f;  // accumulator for output[tid]

    // Shared memory for cross-warp dot product reduction
    const int NUM_WARPS = HEAD_DIM / WARP_SIZE;
    __shared__ float warp_sums[8];  // max 8 warps (HEAD_DIM=256)

    // Iterate over KV positions in this split
    for (int kv_pos = kv_start; kv_pos < kv_end; kv_pos++) {
        // Paged addressing: find physical location
        const int block_idx = block_table[seq_idx * max_blocks_per_seq + kv_pos / block_size];
        const int offset_in_block = kv_pos % block_size;

        // Base pointer into cache: [block_idx, offset_in_block, kv_head, :]
        const int cache_base = ((block_idx * block_size + offset_in_block) * num_kv_heads + kv_head) * HEAD_DIM;

        // Step 1: Compute dot product Q·K for this KV position
        float partial_dot = 0.0f;
        if (tid < HEAD_DIM) {
            float k_val = __bfloat162float(key_cache[cache_base + tid]);
            partial_dot = q_smem[tid] * k_val;
        }

        // Warp-level reduction
        for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
            partial_dot += __shfl_xor_sync(0xffffffff, partial_dot, mask);

        // Cross-warp reduction
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        if (lane_id == 0) warp_sums[warp_id] = partial_dot;
        __syncthreads();

        float score;
        if (tid < NUM_WARPS) {
            score = warp_sums[tid];
            // Reduce across warps (NUM_WARPS is power of 2: 4 or 8)
            for (int mask = NUM_WARPS / 2; mask > 0; mask >>= 1)
                score += __shfl_xor_sync((1u << NUM_WARPS) - 1, score, mask);
            warp_sums[0] = score;
        }
        __syncthreads();
        score = warp_sums[0] * scale;

        // Step 2: Online softmax update
        float new_max = fmaxf(rowmax, score);
        float rescale = __expf(rowmax - new_max);
        float p = __expf(score - new_max);

        rowsum = rowsum * rescale + p;

        // Step 3: Accumulate weighted V
        if (tid < HEAD_DIM) {
            float v_val = __bfloat162float(val_cache[cache_base + tid]);
            o_val = o_val * rescale + p * v_val;
        }
        rowmax = new_max;
    }

    // Normalize partial output by rowsum before writing
    // This makes the reduction kernel work correctly with exp(lse) weights
    if (rowsum > 0.0f) {
        o_val /= rowsum;
    }

    // Write partial results
    const int total_heads = gridDim.y * num_q_heads;
    const int head_linear = seq_idx * num_q_heads + q_head;

    if (tid < HEAD_DIM) {
        partial_O[split_idx * total_heads * HEAD_DIM + head_linear * HEAD_DIM + tid] = o_val;
    }
    if (tid == 0) {
        float lse = (rowsum > 0.0f) ? rowmax + logf(rowsum) : -FLT_MAX;
        partial_lse[split_idx * total_heads + head_linear] = lse;
    }
}

// ==========================================================================
// Phase 2: Reduction kernel — combine split results
// Grid: (batch_size * num_q_heads, 1)
// Block: (HEAD_DIM)
// ==========================================================================
template <int HEAD_DIM>
__global__ void split_kv_reduce_paged(
    const float* __restrict__ partial_O,    // [num_splits, batch*num_q_heads, HEAD_DIM]
    const float* __restrict__ partial_lse,  // [num_splits, batch*num_q_heads]
    __nv_bfloat16* __restrict__ O,          // [batch*num_q_heads, HEAD_DIM]
    int num_splits,
    int total_heads                          // batch * num_q_heads
) {
    const int head_linear = blockIdx.x;     // which (seq, q_head) pair
    const int tid = threadIdx.x;            // 0..HEAD_DIM-1

    if (head_linear >= total_heads || tid >= HEAD_DIM) return;

    // Find global max LSE across splits
    float max_lse = -FLT_MAX;
    for (int s = 0; s < num_splits; s++) {
        float lse = partial_lse[s * total_heads + head_linear];
        max_lse = fmaxf(max_lse, lse);
    }

    // Combine using log-sum-exp rescaling
    float sum_exp = 0.0f;
    float sum_val = 0.0f;

    for (int s = 0; s < num_splits; s++) {
        float lse = partial_lse[s * total_heads + head_linear];
        float w = (lse > -FLT_MAX + 1.0f) ? __expf(lse - max_lse) : 0.0f;
        sum_exp += w;
        sum_val += w * partial_O[s * total_heads * HEAD_DIM + head_linear * HEAD_DIM + tid];
    }

    float result = (sum_exp > 0.0f) ? sum_val / sum_exp : 0.0f;
    O[head_linear * HEAD_DIM + tid] = __float2bfloat16(result);
}

// ==========================================================================
// Host-callable launcher
// ==========================================================================
extern "C" void sm120_flash_decode_paged_launch(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* key_cache,
    const __nv_bfloat16* val_cache,
    const int* block_table,
    const int* seq_lens,
    __nv_bfloat16* O,
    float* partial_O,       // pre-allocated: [max_splits, batch*num_q_heads, HEAD_DIM]
    float* partial_lse,     // pre-allocated: [max_splits, batch*num_q_heads]
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int block_size,
    int max_blocks_per_seq,
    int max_splits,
    cudaStream_t stream
) {
    float scale = 1.0f / sqrtf((float)head_dim);

    // Adaptive split count based on max sequence length
    int num_splits;
    if (max_seq_len <= 256) num_splits = 1;
    else if (max_seq_len <= 1024) num_splits = 4;
    else if (max_seq_len <= 4096) num_splits = 8;
    else if (max_seq_len <= 16384) num_splits = 16;
    else num_splits = 32;

    // Clamp to pre-allocated buffer size
    if (num_splits > max_splits) num_splits = max_splits;

    int kv_per_split = (max_seq_len + num_splits - 1) / num_splits;
    int total_heads = batch_size * num_q_heads;

    // Phase 1: partial attention with paged KV
    dim3 grid1(num_splits, batch_size, num_q_heads);

    if (head_dim == 256) {
        dim3 block1(256);
        split_kv_partial_paged<256><<<grid1, block1, 0, stream>>>(
            Q, key_cache, val_cache, block_table, seq_lens,
            partial_O, partial_lse,
            num_q_heads, num_kv_heads, block_size, max_blocks_per_seq,
            scale, kv_per_split
        );
    } else {
        dim3 block1(128);
        split_kv_partial_paged<128><<<grid1, block1, 0, stream>>>(
            Q, key_cache, val_cache, block_table, seq_lens,
            partial_O, partial_lse,
            num_q_heads, num_kv_heads, block_size, max_blocks_per_seq,
            scale, kv_per_split
        );
    }

    // Phase 2: reduce across splits
    dim3 grid2(total_heads);

    if (head_dim == 256) {
        dim3 block2(256);
        split_kv_reduce_paged<256><<<grid2, block2, 0, stream>>>(
            partial_O, partial_lse, O, num_splits, total_heads
        );
    } else {
        dim3 block2(128);
        split_kv_reduce_paged<128><<<grid2, block2, 0, stream>>>(
            partial_O, partial_lse, O, num_splits, total_heads
        );
    }
}
