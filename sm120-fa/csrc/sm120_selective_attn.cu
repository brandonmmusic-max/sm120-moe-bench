/**
 * SM120 Selective Attention — CUDA-side routing + exact attention
 *
 * Phase 1 kernel: build block summaries + score + select blocks
 * Phase 2: gather selected K/V and run exact attention
 *
 * All routing happens on GPU — no Python overhead.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define HEAD_DIM 128
#define WARP_SIZE 32

// ============================================================================
// Kernel 1: Build block summaries + score against Q + select top-k
//
// Grid: (num_q_rows, batch * num_heads)
// Each block processes one Q row, scores it against all KV block summaries,
// and outputs the selected block indices.
//
// For decode (Q=1), this is one block per head — very parallel.
// ============================================================================

__global__ void selective_route(
    const __nv_bfloat16* __restrict__ Q,        // [B*Hq, Sq, D]
    const __nv_bfloat16* __restrict__ K,        // [B*Hkv, Skv, D]
    int* __restrict__ selected_blocks,           // [B*Hq, Sq, max_selected]
    int* __restrict__ num_selected,              // [B*Hq, Sq]
    int Sq, int Skv, int Hq, int Hkv,
    int block_size, int top_k, int local_window,
    int max_selected, float scale
) {
    const int q_row = blockIdx.x;
    const int head = blockIdx.y;
    const int kv_head = head / (Hq / Hkv);
    const int tid = threadIdx.x;  // 0..127

    if (q_row >= Sq) return;

    const int num_blocks = (Skv + block_size - 1) / block_size;
    const __nv_bfloat16* q_ptr = Q + head * Sq * HEAD_DIM + q_row * HEAD_DIM;
    const __nv_bfloat16* k_ptr = K + kv_head * Skv * HEAD_DIM;

    // ========================================================================
    // Step 1: Compute coarse scores — Q dot mean(K_block) for each block
    // Each thread handles a subset of blocks
    // ========================================================================

    // Shared memory for Q row and block scores
    __shared__ float q_fp32[HEAD_DIM];
    __shared__ float block_scores[2048];  // Up to 2048 blocks (128K tokens at bs=64)

    // Load Q to shared memory in FP32
    if (tid < HEAD_DIM) {
        q_fp32[tid] = __bfloat162float(q_ptr[tid]);
    }
    __syncthreads();

    // Each thread computes scores for multiple blocks
    for (int blk = tid; blk < num_blocks; blk += blockDim.x) {
        int kv_start = blk * block_size;
        int kv_end = min(kv_start + block_size, Skv);
        int count = kv_end - kv_start;

        // Compute mean(K_block) dot Q — incremental dot product
        float score = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++) {
            // Compute mean K[d] for this block
            float k_sum = 0.0f;
            for (int kv = kv_start; kv < kv_end; kv++) {
                k_sum += __bfloat162float(k_ptr[kv * HEAD_DIM + d]);
            }
            float k_mean = k_sum / (float)count;
            score += q_fp32[d] * k_mean;
        }
        block_scores[blk] = score * scale;
    }
    __syncthreads();

    // ========================================================================
    // Step 2: Select blocks — local window + top-k by score
    // Only thread 0 does the selection (sequential but fast for <1024 blocks)
    // ========================================================================
    if (tid == 0) {
        int out_idx = head * Sq * max_selected + q_row * max_selected;
        int count = 0;

        // Mark local window blocks (most recent)
        int local_start = max(0, num_blocks - local_window);
        bool is_local[2048];
        for (int i = 0; i < num_blocks; i++)
            is_local[i] = (i >= local_start);

        // Add local blocks first
        for (int i = local_start; i < num_blocks && count < max_selected; i++) {
            selected_blocks[out_idx + count] = i;
            count++;
        }

        // Find top-k non-local blocks by score
        // Simple selection sort (good enough for <1024 blocks)
        for (int k = 0; k < top_k && count < max_selected; k++) {
            float best_score = -FLT_MAX;
            int best_idx = -1;
            for (int i = 0; i < num_blocks; i++) {
                if (!is_local[i] && block_scores[i] > best_score) {
                    // Check if already selected
                    bool already = false;
                    for (int j = local_window; j < count; j++) {
                        if (selected_blocks[out_idx + j] == i) {
                            already = true;
                            break;
                        }
                    }
                    if (!already) {
                        best_score = block_scores[i];
                        best_idx = i;
                    }
                }
            }
            if (best_idx >= 0) {
                selected_blocks[out_idx + count] = best_idx;
                count++;
            }
        }

        // Sort selected blocks for sequential access
        // Simple insertion sort
        for (int i = 1; i < count; i++) {
            int key = selected_blocks[out_idx + i];
            int j = i - 1;
            while (j >= 0 && selected_blocks[out_idx + j] > key) {
                selected_blocks[out_idx + j + 1] = selected_blocks[out_idx + j];
                j--;
            }
            selected_blocks[out_idx + j + 1] = key;
        }

        num_selected[head * Sq + q_row] = count;
    }
}


// ============================================================================
// Kernel 2: Gather selected K/V blocks into contiguous buffer
//
// Grid: (num_selected_total, batch * num_kv_heads)
// Each block copies one selected KV block
// ============================================================================

__global__ void gather_selected_kv(
    const __nv_bfloat16* __restrict__ K,    // [B*Hkv, Skv, D]
    const __nv_bfloat16* __restrict__ V,    // [B*Hkv, Skv, D]
    __nv_bfloat16* __restrict__ K_sel,      // [B*Hkv, max_sel*block_size, D]
    __nv_bfloat16* __restrict__ V_sel,      // [B*Hkv, max_sel*block_size, D]
    const int* __restrict__ selected_blocks, // [B*Hq, Sq, max_selected]
    const int* __restrict__ num_selected,    // [B*Hq, Sq]
    int Sq, int Skv, int Hq, int Hkv,
    int block_size, int max_selected,
    int q_row_for_selection  // Which Q row's selection to use (0 for decode)
) {
    const int sel_idx = blockIdx.x;   // Which selected block to copy
    const int kv_head = blockIdx.y;   // KV head index
    const int tid = threadIdx.x;      // 0..127

    // Use first Q head's selection (all Q heads share same KV for same KV head)
    const int head = kv_head * (Hq / Hkv);  // Map to first Q head for this KV head
    const int out_base = head * Sq * max_selected + q_row_for_selection * max_selected;

    int n_sel = num_selected[head * Sq + q_row_for_selection];
    if (sel_idx >= n_sel) return;

    int src_block = selected_blocks[out_base + sel_idx];
    int src_start = src_block * block_size;
    int dst_start = sel_idx * block_size;

    // Copy block_size rows of K and V (HEAD_DIM=128 bf16 per row)
    for (int row = 0; row < block_size; row++) {
        int src_row = src_start + row;
        int dst_row = dst_start + row;
        if (src_row < Skv) {
            for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
                K_sel[kv_head * max_selected * block_size * HEAD_DIM + dst_row * HEAD_DIM + d] =
                    K[kv_head * Skv * HEAD_DIM + src_row * HEAD_DIM + d];
                V_sel[kv_head * max_selected * block_size * HEAD_DIM + dst_row * HEAD_DIM + d] =
                    V[kv_head * Skv * HEAD_DIM + src_row * HEAD_DIM + d];
            }
        } else {
            for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
                K_sel[kv_head * max_selected * block_size * HEAD_DIM + dst_row * HEAD_DIM + d] =
                    __float2bfloat16(0.0f);
                V_sel[kv_head * max_selected * block_size * HEAD_DIM + dst_row * HEAD_DIM + d] =
                    __float2bfloat16(0.0f);
            }
        }
    }
}


// ============================================================================
// Host interface: selective attention forward
// ============================================================================

// Forward declaration of exact attention kernel
extern "C" void sm120_flash_attn_forward(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    int batch, int Hq, int Hkv, int Sq, int Skv, int head_dim,
    cudaStream_t stream
);

// ============================================================================
// Kernel 3: Fast summary scoring — precomputed summaries
// Summaries are computed ONCE and cached. Routing just does Q dot summary.
// Grid: (num_q_rows, batch * num_heads)
// ============================================================================
__global__ void build_block_summaries(
    const __nv_bfloat16* __restrict__ K,     // [B*Hkv, Skv, D]
    float* __restrict__ summaries,            // [B*Hkv, num_blocks, D]
    int Skv, int block_size, int num_blocks
) {
    const int blk = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int tid = threadIdx.x;

    if (blk >= num_blocks || tid >= HEAD_DIM) return;

    int kv_start = blk * block_size;
    int kv_end = min(kv_start + block_size, Skv);
    int count = kv_end - kv_start;

    float sum = 0.0f;
    for (int kv = kv_start; kv < kv_end; kv++) {
        sum += __bfloat162float(K[kv_head * Skv * HEAD_DIM + kv * HEAD_DIM + tid]);
    }
    summaries[kv_head * num_blocks * HEAD_DIM + blk * HEAD_DIM + tid] = sum / (float)count;
}

__global__ void fast_route_from_summaries(
    const __nv_bfloat16* __restrict__ Q,     // [B*Hq, Sq, D]
    const float* __restrict__ summaries,      // [B*Hkv, num_blocks, D]
    int* __restrict__ selected_blocks,         // [B*Hq, Sq, max_selected]
    int* __restrict__ num_selected,            // [B*Hq, Sq]
    int Sq, int Hq, int Hkv,
    int num_blocks, int top_k, int local_window, int max_selected,
    float scale
) {
    const int q_row = blockIdx.x;
    const int head = blockIdx.y;
    const int kv_head = head / (Hq / Hkv);
    const int tid = threadIdx.x;

    if (q_row >= Sq) return;

    __shared__ float q_fp32[HEAD_DIM];
    __shared__ float block_scores[1024];

    const __nv_bfloat16* q_ptr = Q + head * Sq * HEAD_DIM + q_row * HEAD_DIM;
    if (tid < HEAD_DIM) q_fp32[tid] = __bfloat162float(q_ptr[tid]);
    __syncthreads();

    // Score Q against precomputed summaries — just a dot product per block
    for (int blk = tid; blk < num_blocks; blk += blockDim.x) {
        float score = 0.0f;
        const float* sum_ptr = summaries + kv_head * num_blocks * HEAD_DIM + blk * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d++) {
            score += q_fp32[d] * sum_ptr[d];
        }
        block_scores[blk] = score * scale;
    }
    __syncthreads();

    // Selection (same as before, thread 0 only)
    if (tid == 0) {
        int out_idx = head * Sq * max_selected + q_row * max_selected;
        int count = 0;

        int local_start = max(0, num_blocks - local_window);

        // Local blocks
        for (int i = local_start; i < num_blocks && count < max_selected; i++) {
            selected_blocks[out_idx + count] = i;
            count++;
        }

        // Top-k
        for (int k = 0; k < top_k && count < max_selected; k++) {
            float best = -FLT_MAX;
            int best_idx = -1;
            for (int i = 0; i < local_start; i++) {
                bool already = false;
                for (int j = local_window; j < count; j++)
                    if (selected_blocks[out_idx + j] == i) { already = true; break; }
                if (!already && block_scores[i] > best) {
                    best = block_scores[i];
                    best_idx = i;
                }
            }
            if (best_idx >= 0) {
                selected_blocks[out_idx + count] = best_idx;
                count++;
            }
        }

        // Sort
        for (int i = 1; i < count; i++) {
            int key = selected_blocks[out_idx + i];
            int j = i - 1;
            while (j >= 0 && selected_blocks[out_idx + j] > key) {
                selected_blocks[out_idx + j + 1] = selected_blocks[out_idx + j];
                j--;
            }
            selected_blocks[out_idx + j + 1] = key;
        }
        num_selected[head * Sq + q_row] = count;
    }
}

extern "C" void sm120_selective_attn_forward(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    int batch, int Hq, int Hkv, int Sq, int Skv, int head_dim,
    int block_size, int top_k, int local_window, int max_selected,
    cudaStream_t stream
) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int total_heads = batch * Hq;
    int total_kv_heads = batch * Hkv;
    int num_blocks = (Skv + block_size - 1) / block_size;

    if (num_blocks <= top_k + local_window) {
        sm120_flash_attn_forward(Q, K, V, O, L, batch, Hq, Hkv, Sq, Skv, head_dim, stream);
        return;
    }

    // Pre-allocate all buffers in one cudaMalloc
    size_t summary_bytes = total_kv_heads * num_blocks * head_dim * sizeof(float);
    size_t sel_bytes = total_heads * Sq * max_selected * sizeof(int);
    size_t nsel_bytes = total_heads * Sq * sizeof(int);
    int sel_kv_len = max_selected * block_size;
    size_t kv_sel_bytes = total_kv_heads * sel_kv_len * head_dim * sizeof(__nv_bfloat16);

    char* d_workspace;
    cudaMalloc(&d_workspace, summary_bytes + sel_bytes + nsel_bytes + 2 * kv_sel_bytes);

    float* d_summaries = reinterpret_cast<float*>(d_workspace);
    int* d_selected = reinterpret_cast<int*>(d_workspace + summary_bytes);
    int* d_num_sel = reinterpret_cast<int*>(d_workspace + summary_bytes + sel_bytes);
    __nv_bfloat16* d_K_sel = reinterpret_cast<__nv_bfloat16*>(d_workspace + summary_bytes + sel_bytes + nsel_bytes);
    __nv_bfloat16* d_V_sel = d_K_sel + total_kv_heads * sel_kv_len * head_dim;

    // Phase 1: Build summaries (could be cached across calls)
    build_block_summaries<<<dim3(num_blocks, total_kv_heads), HEAD_DIM, 0, stream>>>(
        K, d_summaries, Skv, block_size, num_blocks);

    // Phase 2: Fast routing from precomputed summaries
    fast_route_from_summaries<<<dim3(Sq, total_heads), 128, 0, stream>>>(
        Q, d_summaries, d_selected, d_num_sel,
        Sq, Hq, Hkv, num_blocks, top_k, local_window, max_selected, scale);

    // Phase 3: Gather
    gather_selected_kv<<<dim3(max_selected, total_kv_heads), 128, 0, stream>>>(
        K, V, d_K_sel, d_V_sel, d_selected, d_num_sel,
        Sq, Skv, Hq, Hkv, block_size, max_selected, 0);

    // Phase 4: Exact attention on selected blocks
    sm120_flash_attn_forward(Q, d_K_sel, d_V_sel, O, L,
                              batch, Hq, Hkv, Sq, sel_kv_len, head_dim, stream);

    cudaFree(d_workspace);
}
