/**
 * SM120 Selective Attention v3 — Amortized summaries + adaptive routing
 *
 * Key improvements:
 *   1. Separate build_summaries() entry point for amortization
 *   2. forward_with_cache() uses precomputed summaries
 *   3. Adaptive top_k based on score concentration
 *   4. Fused routing with early-exit when scores are concentrated
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define HEAD_DIM 128

// Forward decl of exact kernel
extern "C" void sm120_flash_attn_forward(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    int batch, int Hq, int Hkv, int Sq, int Skv, int head_dim,
    cudaStream_t stream
);

// ============================================================================
// Build summaries (called once when KV cache grows)
// ============================================================================
__global__ void build_summaries_cached(
    const __nv_bfloat16* __restrict__ K,
    float* __restrict__ summaries,
    float* __restrict__ norms,          // Optional: L2 norm per block for confidence
    int Skv, int block_size, int num_blocks, int max_blocks
) {
    const int blk = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int d = threadIdx.x;

    if (blk >= num_blocks || d >= HEAD_DIM) return;

    int kv_start = blk * block_size;
    int kv_end = min(kv_start + block_size, Skv);
    int count = kv_end - kv_start;
    if (count <= 0) return;

    float sum = 0.0f;
    float sq_sum = 0.0f;
    for (int kv = kv_start; kv < kv_end; kv++) {
        float val = __bfloat162float(K[kv_head * Skv * HEAD_DIM + kv * HEAD_DIM + d]);
        sum += val;
        sq_sum += val * val;
    }

    float mean = sum / (float)count;
    summaries[kv_head * max_blocks * HEAD_DIM + blk * HEAD_DIM + d] = mean;

    // Store norm (only thread 0 per block, reduced via atomicAdd)
    if (norms != nullptr) {
        float norm_contrib = sq_sum;
        atomicAdd(&norms[kv_head * max_blocks + blk], norm_contrib);
    }
}

// ============================================================================
// Route with adaptive top_k
// ============================================================================
__global__ void adaptive_route(
    const __nv_bfloat16* __restrict__ Q,
    const float* __restrict__ summaries,
    int* __restrict__ selected_blocks,
    int* __restrict__ num_selected,
    int Sq, int Hq, int Hkv,
    int num_blocks, int max_blocks,
    int min_k, int max_k, int local_window, int max_selected,
    float scale
) {
    const int q_row = blockIdx.x;
    const int head = blockIdx.y;
    const int kv_head = head / (Hq / Hkv);
    const int tid = threadIdx.x;

    if (q_row >= Sq) return;

    extern __shared__ char dsmem[];
    float* scores = reinterpret_cast<float*>(dsmem);
    float* q_fp32 = scores + num_blocks;

    const __nv_bfloat16* q_ptr = Q + head * Sq * HEAD_DIM + q_row * HEAD_DIM;
    if (tid < HEAD_DIM) q_fp32[tid] = __bfloat162float(q_ptr[tid]);
    __syncthreads();

    // Score
    const float* sum_base = summaries + kv_head * max_blocks * HEAD_DIM;
    for (int blk = tid; blk < num_blocks; blk += blockDim.x) {
        float dot = 0.0f;
        const float* s = sum_base + blk * HEAD_DIM;
        #pragma unroll 8
        for (int d = 0; d < HEAD_DIM; d++)
            dot += q_fp32[d] * s[d];
        scores[blk] = dot * scale;
    }
    __syncthreads();

    if (tid == 0) {
        int out_base = head * Sq * max_selected + q_row * max_selected;
        int count = 0;
        int local_start = max(0, num_blocks - local_window);

        // Local blocks
        for (int i = local_start; i < num_blocks && count < max_selected; i++)
            selected_blocks[out_base + count++] = i;

        // Adaptive k: check score concentration
        // Find max and second-max scores
        float max1 = -FLT_MAX, max2 = -FLT_MAX;
        for (int i = 0; i < local_start; i++) {
            if (scores[i] > max1) { max2 = max1; max1 = scores[i]; }
            else if (scores[i] > max2) max2 = scores[i];
        }

        // Score gap ratio: large gap = concentrated = fewer blocks needed
        float gap_ratio = (max1 - max2) / (fabsf(max1) + 1e-6f);
        int adaptive_k;
        if (gap_ratio > 0.5f) adaptive_k = min_k;       // Very concentrated
        else if (gap_ratio > 0.2f) adaptive_k = min_k * 2;
        else if (gap_ratio > 0.1f) adaptive_k = min_k * 4;
        else adaptive_k = max_k;                          // Uniform → need more

        adaptive_k = min(adaptive_k, max_k);

        // Top-k selection
        for (int k = 0; k < adaptive_k && count < max_selected; k++) {
            float best = -FLT_MAX;
            int best_idx = -1;
            for (int i = 0; i < local_start; i++) {
                if (scores[i] > best) {
                    bool dup = false;
                    for (int j = local_window; j < count; j++)
                        if (selected_blocks[out_base + j] == i) { dup = true; break; }
                    if (!dup) { best = scores[i]; best_idx = i; }
                }
            }
            if (best_idx >= 0) {
                selected_blocks[out_base + count++] = best_idx;
                scores[best_idx] = -FLT_MAX;
            }
        }

        // Sort
        for (int i = 1; i < count; i++) {
            int key = selected_blocks[out_base + i];
            int j = i - 1;
            while (j >= 0 && selected_blocks[out_base + j] > key) {
                selected_blocks[out_base + j + 1] = selected_blocks[out_base + j];
                j--;
            }
            selected_blocks[out_base + j + 1] = key;
        }
        num_selected[head * Sq + q_row] = count;
    }
}

// ============================================================================
// Vectorized gather (same as v2)
// ============================================================================
__global__ void fast_gather_kv_v3(
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ K_sel,
    __nv_bfloat16* __restrict__ V_sel,
    const int* __restrict__ selected_blocks,
    const int* __restrict__ num_selected,
    int Sq, int Skv, int Hq, int Hkv,
    int block_size, int max_selected, int q_row
) {
    const int sel_idx = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int tid = threadIdx.x;

    const int head = kv_head * (Hq / Hkv);
    const int out_base = head * Sq * max_selected + q_row * max_selected;
    int n_sel = num_selected[head * Sq + q_row];
    if (sel_idx >= n_sel) return;

    int src_block = selected_blocks[out_base + sel_idx];
    int src_start = src_block * block_size;
    int dst_start = sel_idx * block_size;
    int sel_stride = max_selected * block_size * HEAD_DIM;

    for (int row = 0; row < block_size; row++) {
        int src_row = src_start + row;
        int dst_row = dst_start + row;
        for (int d = tid * 8; d < HEAD_DIM; d += blockDim.x * 8) {
            if (d + 7 < HEAD_DIM) {
                if (src_row < Skv) {
                    *reinterpret_cast<uint4*>(&K_sel[kv_head * sel_stride + dst_row * HEAD_DIM + d]) =
                        *reinterpret_cast<const uint4*>(&K[kv_head * Skv * HEAD_DIM + src_row * HEAD_DIM + d]);
                    *reinterpret_cast<uint4*>(&V_sel[kv_head * sel_stride + dst_row * HEAD_DIM + d]) =
                        *reinterpret_cast<const uint4*>(&V[kv_head * Skv * HEAD_DIM + src_row * HEAD_DIM + d]);
                } else {
                    *reinterpret_cast<uint4*>(&K_sel[kv_head * sel_stride + dst_row * HEAD_DIM + d]) = make_uint4(0,0,0,0);
                    *reinterpret_cast<uint4*>(&V_sel[kv_head * sel_stride + dst_row * HEAD_DIM + d]) = make_uint4(0,0,0,0);
                }
            }
        }
    }
}

// ============================================================================
// Public API: build_summaries (separate, for amortization)
// ============================================================================
extern "C" void sm120_build_summaries(
    const __nv_bfloat16* K, float* summaries, float* norms,
    int batch_kv_heads, int Skv, int block_size,
    cudaStream_t stream
) {
    int num_blocks = (Skv + block_size - 1) / block_size;
    if (norms) cudaMemsetAsync(norms, 0, batch_kv_heads * num_blocks * sizeof(float), stream);
    build_summaries_cached<<<dim3(num_blocks, batch_kv_heads), HEAD_DIM, 0, stream>>>(
        K, summaries, norms, Skv, block_size, num_blocks, num_blocks);
}

// ============================================================================
// Public API: forward with precomputed summaries
// ============================================================================
extern "C" void sm120_selective_with_cache(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const float* summaries,  // precomputed by sm120_build_summaries
    __nv_bfloat16* O, float* L,
    int batch, int Hq, int Hkv, int Sq, int Skv, int head_dim,
    int block_size, int min_k, int max_k, int local_window, int max_selected,
    cudaStream_t stream
) {
    float scale = 1.0f / sqrtf((float)head_dim);
    int total_heads = batch * Hq;
    int total_kv_heads = batch * Hkv;
    int num_blocks = (Skv + block_size - 1) / block_size;

    if (num_blocks <= max_k + local_window) {
        sm120_flash_attn_forward(Q, K, V, O, L, batch, Hq, Hkv, Sq, Skv, head_dim, stream);
        return;
    }

    // Workspace (no summaries — they're precomputed!)
    size_t sel_bytes = total_heads * Sq * max_selected * sizeof(int);
    size_t nsel_bytes = total_heads * Sq * sizeof(int);
    int sel_kv_len = max_selected * block_size;
    size_t kv_bytes = total_kv_heads * sel_kv_len * head_dim * sizeof(__nv_bfloat16);

    char* workspace;
    cudaMalloc(&workspace, sel_bytes + nsel_bytes + 2 * kv_bytes);

    int* d_selected = reinterpret_cast<int*>(workspace);
    int* d_num_sel = reinterpret_cast<int*>(workspace + sel_bytes);
    __nv_bfloat16* d_K_sel = reinterpret_cast<__nv_bfloat16*>(workspace + sel_bytes + nsel_bytes);
    __nv_bfloat16* d_V_sel = d_K_sel + total_kv_heads * sel_kv_len * head_dim;

    // Route (using precomputed summaries — no summary rebuild!)
    int route_smem = (num_blocks + HEAD_DIM) * sizeof(float);
    adaptive_route<<<dim3(Sq, total_heads), 128, route_smem, stream>>>(
        Q, summaries, d_selected, d_num_sel,
        Sq, Hq, Hkv, num_blocks, num_blocks,
        min_k, max_k, local_window, max_selected, scale);

    // Gather
    fast_gather_kv_v3<<<dim3(max_selected, total_kv_heads), 128, 0, stream>>>(
        K, V, d_K_sel, d_V_sel, d_selected, d_num_sel,
        Sq, Skv, Hq, Hkv, block_size, max_selected, 0);

    // Exact attention
    sm120_flash_attn_forward(Q, d_K_sel, d_V_sel, O, L,
                              batch, Hq, Hkv, Sq, sel_kv_len, head_dim, stream);

    cudaFree(workspace);
}

// ============================================================================
// Original API (builds summaries each call — for compatibility)
// ============================================================================
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

    size_t sum_bytes = total_kv_heads * num_blocks * head_dim * sizeof(float);
    size_t sel_bytes = total_heads * Sq * max_selected * sizeof(int);
    size_t nsel_bytes = total_heads * Sq * sizeof(int);
    int sel_kv_len = max_selected * block_size;
    size_t kv_bytes = total_kv_heads * sel_kv_len * head_dim * sizeof(__nv_bfloat16);

    char* workspace;
    cudaMalloc(&workspace, sum_bytes + sel_bytes + nsel_bytes + 2 * kv_bytes);

    float* d_summaries = reinterpret_cast<float*>(workspace);
    int* d_selected = reinterpret_cast<int*>(workspace + sum_bytes);
    int* d_num_sel = reinterpret_cast<int*>(workspace + sum_bytes + sel_bytes);
    __nv_bfloat16* d_K_sel = reinterpret_cast<__nv_bfloat16*>(workspace + sum_bytes + sel_bytes + nsel_bytes);
    __nv_bfloat16* d_V_sel = d_K_sel + total_kv_heads * sel_kv_len * head_dim;

    build_summaries_cached<<<dim3(num_blocks, total_kv_heads), HEAD_DIM, 0, stream>>>(
        K, d_summaries, nullptr, Skv, block_size, num_blocks, num_blocks);

    int route_smem = (num_blocks + HEAD_DIM) * sizeof(float);
    adaptive_route<<<dim3(Sq, total_heads), 128, route_smem, stream>>>(
        Q, d_summaries, d_selected, d_num_sel,
        Sq, Hq, Hkv, num_blocks, num_blocks,
        top_k, top_k, local_window, max_selected, scale);

    fast_gather_kv_v3<<<dim3(max_selected, total_kv_heads), 128, 0, stream>>>(
        K, V, d_K_sel, d_V_sel, d_selected, d_num_sel,
        Sq, Skv, Hq, Hkv, block_size, max_selected, 0);

    sm120_flash_attn_forward(Q, d_K_sel, d_V_sel, O, L,
                              batch, Hq, Hkv, Sq, sel_kv_len, head_dim, stream);

    cudaFree(workspace);
}
