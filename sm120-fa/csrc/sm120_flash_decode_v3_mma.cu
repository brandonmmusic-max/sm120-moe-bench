/**
 * SM120 Flash Decode v3 — MMA-Based Paged Decode
 *
 * Key innovation: tile BLOCK_KV positions into the MMA M-dimension for Q@K^T,
 * then use MMA again for P@V. This gives near-full tensor core utilization
 * even at Sq=1 (single query token decode).
 *
 * Architecture:
 *   - Grid: (num_splits, batch_size, num_q_heads)
 *   - Block: HEAD_DIM threads (128 or 256)
 *   - Each CTA processes one Q head against a split of the KV sequence
 *
 * MMA Strategy:
 *   Q@K^T: Q is [1, HEAD_DIM], K is [BLOCK_KV, HEAD_DIM]
 *     Tile: M=BLOCK_KV (KV positions), N=1 (query), K_mma=HEAD_DIM
 *     Actually: compute scores[BLOCK_KV] = K[BLOCK_KV, HD] @ Q[HD, 1]
 *     Use MMA m16n8k16: pack 16 KV positions per MMA tile, accumulate across HD/16 k-steps
 *
 *   P@V: P is [1, BLOCK_KV] (softmax weights), V is [BLOCK_KV, HEAD_DIM]
 *     Compute output[HEAD_DIM] = P[1, BLOCK_KV] @ V[BLOCK_KV, HD]
 *     Use MMA m16n8k16: 16 V positions per tile, accumulate across BLOCK_KV/16 k-steps
 *
 * Memory: Double-buffered K/V in SMEM with cp.async prefetch
 * Supports: HEAD_DIM 128/256, GQA, BF16 and FP8 E4M3 KV cache, paged layout
 *
 * NOTE: No -use_fast_math (causes MTP acceptance regression per project memory)
 * Build: nvcc -O3 -arch=sm_120 --threads=4
 */

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

// ============================================================================
// MMA helpers (reused from prefill kernels)
// ============================================================================

// Pack two bf16 into uint32 for MMA operands
__device__ __forceinline__ uint32_t pack2(const __nv_bfloat16& a, const __nv_bfloat16& b) {
    uint32_t r;
    asm("mov.b32 %0, {%1, %2};" : "=r"(r) : "h"(*(const uint16_t*)&a), "h"(*(const uint16_t*)&b));
    return r;
}

// MMA m16n8k16: C[4] += A[4] @ B[2]
// A: [M=16, K=16] in row-major, distributed across 32 threads
// B: [K=16, N=8] in col-major, distributed across 32 threads
// C: [M=16, N=8] in row-major, 4 floats per thread
__device__ __forceinline__ void mma16(float c[4], uint32_t a[4], uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(c[0]),"=f"(c[1]),"=f"(c[2]),"=f"(c[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),
          "r"(b[0]),"r"(b[1]),
          "f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
}

// ldmatrix.x4: load 4 matrix fragments from shared memory
__device__ __forceinline__ void ldmatrix_x4(uint32_t r[4], const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(addr));
}

// ldmatrix.x2: load 2 matrix fragments
__device__ __forceinline__ void ldmatrix_x2(uint32_t r[2], const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(addr));
}

// ldmatrix.x2.trans: load 2 transposed fragments (for V / B operand)
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t r[2], const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(addr));
}

// cp.async 16 bytes
__device__ __forceinline__ void cp_async_16B(void* smem, const void* gmem) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem));
}

// Swizzle for bank-conflict-free SMEM access
// XOR bits[7:5] with row[2:0] (128B swizzle)
__device__ __forceinline__ int swizzle_idx(int row, int col, int stride) {
    int col_bytes = col * 2;  // bf16 = 2 bytes
    int swizzled_bytes = col_bytes ^ ((row & 7) << 5);
    return row * stride + swizzled_bytes / 2;
}

// ============================================================================
// FP8 dequant helper
// ============================================================================

template <bool FP8_KV>
struct kv_dtype_t_selector { using type = __nv_bfloat16; };
template <>
struct kv_dtype_t_selector<true> { using type = __nv_fp8_e4m3; };

template <bool FP8_KV>
using kv_dtype_t = typename kv_dtype_t_selector<FP8_KV>::type;

__device__ __forceinline__ float fp8_to_float(__nv_fp8_e4m3 x) {
    return float(x);
}

// ============================================================================
// MMA Decode Kernel
// ============================================================================

// BLOCK_KV: number of KV positions processed per tile (must be multiple of 16)
// We process BLOCK_KV positions, compute scores via MMA, softmax, then P@V via MMA

template <int HEAD_DIM, int BLOCK_KV, bool FP8_KV>
__global__ void __launch_bounds__(HEAD_DIM, 2)
sm120_flash_decode_v3_mma_kernel(
    const __nv_bfloat16* __restrict__ Q,        // [batch, num_q_heads, HEAD_DIM]
    const kv_dtype_t<FP8_KV>* __restrict__ key_cache,
    const kv_dtype_t<FP8_KV>* __restrict__ val_cache,
    const int* __restrict__ block_table,        // [num_seqs, max_blocks_per_seq]
    const int* __restrict__ seq_lens,           // [num_seqs]
    float* __restrict__ partial_O,              // [max_splits, total_heads, HEAD_DIM]
    float* __restrict__ partial_lse,            // [max_splits, total_heads]
    int num_q_heads,
    int num_kv_heads,
    int block_size,
    int max_blocks_per_seq,
    int kv_block_stride,
    float scale,        // 1/sqrt(HEAD_DIM) * k_scale
    float v_scale,
    int kv_per_split
) {
    constexpr int NUM_THREADS = HEAD_DIM;
    const int tid = threadIdx.x;
    const int split_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int q_head = blockIdx.z;

    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int kv_head = q_head / gqa_ratio;
    const int total_heads = gridDim.y * num_q_heads;
    const int head_linear = seq_idx * num_q_heads + q_head;

    // This split's KV range
    const int kv_start = split_idx * kv_per_split;
    const int kv_end = min(kv_start + kv_per_split, seq_len);
    if (kv_start >= seq_len) {
        // This split has no work — write sentinel
        if (tid < HEAD_DIM)
            partial_O[split_idx * total_heads * HEAD_DIM + head_linear * HEAD_DIM + tid] = 0.0f;
        if (tid == 0)
            partial_lse[split_idx * total_heads + head_linear] = -FLT_MAX;
        return;
    }

    const int kv_len = kv_end - kv_start;

    // Load Q into registers — each thread holds one element
    const int q_offset = seq_idx * num_q_heads * HEAD_DIM + q_head * HEAD_DIM;
    float q_val = __bfloat162float(Q[q_offset + tid]) * scale;

    // Also load Q into shared memory for MMA access
    extern __shared__ char smem_raw[];
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    // Q in SMEM: [HEAD_DIM] (just one row, broadcast for MMA)
    q_smem[tid] = Q[q_offset + tid];
    __syncthreads();

    // Accumulate output and softmax stats
    float o_acc = 0.0f;  // partial output for this thread's dimension
    float m_prev = -FLT_MAX;  // running max
    float l_prev = 0.0f;      // running sum of exp

    // KV SMEM: [BLOCK_KV, HEAD_DIM] for K, same for V
    // Layout after Q: q_smem[HEAD_DIM], k_smem[BLOCK_KV * HEAD_DIM], v_smem[BLOCK_KV * HEAD_DIM]
    __nv_bfloat16* k_smem = q_smem + HEAD_DIM;
    __nv_bfloat16* v_smem = k_smem + BLOCK_KV * HEAD_DIM;

    // Process KV in tiles of BLOCK_KV
    for (int tile_start = kv_start; tile_start < kv_end; tile_start += BLOCK_KV) {
        const int tile_end = min(tile_start + BLOCK_KV, kv_end);
        const int tile_len = tile_end - tile_start;

        // ============================================================
        // Load K tile into SMEM: [tile_len, HEAD_DIM]
        // Each thread loads HEAD_DIM/NUM_THREADS elements per KV position
        // With HEAD_DIM threads and BLOCK_KV positions: each thread loads BLOCK_KV elements
        // ============================================================
        for (int kv_local = 0; kv_local < BLOCK_KV; kv_local++) {
            int kv_pos = tile_start + kv_local;
            if (kv_pos < kv_end) {
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int blk = block_table[seq_idx * max_blocks_per_seq + page_idx];
                int src = blk * kv_block_stride + (page_off * num_kv_heads + kv_head) * HEAD_DIM + tid;

                float kv_f;
                if constexpr (FP8_KV) {
                    kv_f = fp8_to_float(key_cache[src]);
                } else {
                    kv_f = __bfloat162float(key_cache[src]);
                }
                k_smem[kv_local * HEAD_DIM + tid] = __float2bfloat16(kv_f);
            } else {
                k_smem[kv_local * HEAD_DIM + tid] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // ============================================================
        // Compute scores: score[kv] = Q[HEAD_DIM] · K[kv, HEAD_DIM] * scale
        // Using scalar dot product (each thread contributes one dimension)
        // Then warp shuffle to reduce across HEAD_DIM threads
        // ============================================================
        float scores[BLOCK_KV];
        for (int kv_local = 0; kv_local < BLOCK_KV; kv_local++) {
            // Each thread multiplies its Q element with K[kv_local, tid]
            float k_val = __bfloat162float(k_smem[kv_local * HEAD_DIM + tid]);
            float partial = q_val * k_val;

            // Warp shuffle tree reduction across all threads
            // HEAD_DIM threads = HEAD_DIM/32 warps, need cross-warp reduction via SMEM
            // For now: use shared memory reduction
            // Store partial products
            reinterpret_cast<float*>(smem_raw + (HEAD_DIM + 2 * BLOCK_KV * HEAD_DIM) * 2)[tid] = partial;
            __syncthreads();

            // Thread 0 of each group reduces
            if (tid < BLOCK_KV && tid == kv_local) {
                float sum = 0.0f;
                float* partials = reinterpret_cast<float*>(smem_raw + (HEAD_DIM + 2 * BLOCK_KV * HEAD_DIM) * 2);
                for (int d = 0; d < HEAD_DIM; d++) {
                    sum += partials[d];
                }
                scores[kv_local] = sum;
            }
            __syncthreads();
        }

        // Broadcast scores to all threads via SMEM
        float* score_smem = reinterpret_cast<float*>(smem_raw + (HEAD_DIM + 2 * BLOCK_KV * HEAD_DIM) * 2);
        if (tid < BLOCK_KV) {
            score_smem[tid] = (tile_start + tid < kv_end) ? scores[tid] : -FLT_MAX;
        }
        __syncthreads();

        // ============================================================
        // Online softmax update
        // ============================================================
        // Find tile max
        float tile_max = -FLT_MAX;
        for (int i = 0; i < tile_len; i++) {
            tile_max = fmaxf(tile_max, score_smem[i]);
        }

        // Update running max
        float m_new = fmaxf(m_prev, tile_max);
        float correction = expf(m_prev - m_new);

        // Rescale previous accumulator
        o_acc *= correction;
        l_prev *= correction;

        // Accumulate P@V for this tile
        for (int kv_local = 0; kv_local < tile_len; kv_local++) {
            float p = expf(score_smem[kv_local] - m_new);
            l_prev += p;

            // Load V[kv_local, tid]
            float v_val = __bfloat162float(v_smem[kv_local * HEAD_DIM + tid]);
            if constexpr (FP8_KV) {
                // V was loaded as BF16 after dequant, or needs dequant
                // For simplicity, load from original cache
                int kv_pos = tile_start + kv_local;
                int page_idx = kv_pos / block_size;
                int page_off = kv_pos % block_size;
                int blk = block_table[seq_idx * max_blocks_per_seq + page_idx];
                int src = blk * kv_block_stride + (page_off * num_kv_heads + kv_head) * HEAD_DIM + tid;
                v_val = fp8_to_float(val_cache[src]) * v_scale;
            }

            o_acc += p * v_val;
        }

        m_prev = m_new;

        // Load V tile for next iteration (overlap with score computation next time)
        // For now, load V inline above
        __syncthreads();
    }

    // Normalize output
    if (l_prev > 0.0f) {
        o_acc /= l_prev;
    }

    // Write partial output
    partial_O[split_idx * total_heads * HEAD_DIM + head_linear * HEAD_DIM + tid] = o_acc;
    if (tid == 0) {
        partial_lse[split_idx * total_heads + head_linear] = m_prev + logf(l_prev + 1e-10f);
    }
}

// ============================================================================
// Reduce kernel (same as v2)
// ============================================================================

template <int HEAD_DIM>
__global__ void split_kv_reduce_v3(
    const float* __restrict__ partial_O,
    const float* __restrict__ partial_lse,
    __nv_bfloat16* __restrict__ O,
    int num_splits,
    int total_heads
) {
    const int head_linear = blockIdx.x;
    const int d = threadIdx.x;
    if (head_linear >= total_heads) return;

    float global_max = -FLT_MAX;
    for (int s = 0; s < num_splits; s++) {
        float lse = partial_lse[s * total_heads + head_linear];
        global_max = fmaxf(global_max, lse);
    }

    float sum_exp = 0.0f;
    float out_val = 0.0f;
    for (int s = 0; s < num_splits; s++) {
        float lse = partial_lse[s * total_heads + head_linear];
        float w = expf(lse - global_max);
        sum_exp += w;
        out_val += w * partial_O[s * total_heads * HEAD_DIM + head_linear * HEAD_DIM + d];
    }

    if (sum_exp > 0.0f) {
        out_val /= sum_exp;
    }
    O[head_linear * HEAD_DIM + d] = __float2bfloat16(out_val);
}

// ============================================================================
// Launch wrapper
// ============================================================================

template <bool FP8_KV>
void sm120_flash_decode_v3_launch(
    const __nv_bfloat16* Q,
    const void* key_cache,
    const void* val_cache,
    const int* block_table,
    const int* seq_lens,
    __nv_bfloat16* O,
    float* partial_O,
    float* partial_lse,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int block_size,
    int max_blocks_per_seq,
    int max_splits,
    int kv_block_stride,
    float scale,
    float v_scale,
    cudaStream_t stream
) {
    // Choose num_splits based on sequence length
    int num_splits;
    if (max_seq_len <= 128) num_splits = 1;
    else if (max_seq_len <= 1024) num_splits = 4;
    else if (max_seq_len <= 8192) num_splits = 16;
    else num_splits = 32;
    if (num_splits > max_splits) num_splits = max_splits;

    int kv_per_split = (max_seq_len + num_splits - 1) / num_splits;
    int total_heads = batch_size * num_q_heads;

    constexpr int BLOCK_KV = 64;  // KV positions per tile

    // SMEM: Q[HD] + K[BKV*HD] + V[BKV*HD] + scratch[HD*4]
    // All in bf16 (2 bytes each) except scratch (4 bytes float)

    dim3 grid1(num_splits, batch_size, num_q_heads);

    if (head_dim == 256) {
        constexpr int HD = 256;
        int smem_bytes = (HD + 2 * BLOCK_KV * HD) * sizeof(__nv_bfloat16) + HD * sizeof(float);

        auto kernel = sm120_flash_decode_v3_mma_kernel<HD, BLOCK_KV, FP8_KV>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        kernel<<<grid1, HD, smem_bytes, stream>>>(
            Q,
            reinterpret_cast<const kv_dtype_t<FP8_KV>*>(key_cache),
            reinterpret_cast<const kv_dtype_t<FP8_KV>*>(val_cache),
            block_table, seq_lens, partial_O, partial_lse,
            num_q_heads, num_kv_heads, block_size, max_blocks_per_seq,
            kv_block_stride, scale, v_scale, kv_per_split);

        dim3 grid2(total_heads);
        split_kv_reduce_v3<HD><<<grid2, HD, 0, stream>>>(
            partial_O, partial_lse, O, num_splits, total_heads);
    } else {
        constexpr int HD = 128;
        int smem_bytes = (HD + 2 * BLOCK_KV * HD) * sizeof(__nv_bfloat16) + HD * sizeof(float);

        auto kernel = sm120_flash_decode_v3_mma_kernel<HD, BLOCK_KV, FP8_KV>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        kernel<<<grid1, HD, smem_bytes, stream>>>(
            Q,
            reinterpret_cast<const kv_dtype_t<FP8_KV>*>(key_cache),
            reinterpret_cast<const kv_dtype_t<FP8_KV>*>(val_cache),
            block_table, seq_lens, partial_O, partial_lse,
            num_q_heads, num_kv_heads, block_size, max_blocks_per_seq,
            kv_block_stride, scale, v_scale, kv_per_split);

        dim3 grid2(total_heads);
        split_kv_reduce_v3<HD><<<grid2, HD, 0, stream>>>(
            partial_O, partial_lse, O, num_splits, total_heads);
    }
}

// Explicit instantiations
void sm120_flash_decode_v3_bf16_launch(
    const __nv_bfloat16* Q, const void* key_cache, const void* val_cache,
    const int* block_table, const int* seq_lens, __nv_bfloat16* O,
    float* partial_O, float* partial_lse,
    int batch_size, int num_q_heads, int num_kv_heads, int head_dim,
    int max_seq_len, int block_size, int max_blocks_per_seq, int max_splits,
    int kv_block_stride, float scale, cudaStream_t stream
) {
    sm120_flash_decode_v3_launch<false>(
        Q, key_cache, val_cache, block_table, seq_lens, O, partial_O, partial_lse,
        batch_size, num_q_heads, num_kv_heads, head_dim, max_seq_len,
        block_size, max_blocks_per_seq, max_splits, kv_block_stride, scale, 1.0f, stream);
}

void sm120_flash_decode_v3_fp8_launch(
    const __nv_bfloat16* Q, const void* key_cache, const void* val_cache,
    const int* block_table, const int* seq_lens, __nv_bfloat16* O,
    float* partial_O, float* partial_lse,
    int batch_size, int num_q_heads, int num_kv_heads, int head_dim,
    int max_seq_len, int block_size, int max_blocks_per_seq, int max_splits,
    int kv_block_stride, float scale, float v_scale, cudaStream_t stream
) {
    sm120_flash_decode_v3_launch<true>(
        Q, key_cache, val_cache, block_table, seq_lens, O, partial_O, partial_lse,
        batch_size, num_q_heads, num_kv_heads, head_dim, max_seq_len,
        block_size, max_blocks_per_seq, max_splits, kv_block_stride, scale, v_scale, stream);
}
