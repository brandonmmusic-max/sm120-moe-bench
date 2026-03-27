/**
 * SM120 Flash Decode v2 -- Tiled vectorized paged KV cache decode
 *
 * Key optimizations over v1 (scalar):
 *   1. Processes BLOCK_KV positions per tile (not 1 at a time)
 *   2. cp.async 16B vectorized loads from paged cache to SMEM
 *   3. Only 3 __syncthreads per tile (vs 2 per KV position in v1)
 *   4. Thread 0 softmax overlapped with V tile cp.async loading
 *   5. Fully unrollable P@V inner loop (BLOCK_KV is compile-time)
 *   6. CUDA graph compatible (pre-allocated workspace)
 *
 * Supports: HEAD_DIM 128/256, GQA, variable seqlen, BF16 and FP8 E4M3 KV cache.
 * KV cache layout (vLLM): [num_blocks, block_size, num_kv_heads, head_dim]
 * Supports interleaved K/V via kv_block_stride parameter.
 *
 * FP8 KV cache support:
 *   - Template parameter FP8_KV selects BF16 (2B) or FP8 E4M3 (1B) KV dtype
 *   - FP8 halves HBM bandwidth for KV loads (1B vs 2B per element)
 *   - Dequantization: FP8 → float on SMEM read (negligible ALU cost)
 *   - k_scale absorbed into QK softmax scale (one multiply, not N)
 *   - v_scale applied at output normalization (one multiply per dim)
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define WARP_SIZE 32

// ---- PTX math intrinsics (match FlashInfer's math.cuh) ----
__device__ __forceinline__ float ptx_exp2(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float ptx_log2(float x) {
    float y;
    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float ptx_rcp(float x) {
    float y;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Match FlashInfer's sentinel value (safe for ex2.approx input range)
constexpr float NEG_INF = -5e4f;

// ---- Type helpers ----
template <bool FP8_KV>
struct KVType {
    using type = __nv_bfloat16;
};
template <>
struct KVType<true> {
    using type = __nv_fp8_e4m3;
};

template <bool FP8_KV>
using kv_dtype_t = typename KVType<FP8_KV>::type;

// Convert KV element to float
__device__ __forceinline__ float kv_to_float(__nv_bfloat16 val) {
    return __bfloat162float(val);
}
__device__ __forceinline__ float kv_to_float(__nv_fp8_e4m3 val) {
    return float(val);
}

// ---- cp.async intrinsics ----
__device__ __forceinline__ void cp_async_16B(void* smem_dst, const void* gmem_src) {
    uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(gmem_src));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n");
}

// ============================================================================
// Paged tile loader: gathers BLOCK_KV rows from paged cache into SMEM
// Each row is HEAD_DIM elements of kv_dtype_t; rows may span different pages.
// Uses cp.async 16B vectorized transfer.
//   BF16: 8 elements per 16B load, FP8: 16 elements per 16B load
// ============================================================================
template <int HEAD_DIM, int BLOCK_KV, bool FP8_KV>
__device__ __forceinline__ void load_paged_tile(
    kv_dtype_t<FP8_KV>* __restrict__ kv_smem,           // [BLOCK_KV, HEAD_DIM] destination
    const kv_dtype_t<FP8_KV>* __restrict__ cache,       // global KV cache
    const int* __restrict__ block_table,
    int seq_idx,
    int tile_start,
    int tile_end,
    int kv_head,
    int block_size,
    int num_kv_heads,
    int max_blocks_per_seq,
    int kv_block_stride,   // elements per block (contiguous: block_size*kv_heads*head_dim; interleaved: 2x)
    int tid,
    int num_threads
) {
    constexpr int ELEM_BYTES = sizeof(kv_dtype_t<FP8_KV>);  // 1 for FP8, 2 for BF16
    constexpr int CHUNK_ELEMS = 16 / ELEM_BYTES;             // 16 for FP8, 8 for BF16
    constexpr int TOTAL_CHUNKS = BLOCK_KV * HEAD_DIM / CHUNK_ELEMS;

    #pragma unroll 4
    for (int i = tid; i < TOTAL_CHUNKS; i += num_threads) {
        const int kv_local = (i * CHUNK_ELEMS) / HEAD_DIM;
        const int d = (i * CHUNK_ELEMS) % HEAD_DIM;
        const int kv_pos = tile_start + kv_local;

        if (kv_pos < tile_end) {
            const int page_idx = kv_pos / block_size;
            const int page_off = kv_pos % block_size;
            const int blk = block_table[seq_idx * max_blocks_per_seq + page_idx];
            // HND layout: [num_blocks, num_kv_heads, block_size, head_dim]
            const int src = blk * kv_block_stride + (kv_head * block_size + page_off) * HEAD_DIM + d;
            cp_async_16B(&kv_smem[kv_local * HEAD_DIM + d], &cache[src]);
        } else {
            // Zero-fill out-of-bounds positions (16 bytes regardless of dtype)
            uint4* dst = reinterpret_cast<uint4*>(&kv_smem[kv_local * HEAD_DIM + d]);
            *dst = make_uint4(0, 0, 0, 0);
        }
    }
    cp_async_commit();
}


// ============================================================================
// Phase 1: Tiled partial attention with paged KV cache
//
// Grid:  (num_splits, batch_size, num_q_heads)
// Block: HEAD_DIM threads
//
// Template: FP8_KV=false → BF16 KV, FP8_KV=true → FP8 E4M3 KV
// When FP8_KV: scale includes k_scale factor, v_scale applied at output.
// ============================================================================
template <int HEAD_DIM, int BLOCK_KV, bool FP8_KV>
__global__ void __launch_bounds__(HEAD_DIM, 2)
tiled_split_kv_partial_paged(
    const __nv_bfloat16* __restrict__ Q,
    const kv_dtype_t<FP8_KV>* __restrict__ key_cache,
    const kv_dtype_t<FP8_KV>* __restrict__ val_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    float* __restrict__ partial_O,
    float* __restrict__ partial_lse,
    int num_q_heads,
    int num_kv_heads,
    int block_size,
    int max_blocks_per_seq,
    int kv_block_stride,  // elements per block in cache (supports interleaved K/V)
    float scale,          // = 1/sqrt(d) * k_scale when FP8_KV
    float v_scale,        // applied at output (1.0 for BF16)
    int kv_per_split
) {
    const int split_idx = blockIdx.x;
    const int seq_idx   = blockIdx.y;
    const int q_head    = blockIdx.z;
    const int tid       = threadIdx.x;
    constexpr int NUM_THREADS = HEAD_DIM;

    // GQA mapping
    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int kv_head = q_head / gqa_ratio;

    // KV range for this split
    const int Skv = seq_lens[seq_idx];
    const int kv_start = split_idx * kv_per_split;
    const int kv_end = (kv_start + kv_per_split < Skv) ? kv_start + kv_per_split : Skv;

    // Output addressing
    const int total_heads = gridDim.y * num_q_heads;
    const int head_linear = seq_idx * num_q_heads + q_head;

    // Early exit for empty splits
    if (kv_start >= Skv) {
        if (tid < HEAD_DIM)
            partial_O[split_idx * total_heads * HEAD_DIM + head_linear * HEAD_DIM + tid] = 0.0f;
        if (tid == 0)
            partial_lse[split_idx * total_heads + head_linear] = NEG_INF;
        return;
    }

    // ---- Shared memory layout ----
    // q_smem:  [HEAD_DIM] float                            -- Q vector (FP32)
    // kv_smem: [BLOCK_KV * HEAD_DIM] kv_dtype              -- K or V tile (reused)
    // p_smem:  [BLOCK_KV + 4] float                        -- scores + softmax metadata
    constexpr int KV_SMEM_BYTES = BLOCK_KV * HEAD_DIM * sizeof(kv_dtype_t<FP8_KV>);

    extern __shared__ char smem_raw[];
    float* q_smem = reinterpret_cast<float*>(smem_raw);
    kv_dtype_t<FP8_KV>* kv_smem = reinterpret_cast<kv_dtype_t<FP8_KV>*>(
                              smem_raw + HEAD_DIM * sizeof(float));
    float* p_smem = reinterpret_cast<float*>(
                              smem_raw + HEAD_DIM * sizeof(float) + KV_SMEM_BYTES);

    // Load Q into shared memory (FP32 for dot product precision)
    {
        const int q_offset = seq_idx * num_q_heads * HEAD_DIM + q_head * HEAD_DIM;
        if (tid < HEAD_DIM)
            q_smem[tid] = __bfloat162float(Q[q_offset + tid]);
    }
    __syncthreads();

    // ---- Thread assignment for Q@K^T ----
    constexpr int R = HEAD_DIM / BLOCK_KV;
    static_assert(R >= 1 && (R & (R - 1)) == 0, "R must be power of 2");
    constexpr int D_PER_THREAD = HEAD_DIM / R;

    const int score_idx   = tid / R;
    const int sub_tid     = tid % R;
    const int d_start_qk  = sub_tid * D_PER_THREAD;

    // Online softmax state (match FlashInfer: m=-5e4, d=1.0)
    float rowmax = NEG_INF;
    float rowsum = 1.0f;
    float o_val  = 0.0f;

    // Tile loop
    const int total_kv  = kv_end - kv_start;
    const int num_tiles = (total_kv + BLOCK_KV - 1) / BLOCK_KV;

    for (int tile = 0; tile < num_tiles; tile++) {
        const int tile_start = kv_start + tile * BLOCK_KV;
        const int tile_end   = (tile_start + BLOCK_KV < kv_end)
                               ? tile_start + BLOCK_KV : kv_end;
        const int tile_len   = tile_end - tile_start;

        // ==============================================================
        // Step 1: Load K tile from paged cache
        // ==============================================================
        load_paged_tile<HEAD_DIM, BLOCK_KV, FP8_KV>(
            kv_smem, key_cache, block_table, seq_idx,
            tile_start, tile_end, kv_head,
            block_size, num_kv_heads, max_blocks_per_seq, kv_block_stride,
            tid, NUM_THREADS);
        cp_async_wait_all();
        __syncthreads();

        // ==============================================================
        // Step 2: Q @ K^T -- compute BLOCK_KV scores
        // ==============================================================
        float dot = 0.0f;
        if (score_idx < tile_len) {
            const kv_dtype_t<FP8_KV>* k_row = &kv_smem[score_idx * HEAD_DIM + d_start_qk];
            const float* q_row = &q_smem[d_start_qk];

            #pragma unroll 16
            for (int d = 0; d < D_PER_THREAD; d++) {
                dot += q_row[d] * kv_to_float(k_row[d]);
            }

            // Reduce within R-thread group via shuffle
            if constexpr (R >= 2) dot += __shfl_xor_sync(0xffffffff, dot, 1);
            if constexpr (R >= 4) dot += __shfl_xor_sync(0xffffffff, dot, 2);
            if constexpr (R >= 8) dot += __shfl_xor_sync(0xffffffff, dot, 4);
        }

        // scale is already in log2 domain: (1/sqrt(d) * k_scale) * log2(e)
        float score = dot * scale;
        if (score_idx >= tile_len) score = NEG_INF;

        if (sub_tid == 0 && score_idx < BLOCK_KV)
            p_smem[score_idx] = score;
        __syncthreads();

        // ==============================================================
        // Step 3: Start V tile load (overlapped with softmax)
        // ==============================================================
        load_paged_tile<HEAD_DIM, BLOCK_KV, FP8_KV>(
            kv_smem, val_cache, block_table, seq_idx,
            tile_start, tile_end, kv_head,
            block_size, num_kv_heads, max_blocks_per_seq, kv_block_stride,
            tid, NUM_THREADS);

        // ==============================================================
        // Step 4: Online softmax (thread 0, overlapped with V load)
        //         All values in log-base-2 domain; use ptx_exp2
        // ==============================================================
        if (tid == 0) {
            float tile_max = NEG_INF;
            for (int n = 0; n < tile_len; n++)
                tile_max = fmaxf(tile_max, p_smem[n]);

            float new_max = fmaxf(rowmax, tile_max);

            float tile_sum = 0.0f;
            for (int n = 0; n < tile_len; n++) {
                float p = ptx_exp2(p_smem[n] - new_max);
                p_smem[n] = p;
                tile_sum += p;
            }
            for (int n = tile_len; n < BLOCK_KV; n++)
                p_smem[n] = 0.0f;

            p_smem[BLOCK_KV]     = new_max;
            p_smem[BLOCK_KV + 1] = tile_sum;
        }

        cp_async_wait_all();
        __syncthreads();

        // ==============================================================
        // Step 5: Rescale existing accumulators + P@V
        // ==============================================================
        {
            const float tile_new_max = p_smem[BLOCK_KV];
            const float tile_sum     = p_smem[BLOCK_KV + 1];
            const float rescale      = ptx_exp2(rowmax - tile_new_max);

            o_val  *= rescale;
            rowsum  = rowsum * rescale + tile_sum;
            rowmax  = tile_new_max;
        }

        if (tid < HEAD_DIM) {
            const int d = tid;
            float pv_sum = 0.0f;

            #pragma unroll 16
            for (int n = 0; n < BLOCK_KV; n++) {
                pv_sum += p_smem[n] * kv_to_float(kv_smem[n * HEAD_DIM + d]);
            }
            o_val += pv_sum;
        }
        __syncthreads();
    }  // end tile loop

    // ==============================================================
    // Normalize and write partial results
    // Match FlashInfer: rcp.approx for division, base-2 LSE
    // v_scale is NOT applied here — applied post-kernel in Python (matches FlashInfer)
    // ==============================================================
    {
        float d_rcp = (rowmax != NEG_INF) ? ptx_rcp(rowsum) : 0.f;
        o_val *= d_rcp;
    }

    if (tid < HEAD_DIM)
        partial_O[split_idx * total_heads * HEAD_DIM + head_linear * HEAD_DIM + tid] = o_val;

    if (tid == 0) {
        // Base-2 LSE: m_log2 + log2(d), matching FlashInfer's state_t::get_lse()
        float lse = (rowmax != NEG_INF) ? rowmax + ptx_log2(rowsum) : NEG_INF;
        partial_lse[split_idx * total_heads + head_linear] = lse;
    }
}


// ============================================================================
// Phase 2: Reduction kernel -- combine split results via log-sum-exp
// Grid: (total_heads)   Block: HEAD_DIM
// ============================================================================
template <int HEAD_DIM>
__global__ void split_kv_reduce_v2(
    const float* __restrict__ partial_O,
    const float* __restrict__ partial_lse,
    __nv_bfloat16* __restrict__ O,
    int num_splits,
    int total_heads
) {
    const int head_linear = blockIdx.x;
    const int tid = threadIdx.x;

    if (head_linear >= total_heads || tid >= HEAD_DIM) return;

    // LSE values are in base-2 domain; use ptx_exp2 for merge (matches FlashInfer cascade.cuh)
    float max_lse = NEG_INF;
    for (int s = 0; s < num_splits; s++) {
        float lse = partial_lse[s * total_heads + head_linear];
        max_lse = fmaxf(max_lse, lse);
    }

    float sum_exp = 0.0f;
    float sum_val = 0.0f;
    for (int s = 0; s < num_splits; s++) {
        float lse = partial_lse[s * total_heads + head_linear];
        float w = (lse > NEG_INF + 1.0f) ? ptx_exp2(lse - max_lse) : 0.0f;
        sum_exp += w;
        sum_val += w * partial_O[s * total_heads * HEAD_DIM + head_linear * HEAD_DIM + tid];
    }

    // Match FlashInfer: __fdividef for final normalization
    float result = (sum_exp > 0.0f) ? __fdividef(sum_val, sum_exp) : 0.0f;
    O[head_linear * HEAD_DIM + tid] = __float2bfloat16(result);
}


// ============================================================================
// Common launcher logic (templated on FP8_KV)
// ============================================================================
template <bool FP8_KV>
static void launch_decode(
    const __nv_bfloat16* Q,
    const kv_dtype_t<FP8_KV>* key_cache,
    const kv_dtype_t<FP8_KV>* val_cache,
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
    float k_scale,
    float v_scale,
    cudaStream_t stream
) {
    // Absorb k_scale into softmax scale, pre-multiply by log2(e)
    // to match FlashInfer's log-base-2 domain (variants.cuh: sm_scale_log2)
    float scale = (1.0f / sqrtf((float)head_dim)) * k_scale * 1.44269504088896340736f;

    // Adaptive split count
    int num_splits;
    if (max_seq_len <= 256)        num_splits = 1;
    else if (max_seq_len <= 1024)  num_splits = 4;
    else if (max_seq_len <= 4096)  num_splits = 8;
    else if (max_seq_len <= 16384) num_splits = 16;
    else                           num_splits = 32;
    if (num_splits > max_splits)   num_splits = max_splits;

    int kv_per_split = (max_seq_len + num_splits - 1) / num_splits;
    int total_heads = batch_size * num_q_heads;

    dim3 grid1(num_splits, batch_size, num_q_heads);

    if (head_dim == 256) {
        constexpr int HD = 256;
        constexpr int BKV = 128;
        int smem_bytes = HD * (int)sizeof(float)
                       + BKV * HD * (int)sizeof(kv_dtype_t<FP8_KV>)
                       + (BKV + 4) * (int)sizeof(float);

        auto kernel = tiled_split_kv_partial_paged<HD, BKV, FP8_KV>;
        cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        kernel<<<grid1, HD, smem_bytes, stream>>>(
            Q, key_cache, val_cache, block_table, seq_lens,
            partial_O, partial_lse,
            num_q_heads, num_kv_heads, block_size, max_blocks_per_seq, kv_block_stride,
            scale, v_scale, kv_per_split);

        dim3 grid2(total_heads);
        split_kv_reduce_v2<HD><<<grid2, HD, 0, stream>>>(
            partial_O, partial_lse, O, num_splits, total_heads);
    } else {
        constexpr int HD = 128;
        constexpr int BKV = 128;
        int smem_bytes = HD * (int)sizeof(float)
                       + BKV * HD * (int)sizeof(kv_dtype_t<FP8_KV>)
                       + (BKV + 4) * (int)sizeof(float);

        auto kernel = tiled_split_kv_partial_paged<HD, BKV, FP8_KV>;
        cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        kernel<<<grid1, HD, smem_bytes, stream>>>(
            Q, key_cache, val_cache, block_table, seq_lens,
            partial_O, partial_lse,
            num_q_heads, num_kv_heads, block_size, max_blocks_per_seq, kv_block_stride,
            scale, v_scale, kv_per_split);

        dim3 grid2(total_heads);
        split_kv_reduce_v2<HD><<<grid2, HD, 0, stream>>>(
            partial_O, partial_lse, O, num_splits, total_heads);
    }
}


// ============================================================================
// Host-callable launchers (extern "C" for torch extension)
// ============================================================================

// BF16 launcher (backward compatible — same signature as before + k_scale/v_scale)
extern "C" void sm120_flash_decode_paged_launch(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* key_cache,
    const __nv_bfloat16* val_cache,
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
    cudaStream_t stream
) {
    launch_decode<false>(
        Q, key_cache, val_cache, block_table, seq_lens, O,
        partial_O, partial_lse,
        batch_size, num_q_heads, num_kv_heads, head_dim,
        max_seq_len, block_size, max_blocks_per_seq, max_splits,
        kv_block_stride, 1.0f, 1.0f, stream);
}

// FP8 E4M3 launcher
extern "C" void sm120_flash_decode_paged_fp8_launch(
    const __nv_bfloat16* Q,
    const __nv_fp8_e4m3* key_cache,
    const __nv_fp8_e4m3* val_cache,
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
    float k_scale,
    float v_scale,
    cudaStream_t stream
) {
    launch_decode<true>(
        Q, key_cache, val_cache, block_table, seq_lens, O,
        partial_O, partial_lse,
        batch_size, num_q_heads, num_kv_heads, head_dim,
        max_seq_len, block_size, max_blocks_per_seq, max_splits,
        kv_block_stride, k_scale, v_scale, stream);
}
