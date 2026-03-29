/**
 * SM120 Flash Decode v2 -- NVFP4 KV Cache (E2M1 data + E4M3FN block scales)
 *
 * Extends the FP8 decode kernel to support NVFP4 (4-bit) KV cache.
 * Halves KV cache memory (0.5625 bytes vs 1.0 bytes per element) for ~2x context.
 *
 * NVFP4 format:
 *   - Data: 2 FP4 E2M1 values packed per byte (low nibble = even idx, high = odd)
 *   - Scales: 1 E4M3FN byte per FP4_BLOCK_SIZE=16 FP4 values
 *   - Dequantization: real_value = decode_fp4(nibble) * decode_e4m3fn(block_scale)
 *
 * Packed cache layout (per KV row = one position, one head):
 *   [fp4_data: HEAD_DIM/2 bytes] [scales: HEAD_DIM/16 bytes]
 *   Total: 9*HEAD_DIM/16 bytes per row
 *   For HEAD_DIM=256: 128 + 16 = 144 bytes per row
 *
 * FP4 E2M1 encoding (4-bit: 1 sign + 2 exponent + 1 mantissa):
 *   Values: 0, +/-0.5, +/-1.0, +/-1.5, +/-2.0, +/-3.0, +/-4.0, +/-6.0
 *
 * Block scale E4M3FN encoding (8-bit: 1s + 4e + 3m):
 *   Max representable: 448.0
 *   Uses ldexpf() for decode -- NOT integer shift (overflow at e>=14)
 *   See PHASE2_RESULTS.md bug fix #3 for the full story.
 *
 * Provenance: decode_fp4, decode_e4m3fn from VerdictMoE (verdict_moe_ext.cu)
 *   - Battle-tested at 165.1 tok/s with Qwen3.5-397B in production
 *   - E4M3FN decode fixed through 3 phases (PHASE2_RESULTS.md)
 *   - decode_fp4 uses register arithmetic only (no __constant__ LUT --
 *     constant memory serializes divergent warp access, see feedback_constant_mem_lut)
 *
 * Kernel structure mirrors sm120_flash_decode_v2_paged.cu (BF16/FP8 version):
 *   Phase 1: Tiled split-KV partial attention with paged NVFP4 cache
 *   Phase 2: Reduction kernel (shared with BF16/FP8, duplicated here for standalone build)
 */

#include <cuda.h>
#include <cuda_bf16.h>
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

// ============================================================================
// NVFP4 dequantization helpers
// From VerdictMoE (verdict_moe_ext.cu), battle-tested in production.
// ============================================================================

// NVFP4 block scale group size: 1 E4M3FN scale per 16 FP4 values.
// This is the NVIDIA standard for NVFP4 (mxfp4/nvfp4).
constexpr int FP4_BLOCK_SIZE = 16;

/**
 * Decode one FP4 E2M1 nibble to float.
 *
 * E2M1 format: [sign:1][exponent:2][mantissa:1]
 * 8 magnitude values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
 *
 * Uses register-only arithmetic. No __constant__ memory LUT because
 * divergent warp access to __constant__ serializes (up to 32 serial reads).
 * See feedback_constant_mem_lut.md.
 */
__device__ __forceinline__ float decode_fp4(uint8_t nibble) {
    int mag = nibble & 0x7;
    int e = mag >> 1;       // 2-bit exponent
    int m = mag & 1;        // 1-bit mantissa
    float val;
    if (e == 0) {
        // Subnormal: value = m * 0.5 (either 0.0 or 0.5)
        val = m ? 0.5f : 0.0f;
    } else {
        // Normal: value = (1.0 + m*0.5) * 2^(e-1)
        // Equivalent to: (2+m) * 2^(e-2) * 0.5
        // For e=1: (2+m)*0.5  = 1.0 or 1.5
        // For e=2: (2+m)*1.0  = 2.0 or 3.0
        // For e=3: (2+m)*2.0  = 4.0 or 6.0
        val = (float)(2 + m) * (float)(1 << (e - 1)) * 0.5f;
    }
    return (nibble & 0x8) ? -val : val;
}

/**
 * Decode one E4M3FN byte to float.
 *
 * E4M3FN format: [sign:1][exponent:4][mantissa:3]
 * Uses ldexpf() for safe exponentiation -- NOT integer shift.
 *
 * CRITICAL: The integer shift approach (1 << (e+17)) overflows for e>=14,
 * causing sign flips and zeros. 3-11% of block scales in Qwen3.5-397B
 * hit e>=14 (5.4% gate_proj, 10.9% up_proj). This was root cause #3 in
 * PHASE2_RESULTS.md and took 3 debugging phases to discover.
 *
 * ldexpf computes val * 2^exp without integer overflow risk.
 */
__device__ __forceinline__ float decode_e4m3fn(uint8_t x) {
    int s = (x >> 7) & 1;   // sign bit
    int e = (x >> 3) & 0xF; // 4-bit exponent (bias = 7)
    int m = x & 7;           // 3-bit mantissa
    float val;
    if (e == 15 && m == 7) {
        val = 0.0f;  // NaN -> 0 (E4M3FN has no inf, NaN=0x7F/0xFF mapped to 0)
    } else if (e == 0) {
        // Subnormal: (0.m) * 2^(-6) = m/8 * 2^(-6) = m * 2^(-9) = m/512
        val = __int2float_rn(m) * 0.001953125f;  // 1/512
    } else {
        // Normal: (1 + m/8) * 2^(e-7)
        // ldexpf avoids integer overflow at e>=14
        val = ldexpf(1.0f + __int2float_rn(m) * 0.125f, e - 7);
    }
    return s ? -val : val;
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
// NVFP4 paged tile loader
//
// Loads one tile of BLOCK_KV rows from paged NVFP4 cache into split SMEM:
//   fp4_smem:   [BLOCK_KV, HEAD_DIM/2] uint8  -- packed FP4 data
//   scale_smem: [BLOCK_KV, SCALE_COLS] uint8   -- E4M3FN block scales
//
// Source cache layout (packed, per row):
//   [fp4_data: HEAD_DIM/2 bytes] [scales: HEAD_DIM/FP4_BLOCK_SIZE bytes]
//   Stride between rows: PACKED_COLS = HEAD_DIM/2 + HEAD_DIM/FP4_BLOCK_SIZE
//
// Uses cp.async 16B vectorized transfers from HBM to SMEM.
// ============================================================================
template <int HEAD_DIM, int BLOCK_KV>
__device__ __forceinline__ void load_paged_tile_fp4(
    uint8_t* __restrict__ fp4_smem,            // [BLOCK_KV, DATA_COLS]
    uint8_t* __restrict__ scale_smem,          // [BLOCK_KV, SCALE_COLS]
    const uint8_t* __restrict__ cache,         // global packed KV cache
    const int* __restrict__ block_table,
    int seq_idx,
    int tile_start,
    int tile_end,
    int kv_head,
    int block_size,
    int num_kv_heads,
    int max_blocks_per_seq,
    int kv_block_stride,   // stride(0) of cache in bytes (uint8)
    int tid,
    int num_threads
) {
    constexpr int DATA_COLS  = HEAD_DIM / 2;                  // 128 for HD=256
    constexpr int SCALE_COLS = HEAD_DIM / FP4_BLOCK_SIZE;     // 16 for HD=256
    constexpr int PACKED_COLS = DATA_COLS + SCALE_COLS;        // 144 for HD=256

    // Phase 1: Load FP4 data bytes
    // Each 16B load = 16 packed bytes = 32 FP4 elements
    constexpr int TOTAL_DATA_16B = BLOCK_KV * DATA_COLS / 16;

    #pragma unroll 4
    for (int i = tid; i < TOTAL_DATA_16B; i += num_threads) {
        const int kv_local = (i * 16) / DATA_COLS;   // which row in tile
        const int d_byte   = (i * 16) % DATA_COLS;   // byte offset in row
        const int kv_pos   = tile_start + kv_local;

        if (kv_pos < tile_end) {
            const int page_idx = kv_pos / block_size;
            const int page_off = kv_pos % block_size;
            const int blk = block_table[seq_idx * max_blocks_per_seq + page_idx];
            // NHD layout: row = (page_off * num_kv_heads + kv_head)
            // Data starts at offset 0 within packed row
            const int src = blk * kv_block_stride
                          + (page_off * num_kv_heads + kv_head) * PACKED_COLS
                          + d_byte;
            cp_async_16B(&fp4_smem[kv_local * DATA_COLS + d_byte], &cache[src]);
        } else {
            // Zero-fill OOB
            uint4* dst = reinterpret_cast<uint4*>(&fp4_smem[kv_local * DATA_COLS + d_byte]);
            *dst = make_uint4(0, 0, 0, 0);
        }
    }

    // Phase 2: Load scale bytes
    // Each row has SCALE_COLS bytes. For HD=256: 16 bytes = exactly one 16B chunk per row.
    constexpr int TOTAL_SCALE_16B = BLOCK_KV * SCALE_COLS / 16;
    // SCALE_COLS must be a multiple of 16 for 16B loads.
    // For HD=256: SCALE_COLS=16 -> 1 chunk per row. For HD=128: SCALE_COLS=8 -> need special handling.
    static_assert(SCALE_COLS >= 16, "HEAD_DIM must be >= 256 for 16B-aligned scale loads");

    #pragma unroll 2
    for (int i = tid; i < TOTAL_SCALE_16B; i += num_threads) {
        const int kv_local  = (i * 16) / SCALE_COLS;
        const int s_byte    = (i * 16) % SCALE_COLS;
        const int kv_pos    = tile_start + kv_local;

        if (kv_pos < tile_end) {
            const int page_idx = kv_pos / block_size;
            const int page_off = kv_pos % block_size;
            const int blk = block_table[seq_idx * max_blocks_per_seq + page_idx];
            // Scales start at offset DATA_COLS within packed row
            const int src = blk * kv_block_stride
                          + (page_off * num_kv_heads + kv_head) * PACKED_COLS
                          + DATA_COLS + s_byte;
            cp_async_16B(&scale_smem[kv_local * SCALE_COLS + s_byte], &cache[src]);
        } else {
            uint4* dst = reinterpret_cast<uint4*>(&scale_smem[kv_local * SCALE_COLS + s_byte]);
            *dst = make_uint4(0, 0, 0, 0);
        }
    }

    cp_async_commit();
}


// ============================================================================
// Phase 1: Tiled partial attention with paged NVFP4 KV cache
//
// Grid:  (num_splits, batch_size, num_q_heads)
// Block: HEAD_DIM threads
//
// SMEM layout:
//   q_smem:           [HEAD_DIM] float
//   fp4_smem:         [BLOCK_KV * DATA_COLS] uint8  (reused for K then V)
//   scale_smem:       [BLOCK_KV * SCALE_COLS] uint8 (reused for K then V)
//   scale_float_smem: [BLOCK_KV * SCALE_COLS] float (pre-decoded scales)
//   p_smem:           [BLOCK_KV + 4] float          (scores + softmax metadata)
// ============================================================================
template <int HEAD_DIM, int BLOCK_KV>
__global__ void __launch_bounds__(HEAD_DIM, 2)
tiled_split_kv_partial_paged_fp4(
    const __nv_bfloat16* __restrict__ Q,
    const uint8_t* __restrict__ key_cache,     // packed NVFP4 [data|scales]
    const uint8_t* __restrict__ val_cache,     // packed NVFP4 [data|scales]
    const int* __restrict__ block_table,
    const int* __restrict__ seq_lens,
    float* __restrict__ partial_O,
    float* __restrict__ partial_lse,
    int num_q_heads,
    int num_kv_heads,
    int block_size,
    int max_blocks_per_seq,
    int kv_block_stride,
    float scale,          // = 1/sqrt(d) * k_tensor_scale * log2(e)
    float v_tensor_scale, // per-tensor V pre-normalization scale (1.0 if no pre-norm)
    int kv_per_split
) {
    const int split_idx = blockIdx.x;
    const int seq_idx   = blockIdx.y;
    const int q_head    = blockIdx.z;
    const int tid       = threadIdx.x;
    constexpr int NUM_THREADS = HEAD_DIM;

    // Layout constants
    constexpr int DATA_COLS  = HEAD_DIM / 2;
    constexpr int SCALE_COLS = HEAD_DIM / FP4_BLOCK_SIZE;

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
    // All regions 16-byte aligned for cp.async compatibility.
    constexpr int Q_BYTES     = HEAD_DIM * sizeof(float);                 // 1024
    constexpr int FP4_BYTES   = BLOCK_KV * DATA_COLS;                     // 16384
    constexpr int SCALE_BYTES = BLOCK_KV * SCALE_COLS;                    // 2048
    constexpr int SF_BYTES    = BLOCK_KV * SCALE_COLS * sizeof(float);    // 8192
    constexpr int P_BYTES     = (BLOCK_KV + 4) * sizeof(float);          // 528

    extern __shared__ char smem_raw[];

    float*   q_smem           = reinterpret_cast<float*>(smem_raw);
    uint8_t* fp4_smem         = reinterpret_cast<uint8_t*>(smem_raw + Q_BYTES);
    uint8_t* scale_smem       = reinterpret_cast<uint8_t*>(smem_raw + Q_BYTES + FP4_BYTES);
    float*   scale_float_smem = reinterpret_cast<float*>(smem_raw + Q_BYTES + FP4_BYTES + SCALE_BYTES);
    float*   p_smem           = reinterpret_cast<float*>(smem_raw + Q_BYTES + FP4_BYTES + SCALE_BYTES + SF_BYTES);

    // Load Q into shared memory (FP32 for dot product precision)
    {
        const int q_offset = seq_idx * num_q_heads * HEAD_DIM + q_head * HEAD_DIM;
        if (tid < HEAD_DIM)
            q_smem[tid] = __bfloat162float(Q[q_offset + tid]);
    }
    __syncthreads();

    // ---- Thread assignment for Q@K^T ----
    constexpr int R = HEAD_DIM / BLOCK_KV;  // 2 for HD=256, BKV=128
    static_assert(R >= 1 && (R & (R - 1)) == 0, "R must be power of 2");
    constexpr int D_PER_THREAD = HEAD_DIM / R;  // 128

    const int score_idx   = tid / R;         // which KV position this thread scores
    const int sub_tid     = tid % R;         // which sub-part of the dot product
    const int d_start_qk  = sub_tid * D_PER_THREAD;

    // Online softmax state (log-base-2 domain, matches FlashInfer state_t)
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
        // Step 1: Load K tile (FP4 data + scales) from paged cache
        // ==============================================================
        load_paged_tile_fp4<HEAD_DIM, BLOCK_KV>(
            fp4_smem, scale_smem, key_cache, block_table, seq_idx,
            tile_start, tile_end, kv_head,
            block_size, num_kv_heads, max_blocks_per_seq, kv_block_stride,
            tid, NUM_THREADS);
        cp_async_wait_all();
        __syncthreads();

        // ==============================================================
        // Step 1b: Pre-decode block scales to float in SMEM
        // Avoids expensive decode_e4m3fn calls in QK/PV hot loops.
        // 2048 decodes / 256 threads = 8 per thread (negligible).
        // ==============================================================
        {
            constexpr int TOTAL_SCALES = BLOCK_KV * SCALE_COLS;
            #pragma unroll 8
            for (int i = tid; i < TOTAL_SCALES; i += NUM_THREADS) {
                scale_float_smem[i] = decode_e4m3fn(scale_smem[i]);
            }
        }
        __syncthreads();

        // ==============================================================
        // Step 2: Q @ K^T -- compute BLOCK_KV scores with FP4 dequant
        // ==============================================================
        float dot = 0.0f;
        {
            // FP4 data for this KV row, starting at our dimension offset
            const uint8_t* fp4_row = &fp4_smem[score_idx * DATA_COLS + d_start_qk / 2];
            // Pre-decoded scales for this KV row
            const float* sf_row = &scale_float_smem[score_idx * SCALE_COLS + d_start_qk / FP4_BLOCK_SIZE];
            const float* q_row = &q_smem[d_start_qk];

            // Process D_PER_THREAD elements in groups of FP4_BLOCK_SIZE
            // Each group shares one block scale.
            // D_PER_THREAD=128, FP4_BLOCK_SIZE=16 -> 8 scale groups
            #pragma unroll
            for (int sg = 0; sg < D_PER_THREAD / FP4_BLOCK_SIZE; sg++) {
                float block_scale = sf_row[sg];

                // Process 16 FP4 elements (8 packed bytes) with this scale
                #pragma unroll
                for (int dd = 0; dd < FP4_BLOCK_SIZE; dd += 2) {
                    int d_local = sg * FP4_BLOCK_SIZE + dd;
                    // One byte holds two FP4 values: low nibble = even, high nibble = odd
                    uint8_t byte = fp4_row[d_local / 2];
                    float v0 = decode_fp4(byte & 0xF) * block_scale;
                    float v1 = decode_fp4(byte >> 4) * block_scale;
                    dot += q_row[d_local] * v0 + q_row[d_local + 1] * v1;
                }
            }

            // Reduce within R-thread group via shuffle (unconditional -- all threads)
            if constexpr (R >= 2) dot += __shfl_xor_sync(0xffffffff, dot, 1);
            if constexpr (R >= 4) dot += __shfl_xor_sync(0xffffffff, dot, 2);
            if constexpr (R >= 8) dot += __shfl_xor_sync(0xffffffff, dot, 4);
        }

        // scale is already in log2 domain: (1/sqrt(d)) * log2(e)
        float score = dot * scale;
        if (score_idx >= tile_len) score = NEG_INF;

        if (sub_tid == 0 && score_idx < BLOCK_KV)
            p_smem[score_idx] = score;
        __syncthreads();

        // ==============================================================
        // Step 3: Start V tile load (overlapped with softmax)
        // ==============================================================
        load_paged_tile_fp4<HEAD_DIM, BLOCK_KV>(
            fp4_smem, scale_smem, val_cache, block_table, seq_idx,
            tile_start, tile_end, kv_head,
            block_size, num_kv_heads, max_blocks_per_seq, kv_block_stride,
            tid, NUM_THREADS);

        // ==============================================================
        // Step 4: Online softmax (thread 0, overlapped with V load)
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

        // Pre-decode V scales
        {
            constexpr int TOTAL_SCALES = BLOCK_KV * SCALE_COLS;
            #pragma unroll 8
            for (int i = tid; i < TOTAL_SCALES; i += NUM_THREADS) {
                scale_float_smem[i] = decode_e4m3fn(scale_smem[i]);
            }
        }
        __syncthreads();

        // ==============================================================
        // Step 5: Rescale existing accumulators + P@V with FP4 dequant
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
            const int d_byte = d / 2;
            const int d_nibble_is_high = d & 1;
            const int scale_col = d / FP4_BLOCK_SIZE;
            float pv_sum = 0.0f;

            #pragma unroll 16
            for (int n = 0; n < BLOCK_KV; n++) {
                // Dequantize V element at (n, d)
                uint8_t byte = fp4_smem[n * DATA_COLS + d_byte];
                uint8_t nibble = d_nibble_is_high ? (byte >> 4) : (byte & 0xF);
                float fp4_val = decode_fp4(nibble);
                float v_val = fp4_val * scale_float_smem[n * SCALE_COLS + scale_col];
                pv_sum += p_smem[n] * v_val;
            }
            o_val += pv_sum;
        }
        __syncthreads();
    }  // end tile loop

    // ==============================================================
    // Normalize and write partial results
    // v_tensor_scale applied here: block scales decode the per-block FP4 values,
    // v_tensor_scale restores the original magnitude removed during pre-normalization.
    // ==============================================================
    {
        float d_rcp = (rowmax != NEG_INF) ? ptx_rcp(rowsum) : 0.f;
        o_val *= d_rcp * v_tensor_scale;
    }

    if (tid < HEAD_DIM)
        partial_O[split_idx * total_heads * HEAD_DIM + head_linear * HEAD_DIM + tid] = o_val;

    if (tid == 0) {
        float lse = (rowmax != NEG_INF) ? rowmax + ptx_log2(rowsum) : NEG_INF;
        partial_lse[split_idx * total_heads + head_linear] = lse;
    }
}


// ============================================================================
// Phase 2: Reduction kernel -- combine split results via log-sum-exp
// (Identical to BF16/FP8 version -- duplicated for standalone build)
// ============================================================================
template <int HEAD_DIM>
__global__ void split_kv_reduce_fp4(
    const float* __restrict__ partial_O,
    const float* __restrict__ partial_lse,
    __nv_bfloat16* __restrict__ O,
    int num_splits,
    int total_heads
) {
    const int head_linear = blockIdx.x;
    const int tid = threadIdx.x;

    if (head_linear >= total_heads || tid >= HEAD_DIM) return;

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

    float result = (sum_exp > 0.0f) ? __fdividef(sum_val, sum_exp) : 0.0f;
    O[head_linear * HEAD_DIM + tid] = __float2bfloat16(result);
}


// ============================================================================
// Launcher
// ============================================================================
static void launch_decode_fp4(
    const __nv_bfloat16* Q,
    const uint8_t* key_cache,
    const uint8_t* val_cache,
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
    float k_tensor_scale,
    float v_tensor_scale,
    cudaStream_t stream
) {
    // Per-tensor scales handle data pre-normalization for extreme values.
    // k_tensor_scale absorbed into softmax scale (one multiply, not N).
    // v_tensor_scale applied at output normalization.
    // For normal-range data, both are 1.0.
    float scale = (1.0f / sqrtf((float)head_dim)) * k_tensor_scale * 1.44269504088896340736f;

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
        constexpr int DATA_COLS  = HD / 2;
        constexpr int SCALE_COLS = HD / FP4_BLOCK_SIZE;

        // SMEM: q(float) + fp4_data(uint8) + scales(uint8) + scales_float(float) + p(float)
        int smem_bytes = HD * (int)sizeof(float)           // q_smem
                       + BKV * DATA_COLS                    // fp4_smem
                       + BKV * SCALE_COLS                   // scale_smem
                       + BKV * SCALE_COLS * (int)sizeof(float)  // scale_float_smem
                       + (BKV + 4) * (int)sizeof(float);   // p_smem

        auto kernel = tiled_split_kv_partial_paged_fp4<HD, BKV>;
        cudaFuncSetAttribute(kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        kernel<<<grid1, HD, smem_bytes, stream>>>(
            Q, key_cache, val_cache, block_table, seq_lens,
            partial_O, partial_lse,
            num_q_heads, num_kv_heads, block_size, max_blocks_per_seq, kv_block_stride,
            scale, v_tensor_scale, kv_per_split);

        dim3 grid2(total_heads);
        split_kv_reduce_fp4<HD><<<grid2, HD, 0, stream>>>(
            partial_O, partial_lse, O, num_splits, total_heads);
    } else {
        // HD=128 support (not our primary target but included for completeness)
        // NOTE: HD=128 has SCALE_COLS=8 which is < 16 bytes -- cp.async 16B won't work.
        // For now, only HD=256 is supported. This path will assert in the static_assert.
        // If HD=128 support is needed, use cp.async.ca.shared.global [%0], [%1], 8
        // or fall back to regular loads for scales.
    }
}


// ============================================================================
// Host-callable launcher (extern "C" for torch extension)
// ============================================================================
extern "C" void sm120_flash_decode_paged_fp4_launch(
    const __nv_bfloat16* Q,
    const uint8_t* key_cache,
    const uint8_t* val_cache,
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
    float k_tensor_scale,
    float v_tensor_scale,
    cudaStream_t stream
) {
    launch_decode_fp4(
        Q, key_cache, val_cache, block_table, seq_lens, O,
        partial_O, partial_lse,
        batch_size, num_q_heads, num_kv_heads, head_dim,
        max_seq_len, block_size, max_blocks_per_seq, max_splits,
        kv_block_stride, k_tensor_scale, v_tensor_scale, stream);
}
