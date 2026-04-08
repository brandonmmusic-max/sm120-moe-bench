/**
 * reshape_and_cache_nvfp4.cu
 *
 * Fused scatter-write + BF16→NVFP4 quantization for paged KV cache.
 * Drop-in replacement for vLLM's reshape_and_cache_flash when kv_cache_dtype="nvfp4".
 *
 * For each token:
 *   1. Read BF16 K/V from model output [num_tokens, num_heads, head_dim]
 *   2. Compute per-block (16-element) max_abs → E4M3FN scale
 *   3. Quantize 16 BF16 values → 8 packed FP4 bytes (2 nibbles per byte)
 *   4. Scatter-write packed data + block scale to the correct cache slot
 *
 * Cache layout (packed, per key or value):
 *   [num_blocks, block_size, num_heads, packed_dim]
 *   where packed_dim = head_dim/2 + head_dim/16
 *   First head_dim/2 bytes: packed FP4 data (low nibble = even idx, high = odd)
 *   Last head_dim/16 bytes: E4M3FN block scales (one per 16 elements)
 *
 * Hardware: SM120 (Blackwell). Uses __nv_bfloat16, no PTX FP4 convert
 * (cvt.rn.satfinite.e2m1x2 requires SM100 cubins). Software LUT approach.
 *
 * Build: nvcc -O3 -arch=sm_120 --threads=4
 */

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// FP4 E2M1 magnitude lookup: {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
__device__ __constant__ float FP4_MAGS[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

// E4M3FN encode: float → uint8 code
// Uses ldexpf-based encode to avoid integer overflow at e>=14
__device__ __forceinline__ uint8_t encode_e4m3fn(float val) {
    if (val <= 0.0f) return 0;
    if (val >= 448.0f) return 0xFE;  // max normal (e=15, m=6)

    // Binary search through E4M3FN representable values
    // E4M3FN: sign(1) + exp(4, bias=7) + mantissa(3)
    // Positive values: (1 + m/8) * 2^(e-7) for e>0, m/8 * 2^(-6) for e=0
    int best_code = 0;
    float best_dist = val;

    // Subnormals (e=0): m * 2^(-9) for m=1..7
    for (int m = 1; m <= 7; m++) {
        float decoded = (float)m * (1.0f / 512.0f);
        float dist = fabsf(val - decoded);
        if (dist < best_dist) { best_dist = dist; best_code = m; }
    }

    // Normals (e=1..15): (1 + m/8) * 2^(e-7)
    for (int e = 1; e <= 15; e++) {
        int m_max = (e == 15) ? 6 : 7;  // E4M3FN: e=15,m=7 is NaN
        for (int m = 0; m <= m_max; m++) {
            float decoded = ldexpf(1.0f + (float)m / 8.0f, e - 7);
            float dist = fabsf(val - decoded);
            if (dist < best_dist) {
                best_dist = dist;
                best_code = (e << 3) | m;
            }
        }
    }
    return (uint8_t)best_code;
}

// E4M3FN decode: uint8 code → float (must match encode exactly)
__device__ __forceinline__ float decode_e4m3fn(uint8_t code) {
    int e = (code >> 3) & 0xF;
    int m = code & 0x7;
    if (e == 0) return (float)m * (1.0f / 512.0f);  // subnormal
    if (e == 15 && m == 7) return 0.0f;               // NaN → 0
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

// Quantize a single float to unsigned FP4 nibble (0..7 magnitude index)
__device__ __forceinline__ uint8_t quantize_fp4_unsigned(float abs_val) {
    // Find nearest FP4 magnitude via binary search on sorted LUT
    // LUT: {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    if (abs_val <= 0.25f) return 0;  // → 0.0
    if (abs_val <= 0.75f) return 1;  // → 0.5
    if (abs_val <= 1.25f) return 2;  // → 1.0
    if (abs_val <= 1.75f) return 3;  // → 1.5
    if (abs_val <= 2.50f) return 4;  // → 2.0
    if (abs_val <= 3.50f) return 5;  // → 3.0
    if (abs_val <= 5.00f) return 6;  // → 4.0
    return 7;                         // → 6.0
}

// Full FP4 encode: float → 4-bit code (sign + 3-bit magnitude index)
__device__ __forceinline__ uint8_t encode_fp4(float val) {
    uint8_t sign = (val < 0.0f) ? 0x8 : 0x0;
    uint8_t mag = quantize_fp4_unsigned(fabsf(val));
    return sign | mag;
}

// Block size for NVFP4 scales (16 elements per block scale)
constexpr int FP4_BLOCK_SIZE = 16;

/**
 * Fused scatter-write + NVFP4 quantization kernel.
 *
 * One CUDA block per token. Threads cooperatively process all heads.
 * For each head: compute block scales, quantize, pack, write.
 *
 * Template params:
 *   HEAD_DIM: head dimension (128 or 256)
 */
template <int HEAD_DIM>
__global__ void reshape_and_cache_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ src,     // [num_tokens, num_heads, HEAD_DIM]
    uint8_t* __restrict__ cache,                // [num_blocks, block_size, num_heads, packed_dim]
    const int64_t* __restrict__ slot_mapping,   // [num_tokens]
    float global_scale,                         // per-tensor normalization (1.0 for normal range)
    int num_heads,
    int block_size,
    int64_t cache_stride_block,     // stride between blocks in cache (elements)
    int64_t cache_stride_page,      // stride between positions within a block
    int64_t cache_stride_head       // stride between heads
) {
    constexpr int DATA_BYTES = HEAD_DIM / 2;           // packed FP4 data per head
    constexpr int SCALE_BYTES = HEAD_DIM / FP4_BLOCK_SIZE;  // block scales per head
    constexpr int PACKED_DIM = DATA_BYTES + SCALE_BYTES;
    constexpr int NUM_BLOCKS_PER_HEAD = HEAD_DIM / FP4_BLOCK_SIZE;

    const int token_idx = blockIdx.x;
    const int64_t slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) return;  // padding token

    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;

    // Source: BF16 [num_tokens, num_heads, HEAD_DIM]
    const __nv_bfloat16* token_src = src + (int64_t)token_idx * num_heads * HEAD_DIM;

    // Destination: packed NVFP4 in cache
    uint8_t* token_dst = cache + block_idx * cache_stride_block
                               + block_offset * cache_stride_page;

    const float inv_global_scale = (global_scale != 0.0f) ? (1.0f / global_scale) : 1.0f;

    // Process heads cooperatively: each warp handles one head
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;

    for (int head = warp_id; head < num_heads; head += warps_per_block) {
        const __nv_bfloat16* h_src = token_src + head * HEAD_DIM;
        uint8_t* h_dst = token_dst + (int64_t)head * cache_stride_head;

        // Load BF16 values into registers and apply global scale
        float vals[HEAD_DIM / 32];  // each thread handles HEAD_DIM/32 elements
        const int elems_per_thread = HEAD_DIM / 32;

        #pragma unroll
        for (int i = 0; i < elems_per_thread; i++) {
            int idx = lane + i * 32;
            if (idx < HEAD_DIM) {
                vals[i] = __bfloat162float(h_src[idx]) * inv_global_scale;
            }
        }

        // For each FP4 block of 16 elements: compute max_abs, find scale, quantize
        #pragma unroll
        for (int blk = 0; blk < NUM_BLOCKS_PER_HEAD; blk++) {
            int blk_start = blk * FP4_BLOCK_SIZE;

            // Step 1: Compute max_abs across 16 elements using warp shuffle
            // Only threads 0..15 of the group that owns these elements participate
            float my_abs = 0.0f;
            int my_local = lane - (blk_start % 32);  // offset within this block

            // Each of the 16 elements in the block is owned by a specific thread
            // Map: element i (0..15) → thread (blk_start + i) % 32
            // Use warp shuffle to share values

            // Simpler approach: each thread computes its contribution
            float local_max = 0.0f;
            #pragma unroll
            for (int i = 0; i < elems_per_thread; i++) {
                int idx = lane + i * 32;
                if (idx >= blk_start && idx < blk_start + FP4_BLOCK_SIZE) {
                    local_max = fmaxf(local_max, fabsf(vals[i]));
                }
            }

            // Warp-wide max reduction
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, offset));
            }
            // Now all threads in warp have the block max_abs

            // Step 2: Compute E4M3FN block scale = max_abs / FP4_MAX_MAG
            float target_scale = local_max / 6.0f;  // 6.0 = max FP4 magnitude
            target_scale = fmaxf(target_scale, 1.0f / 512.0f);  // clamp to min subnormal

            // Encode as E4M3FN (nearest representable)
            uint8_t scale_code = encode_e4m3fn(target_scale);
            float scale_decoded = decode_e4m3fn(scale_code);
            float inv_scale = (scale_decoded > 0.0f) ? (1.0f / scale_decoded) : 0.0f;

            // Step 3: Quantize and pack FP4 pairs
            // Thread 0 of the warp writes the packed bytes and scale
            if (lane == 0) {
                // Write block scale
                h_dst[DATA_BYTES + blk] = scale_code;
            }

            // Each pair of elements → 1 packed byte
            // Elements blk_start..blk_start+15 → bytes blk_start/2..blk_start/2+7
            #pragma unroll
            for (int i = 0; i < elems_per_thread; i++) {
                int idx = lane + i * 32;
                if (idx >= blk_start && idx < blk_start + FP4_BLOCK_SIZE && (idx % 2 == 0)) {
                    // Get this element and the next
                    float v0 = 0.0f, v1 = 0.0f;

                    // Find v0 (idx) and v1 (idx+1) — they might be in different threads
                    // Since we need pairs, use shared memory or shuffle
                    // For simplicity with the register layout, reconstruct from source
                    v0 = __bfloat162float(h_src[idx]) * inv_global_scale;
                    v1 = __bfloat162float(h_src[idx + 1]) * inv_global_scale;

                    uint8_t nibble0 = encode_fp4(v0 * inv_scale);
                    uint8_t nibble1 = encode_fp4(v1 * inv_scale);
                    uint8_t packed = nibble0 | (nibble1 << 4);

                    h_dst[idx / 2] = packed;
                }
            }
        }
    }
}

// Launch wrapper
void reshape_and_cache_nvfp4_launch(
    const __nv_bfloat16* src,       // [num_tokens, num_heads, head_dim] - key or value
    uint8_t* cache,                  // paged cache
    const int64_t* slot_mapping,
    float global_scale,
    int num_tokens,
    int num_heads,
    int head_dim,
    int block_size,
    int64_t cache_stride_block,
    int64_t cache_stride_page,
    int64_t cache_stride_head,
    cudaStream_t stream
) {
    if (num_tokens == 0) return;

    // 128 threads per block (4 warps)
    constexpr int THREADS = 128;
    dim3 grid(num_tokens);
    dim3 block(THREADS);

    if (head_dim == 256) {
        reshape_and_cache_nvfp4_kernel<256><<<grid, block, 0, stream>>>(
            src, cache, slot_mapping, global_scale,
            num_heads, block_size,
            cache_stride_block, cache_stride_page, cache_stride_head
        );
    } else if (head_dim == 128) {
        reshape_and_cache_nvfp4_kernel<128><<<grid, block, 0, stream>>>(
            src, cache, slot_mapping, global_scale,
            num_heads, block_size,
            cache_stride_block, cache_stride_page, cache_stride_head
        );
    }
}
