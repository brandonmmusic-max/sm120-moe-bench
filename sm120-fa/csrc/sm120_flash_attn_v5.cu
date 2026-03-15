/**
 * SM120 Flash Attention v5 — Swizzled SMEM + tile autotuning
 *
 * Key changes from v3:
 *   - XOR swizzle on Q/K/V shared memory layouts
 *   - Swizzle applied consistently to writes (cp.async) and reads (pack2/ldmatrix)
 *   - Score masking for non-aligned sequences
 *   - Cross-thread softmax reduction
 *
 * Swizzle pattern: 128B XOR swizzle (Swizzle<3,3,3>)
 *   For a row with stride S bytes, the swizzled byte offset is:
 *     swizzled = offset ^ ((row & 7) << 4)
 *   This XORs bits [6:4] of the column byte offset with row[2:0],
 *   ensuring threads in different rows access different banks.
 *
 * Performance target: 80-100 TFLOPS (up from 54 TFLOPS)
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define BLOCK_M 64
#define BLOCK_N 64
#define HEAD_DIM 128
#define NUM_STAGES 2
#define WARP_SIZE 32
#define NUM_WARPS 4
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE)
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_M MMA_M

// SMEM stride: padded to 136 bf16 elements (272 bytes) per row
// to break bank conflict patterns. 136 = 128 + 8 padding.
// Actually, with XOR swizzle we don't need padding — the swizzle
// itself breaks conflicts. Use HEAD_DIM as stride.
#define SMEM_STRIDE HEAD_DIM  // 128 bf16 per row

#define Q_ELEMS (BLOCK_M * SMEM_STRIDE)
#define KV_ELEMS (BLOCK_N * SMEM_STRIDE)
#define P_ELEMS (BLOCK_M * BLOCK_N)
#define SMEM_BYTES ((Q_ELEMS + NUM_STAGES * 2 * KV_ELEMS + P_ELEMS) * 2)

// ============================================================================
// XOR Swizzle: remap SMEM addresses to avoid bank conflicts
//
// For 128-byte rows (64 bf16), swizzle bits [6:4] with row[2:0]:
//   swizzled_byte_offset = byte_offset ^ ((row & 7) << 4)
//
// For 256-byte rows (128 bf16 = HEAD_DIM), we use a wider swizzle:
//   swizzled_byte_offset = byte_offset ^ ((row & 7) << 5)
// This XORs bits [7:5] with row[2:0], covering 32-byte (16 bf16) groups.
// ============================================================================

// Convert (row, col_in_bf16) to swizzled SMEM index (in bf16 elements)
__device__ __forceinline__ int swizzle_idx(int row, int col, int stride) {
    // Convert col to byte offset, apply XOR, convert back
    int col_bytes = col * 2;  // bf16 = 2 bytes
    int swizzled_bytes = col_bytes ^ ((row & 7) << 5);  // XOR bits [7:5]
    return row * stride + swizzled_bytes / 2;
}

// Swizzle for 16-byte (8 bf16) aligned access (cp.async granularity)
// Returns swizzled base index for a 16-byte chunk starting at (row, col)
__device__ __forceinline__ int swizzle_16B_idx(int row, int col, int stride) {
    // col must be 8-aligned (16 bytes)
    int col_bytes = col * 2;
    int swizzled_bytes = col_bytes ^ ((row & 7) << 5);
    return row * stride + swizzled_bytes / 2;
}

__device__ __forceinline__ uint32_t pack2(const __nv_bfloat16& a, const __nv_bfloat16& b) {
    uint32_t r;
    asm("mov.b32 %0, {%1, %2};" : "=r"(r) : "h"(*(const uint16_t*)&a), "h"(*(const uint16_t*)&b));
    return r;
}

__device__ __forceinline__ void cp_async_16B(void* s, const void* g) {
    uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(s));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(g));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n"); }
__device__ __forceinline__ void cp_async_wait_one() { asm volatile("cp.async.wait_group 1;\n"); }

// ============================================================================
// Load A fragment with swizzle
// ============================================================================
__device__ __forceinline__ void load_A_frag_swizzled(
    uint32_t frag[4],
    const __nv_bfloat16* smem,
    int row_offset, int k_offset, int stride, int lane
) {
    int g = lane / 4;
    int t = lane % 4;
    int r0 = row_offset + g;
    int r1 = row_offset + g + 8;
    int c0 = k_offset + t * 2;
    int c1 = k_offset + t * 2 + 8;

    frag[0] = pack2(smem[swizzle_idx(r0, c0, stride)],     smem[swizzle_idx(r0, c0 + 1, stride)]);
    frag[1] = pack2(smem[swizzle_idx(r1, c0, stride)],     smem[swizzle_idx(r1, c0 + 1, stride)]);
    frag[2] = pack2(smem[swizzle_idx(r0, c1, stride)],     smem[swizzle_idx(r0, c1 + 1, stride)]);
    frag[3] = pack2(smem[swizzle_idx(r1, c1, stride)],     smem[swizzle_idx(r1, c1 + 1, stride)]);
}

// ============================================================================
// Load B fragment with swizzle
// ============================================================================
__device__ __forceinline__ void load_B_frag_swizzled(
    uint32_t frag[2],
    const __nv_bfloat16* smem,
    int n_offset, int k_offset, int stride, int lane
) {
    int g = lane / 4;
    int t = lane % 4;
    int n = n_offset + g;
    int k0 = k_offset + t * 2;
    int k1 = k_offset + t * 2 + 8;

    frag[0] = pack2(smem[swizzle_idx(n, k0, stride)],     smem[swizzle_idx(n, k0 + 1, stride)]);
    frag[1] = pack2(smem[swizzle_idx(n, k1, stride)],     smem[swizzle_idx(n, k1 + 1, stride)]);
}

// MMA
__device__ __forceinline__ void mma_m16n8k16(float c[4], uint32_t a[4], uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\n"
        "    {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

// ============================================================================
// Main kernel
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
sm120_flash_attn_fwd_v5(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int Sq, int Skv, int Hq, int Hkv, float scale
) {
    const int bm = blockIdx.x;
    const int head = blockIdx.y;
    const int kv_head = head / (Hq / Hkv);
    const int m_start = bm * BLOCK_M;
    const int tid = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int g = lane / 4;
    const int t = lane % 4;

    const __nv_bfloat16* q_ptr = Q + head * Sq * HEAD_DIM;
    const __nv_bfloat16* k_ptr = K + kv_head * Skv * HEAD_DIM;
    const __nv_bfloat16* v_ptr = V + kv_head * Skv * HEAD_DIM;

    extern __shared__ char smem[];
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* k_s[2], *v_s[2];
    k_s[0] = q_s + Q_ELEMS;
    v_s[0] = k_s[0] + KV_ELEMS;
    k_s[1] = v_s[0] + KV_ELEMS;
    v_s[1] = k_s[1] + KV_ELEMS;
    __nv_bfloat16* p_s = v_s[1] + KV_ELEMS;

    // Zero-fill Q SMEM
    for (int i = tid; i < Q_ELEMS; i += BLOCK_SIZE)
        q_s[i] = __float2bfloat16(0.0f);
    __syncthreads();

    // Load Q with swizzle: write to swizzled SMEM locations
    for (int i = tid; i < Q_ELEMS / 8; i += BLOCK_SIZE) {
        int row = (i * 8) / HEAD_DIM;
        int col = (i * 8) % HEAD_DIM;
        int qr = m_start + row;
        if (qr < Sq) {
            // cp.async loads 16 bytes into swizzled SMEM address
            int sw_idx = swizzle_16B_idx(row, col, SMEM_STRIDE);
            cp_async_16B(&q_s[sw_idx], &q_ptr[qr * HEAD_DIM + col]);
        }
    }
    cp_async_commit(); cp_async_wait_all(); __syncthreads();

    // Output accumulators
    float o_acc[2][16][2];
    float rowmax[2] = {-FLT_MAX, -FLT_MAX};
    float rowsum[2] = {0.0f, 0.0f};
    #pragma unroll
    for (int rh = 0; rh < 2; rh++)
        #pragma unroll
        for (int nt = 0; nt < 16; nt++)
            o_acc[rh][nt][0] = o_acc[rh][nt][1] = 0.0f;

    const int num_kv = (Skv + BLOCK_N - 1) / BLOCK_N;

    auto load_kv = [&](int blk, int stage) {
        int ns = blk * BLOCK_N;
        for (int i = tid; i < KV_ELEMS / 8; i += BLOCK_SIZE) {
            int row = (i * 8) / HEAD_DIM;
            int col = (i * 8) % HEAD_DIM;
            int kvr = ns + row;
            int sw_idx = swizzle_16B_idx(row, col, SMEM_STRIDE);
            if (kvr < Skv) {
                cp_async_16B(&k_s[stage][sw_idx], &k_ptr[kvr * HEAD_DIM + col]);
                cp_async_16B(&v_s[stage][sw_idx], &v_ptr[kvr * HEAD_DIM + col]);
            } else {
                for (int j = 0; j < 8; j++) {
                    k_s[stage][sw_idx + j] = __float2bfloat16(0.0f);
                    v_s[stage][sw_idx + j] = __float2bfloat16(0.0f);
                }
            }
        }
        cp_async_commit();
    };

    load_kv(0, 0);

    for (int kv = 0; kv < num_kv; kv++) {
        int cs = kv % 2;
        if (kv + 1 < num_kv) load_kv(kv + 1, 1 - cs);
        if (kv + 1 < num_kv) cp_async_wait_one(); else cp_async_wait_all();
        __syncthreads();

        // Q@K^T via MMA with swizzled reads
        float s_acc[2][8][2];
        #pragma unroll
        for (int rh = 0; rh < 2; rh++)
            #pragma unroll
            for (int nt = 0; nt < 8; nt++)
                s_acc[rh][nt][0] = s_acc[rh][nt][1] = 0.0f;

        #pragma unroll
        for (int ki = 0; ki < HEAD_DIM / MMA_K; ki++) {
            uint32_t q_frag[4];
            load_A_frag_swizzled(q_frag, q_s, warp * WARP_M, ki * MMA_K, SMEM_STRIDE, lane);

            #pragma unroll
            for (int ni = 0; ni < BLOCK_N / MMA_N; ni++) {
                uint32_t k_frag[2];
                load_B_frag_swizzled(k_frag, k_s[cs], ni * MMA_N, ki * MMA_K, SMEM_STRIDE, lane);

                float tile[4] = {s_acc[0][ni][0], s_acc[0][ni][1], s_acc[1][ni][0], s_acc[1][ni][1]};
                mma_m16n8k16(tile, q_frag, k_frag);
                s_acc[0][ni][0] = tile[0]; s_acc[0][ni][1] = tile[1];
                s_acc[1][ni][0] = tile[2]; s_acc[1][ni][1] = tile[3];
            }
        }

        // Scale + mask invalid KV
        int kv_start = kv * BLOCK_N;
        #pragma unroll
        for (int rh = 0; rh < 2; rh++)
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                s_acc[rh][nt][0] *= scale;
                s_acc[rh][nt][1] *= scale;
                int kv_idx0 = kv_start + nt * MMA_N + t * 2;
                int kv_idx1 = kv_idx0 + 1;
                if (kv_idx0 >= Skv) s_acc[rh][nt][0] = -FLT_MAX;
                if (kv_idx1 >= Skv) s_acc[rh][nt][1] = -FLT_MAX;
            }

        // Online softmax with cross-thread reduction
        for (int rh = 0; rh < 2; rh++) {
            float thread_max = rowmax[rh];
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                thread_max = fmaxf(thread_max, s_acc[rh][nt][0]);
                thread_max = fmaxf(thread_max, s_acc[rh][nt][1]);
            }
            float new_max = thread_max;
            new_max = fmaxf(new_max, __shfl_xor_sync(0xffffffff, new_max, 1));
            new_max = fmaxf(new_max, __shfl_xor_sync(0xffffffff, new_max, 2));

            float rescale = __expf(rowmax[rh] - new_max);
            rowsum[rh] *= rescale;
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                o_acc[rh][nt][0] *= rescale;
                o_acc[rh][nt][1] *= rescale;
            }

            float local_sum = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                s_acc[rh][nt][0] = __expf(s_acc[rh][nt][0] - new_max);
                s_acc[rh][nt][1] = __expf(s_acc[rh][nt][1] - new_max);
                local_sum += s_acc[rh][nt][0] + s_acc[rh][nt][1];
            }
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, 1);
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, 2);
            rowsum[rh] += local_sum;
            rowmax[rh] = new_max;
        }

        // Write P to SMEM (P doesn't need swizzle — different layout)
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int col0 = nt * MMA_N + t * 2;
            int row0 = warp * WARP_M + g;
            int row1 = warp * WARP_M + g + 8;
            p_s[row0 * BLOCK_N + col0]     = __float2bfloat16(s_acc[0][nt][0]);
            p_s[row0 * BLOCK_N + col0 + 1] = __float2bfloat16(s_acc[0][nt][1]);
            p_s[row1 * BLOCK_N + col0]     = __float2bfloat16(s_acc[1][nt][0]);
            p_s[row1 * BLOCK_N + col0 + 1] = __float2bfloat16(s_acc[1][nt][1]);
        }
        __syncthreads();

        // P@V via MMA — P uses non-swizzled layout, V uses swizzled
        #pragma unroll
        for (int ki = 0; ki < BLOCK_N / MMA_K; ki++) {
            uint32_t p_frag[4];
            // P is not swizzled — use regular load
            {
                int pg = lane / 4, pt = lane % 4;
                int pr0 = warp * WARP_M + pg;
                int pr1 = warp * WARP_M + pg + 8;
                int pc0 = ki * MMA_K + pt * 2;
                int pc1 = ki * MMA_K + pt * 2 + 8;
                p_frag[0] = pack2(p_s[pr0 * BLOCK_N + pc0],     p_s[pr0 * BLOCK_N + pc0 + 1]);
                p_frag[1] = pack2(p_s[pr1 * BLOCK_N + pc0],     p_s[pr1 * BLOCK_N + pc0 + 1]);
                p_frag[2] = pack2(p_s[pr0 * BLOCK_N + pc1],     p_s[pr0 * BLOCK_N + pc1 + 1]);
                p_frag[3] = pack2(p_s[pr1 * BLOCK_N + pc1],     p_s[pr1 * BLOCK_N + pc1 + 1]);
            }

            #pragma unroll
            for (int di = 0; di < HEAD_DIM / MMA_N; di++) {
                // V fragment with swizzle
                uint32_t v_frag[2];
                int vk0 = ki * MMA_K + t * 2;
                int vk1 = ki * MMA_K + t * 2 + 8;
                int vn = di * MMA_N + g;

                v_frag[0] = pack2(v_s[cs][swizzle_idx(vk0, vn, SMEM_STRIDE)],
                                  v_s[cs][swizzle_idx(vk0 + 1, vn, SMEM_STRIDE)]);
                v_frag[1] = pack2(v_s[cs][swizzle_idx(vk1, vn, SMEM_STRIDE)],
                                  v_s[cs][swizzle_idx(vk1 + 1, vn, SMEM_STRIDE)]);

                float o_tile[4] = {o_acc[0][di][0], o_acc[0][di][1],
                                   o_acc[1][di][0], o_acc[1][di][1]};
                mma_m16n8k16(o_tile, p_frag, v_frag);
                o_acc[0][di][0] = o_tile[0]; o_acc[0][di][1] = o_tile[1];
                o_acc[1][di][0] = o_tile[2]; o_acc[1][di][1] = o_tile[3];
            }
        }

        __syncthreads();
    }

    // Normalize and write
    __nv_bfloat16* o_ptr = O + head * Sq * HEAD_DIM;
    for (int rh = 0; rh < 2; rh++) {
        float inv = (rowsum[rh] > 0.0f) ? 1.0f / rowsum[rh] : 0.0f;
        int row = m_start + warp * WARP_M + g + rh * 8;
        if (row < Sq) {
            #pragma unroll
            for (int di = 0; di < 16; di++) {
                int col0 = di * MMA_N + t * 2;
                o_ptr[row * HEAD_DIM + col0]     = __float2bfloat16(o_acc[rh][di][0] * inv);
                o_ptr[row * HEAD_DIM + col0 + 1] = __float2bfloat16(o_acc[rh][di][1] * inv);
            }
        }
    }

    if (LSE && t == 0) {
        for (int rh = 0; rh < 2; rh++) {
            int row = m_start + warp * WARP_M + g + rh * 8;
            if (row < Sq) LSE[head * Sq + row] = rowmax[rh] + logf(rowsum[rh]);
        }
    }
}

extern "C" void sm120_flash_attn_forward(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    int batch, int Hq, int Hkv, int Sq, int Skv, int hd, cudaStream_t stream
) {
    float sc = 1.0f / sqrtf((float)hd);
    dim3 grid((Sq + BLOCK_M - 1) / BLOCK_M, batch * Hq);
    cudaFuncSetAttribute(sm120_flash_attn_fwd_v5,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    sm120_flash_attn_fwd_v5<<<grid, BLOCK_SIZE, SMEM_BYTES, stream>>>(
        Q, K, V, O, L, Sq, Skv, Hq, Hkv, sc);
}
