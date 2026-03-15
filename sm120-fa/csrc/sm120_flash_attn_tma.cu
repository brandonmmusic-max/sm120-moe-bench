/**
 * SM120 Flash Attention — TMA-based kernel
 *
 * Uses cp.async.bulk for global→SMEM copies instead of per-thread cp.async.
 * One thread issues the bulk copy for an entire KV tile, freeing other threads
 * to overlap compute with memory transfer.
 *
 * Key benefits:
 *   - Higher memory bandwidth utilization
 *   - Better compute/memory overlap (only 1 thread stalls on issue)
 *   - Mbarrier-based synchronization (lighter than __syncthreads)
 *
 * BLOCK_M=64, BLOCK_N=64, HEAD_DIM=128, 4 warps
 * Uses swizzled SMEM layout (same as v5)
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

#define SMEM_STRIDE HEAD_DIM
#define Q_ELEMS (BLOCK_M * SMEM_STRIDE)
#define KV_ELEMS (BLOCK_N * SMEM_STRIDE)
#define P_ELEMS (BLOCK_M * BLOCK_N)

// SMEM: Q(16KB) + K×2(32KB) + V×2(32KB) + P(8KB) + mbarriers(64B) = ~88KB
#define KV_TILE_BYTES (BLOCK_N * HEAD_DIM * 2)  // 16KB per K or V tile
#define SMEM_BYTES ((Q_ELEMS + NUM_STAGES * 2 * KV_ELEMS + P_ELEMS) * 2 + 128)

__device__ __forceinline__ int swizzle_idx(int row, int col, int stride) {
    int col_bytes = col * 2;
    int swizzled_bytes = col_bytes ^ ((row & 7) << 5);
    return row * stride + swizzled_bytes / 2;
}

__device__ __forceinline__ uint32_t pack2(const __nv_bfloat16& a, const __nv_bfloat16& b) {
    uint32_t r;
    asm("mov.b32 %0, {%1, %2};" : "=r"(r) : "h"(*(const uint16_t*)&a), "h"(*(const uint16_t*)&b));
    return r;
}

__device__ __forceinline__ void mma_m16n8k16(float c[4], uint32_t a[4], uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\n"
        "    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

// ============================================================================
// TMA helpers
// ============================================================================

// Initialize mbarrier
__device__ __forceinline__ void mbar_init(uint64_t* mbar, int arrival_count) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(addr), "r"(arrival_count));
}

// Set expected TX bytes on mbarrier
__device__ __forceinline__ void mbar_expect_tx(uint64_t* mbar, int nbytes) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(addr), "r"(nbytes));
}

// Issue bulk copy with mbarrier tracking
__device__ __forceinline__ void tma_bulk_copy(
    void* smem_dst, const void* gmem_src, int nbytes, uint64_t* mbar
) {
    uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
        :: "r"(dst), "l"(gmem_src), "r"(nbytes), "r"(mbar_addr)
    );
}

// Wait for mbarrier (spin on try_wait)
__device__ __forceinline__ void mbar_wait(uint64_t* mbar, int phase) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "MBAR_WAIT:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra MBAR_WAIT;\n"
        "}\n"
        :: "r"(addr), "r"(phase)
    );
}

// Invalidate mbarrier for next phase
__device__ __forceinline__ void mbar_arrive(uint64_t* mbar) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" :: "r"(addr));
}

// ============================================================================
// Main TMA kernel
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
sm120_flash_attn_tma(
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
    const int g = lane / 4, t = lane % 4;

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

    // Mbarriers for KV loading (one per stage)
    uint64_t* mbar_kv = reinterpret_cast<uint64_t*>(
        reinterpret_cast<char*>(p_s) + P_ELEMS * 2);

    const int num_kv = (Skv + BLOCK_N - 1) / BLOCK_N;

    // ========================================================================
    // Load Q via per-thread cp.async (Q is loaded once, not performance-critical)
    // ========================================================================
    for (int i = tid; i < Q_ELEMS; i += BLOCK_SIZE)
        q_s[i] = __float2bfloat16(0.0f);
    __syncthreads();

    // Q doesn't need swizzle for TMA since we read it with swizzle_idx
    // Load Q row-by-row with swizzle
    for (int row = 0; row < BLOCK_M; row++) {
        int qr = m_start + row;
        if (qr < Sq) {
            for (int col = tid; col < HEAD_DIM; col += BLOCK_SIZE) {
                int sw = swizzle_idx(row, col, SMEM_STRIDE);
                q_s[sw] = q_ptr[qr * HEAD_DIM + col];
            }
        }
    }
    __syncthreads();

    // ========================================================================
    // Initialize mbarriers
    // ========================================================================
    if (tid == 0) {
        for (int s = 0; s < NUM_STAGES; s++) {
            mbar_init(&mbar_kv[s], 1);  // 1 arrival expected
        }
    }
    __syncthreads();

    // ========================================================================
    // Load first KV via TMA bulk copy
    // We can't use TMA with swizzled layout directly — TMA copies contiguous bytes.
    // So we load K/V without swizzle, then the compute path reads with non-swizzled idx.
    // OR: we load contiguously and apply swizzle during compute reads.
    //
    // Simplest: load K/V row-by-row via TMA bulk (each row = 256 bytes = contiguous)
    // One TMA per row, thread 0 issues all.
    // ========================================================================

    // Load KV block 0
    if (tid == 0) {
        mbar_expect_tx(&mbar_kv[0], 2 * KV_TILE_BYTES);  // K + V

        for (int row = 0; row < BLOCK_N; row++) {
            int kvr = row;
            if (kvr < Skv) {
                // K row
                tma_bulk_copy(&k_s[0][row * HEAD_DIM], &k_ptr[kvr * HEAD_DIM],
                              HEAD_DIM * 2, &mbar_kv[0]);
                // V row
                tma_bulk_copy(&v_s[0][row * HEAD_DIM], &v_ptr[kvr * HEAD_DIM],
                              HEAD_DIM * 2, &mbar_kv[0]);
            }
        }
    }
    // Wait for first KV block
    mbar_wait(&mbar_kv[0], 0);
    __syncthreads();

    // Output accumulators
    float o_acc[2][16][2];
    float rowmax[2] = {-FLT_MAX, -FLT_MAX};
    float rowsum[2] = {0.0f, 0.0f};
    #pragma unroll
    for (int rh = 0; rh < 2; rh++)
        #pragma unroll
        for (int nt = 0; nt < 16; nt++)
            o_acc[rh][nt][0] = o_acc[rh][nt][1] = 0.0f;

    for (int kv = 0; kv < num_kv; kv++) {
        int cs = kv % 2;
        int ns = 1 - cs;

        // Prefetch next KV block via TMA (thread 0 only)
        if (kv + 1 < num_kv) {
            if (tid == 0) {
                // Re-init mbarrier for next stage
                mbar_init(&mbar_kv[ns], 1);
            }
            __syncwarp();
            if (tid == 0) {
                int next_start = (kv + 1) * BLOCK_N;
                mbar_expect_tx(&mbar_kv[ns], 2 * KV_TILE_BYTES);

                for (int row = 0; row < BLOCK_N; row++) {
                    int kvr = next_start + row;
                    if (kvr < Skv) {
                        tma_bulk_copy(&k_s[ns][row * HEAD_DIM], &k_ptr[kvr * HEAD_DIM],
                                      HEAD_DIM * 2, &mbar_kv[ns]);
                        tma_bulk_copy(&v_s[ns][row * HEAD_DIM], &v_ptr[kvr * HEAD_DIM],
                                      HEAD_DIM * 2, &mbar_kv[ns]);
                    }
                }
            }
        }

        // ================================================================
        // Compute Q@K^T on current KV (no swizzle on K/V — loaded contiguously)
        // ================================================================
        float s_acc[2][8][2];
        #pragma unroll
        for (int rh = 0; rh < 2; rh++)
            #pragma unroll
            for (int nt = 0; nt < 8; nt++)
                s_acc[rh][nt][0] = s_acc[rh][nt][1] = 0.0f;

        #pragma unroll
        for (int ki = 0; ki < HEAD_DIM / MMA_K; ki++) {
            // Q fragment (swizzled)
            uint32_t q_frag[4];
            {
                int r0 = warp * WARP_M + g, r1 = r0 + 8;
                int c0 = ki * MMA_K + t * 2, c1 = c0 + 8;
                q_frag[0] = pack2(q_s[swizzle_idx(r0,c0,SMEM_STRIDE)],
                                  q_s[swizzle_idx(r0,c0+1,SMEM_STRIDE)]);
                q_frag[1] = pack2(q_s[swizzle_idx(r1,c0,SMEM_STRIDE)],
                                  q_s[swizzle_idx(r1,c0+1,SMEM_STRIDE)]);
                q_frag[2] = pack2(q_s[swizzle_idx(r0,c1,SMEM_STRIDE)],
                                  q_s[swizzle_idx(r0,c1+1,SMEM_STRIDE)]);
                q_frag[3] = pack2(q_s[swizzle_idx(r1,c1,SMEM_STRIDE)],
                                  q_s[swizzle_idx(r1,c1+1,SMEM_STRIDE)]);
            }

            #pragma unroll
            for (int ni = 0; ni < BLOCK_N / MMA_N; ni++) {
                // K fragment (NOT swizzled — TMA loads contiguous)
                uint32_t k_frag[2];
                int kn = ni * MMA_N + g;
                int kk0 = ki * MMA_K + t * 2, kk1 = kk0 + 8;
                k_frag[0] = pack2(k_s[cs][kn * HEAD_DIM + kk0],
                                  k_s[cs][kn * HEAD_DIM + kk0 + 1]);
                k_frag[1] = pack2(k_s[cs][kn * HEAD_DIM + kk1],
                                  k_s[cs][kn * HEAD_DIM + kk1 + 1]);

                float tile[4] = {s_acc[0][ni][0], s_acc[0][ni][1],
                                 s_acc[1][ni][0], s_acc[1][ni][1]};
                mma_m16n8k16(tile, q_frag, k_frag);
                s_acc[0][ni][0] = tile[0]; s_acc[0][ni][1] = tile[1];
                s_acc[1][ni][0] = tile[2]; s_acc[1][ni][1] = tile[3];
            }
        }

        // Scale + mask
        int kv_start = kv * BLOCK_N;
        #pragma unroll
        for (int rh = 0; rh < 2; rh++)
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                s_acc[rh][nt][0] *= scale;
                s_acc[rh][nt][1] *= scale;
                int kv_idx0 = kv_start + nt * MMA_N + t * 2;
                if (kv_idx0 >= Skv) s_acc[rh][nt][0] = -FLT_MAX;
                if (kv_idx0 + 1 >= Skv) s_acc[rh][nt][1] = -FLT_MAX;
            }

        // Softmax with cross-thread reduction
        for (int rh = 0; rh < 2; rh++) {
            float tm = rowmax[rh];
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                tm = fmaxf(tm, s_acc[rh][nt][0]);
                tm = fmaxf(tm, s_acc[rh][nt][1]);
            }
            float nm = tm;
            nm = fmaxf(nm, __shfl_xor_sync(0xffffffff, nm, 1));
            nm = fmaxf(nm, __shfl_xor_sync(0xffffffff, nm, 2));
            float rs = __expf(rowmax[rh] - nm);
            rowsum[rh] *= rs;
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                o_acc[rh][nt][0] *= rs; o_acc[rh][nt][1] *= rs;
            }
            float ls = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                s_acc[rh][nt][0] = __expf(s_acc[rh][nt][0] - nm);
                s_acc[rh][nt][1] = __expf(s_acc[rh][nt][1] - nm);
                ls += s_acc[rh][nt][0] + s_acc[rh][nt][1];
            }
            ls += __shfl_xor_sync(0xffffffff, ls, 1);
            ls += __shfl_xor_sync(0xffffffff, ls, 2);
            rowsum[rh] += ls;
            rowmax[rh] = nm;
        }

        // P staging
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int col0 = nt * MMA_N + t * 2;
            int row0 = warp * WARP_M + g, row1 = row0 + 8;
            p_s[row0 * BLOCK_N + col0]     = __float2bfloat16(s_acc[0][nt][0]);
            p_s[row0 * BLOCK_N + col0 + 1] = __float2bfloat16(s_acc[0][nt][1]);
            p_s[row1 * BLOCK_N + col0]     = __float2bfloat16(s_acc[1][nt][0]);
            p_s[row1 * BLOCK_N + col0 + 1] = __float2bfloat16(s_acc[1][nt][1]);
        }
        __syncwarp();

        // P@V (V NOT swizzled — contiguous from TMA)
        #pragma unroll
        for (int ki = 0; ki < BLOCK_N / MMA_K; ki++) {
            uint32_t p_frag[4];
            {
                int pr0 = warp * WARP_M + g, pr1 = pr0 + 8;
                int pc0 = ki * MMA_K + t * 2, pc1 = pc0 + 8;
                p_frag[0] = pack2(p_s[pr0*BLOCK_N+pc0], p_s[pr0*BLOCK_N+pc0+1]);
                p_frag[1] = pack2(p_s[pr1*BLOCK_N+pc0], p_s[pr1*BLOCK_N+pc0+1]);
                p_frag[2] = pack2(p_s[pr0*BLOCK_N+pc1], p_s[pr0*BLOCK_N+pc1+1]);
                p_frag[3] = pack2(p_s[pr1*BLOCK_N+pc1], p_s[pr1*BLOCK_N+pc1+1]);
            }

            #pragma unroll
            for (int di = 0; di < HEAD_DIM / MMA_N; di++) {
                uint32_t v_frag[2];
                int vk0 = ki * MMA_K + t * 2, vk1 = vk0 + 8;
                int vn = di * MMA_N + g;
                // V is NOT swizzled — direct row-major access
                v_frag[0] = pack2(v_s[cs][vk0 * HEAD_DIM + vn],
                                  v_s[cs][(vk0+1) * HEAD_DIM + vn]);
                v_frag[1] = pack2(v_s[cs][vk1 * HEAD_DIM + vn],
                                  v_s[cs][(vk1+1) * HEAD_DIM + vn]);

                float ot[4] = {o_acc[0][di][0], o_acc[0][di][1],
                               o_acc[1][di][0], o_acc[1][di][1]};
                mma_m16n8k16(ot, p_frag, v_frag);
                o_acc[0][di][0] = ot[0]; o_acc[0][di][1] = ot[1];
                o_acc[1][di][0] = ot[2]; o_acc[1][di][1] = ot[3];
            }
        }

        // Wait for next KV to be ready before next iteration
        if (kv + 1 < num_kv) {
            mbar_wait(&mbar_kv[ns], 0);
        }
        __syncthreads();
    }

    // Write output
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
    cudaFuncSetAttribute(sm120_flash_attn_tma,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    sm120_flash_attn_tma<<<grid, BLOCK_SIZE, SMEM_BYTES, stream>>>(
        Q, K, V, O, L, Sq, Skv, Hq, Hkv, sc);
}
