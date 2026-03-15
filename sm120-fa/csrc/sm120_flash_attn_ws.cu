/**
 * SM120 Flash Attention — Warp-Specialized Architecture
 *
 * 4 warps split into 2 roles:
 *   Warp 0-1: PRODUCERS — load K/V from global→SMEM via cp.async
 *   Warp 2-3: CONSUMERS — compute Q@K^T, softmax, P@V via MMA
 *
 * Key insight: producers and consumers operate on DIFFERENT KV blocks
 * simultaneously. While consumers compute on block N, producers load
 * block N+1. This overlaps memory and compute.
 *
 * Each consumer warp handles 16 rows of M (same as v5).
 * BLOCK_M=32 (2 consumer warps × 16 rows), BLOCK_N=64, HEAD_DIM=128.
 *
 * Synchronization: named barriers (bar.sync) separate producer/consumer phases.
 *
 * Expected improvement: ~50-100% over v5 by overlapping memory latency
 * with tensor core compute.
 *
 * SMEM: Q(32×128=8KB) + K×3(64×128×3=48KB) + V×3(48KB) = too big
 *       Q(8KB) + K×2(32KB) + V×2(32KB) + P(32×64=4KB) = 76KB ✓
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define BLOCK_M 32       // Reduced from 64: 2 consumer warps × 16
#define BLOCK_N 64
#define HEAD_DIM 128
#define NUM_STAGES 2
#define WARP_SIZE 32
#define NUM_WARPS 4       // 2 producers + 2 consumers
#define NUM_PRODUCERS 2
#define NUM_CONSUMERS 2
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE)  // 128 threads

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_M MMA_M

#define SMEM_STRIDE HEAD_DIM
#define Q_ELEMS (BLOCK_M * SMEM_STRIDE)
#define KV_ELEMS (BLOCK_N * SMEM_STRIDE)
#define P_ELEMS (BLOCK_M * BLOCK_N)
// Q(8KB) + K×2(32KB) + V×2(32KB) + P(4KB) = 76KB < 99KB ✓
#define SMEM_BYTES ((Q_ELEMS + NUM_STAGES * 2 * KV_ELEMS + P_ELEMS) * 2)

// Swizzle
__device__ __forceinline__ int swizzle_idx(int row, int col, int stride) {
    int col_bytes = col * 2;
    int swizzled_bytes = col_bytes ^ ((row & 7) << 5);
    return row * stride + swizzled_bytes / 2;
}

__device__ __forceinline__ int swizzle_16B_idx(int row, int col, int stride) {
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

// Named barriers for warp specialization
// barrier 0: producer→consumer signal (KV data ready)
// barrier 1: consumer→producer signal (KV data consumed, safe to overwrite)
__device__ __forceinline__ void bar_arrive(int bar_id, int count) {
    asm volatile("bar.arrive %0, %1;\n" :: "r"(bar_id), "r"(count));
}
__device__ __forceinline__ void bar_wait(int bar_id, int count) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(bar_id), "r"(count));
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
// Main warp-specialized kernel
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
sm120_flash_attn_ws(
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

    const int is_producer = (warp < NUM_PRODUCERS);
    const int is_consumer = !is_producer;
    const int consumer_warp = warp - NUM_PRODUCERS;  // 0 or 1 for consumers

    const __nv_bfloat16* q_ptr = Q + head * Sq * HEAD_DIM;
    const __nv_bfloat16* k_ptr = K + kv_head * Skv * HEAD_DIM;
    const __nv_bfloat16* v_ptr = V + kv_head * Skv * HEAD_DIM;

    extern __shared__ char smem[];
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* k_s[NUM_STAGES], *v_s[NUM_STAGES];
    k_s[0] = q_s + Q_ELEMS;
    v_s[0] = k_s[0] + KV_ELEMS;
    k_s[1] = v_s[0] + KV_ELEMS;
    v_s[1] = k_s[1] + KV_ELEMS;
    __nv_bfloat16* p_s = v_s[1] + KV_ELEMS;

    const int num_kv = (Skv + BLOCK_N - 1) / BLOCK_N;

    // ========================================================================
    // Phase 0: All warps cooperate to load Q
    // ========================================================================
    for (int i = tid; i < Q_ELEMS; i += BLOCK_SIZE)
        q_s[i] = __float2bfloat16(0.0f);
    __syncthreads();

    for (int i = tid; i < Q_ELEMS / 8; i += BLOCK_SIZE) {
        int row = (i * 8) / HEAD_DIM;
        int col = (i * 8) % HEAD_DIM;
        int qr = m_start + row;
        if (qr < Sq)
            cp_async_16B(&q_s[swizzle_16B_idx(row, col, SMEM_STRIDE)],
                          &q_ptr[qr * HEAD_DIM + col]);
    }
    cp_async_commit(); cp_async_wait_all();
    __syncthreads();

    // ========================================================================
    // Warp-specialized main loop
    // ========================================================================

    if (is_producer) {
        // ====================================================================
        // PRODUCER WARPS (0-1): load KV blocks
        // ====================================================================
        int prod_tid = warp * WARP_SIZE + lane;  // thread index within producers
        int prod_threads = NUM_PRODUCERS * WARP_SIZE;  // 64 producer threads

        for (int kv = 0; kv < num_kv; kv++) {
            int stage = kv % NUM_STAGES;
            int ns = kv * BLOCK_N;

            // Wait for consumers to finish with this stage's data
            if (kv >= NUM_STAGES) {
                bar_wait(1, BLOCK_SIZE);  // consumer→producer: safe to overwrite
            }

            // Load KV block
            for (int i = prod_tid; i < KV_ELEMS / 8; i += prod_threads) {
                int row = (i * 8) / HEAD_DIM;
                int col = (i * 8) % HEAD_DIM;
                int kvr = ns + row;
                int sw = swizzle_16B_idx(row, col, SMEM_STRIDE);
                if (kvr < Skv) {
                    cp_async_16B(&k_s[stage][sw], &k_ptr[kvr * HEAD_DIM + col]);
                    cp_async_16B(&v_s[stage][sw], &v_ptr[kvr * HEAD_DIM + col]);
                } else {
                    for (int j = 0; j < 8; j++) {
                        k_s[stage][sw + j] = __float2bfloat16(0.0f);
                        v_s[stage][sw + j] = __float2bfloat16(0.0f);
                    }
                }
            }
            cp_async_commit();
            cp_async_wait_all();

            // Signal consumers: KV data ready
            bar_arrive(0, BLOCK_SIZE);  // producer→consumer: data ready
        }

    } else {
        // ====================================================================
        // CONSUMER WARPS (2-3): compute Q@K^T, softmax, P@V
        // Each consumer warp handles 16 rows of M
        // ====================================================================

        float o_acc[2][16][2];
        float rowmax[2] = {-FLT_MAX, -FLT_MAX};
        float rowsum[2] = {0.0f, 0.0f};
        #pragma unroll
        for (int rh = 0; rh < 2; rh++)
            #pragma unroll
            for (int nt = 0; nt < 16; nt++)
                o_acc[rh][nt][0] = o_acc[rh][nt][1] = 0.0f;

        for (int kv = 0; kv < num_kv; kv++) {
            int cs = kv % NUM_STAGES;

            // Wait for producers to finish loading
            bar_wait(0, BLOCK_SIZE);  // producer→consumer: data ready

            // Q@K^T
            float s_acc[2][8][2];
            #pragma unroll
            for (int rh = 0; rh < 2; rh++)
                #pragma unroll
                for (int nt = 0; nt < 8; nt++)
                    s_acc[rh][nt][0] = s_acc[rh][nt][1] = 0.0f;

            #pragma unroll
            for (int ki = 0; ki < HEAD_DIM / MMA_K; ki++) {
                uint32_t q_frag[4];
                {
                    int r0 = consumer_warp * WARP_M + g;
                    int r1 = consumer_warp * WARP_M + g + 8;
                    int c0 = ki * MMA_K + t * 2;
                    int c1 = ki * MMA_K + t * 2 + 8;
                    q_frag[0] = pack2(q_s[swizzle_idx(r0, c0, SMEM_STRIDE)],
                                      q_s[swizzle_idx(r0, c0+1, SMEM_STRIDE)]);
                    q_frag[1] = pack2(q_s[swizzle_idx(r1, c0, SMEM_STRIDE)],
                                      q_s[swizzle_idx(r1, c0+1, SMEM_STRIDE)]);
                    q_frag[2] = pack2(q_s[swizzle_idx(r0, c1, SMEM_STRIDE)],
                                      q_s[swizzle_idx(r0, c1+1, SMEM_STRIDE)]);
                    q_frag[3] = pack2(q_s[swizzle_idx(r1, c1, SMEM_STRIDE)],
                                      q_s[swizzle_idx(r1, c1+1, SMEM_STRIDE)]);
                }

                #pragma unroll
                for (int ni = 0; ni < BLOCK_N / MMA_N; ni++) {
                    uint32_t k_frag[2];
                    int kn = ni * MMA_N + g;
                    int kk0 = ki * MMA_K + t * 2;
                    int kk1 = ki * MMA_K + t * 2 + 8;
                    k_frag[0] = pack2(k_s[cs][swizzle_idx(kn, kk0, SMEM_STRIDE)],
                                      k_s[cs][swizzle_idx(kn, kk0+1, SMEM_STRIDE)]);
                    k_frag[1] = pack2(k_s[cs][swizzle_idx(kn, kk1, SMEM_STRIDE)],
                                      k_s[cs][swizzle_idx(kn, kk1+1, SMEM_STRIDE)]);

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

            // Online softmax
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
                    o_acc[rh][nt][0] *= rs;
                    o_acc[rh][nt][1] *= rs;
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

            // P staging (consumer warps write their own rows)
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                int col0 = nt * MMA_N + t * 2;
                int row0 = consumer_warp * WARP_M + g;
                int row1 = consumer_warp * WARP_M + g + 8;
                p_s[row0 * BLOCK_N + col0]     = __float2bfloat16(s_acc[0][nt][0]);
                p_s[row0 * BLOCK_N + col0 + 1] = __float2bfloat16(s_acc[0][nt][1]);
                p_s[row1 * BLOCK_N + col0]     = __float2bfloat16(s_acc[1][nt][0]);
                p_s[row1 * BLOCK_N + col0 + 1] = __float2bfloat16(s_acc[1][nt][1]);
            }
            __syncwarp();  // P is warp-private — syncwarp sufficient

            // P@V
            #pragma unroll
            for (int ki = 0; ki < BLOCK_N / MMA_K; ki++) {
                uint32_t p_frag[4];
                {
                    int pr0 = consumer_warp * WARP_M + g;
                    int pr1 = consumer_warp * WARP_M + g + 8;
                    int pc0 = ki * MMA_K + t * 2;
                    int pc1 = ki * MMA_K + t * 2 + 8;
                    p_frag[0] = pack2(p_s[pr0*BLOCK_N+pc0], p_s[pr0*BLOCK_N+pc0+1]);
                    p_frag[1] = pack2(p_s[pr1*BLOCK_N+pc0], p_s[pr1*BLOCK_N+pc0+1]);
                    p_frag[2] = pack2(p_s[pr0*BLOCK_N+pc1], p_s[pr0*BLOCK_N+pc1+1]);
                    p_frag[3] = pack2(p_s[pr1*BLOCK_N+pc1], p_s[pr1*BLOCK_N+pc1+1]);
                }

                #pragma unroll
                for (int di = 0; di < HEAD_DIM / MMA_N; di++) {
                    uint32_t v_frag[2];
                    int vk0 = ki * MMA_K + t * 2;
                    int vk1 = ki * MMA_K + t * 2 + 8;
                    int vn = di * MMA_N + g;

                    v_frag[0] = pack2(v_s[cs][swizzle_idx(vk0, vn, SMEM_STRIDE)],
                                      v_s[cs][swizzle_idx(vk0+1, vn, SMEM_STRIDE)]);
                    v_frag[1] = pack2(v_s[cs][swizzle_idx(vk1, vn, SMEM_STRIDE)],
                                      v_s[cs][swizzle_idx(vk1+1, vn, SMEM_STRIDE)]);

                    float ot[4] = {o_acc[0][di][0], o_acc[0][di][1],
                                   o_acc[1][di][0], o_acc[1][di][1]};
                    mma_m16n8k16(ot, p_frag, v_frag);
                    o_acc[0][di][0] = ot[0]; o_acc[0][di][1] = ot[1];
                    o_acc[1][di][0] = ot[2]; o_acc[1][di][1] = ot[3];
                }
            }

            // Signal producers: done with this stage's KV data
            bar_arrive(1, BLOCK_SIZE);
        }

        // Write output (consumers only)
        __nv_bfloat16* o_ptr = O + head * Sq * HEAD_DIM;
        for (int rh = 0; rh < 2; rh++) {
            float inv = (rowsum[rh] > 0.0f) ? 1.0f / rowsum[rh] : 0.0f;
            int row = m_start + consumer_warp * WARP_M + g + rh * 8;
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
                int row = m_start + consumer_warp * WARP_M + g + rh * 8;
                if (row < Sq) LSE[head * Sq + row] = rowmax[rh] + logf(rowsum[rh]);
            }
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
    cudaFuncSetAttribute(sm120_flash_attn_ws,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    cudaFuncSetAttribute(sm120_flash_attn_ws,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);
    sm120_flash_attn_ws<<<grid, BLOCK_SIZE, SMEM_BYTES, stream>>>(
        Q, K, V, O, L, Sq, Skv, Hq, Hkv, sc);
}
