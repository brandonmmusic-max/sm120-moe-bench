/**
 * SM120 Flash Attention — Prefill with Causal Masking
 *
 * Based on v6 structure (cp.async, software swizzle) with:
 *   - Single-pass full HEAD_DIM (no 2-pass register optimization)
 *   - Template causal masking with early KV block termination
 *   - Double-buffered KV pipeline
 *   - BM=64, BN=64, HD=128, 4 warps (128 threads)
 *
 * Layout: Q[batch*Hq, Sq, HD], K[batch*Hkv, Skv, HD], V[batch*Hkv, Skv, HD]
 * Output: O[batch*Hq, Sq, HD], LSE[batch*Hq, Sq] (optional)
 *
 * NOTE: No -use_fast_math (per memory: causes MTP acceptance regression)
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
// Q(16KB) + 2×K(16KB) + 2×V(16KB) + P(8KB) = 88KB
#define SMEM_BYTES ((Q_ELEMS + NUM_STAGES * 2 * KV_ELEMS + P_ELEMS) * 2)

// Software swizzle: 256-byte rows (128 BF16 = 256 bytes), XOR bits[7:5] with row[2:0]
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

__device__ __forceinline__ void cp_async_16B(void* s, const void* g) {
    uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(s));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(g));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n"); }
__device__ __forceinline__ void cp_async_wait_one() { asm volatile("cp.async.wait_group 1;\n"); }

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

// Load A fragment (Q or P) from swizzled SMEM
__device__ __forceinline__ void load_A_frag(
    uint32_t frag[4], const __nv_bfloat16* smem,
    int row_off, int k_off, int stride, int lane
) {
    int g = lane / 4, t = lane % 4;
    int r0 = row_off + g, r1 = row_off + g + 8;
    int c0 = k_off + t * 2, c1 = k_off + t * 2 + 8;
    frag[0] = pack2(smem[swizzle_idx(r0, c0, stride)], smem[swizzle_idx(r0, c0+1, stride)]);
    frag[1] = pack2(smem[swizzle_idx(r1, c0, stride)], smem[swizzle_idx(r1, c0+1, stride)]);
    frag[2] = pack2(smem[swizzle_idx(r0, c1, stride)], smem[swizzle_idx(r0, c1+1, stride)]);
    frag[3] = pack2(smem[swizzle_idx(r1, c1, stride)], smem[swizzle_idx(r1, c1+1, stride)]);
}

// Load B fragment (K or V) from swizzled SMEM
__device__ __forceinline__ void load_B_frag(
    uint32_t frag[2], const __nv_bfloat16* smem,
    int n_off, int k_off, int stride, int lane
) {
    int g = lane / 4, t = lane % 4;
    int n = n_off + g, k0 = k_off + t * 2, k1 = k_off + t * 2 + 8;
    frag[0] = pack2(smem[swizzle_idx(n, k0, stride)], smem[swizzle_idx(n, k0+1, stride)]);
    frag[1] = pack2(smem[swizzle_idx(n, k1, stride)], smem[swizzle_idx(n, k1+1, stride)]);
}


// ============================================================================
// Main prefill kernel
// Grid: (num_m_blocks, batch * Hq)
// Block: BLOCK_SIZE threads (4 warps)
// ============================================================================
template <bool CAUSAL>
__global__ void __launch_bounds__(BLOCK_SIZE, 2)
sm120_flash_prefill_fwd(
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

    // ---- Load Q into SMEM with software swizzle ----
    for (int i = tid; i < Q_ELEMS; i += BLOCK_SIZE)
        q_s[i] = __float2bfloat16(0.0f);
    __syncthreads();

    for (int i = tid; i < Q_ELEMS / 8; i += BLOCK_SIZE) {
        int row = (i * 8) / HEAD_DIM, col = (i * 8) % HEAD_DIM;
        int qr = m_start + row;
        if (qr < Sq) cp_async_16B(&q_s[swizzle_idx(row, col, SMEM_STRIDE)],
                                    &q_ptr[qr * HEAD_DIM + col]);
    }
    cp_async_commit(); cp_async_wait_all(); __syncthreads();

    // ---- Output accumulators: full HEAD_DIM ----
    float o_acc[2][HEAD_DIM / MMA_N][2];  // [2 rh][16 d-tiles][2 elems] = 64 floats
    float rowmax[2] = {-FLT_MAX, -FLT_MAX};
    float rowsum[2] = {0.0f, 0.0f};

    #pragma unroll
    for (int rh = 0; rh < 2; rh++)
        #pragma unroll
        for (int nt = 0; nt < HEAD_DIM / MMA_N; nt++)
            o_acc[rh][nt][0] = o_acc[rh][nt][1] = 0.0f;

    // ---- KV tile count (with causal early termination) ----
    int num_kv = (Skv + BLOCK_N - 1) / BLOCK_N;
    if (CAUSAL) {
        // Last KV block that could have valid (non-masked) positions
        int max_kv_needed = (m_start + BLOCK_M - 1 + BLOCK_N) / BLOCK_N;
        if (max_kv_needed < num_kv) num_kv = max_kv_needed;
    }

    // ---- KV loading lambda ----
    auto load_kv = [&](int blk, int stage) {
        int ns = blk * BLOCK_N;
        for (int i = tid; i < KV_ELEMS / 8; i += BLOCK_SIZE) {
            int row = (i * 8) / HEAD_DIM, col = (i * 8) % HEAD_DIM;
            int kvr = ns + row;
            int sw = swizzle_idx(row, col, SMEM_STRIDE);
            if (kvr < Skv) {
                cp_async_16B(&k_s[stage][sw], &k_ptr[kvr * HEAD_DIM + col]);
                cp_async_16B(&v_s[stage][sw], &v_ptr[kvr * HEAD_DIM + col]);
            } else {
                // Zero-fill OOB
                for (int j = 0; j < 8; j++) {
                    k_s[stage][sw + j] = __float2bfloat16(0.0f);
                    v_s[stage][sw + j] = __float2bfloat16(0.0f);
                }
            }
        }
        cp_async_commit();
    };

    // Prefetch first KV tile
    if (num_kv > 0) load_kv(0, 0);

    // ---- Main KV loop ----
    for (int kv = 0; kv < num_kv; kv++) {
        int cs = kv % 2;

        // Prefetch next KV tile
        if (kv + 1 < num_kv) load_kv(kv + 1, 1 - cs);

        // Wait for current tile
        if (kv + 1 < num_kv) cp_async_wait_one(); else cp_async_wait_all();
        __syncthreads();

        // ============ Q@K^T (full HEAD_DIM) ============
        float s_acc[2][BLOCK_N / MMA_N][2];
        #pragma unroll
        for (int rh = 0; rh < 2; rh++)
            #pragma unroll
            for (int nt = 0; nt < BLOCK_N / MMA_N; nt++)
                s_acc[rh][nt][0] = s_acc[rh][nt][1] = 0.0f;

        #pragma unroll
        for (int ki = 0; ki < HEAD_DIM / MMA_K; ki++) {
            uint32_t q_frag[4];
            load_A_frag(q_frag, q_s, warp * WARP_M, ki * MMA_K, SMEM_STRIDE, lane);

            #pragma unroll
            for (int ni = 0; ni < BLOCK_N / MMA_N; ni++) {
                uint32_t k_frag[2];
                load_B_frag(k_frag, k_s[cs], ni * MMA_N, ki * MMA_K, SMEM_STRIDE, lane);

                float tile[4] = {s_acc[0][ni][0], s_acc[0][ni][1],
                                 s_acc[1][ni][0], s_acc[1][ni][1]};
                mma_m16n8k16(tile, q_frag, k_frag);
                s_acc[0][ni][0] = tile[0]; s_acc[0][ni][1] = tile[1];
                s_acc[1][ni][0] = tile[2]; s_acc[1][ni][1] = tile[3];
            }
        }

        // ============ Scale + Mask ============
        int kv_start = kv * BLOCK_N;
        #pragma unroll
        for (int rh = 0; rh < 2; rh++) {
            #pragma unroll
            for (int nt = 0; nt < BLOCK_N / MMA_N; nt++) {
                s_acc[rh][nt][0] *= scale;
                s_acc[rh][nt][1] *= scale;

                int ki0 = kv_start + nt * MMA_N + t * 2;
                // OOB masking
                if (ki0 >= Skv) s_acc[rh][nt][0] = -FLT_MAX;
                if (ki0 + 1 >= Skv) s_acc[rh][nt][1] = -FLT_MAX;

                // Causal masking: mask if kv_pos > query_pos
                if (CAUSAL) {
                    int qi = m_start + warp * WARP_M + g + rh * 8;
                    if (ki0 > qi) s_acc[rh][nt][0] = -FLT_MAX;
                    if (ki0 + 1 > qi) s_acc[rh][nt][1] = -FLT_MAX;
                }
            }
        }

        // ============ Online Softmax ============
        for (int rh = 0; rh < 2; rh++) {
            float tm = rowmax[rh];
            #pragma unroll
            for (int nt = 0; nt < BLOCK_N / MMA_N; nt++) {
                tm = fmaxf(tm, s_acc[rh][nt][0]);
                tm = fmaxf(tm, s_acc[rh][nt][1]);
            }
            float nm = tm;
            nm = fmaxf(nm, __shfl_xor_sync(0xffffffff, nm, 1));
            nm = fmaxf(nm, __shfl_xor_sync(0xffffffff, nm, 2));

            float rs = __expf(rowmax[rh] - nm);
            rowsum[rh] *= rs;
            #pragma unroll
            for (int nt = 0; nt < HEAD_DIM / MMA_N; nt++) {
                o_acc[rh][nt][0] *= rs;
                o_acc[rh][nt][1] *= rs;
            }

            float ls = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < BLOCK_N / MMA_N; nt++) {
                s_acc[rh][nt][0] = __expf(s_acc[rh][nt][0] - nm);
                s_acc[rh][nt][1] = __expf(s_acc[rh][nt][1] - nm);
                ls += s_acc[rh][nt][0] + s_acc[rh][nt][1];
            }
            ls += __shfl_xor_sync(0xffffffff, ls, 1);
            ls += __shfl_xor_sync(0xffffffff, ls, 2);
            rowsum[rh] += ls;
            rowmax[rh] = nm;
        }

        // ============ Stage P to SMEM for P@V ============
        #pragma unroll
        for (int nt = 0; nt < BLOCK_N / MMA_N; nt++) {
            int col0 = nt * MMA_N + t * 2;
            int row0 = warp * WARP_M + g;
            int row1 = warp * WARP_M + g + 8;
            p_s[row0 * BLOCK_N + col0]     = __float2bfloat16(s_acc[0][nt][0]);
            p_s[row0 * BLOCK_N + col0 + 1] = __float2bfloat16(s_acc[0][nt][1]);
            p_s[row1 * BLOCK_N + col0]     = __float2bfloat16(s_acc[1][nt][0]);
            p_s[row1 * BLOCK_N + col0 + 1] = __float2bfloat16(s_acc[1][nt][1]);
        }
        __syncthreads();

        // ============ P@V (full HEAD_DIM) ============
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
        __syncthreads();
    }  // end KV loop

    // ============ Normalize and write output ============
    __nv_bfloat16* o_ptr = O + head * Sq * HEAD_DIM;

    for (int rh = 0; rh < 2; rh++) {
        float inv = (rowsum[rh] > 0.0f) ? 1.0f / rowsum[rh] : 0.0f;
        int row = m_start + warp * WARP_M + g + rh * 8;
        if (row < Sq) {
            #pragma unroll
            for (int di = 0; di < HEAD_DIM / MMA_N; di++) {
                int col0 = di * MMA_N + t * 2;
                o_ptr[row * HEAD_DIM + col0]     = __float2bfloat16(o_acc[rh][di][0] * inv);
                o_ptr[row * HEAD_DIM + col0 + 1] = __float2bfloat16(o_acc[rh][di][1] * inv);
            }
        }
    }

    // Optional LSE output
    if (LSE && t == 0) {
        for (int rh = 0; rh < 2; rh++) {
            int row = m_start + warp * WARP_M + g + rh * 8;
            if (row < Sq)
                LSE[head * Sq + row] = rowmax[rh] + logf(rowsum[rh]);
        }
    }
}


// ============================================================================
// Host launcher
// ============================================================================
extern "C" void sm120_flash_prefill_launch(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    int batch, int Hq, int Hkv, int Sq, int Skv, int hd,
    bool causal, cudaStream_t stream
) {
    float sc = 1.0f / sqrtf((float)hd);
    dim3 grid((Sq + BLOCK_M - 1) / BLOCK_M, batch * Hq);
    dim3 block(BLOCK_SIZE);

    if (causal) {
        cudaFuncSetAttribute(sm120_flash_prefill_fwd<true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        sm120_flash_prefill_fwd<true><<<grid, block, SMEM_BYTES, stream>>>(
            Q, K, V, O, LSE, Sq, Skv, Hq, Hkv, sc);
    } else {
        cudaFuncSetAttribute(sm120_flash_prefill_fwd<false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        sm120_flash_prefill_fwd<false><<<grid, block, SMEM_BYTES, stream>>>(
            Q, K, V, O, LSE, Sq, Skv, Hq, Hkv, sc);
    }
}
