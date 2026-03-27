/**
 * SM120 Flash Attention — Prefill with Causal Masking
 *
 * Two kernel variants:
 *   - HD=128: Single-pass, BM=64 BN=64, double-buffered KV, 88KB SMEM, occ=1
 *   - HD=256: 2-pass over head dim, Q full in SMEM (32KB) + KV_half (16KB) + P (8KB) = 56KB
 *             Loads K and V in two 128-dim halves per KV tile
 *
 * SM120 max SMEM per block: 96KB
 *
 * Layout: Q[batch*Hq, Sq, HD], K[batch*Hkv, Skv, HD], V[batch*Hkv, Skv, HD]
 * Output: O[batch*Hq, Sq, HD]
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
#define WARP_SIZE 32
#define NUM_WARPS 4
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE)
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_M MMA_M

// ============================================================================
// Shared utilities
// ============================================================================

// Software swizzle for 256-byte (128 BF16) row sectors
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

// Load A fragment from swizzled SMEM
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

// Load B fragment from swizzled SMEM
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
// Kernel 1: HD=128, single-pass, double-buffered KV
// SMEM: Q(16KB) + 2×K(16KB) + 2×V(16KB) + P(8KB) = 88KB
// ============================================================================

#define HD128 128
#define SMEM_STRIDE_128 128
#define Q_ELEMS_128 (BLOCK_M * SMEM_STRIDE_128)
#define KV_ELEMS_128 (BLOCK_N * SMEM_STRIDE_128)
#define P_ELEMS (BLOCK_M * BLOCK_N)
#define SMEM_BYTES_128 ((Q_ELEMS_128 + 2 * 2 * KV_ELEMS_128 + P_ELEMS) * 2)

template <bool CAUSAL>
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
sm120_flash_prefill_hd128(
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
    const int t = lane % 4;

    const __nv_bfloat16* q_ptr = Q + head * Sq * HD128;
    const __nv_bfloat16* k_ptr = K + kv_head * Skv * HD128;
    const __nv_bfloat16* v_ptr = V + kv_head * Skv * HD128;

    extern __shared__ char smem[];
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* k_s[2], *v_s[2];
    k_s[0] = q_s + Q_ELEMS_128;
    v_s[0] = k_s[0] + KV_ELEMS_128;
    k_s[1] = v_s[0] + KV_ELEMS_128;
    v_s[1] = k_s[1] + KV_ELEMS_128;
    __nv_bfloat16* p_s = v_s[1] + KV_ELEMS_128;

    // Load Q
    for (int i = tid; i < Q_ELEMS_128; i += BLOCK_SIZE) q_s[i] = __float2bfloat16(0.0f);
    __syncthreads();
    for (int i = tid; i < Q_ELEMS_128 / 8; i += BLOCK_SIZE) {
        int row = (i * 8) / HD128, col = (i * 8) % HD128;
        int qr = m_start + row;
        if (qr < Sq) cp_async_16B(&q_s[swizzle_idx(row, col, SMEM_STRIDE_128)], &q_ptr[qr * HD128 + col]);
    }
    cp_async_commit(); cp_async_wait_all(); __syncthreads();

    float o_acc[2][HD128 / MMA_N][2];
    float rowmax[2] = {-FLT_MAX, -FLT_MAX};
    float rowsum[2] = {0.0f, 0.0f};
    #pragma unroll
    for (int rh = 0; rh < 2; rh++)
        #pragma unroll
        for (int nt = 0; nt < HD128 / MMA_N; nt++)
            o_acc[rh][nt][0] = o_acc[rh][nt][1] = 0.0f;

    int num_kv = (Skv + BLOCK_N - 1) / BLOCK_N;
    if (CAUSAL) {
        int max_kv = (m_start + BLOCK_M - 1 + BLOCK_N) / BLOCK_N;
        if (max_kv < num_kv) num_kv = max_kv;
    }

    auto load_kv128 = [&](int blk, int stage) {
        int ns = blk * BLOCK_N;
        for (int i = tid; i < KV_ELEMS_128 / 8; i += BLOCK_SIZE) {
            int row = (i * 8) / HD128, col = (i * 8) % HD128;
            int kvr = ns + row;
            int sw = swizzle_idx(row, col, SMEM_STRIDE_128);
            if (kvr < Skv) {
                cp_async_16B(&k_s[stage][sw], &k_ptr[kvr * HD128 + col]);
                cp_async_16B(&v_s[stage][sw], &v_ptr[kvr * HD128 + col]);
            } else {
                for (int j = 0; j < 8; j++) {
                    k_s[stage][sw + j] = __float2bfloat16(0.0f);
                    v_s[stage][sw + j] = __float2bfloat16(0.0f);
                }
            }
        }
        cp_async_commit();
    };

    if (num_kv > 0) load_kv128(0, 0);

    for (int kv = 0; kv < num_kv; kv++) {
        int cs = kv % 2;
        if (kv + 1 < num_kv) load_kv128(kv + 1, 1 - cs);
        if (kv + 1 < num_kv) cp_async_wait_one(); else cp_async_wait_all();
        __syncthreads();

        // QK^T
        float s_acc[2][BLOCK_N / MMA_N][2];
        #pragma unroll
        for (int rh = 0; rh < 2; rh++)
            #pragma unroll
            for (int nt = 0; nt < BLOCK_N / MMA_N; nt++)
                s_acc[rh][nt][0] = s_acc[rh][nt][1] = 0.0f;

        #pragma unroll
        for (int ki = 0; ki < HD128 / MMA_K; ki++) {
            uint32_t q_frag[4];
            load_A_frag(q_frag, q_s, warp * WARP_M, ki * MMA_K, SMEM_STRIDE_128, lane);
            #pragma unroll
            for (int ni = 0; ni < BLOCK_N / MMA_N; ni++) {
                uint32_t k_frag[2];
                load_B_frag(k_frag, k_s[cs], ni * MMA_N, ki * MMA_K, SMEM_STRIDE_128, lane);
                float tile[4] = {s_acc[0][ni][0], s_acc[0][ni][1], s_acc[1][ni][0], s_acc[1][ni][1]};
                mma_m16n8k16(tile, q_frag, k_frag);
                s_acc[0][ni][0] = tile[0]; s_acc[0][ni][1] = tile[1];
                s_acc[1][ni][0] = tile[2]; s_acc[1][ni][1] = tile[3];
            }
        }

        // Scale + Mask
        int kv_start = kv * BLOCK_N;
        int g = lane / 4;
        #pragma unroll
        for (int rh = 0; rh < 2; rh++) {
            #pragma unroll
            for (int nt = 0; nt < BLOCK_N / MMA_N; nt++) {
                s_acc[rh][nt][0] *= scale; s_acc[rh][nt][1] *= scale;
                int ki0 = kv_start + nt * MMA_N + t * 2;
                if (ki0 >= Skv) s_acc[rh][nt][0] = -FLT_MAX;
                if (ki0 + 1 >= Skv) s_acc[rh][nt][1] = -FLT_MAX;
                if (CAUSAL) {
                    int qi = m_start + warp * WARP_M + g + rh * 8;
                    if (ki0 > qi) s_acc[rh][nt][0] = -FLT_MAX;
                    if (ki0 + 1 > qi) s_acc[rh][nt][1] = -FLT_MAX;
                }
            }
        }

        // Online softmax
        for (int rh = 0; rh < 2; rh++) {
            float tm = rowmax[rh];
            #pragma unroll
            for (int nt = 0; nt < BLOCK_N / MMA_N; nt++) {
                tm = fmaxf(tm, s_acc[rh][nt][0]); tm = fmaxf(tm, s_acc[rh][nt][1]);
            }
            float nm = tm;
            nm = fmaxf(nm, __shfl_xor_sync(0xffffffff, nm, 1));
            nm = fmaxf(nm, __shfl_xor_sync(0xffffffff, nm, 2));
            float rs = __expf(rowmax[rh] - nm);
            rowsum[rh] *= rs;
            #pragma unroll
            for (int nt = 0; nt < HD128 / MMA_N; nt++) { o_acc[rh][nt][0] *= rs; o_acc[rh][nt][1] *= rs; }
            float ls = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < BLOCK_N / MMA_N; nt++) {
                s_acc[rh][nt][0] = __expf(s_acc[rh][nt][0] - nm);
                s_acc[rh][nt][1] = __expf(s_acc[rh][nt][1] - nm);
                ls += s_acc[rh][nt][0] + s_acc[rh][nt][1];
            }
            ls += __shfl_xor_sync(0xffffffff, ls, 1);
            ls += __shfl_xor_sync(0xffffffff, ls, 2);
            rowsum[rh] += ls; rowmax[rh] = nm;
        }

        // P → SMEM
        #pragma unroll
        for (int nt = 0; nt < BLOCK_N / MMA_N; nt++) {
            int col0 = nt * MMA_N + t * 2;
            int row0 = warp * WARP_M + g, row1 = row0 + 8;
            p_s[row0 * BLOCK_N + col0] = __float2bfloat16(s_acc[0][nt][0]);
            p_s[row0 * BLOCK_N + col0 + 1] = __float2bfloat16(s_acc[0][nt][1]);
            p_s[row1 * BLOCK_N + col0] = __float2bfloat16(s_acc[1][nt][0]);
            p_s[row1 * BLOCK_N + col0 + 1] = __float2bfloat16(s_acc[1][nt][1]);
        }
        __syncthreads();

        // P@V
        #pragma unroll
        for (int ki = 0; ki < BLOCK_N / MMA_K; ki++) {
            uint32_t p_frag[4];
            { int pr0 = warp * WARP_M + g, pr1 = pr0 + 8, pc0 = ki * MMA_K + t * 2, pc1 = pc0 + 8;
              p_frag[0] = pack2(p_s[pr0*BLOCK_N+pc0], p_s[pr0*BLOCK_N+pc0+1]);
              p_frag[1] = pack2(p_s[pr1*BLOCK_N+pc0], p_s[pr1*BLOCK_N+pc0+1]);
              p_frag[2] = pack2(p_s[pr0*BLOCK_N+pc1], p_s[pr0*BLOCK_N+pc1+1]);
              p_frag[3] = pack2(p_s[pr1*BLOCK_N+pc1], p_s[pr1*BLOCK_N+pc1+1]); }
            #pragma unroll
            for (int di = 0; di < HD128 / MMA_N; di++) {
                uint32_t v_frag[2];
                int vk0 = ki * MMA_K + t * 2, vk1 = vk0 + 8, vn = di * MMA_N + g;
                v_frag[0] = pack2(v_s[cs][swizzle_idx(vk0, vn, SMEM_STRIDE_128)],
                                  v_s[cs][swizzle_idx(vk0+1, vn, SMEM_STRIDE_128)]);
                v_frag[1] = pack2(v_s[cs][swizzle_idx(vk1, vn, SMEM_STRIDE_128)],
                                  v_s[cs][swizzle_idx(vk1+1, vn, SMEM_STRIDE_128)]);
                float ot[4] = {o_acc[0][di][0], o_acc[0][di][1], o_acc[1][di][0], o_acc[1][di][1]};
                mma_m16n8k16(ot, p_frag, v_frag);
                o_acc[0][di][0] = ot[0]; o_acc[0][di][1] = ot[1];
                o_acc[1][di][0] = ot[2]; o_acc[1][di][1] = ot[3];
            }
        }
        __syncthreads();
    }

    // Write output
    __nv_bfloat16* o_ptr = O + head * Sq * HD128;
    int g = lane / 4;
    for (int rh = 0; rh < 2; rh++) {
        float inv = (rowsum[rh] > 0.0f) ? 1.0f / rowsum[rh] : 0.0f;
        int row = m_start + warp * WARP_M + g + rh * 8;
        if (row < Sq) {
            #pragma unroll
            for (int di = 0; di < HD128 / MMA_N; di++) {
                int col0 = di * MMA_N + t * 2;
                o_ptr[row * HD128 + col0] = __float2bfloat16(o_acc[rh][di][0] * inv);
                o_ptr[row * HD128 + col0 + 1] = __float2bfloat16(o_acc[rh][di][1] * inv);
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


// ============================================================================
// Kernel 2: HD=256, 2-pass over head dimension
// Q_full[BM, 256] (32KB) + kv_half[BN, 128] (16KB) + P[BM, BN] (8KB) = 56KB
// ============================================================================

#define HD256 256
#define HD_HALF 128
#define SMEM_STRIDE_256 256    // Q has 256 columns
#define SMEM_STRIDE_HALF 128   // KV half has 128 columns
#define Q_ELEMS_256 (BLOCK_M * SMEM_STRIDE_256)
#define KV_HALF_ELEMS (BLOCK_N * SMEM_STRIDE_HALF)
// Q(32KB) + kv_half(16KB) + P(8KB) = 56KB
#define SMEM_BYTES_256 ((Q_ELEMS_256 + KV_HALF_ELEMS + P_ELEMS) * 2)

template <bool CAUSAL>
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
sm120_flash_prefill_hd256(
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

    const __nv_bfloat16* q_ptr = Q + head * Sq * HD256;
    const __nv_bfloat16* k_ptr = K + kv_head * Skv * HD256;
    const __nv_bfloat16* v_ptr = V + kv_head * Skv * HD256;

    extern __shared__ char smem[];
    // Q_full: [BM, 256] swizzled with stride=256
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem);
    // KV half: [BN, 128] swizzled with stride=128
    __nv_bfloat16* kv_s = q_s + Q_ELEMS_256;
    // P: [BM, BN] unswizzled
    __nv_bfloat16* p_s = kv_s + KV_HALF_ELEMS;

    // ---- Load Q (full HD=256) into SMEM ----
    for (int i = tid; i < Q_ELEMS_256; i += BLOCK_SIZE) q_s[i] = __float2bfloat16(0.0f);
    __syncthreads();
    for (int i = tid; i < Q_ELEMS_256 / 8; i += BLOCK_SIZE) {
        int row = (i * 8) / HD256, col = (i * 8) % HD256;
        int qr = m_start + row;
        if (qr < Sq) cp_async_16B(&q_s[swizzle_idx(row, col, SMEM_STRIDE_256)], &q_ptr[qr * HD256 + col]);
    }
    cp_async_commit(); cp_async_wait_all(); __syncthreads();

    // Output accumulators: two halves of HD=256
    // o_lo covers dims [0, 128), o_hi covers dims [128, 256)
    float o_lo[2][HD_HALF / MMA_N][2];  // 64 floats
    float o_hi[2][HD_HALF / MMA_N][2];  // 64 floats
    float rowmax[2] = {-FLT_MAX, -FLT_MAX};
    float rowsum[2] = {0.0f, 0.0f};

    #pragma unroll
    for (int rh = 0; rh < 2; rh++) {
        #pragma unroll
        for (int nt = 0; nt < HD_HALF / MMA_N; nt++) {
            o_lo[rh][nt][0] = o_lo[rh][nt][1] = 0.0f;
            o_hi[rh][nt][0] = o_hi[rh][nt][1] = 0.0f;
        }
    }

    int num_kv = (Skv + BLOCK_N - 1) / BLOCK_N;
    if (CAUSAL) {
        int max_kv = (m_start + BLOCK_M - 1 + BLOCK_N) / BLOCK_N;
        if (max_kv < num_kv) num_kv = max_kv;
    }

    // Helper: load half of KV tile (128 dims starting at dim_off) into kv_s
    auto load_kv_half = [&](const __nv_bfloat16* src, int tile_start, int dim_off) {
        for (int i = tid; i < KV_HALF_ELEMS / 8; i += BLOCK_SIZE) {
            int row = (i * 8) / HD_HALF, col = (i * 8) % HD_HALF;
            int kvr = tile_start + row;
            int sw = swizzle_idx(row, col, SMEM_STRIDE_HALF);
            if (kvr < Skv) {
                cp_async_16B(&kv_s[sw], &src[kvr * HD256 + dim_off + col]);
            } else {
                for (int j = 0; j < 8; j++) kv_s[sw + j] = __float2bfloat16(0.0f);
            }
        }
        cp_async_commit();
    };

    // ---- Main KV loop ----
    for (int kv = 0; kv < num_kv; kv++) {
        int kv_start = kv * BLOCK_N;

        // ======== Phase 1: QK^T = Q[0:128] × K[0:128]^T ========
        load_kv_half(k_ptr, kv_start, 0);
        cp_async_wait_all(); __syncthreads();

        float s_acc[2][BLOCK_N / MMA_N][2];
        #pragma unroll
        for (int rh = 0; rh < 2; rh++)
            #pragma unroll
            for (int nt = 0; nt < BLOCK_N / MMA_N; nt++)
                s_acc[rh][nt][0] = s_acc[rh][nt][1] = 0.0f;

        // Q columns [0, 128) × K_half
        #pragma unroll
        for (int ki = 0; ki < HD_HALF / MMA_K; ki++) {
            uint32_t q_frag[4];
            load_A_frag(q_frag, q_s, warp * WARP_M, ki * MMA_K, SMEM_STRIDE_256, lane);
            #pragma unroll
            for (int ni = 0; ni < BLOCK_N / MMA_N; ni++) {
                uint32_t k_frag[2];
                load_B_frag(k_frag, kv_s, ni * MMA_N, ki * MMA_K, SMEM_STRIDE_HALF, lane);
                float tile[4] = {s_acc[0][ni][0], s_acc[0][ni][1], s_acc[1][ni][0], s_acc[1][ni][1]};
                mma_m16n8k16(tile, q_frag, k_frag);
                s_acc[0][ni][0] = tile[0]; s_acc[0][ni][1] = tile[1];
                s_acc[1][ni][0] = tile[2]; s_acc[1][ni][1] = tile[3];
            }
        }
        __syncthreads();

        // ======== Phase 2: QK^T += Q[128:256] × K[128:256]^T ========
        load_kv_half(k_ptr, kv_start, HD_HALF);
        cp_async_wait_all(); __syncthreads();

        // Q columns [128, 256) × K_half
        #pragma unroll
        for (int ki = 0; ki < HD_HALF / MMA_K; ki++) {
            uint32_t q_frag[4];
            // Offset into second half of Q: columns HD_HALF + ki*MMA_K
            load_A_frag(q_frag, q_s, warp * WARP_M, HD_HALF + ki * MMA_K, SMEM_STRIDE_256, lane);
            #pragma unroll
            for (int ni = 0; ni < BLOCK_N / MMA_N; ni++) {
                uint32_t k_frag[2];
                load_B_frag(k_frag, kv_s, ni * MMA_N, ki * MMA_K, SMEM_STRIDE_HALF, lane);
                float tile[4] = {s_acc[0][ni][0], s_acc[0][ni][1], s_acc[1][ni][0], s_acc[1][ni][1]};
                mma_m16n8k16(tile, q_frag, k_frag);
                s_acc[0][ni][0] = tile[0]; s_acc[0][ni][1] = tile[1];
                s_acc[1][ni][0] = tile[2]; s_acc[1][ni][1] = tile[3];
            }
        }

        // ======== Scale + Mask ========
        #pragma unroll
        for (int rh = 0; rh < 2; rh++) {
            #pragma unroll
            for (int nt = 0; nt < BLOCK_N / MMA_N; nt++) {
                s_acc[rh][nt][0] *= scale; s_acc[rh][nt][1] *= scale;
                int ki0 = kv_start + nt * MMA_N + t * 2;
                if (ki0 >= Skv) s_acc[rh][nt][0] = -FLT_MAX;
                if (ki0 + 1 >= Skv) s_acc[rh][nt][1] = -FLT_MAX;
                if (CAUSAL) {
                    int qi = m_start + warp * WARP_M + g + rh * 8;
                    if (ki0 > qi) s_acc[rh][nt][0] = -FLT_MAX;
                    if (ki0 + 1 > qi) s_acc[rh][nt][1] = -FLT_MAX;
                }
            }
        }

        // ======== Online Softmax ========
        for (int rh = 0; rh < 2; rh++) {
            float tm = rowmax[rh];
            #pragma unroll
            for (int nt = 0; nt < BLOCK_N / MMA_N; nt++) {
                tm = fmaxf(tm, s_acc[rh][nt][0]); tm = fmaxf(tm, s_acc[rh][nt][1]);
            }
            float nm = tm;
            nm = fmaxf(nm, __shfl_xor_sync(0xffffffff, nm, 1));
            nm = fmaxf(nm, __shfl_xor_sync(0xffffffff, nm, 2));
            float rs = __expf(rowmax[rh] - nm);
            rowsum[rh] *= rs;
            // Rescale BOTH halves of output
            #pragma unroll
            for (int nt = 0; nt < HD_HALF / MMA_N; nt++) {
                o_lo[rh][nt][0] *= rs; o_lo[rh][nt][1] *= rs;
                o_hi[rh][nt][0] *= rs; o_hi[rh][nt][1] *= rs;
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
            rowsum[rh] += ls; rowmax[rh] = nm;
        }

        // ======== P → SMEM ========
        #pragma unroll
        for (int nt = 0; nt < BLOCK_N / MMA_N; nt++) {
            int col0 = nt * MMA_N + t * 2;
            int row0 = warp * WARP_M + g, row1 = row0 + 8;
            p_s[row0 * BLOCK_N + col0] = __float2bfloat16(s_acc[0][nt][0]);
            p_s[row0 * BLOCK_N + col0 + 1] = __float2bfloat16(s_acc[0][nt][1]);
            p_s[row1 * BLOCK_N + col0] = __float2bfloat16(s_acc[1][nt][0]);
            p_s[row1 * BLOCK_N + col0 + 1] = __float2bfloat16(s_acc[1][nt][1]);
        }
        __syncthreads();

        // ======== Phase 3: O_lo += P × V[0:128] ========
        // First preload P fragments (shared across both V passes)
        // Load V first half
        load_kv_half(v_ptr, kv_start, 0);
        cp_async_wait_all(); __syncthreads();

        #pragma unroll
        for (int ki = 0; ki < BLOCK_N / MMA_K; ki++) {
            uint32_t p_frag[4];
            { int pr0 = warp * WARP_M + g, pr1 = pr0 + 8, pc0 = ki * MMA_K + t * 2, pc1 = pc0 + 8;
              p_frag[0] = pack2(p_s[pr0*BLOCK_N+pc0], p_s[pr0*BLOCK_N+pc0+1]);
              p_frag[1] = pack2(p_s[pr1*BLOCK_N+pc0], p_s[pr1*BLOCK_N+pc0+1]);
              p_frag[2] = pack2(p_s[pr0*BLOCK_N+pc1], p_s[pr0*BLOCK_N+pc1+1]);
              p_frag[3] = pack2(p_s[pr1*BLOCK_N+pc1], p_s[pr1*BLOCK_N+pc1+1]); }
            #pragma unroll
            for (int di = 0; di < HD_HALF / MMA_N; di++) {
                uint32_t v_frag[2];
                int vk0 = ki * MMA_K + t * 2, vk1 = vk0 + 8, vn = di * MMA_N + g;
                v_frag[0] = pack2(kv_s[swizzle_idx(vk0, vn, SMEM_STRIDE_HALF)],
                                  kv_s[swizzle_idx(vk0+1, vn, SMEM_STRIDE_HALF)]);
                v_frag[1] = pack2(kv_s[swizzle_idx(vk1, vn, SMEM_STRIDE_HALF)],
                                  kv_s[swizzle_idx(vk1+1, vn, SMEM_STRIDE_HALF)]);
                float ot[4] = {o_lo[0][di][0], o_lo[0][di][1], o_lo[1][di][0], o_lo[1][di][1]};
                mma_m16n8k16(ot, p_frag, v_frag);
                o_lo[0][di][0] = ot[0]; o_lo[0][di][1] = ot[1];
                o_lo[1][di][0] = ot[2]; o_lo[1][di][1] = ot[3];
            }
        }
        __syncthreads();

        // ======== Phase 4: O_hi += P × V[128:256] ========
        load_kv_half(v_ptr, kv_start, HD_HALF);
        cp_async_wait_all(); __syncthreads();

        #pragma unroll
        for (int ki = 0; ki < BLOCK_N / MMA_K; ki++) {
            uint32_t p_frag[4];
            { int pr0 = warp * WARP_M + g, pr1 = pr0 + 8, pc0 = ki * MMA_K + t * 2, pc1 = pc0 + 8;
              p_frag[0] = pack2(p_s[pr0*BLOCK_N+pc0], p_s[pr0*BLOCK_N+pc0+1]);
              p_frag[1] = pack2(p_s[pr1*BLOCK_N+pc0], p_s[pr1*BLOCK_N+pc0+1]);
              p_frag[2] = pack2(p_s[pr0*BLOCK_N+pc1], p_s[pr0*BLOCK_N+pc1+1]);
              p_frag[3] = pack2(p_s[pr1*BLOCK_N+pc1], p_s[pr1*BLOCK_N+pc1+1]); }
            #pragma unroll
            for (int di = 0; di < HD_HALF / MMA_N; di++) {
                uint32_t v_frag[2];
                int vk0 = ki * MMA_K + t * 2, vk1 = vk0 + 8, vn = di * MMA_N + g;
                v_frag[0] = pack2(kv_s[swizzle_idx(vk0, vn, SMEM_STRIDE_HALF)],
                                  kv_s[swizzle_idx(vk0+1, vn, SMEM_STRIDE_HALF)]);
                v_frag[1] = pack2(kv_s[swizzle_idx(vk1, vn, SMEM_STRIDE_HALF)],
                                  kv_s[swizzle_idx(vk1+1, vn, SMEM_STRIDE_HALF)]);
                float ot[4] = {o_hi[0][di][0], o_hi[0][di][1], o_hi[1][di][0], o_hi[1][di][1]};
                mma_m16n8k16(ot, p_frag, v_frag);
                o_hi[0][di][0] = ot[0]; o_hi[0][di][1] = ot[1];
                o_hi[1][di][0] = ot[2]; o_hi[1][di][1] = ot[3];
            }
        }
        __syncthreads();
    }  // end KV loop

    // ======== Write output (both halves) ========
    __nv_bfloat16* o_ptr = O + head * Sq * HD256;

    for (int rh = 0; rh < 2; rh++) {
        float inv = (rowsum[rh] > 0.0f) ? 1.0f / rowsum[rh] : 0.0f;
        int row = m_start + warp * WARP_M + g + rh * 8;
        if (row < Sq) {
            // First half [0, 128)
            #pragma unroll
            for (int di = 0; di < HD_HALF / MMA_N; di++) {
                int col0 = di * MMA_N + t * 2;
                o_ptr[row * HD256 + col0] = __float2bfloat16(o_lo[rh][di][0] * inv);
                o_ptr[row * HD256 + col0 + 1] = __float2bfloat16(o_lo[rh][di][1] * inv);
            }
            // Second half [128, 256)
            #pragma unroll
            for (int di = 0; di < HD_HALF / MMA_N; di++) {
                int col0 = HD_HALF + di * MMA_N + t * 2;
                o_ptr[row * HD256 + col0] = __float2bfloat16(o_hi[rh][di][0] * inv);
                o_ptr[row * HD256 + col0 + 1] = __float2bfloat16(o_hi[rh][di][1] * inv);
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


// ============================================================================
// Host launcher — dispatches based on head_dim
// ============================================================================
extern "C" void sm120_flash_prefill_launch(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    int batch, int Hq, int Hkv, int Sq, int Skv, int hd,
    bool causal, cudaStream_t stream
) {
    dim3 grid((Sq + BLOCK_M - 1) / BLOCK_M, batch * Hq);
    dim3 block(BLOCK_SIZE);

    if (hd == 256) {
        float sc = 1.0f / sqrtf(256.0f);
        int smem = SMEM_BYTES_256;
        if (causal) {
            cudaFuncSetAttribute(sm120_flash_prefill_hd256<true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            sm120_flash_prefill_hd256<true><<<grid, block, smem, stream>>>(
                Q, K, V, O, LSE, Sq, Skv, Hq, Hkv, sc);
        } else {
            cudaFuncSetAttribute(sm120_flash_prefill_hd256<false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            sm120_flash_prefill_hd256<false><<<grid, block, smem, stream>>>(
                Q, K, V, O, LSE, Sq, Skv, Hq, Hkv, sc);
        }
    } else {
        float sc = 1.0f / sqrtf((float)hd);
        int smem = SMEM_BYTES_128;
        if (causal) {
            cudaFuncSetAttribute(sm120_flash_prefill_hd128<true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            sm120_flash_prefill_hd128<true><<<grid, block, smem, stream>>>(
                Q, K, V, O, LSE, Sq, Skv, Hq, Hkv, sc);
        } else {
            cudaFuncSetAttribute(sm120_flash_prefill_hd128<false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            sm120_flash_prefill_hd128<false><<<grid, block, smem, stream>>>(
                Q, K, V, O, LSE, Sq, Skv, Hq, Hkv, sc);
        }
    }
}
