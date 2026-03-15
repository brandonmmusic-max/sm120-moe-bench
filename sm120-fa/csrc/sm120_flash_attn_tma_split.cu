/**
 * SM120 Flash Attention — Split-HD TMA with SWIZZLE_128B
 *
 * Key idea: HEAD_DIM=128 bf16 = 256 bytes > 128 byte TMA swizzle limit.
 * Solution: treat the tensor as [total_rows, 2, 64] and use box=[64, BLOCK_N]
 * with two TMA loads per tile (left half and right half).
 *
 * BM128, single-stage, 8 warps.
 * Q loaded with cp.async + software swizzle (as before).
 * K/V loaded with TMA 2D + hardware SWIZZLE_128B.
 *
 * SMEM layout for K/V: each half stored contiguously.
 * K_left[BLOCK_N][64] and K_right[BLOCK_N][64] adjacent in SMEM.
 * Reads use the TMA swizzle pattern (which we match by reading
 * at the same addresses TMA wrote to).
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define BLOCK_M 128
#define BLOCK_N 64
#define HEAD_DIM 128
#define HALF_HD 64
#define WARP_SIZE 32
#define NUM_WARPS 8
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE)
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_M MMA_M

// SMEM layout:
// Q: [BLOCK_M, HEAD_DIM] with software swizzle, stride=HEAD_DIM
// K: [BLOCK_N, HALF_HD] × 2 halves, each with TMA swizzle, stride=HALF_HD
// V: same as K
// P: [BLOCK_M, BLOCK_N] no swizzle
#define Q_STRIDE HEAD_DIM
#define KV_HALF_STRIDE HALF_HD  // 64 elements per half-row
#define Q_ELEMS (BLOCK_M * Q_STRIDE)
#define KV_HALF_ELEMS (BLOCK_N * KV_HALF_STRIDE)  // 64×64 = 4096 per half
#define KV_ELEMS (2 * KV_HALF_ELEMS)  // Both halves = 8192 per K or V
#define P_ELEMS (BLOCK_M * BLOCK_N)
// Q(32KB) + K(16KB) + V(16KB) + P(16KB) + mbar(64B) = 80KB
#define SMEM_BYTES ((Q_ELEMS + KV_ELEMS + KV_ELEMS + P_ELEMS) * 2 + 128)

// Software swizzle for Q (same as v5)
__device__ __forceinline__ int q_swizzle_idx(int row, int col, int stride) {
    int col_bytes = col * 2;
    int swizzled_bytes = col_bytes ^ ((row & 7) << 5);
    return row * stride + swizzled_bytes / 2;
}

// TMA swizzle read: TMA SWIZZLE_128B XORs within 128-byte spans
// For HALF_HD=64 bf16 = 128 bytes per row, the swizzle XORs
// byte offset bits within each 128-byte row with the row index.
// Exact pattern: byte_addr = base + row * 128 + (col_bytes ^ ((row & 7) << 4))
// This is the standard CUTLASS Swizzle<3,3,3> for 128-byte rows.
__device__ __forceinline__ int tma_swizzle_idx(int row, int col, int half_stride) {
    // col is in [0, HALF_HD=64), col_bytes in [0, 128)
    int col_bytes = col * 2;
    int swizzled_bytes = col_bytes ^ ((row & 7) << 4);  // XOR bits [6:4]
    return row * half_stride + swizzled_bytes / 2;
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

__device__ __forceinline__ void tma_load_2d(
    void* smem_dst, const CUtensorMap* desc,
    int coord_x, int coord_y, uint64_t* mbar
) {
    uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    uint64_t desc_addr = reinterpret_cast<uint64_t>(desc);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(dst), "l"(desc_addr), "r"(coord_x), "r"(coord_y), "r"(mbar_addr)
    );
}

__device__ __forceinline__ void mbar_init(uint64_t* m, int n) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(m));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(a), "r"(n));
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* m, int b) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(m));
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(a), "r"(b));
}
__device__ __forceinline__ void mbar_wait(uint64_t* m, int p) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(m));
    asm volatile(
        "{\n.reg .pred P;\nMW:\nmbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n@!P bra MW;\n}\n"
        :: "r"(a), "r"(p));
}

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
sm120_flash_attn_tma_split(
    const __nv_bfloat16* __restrict__ Q,
    const CUtensorMap* __restrict__ K_desc_left,
    const CUtensorMap* __restrict__ K_desc_right,
    const CUtensorMap* __restrict__ V_desc_left,
    const CUtensorMap* __restrict__ V_desc_right,
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

    extern __shared__ char smem[];
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem);
    // K stored as two halves: k_left[BLOCK_N][64] then k_right[BLOCK_N][64]
    __nv_bfloat16* k_left = q_s + Q_ELEMS;
    __nv_bfloat16* k_right = k_left + KV_HALF_ELEMS;
    __nv_bfloat16* v_left = k_right + KV_HALF_ELEMS;
    __nv_bfloat16* v_right = v_left + KV_HALF_ELEMS;
    __nv_bfloat16* p_s = v_right + KV_HALF_ELEMS;
    uint64_t* mbar_kv = reinterpret_cast<uint64_t*>(
        reinterpret_cast<char*>(p_s) + P_ELEMS * 2);

    const int num_kv = (Skv + BLOCK_N - 1) / BLOCK_N;

    // Load Q with software swizzle
    for (int i = tid; i < Q_ELEMS; i += BLOCK_SIZE) q_s[i] = __float2bfloat16(0.0f);
    __syncthreads();
    for (int i = tid; i < Q_ELEMS / 8; i += BLOCK_SIZE) {
        int row = (i * 8) / HEAD_DIM, col = (i * 8) % HEAD_DIM;
        int qr = m_start + row;
        if (qr < Sq) {
            int sw = q_swizzle_idx(row, col, Q_STRIDE);
            cp_async_16B(&q_s[sw], &q_ptr[qr * HEAD_DIM + col]);
        }
    }
    cp_async_commit(); cp_async_wait_all(); __syncthreads();

    if (tid == 0) mbar_init(mbar_kv, 1);
    __syncthreads();

    float o_acc[2][16][2];
    float rowmax[2] = {-FLT_MAX, -FLT_MAX};
    float rowsum[2] = {0.0f, 0.0f};
    #pragma unroll
    for (int rh = 0; rh < 2; rh++)
        #pragma unroll
        for (int nt = 0; nt < 16; nt++)
            o_acc[rh][nt][0] = o_acc[rh][nt][1] = 0.0f;

    for (int kv = 0; kv < num_kv; kv++) {
        int kv_row = kv_head * Skv + kv * BLOCK_N;

        // TMA load: 4 descriptors × 1 load each = 4 TMA instructions
        if (tid == 0) {
            mbar_init(mbar_kv, 1);
        }
        __syncthreads();

        if (tid == 0) {
            int half_bytes = BLOCK_N * HALF_HD * 2;  // 8KB per half
            mbar_expect_tx(mbar_kv, 4 * half_bytes);  // 4 halves total

            tma_load_2d(k_left,  K_desc_left,  0, kv_row, mbar_kv);
            tma_load_2d(k_right, K_desc_right, 0, kv_row, mbar_kv);
            tma_load_2d(v_left,  V_desc_left,  0, kv_row, mbar_kv);
            tma_load_2d(v_right, V_desc_right, 0, kv_row, mbar_kv);
        }

        mbar_wait(mbar_kv, 0);
        __syncthreads();

        // Q@K^T: iterate over HEAD_DIM in steps of MMA_K=16
        float s_acc[2][BLOCK_N / MMA_N][2];
        #pragma unroll
        for (int rh = 0; rh < 2; rh++)
            #pragma unroll
            for (int nt = 0; nt < BLOCK_N / MMA_N; nt++)
                s_acc[rh][nt][0] = s_acc[rh][nt][1] = 0.0f;

        #pragma unroll
        for (int ki = 0; ki < HEAD_DIM / MMA_K; ki++) {
            // Q fragment: software swizzled
            uint32_t q_frag[4];
            {
                int r0 = warp * WARP_M + g, r1 = r0 + 8;
                int c0 = ki * MMA_K + t * 2, c1 = c0 + 8;
                q_frag[0] = pack2(q_s[q_swizzle_idx(r0,c0,Q_STRIDE)], q_s[q_swizzle_idx(r0,c0+1,Q_STRIDE)]);
                q_frag[1] = pack2(q_s[q_swizzle_idx(r1,c0,Q_STRIDE)], q_s[q_swizzle_idx(r1,c0+1,Q_STRIDE)]);
                q_frag[2] = pack2(q_s[q_swizzle_idx(r0,c1,Q_STRIDE)], q_s[q_swizzle_idx(r0,c1+1,Q_STRIDE)]);
                q_frag[3] = pack2(q_s[q_swizzle_idx(r1,c1,Q_STRIDE)], q_s[q_swizzle_idx(r1,c1+1,Q_STRIDE)]);
            }

            // K fragment: determine which half
            int k_col_start = ki * MMA_K;  // 0, 16, 32, ..., 112
            int half = k_col_start / HALF_HD;  // 0 for cols 0-63, 1 for 64-127
            int local_col = k_col_start - half * HALF_HD;
            __nv_bfloat16* k_half = (half == 0) ? k_left : k_right;

            #pragma unroll
            for (int ni = 0; ni < BLOCK_N / MMA_N; ni++) {
                uint32_t k_frag[2];
                int kn = ni * MMA_N + g;
                int kk0 = local_col + t * 2, kk1 = kk0 + 8;

                // TMA swizzle read (128B span for 64-element half-rows)
                k_frag[0] = pack2(k_half[tma_swizzle_idx(kn, kk0, KV_HALF_STRIDE)],
                                  k_half[tma_swizzle_idx(kn, kk0+1, KV_HALF_STRIDE)]);
                k_frag[1] = pack2(k_half[tma_swizzle_idx(kn, kk1, KV_HALF_STRIDE)],
                                  k_half[tma_swizzle_idx(kn, kk1+1, KV_HALF_STRIDE)]);

                float tile[4] = {s_acc[0][ni][0], s_acc[0][ni][1], s_acc[1][ni][0], s_acc[1][ni][1]};
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
            for (int nt = 0; nt < BLOCK_N / MMA_N; nt++) {
                s_acc[rh][nt][0] *= scale;
                s_acc[rh][nt][1] *= scale;
                int kv_idx0 = kv_start + nt * MMA_N + t * 2;
                if (kv_idx0 >= Skv) s_acc[rh][nt][0] = -FLT_MAX;
                if (kv_idx0 + 1 >= Skv) s_acc[rh][nt][1] = -FLT_MAX;
            }

        // Softmax
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
            for (int nt = 0; nt < 16; nt++) { o_acc[rh][nt][0] *= rs; o_acc[rh][nt][1] *= rs; }
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

        // P staging
        #pragma unroll
        for (int nt = 0; nt < BLOCK_N / MMA_N; nt++) {
            int col0 = nt * MMA_N + t * 2;
            int row0 = warp * WARP_M + g, row1 = row0 + 8;
            p_s[row0 * BLOCK_N + col0]     = __float2bfloat16(s_acc[0][nt][0]);
            p_s[row0 * BLOCK_N + col0 + 1] = __float2bfloat16(s_acc[0][nt][1]);
            p_s[row1 * BLOCK_N + col0]     = __float2bfloat16(s_acc[1][nt][0]);
            p_s[row1 * BLOCK_N + col0 + 1] = __float2bfloat16(s_acc[1][nt][1]);
        }
        __syncwarp();

        // P@V — iterate over output HEAD_DIM, using appropriate V half
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
                // Determine which V half this di belongs to
                int d_col = di * MMA_N;  // 0, 8, 16, ..., 120
                int v_half = d_col / HALF_HD;  // 0 for d<64, 1 for d≥64
                int v_local = d_col - v_half * HALF_HD;
                __nv_bfloat16* v_h = (v_half == 0) ? v_left : v_right;

                uint32_t v_frag[2];
                int vk0 = ki * MMA_K + t * 2, vk1 = vk0 + 8;
                int vn = v_local + g;

                v_frag[0] = pack2(v_h[tma_swizzle_idx(vk0, vn, KV_HALF_STRIDE)],
                                  v_h[tma_swizzle_idx(vk0+1, vn, KV_HALF_STRIDE)]);
                v_frag[1] = pack2(v_h[tma_swizzle_idx(vk1, vn, KV_HALF_STRIDE)],
                                  v_h[tma_swizzle_idx(vk1+1, vn, KV_HALF_STRIDE)]);

                float ot[4] = {o_acc[0][di][0], o_acc[0][di][1], o_acc[1][di][0], o_acc[1][di][1]};
                mma_m16n8k16(ot, p_frag, v_frag);
                o_acc[0][di][0] = ot[0]; o_acc[0][di][1] = ot[1];
                o_acc[1][di][0] = ot[2]; o_acc[1][di][1] = ot[3];
            }
        }
        __syncthreads();
    }

    // Output
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

// ============================================================================
// Host: create split-HD TMA descriptors
// ============================================================================
extern "C" void sm120_flash_attn_forward(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    int batch, int Hq, int Hkv, int Sq, int Skv, int hd, cudaStream_t stream
) {
    float sc = 1.0f / sqrtf((float)hd);
    int total_kv_heads = batch * Hkv;
    int total_rows = total_kv_heads * Skv;

    // Create 4 TMA descriptors: K_left, K_right, V_left, V_right
    // Each describes a [total_rows, HALF_HD=64] view of the original [total_rows, HEAD_DIM=128] tensor
    // K_left: starts at K, stride=HEAD_DIM*2 bytes, inner dim=64
    // K_right: starts at K+64, stride=HEAD_DIM*2 bytes, inner dim=64

    CUtensorMap k_desc_l __attribute__((aligned(64)));
    CUtensorMap k_desc_r __attribute__((aligned(64)));
    CUtensorMap v_desc_l __attribute__((aligned(64)));
    CUtensorMap v_desc_r __attribute__((aligned(64)));

    cuuint64_t dims[2] = {(cuuint64_t)HALF_HD, (cuuint64_t)total_rows};
    cuuint64_t strides[1] = {(cuuint64_t)(HEAD_DIM * 2)};  // Row stride = full HEAD_DIM in bytes
    cuuint32_t box[2] = {(cuuint32_t)HALF_HD, (cuuint32_t)BLOCK_N};
    cuuint32_t elem_strides[2] = {1, 1};

    // K left half (cols 0-63)
    cuTensorMapEncodeTiled(&k_desc_l, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        (void*)K, dims, strides, box, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    // K right half (cols 64-127) — offset pointer by 64 elements
    cuTensorMapEncodeTiled(&k_desc_r, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        (void*)(K + HALF_HD), dims, strides, box, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    // V halves
    cuTensorMapEncodeTiled(&v_desc_l, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        (void*)V, dims, strides, box, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    cuTensorMapEncodeTiled(&v_desc_r, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2,
        (void*)(V + HALF_HD), dims, strides, box, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    // Copy descriptors to device
    CUtensorMap* d_descs;
    cudaMalloc(&d_descs, 4 * sizeof(CUtensorMap));
    cudaMemcpyAsync(d_descs, &k_desc_l, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_descs + 1, &k_desc_r, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_descs + 2, &v_desc_l, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_descs + 3, &v_desc_r, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);

    dim3 grid((Sq + BLOCK_M - 1) / BLOCK_M, batch * Hq);
    cudaFuncSetAttribute(sm120_flash_attn_tma_split,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    cudaFuncSetAttribute(sm120_flash_attn_tma_split,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxL1);

    sm120_flash_attn_tma_split<<<grid, BLOCK_SIZE, SMEM_BYTES, stream>>>(
        Q, d_descs, d_descs + 1, d_descs + 2, d_descs + 3,
        O, L, Sq, Skv, Hq, Hkv, sc);

    cudaFree(d_descs);
}
