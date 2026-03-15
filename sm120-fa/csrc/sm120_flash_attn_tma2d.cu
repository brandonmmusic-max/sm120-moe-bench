/**
 * SM120 Flash Attention — 2D TMA Descriptor Kernel
 *
 * Uses cuTensorMapEncode to create tensor descriptors on the host.
 * The GPU loads entire 2D tiles (BLOCK_N × HEAD_DIM) in a single
 * cp.async.bulk.tensor instruction — no per-thread loading needed.
 *
 * BM128+BN64 single-stage + MaxL1 (same config as our 117 TFLOPS best).
 * Expected improvement: eliminate all per-thread cp.async overhead.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define BLOCK_M 128
#define BLOCK_N 64
#define HEAD_DIM 128
#define WARP_SIZE 32
#define NUM_WARPS 8
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE)
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_M MMA_M

#define SMEM_STRIDE HEAD_DIM
#define Q_ELEMS (BLOCK_M * SMEM_STRIDE)
#define KV_ELEMS (BLOCK_N * SMEM_STRIDE)
#define P_ELEMS (BLOCK_M * BLOCK_N)
// Q(32KB) + K(16KB) + V(16KB) + P(16KB) + mbar(16B) = 80KB
#define SMEM_BYTES ((Q_ELEMS + KV_ELEMS + KV_ELEMS + P_ELEMS) * 2 + 128)

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

// TMA 2D tile copy using tensor descriptor
__device__ __forceinline__ void tma_load_2d(
    void* smem_dst,
    const CUtensorMap* desc,
    int coord_x,  // column offset
    int coord_y,  // row offset
    uint64_t* mbar
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

__device__ __forceinline__ void mbar_init(uint64_t* mbar, int count) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(addr), "r"(count));
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* mbar, int bytes) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(addr), "r"(bytes));
}

__device__ __forceinline__ void mbar_wait(uint64_t* mbar, int phase) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    asm volatile(
        "{\n"
        ".reg .pred P;\n"
        "MWAIT:\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
        "@!P bra MWAIT;\n"
        "}\n"
        :: "r"(addr), "r"(phase)
    );
}

// ============================================================================
// Main kernel — receives tensor descriptors as kernel arguments
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
sm120_flash_attn_tma2d(
    const __nv_bfloat16* __restrict__ Q,
    const CUtensorMap* __restrict__ K_desc,  // TMA descriptor for K
    const CUtensorMap* __restrict__ V_desc,  // TMA descriptor for V
    const __nv_bfloat16* __restrict__ K_raw, // Raw K pointer for fallback
    const __nv_bfloat16* __restrict__ V_raw, // Raw V pointer for fallback
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
    const __nv_bfloat16* k_ptr = K_raw + kv_head * Skv * HEAD_DIM;
    const __nv_bfloat16* v_ptr = V_raw + kv_head * Skv * HEAD_DIM;

    extern __shared__ char smem[];
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* k_s = q_s + Q_ELEMS;
    __nv_bfloat16* v_s = k_s + KV_ELEMS;
    __nv_bfloat16* p_s = v_s + KV_ELEMS;
    uint64_t* mbar_kv = reinterpret_cast<uint64_t*>(
        reinterpret_cast<char*>(p_s) + P_ELEMS * 2);

    const int num_kv = (Skv + BLOCK_N - 1) / BLOCK_N;

    // Load Q (standard cp.async — loaded once)
    for (int i = tid; i < Q_ELEMS; i += BLOCK_SIZE)
        q_s[i] = __float2bfloat16(0.0f);
    __syncthreads();
    for (int i = tid; i < Q_ELEMS / 8; i += BLOCK_SIZE) {
        int row = (i * 8) / HEAD_DIM, col = (i * 8) % HEAD_DIM;
        int qr = m_start + row;
        if (qr < Sq)
            cp_async_16B(&q_s[swizzle_idx(row, col, SMEM_STRIDE)],
                          &q_ptr[qr * HEAD_DIM + col]);
    }
    cp_async_commit(); cp_async_wait_all(); __syncthreads();

    // Init mbarrier
    if (tid == 0) mbar_init(mbar_kv, 1);
    __syncthreads();

    float o_acc[2][16][2];
    float rowmax[2] = {-FLT_MAX, -FLT_MAX};
    float rowsum[2] = {0.0f, 0.0f};
    #pragma unroll 1
    for (int rh = 0; rh < 2; rh++)
        for (int nt = 0; nt < 16; nt++)
            o_acc[rh][nt][0] = o_acc[rh][nt][1] = 0.0f;

    for (int kv = 0; kv < num_kv; kv++) {
        int kv_start_row = kv * BLOCK_N;

        // ================================================================
        // Load KV via TMA 2D descriptor (thread 0 only!)
        // ================================================================
        if (tid == 0) {
            mbar_init(mbar_kv, 1);
        }
        __syncthreads();  // All threads must see mbar init

        if (tid == 0) {
            int kv_bytes = BLOCK_N * HEAD_DIM * 2;
            mbar_expect_tx(mbar_kv, 2 * kv_bytes);

            tma_load_2d(k_s, K_desc, 0, kv_head * Skv + kv_start_row, mbar_kv);
            tma_load_2d(v_s, V_desc, 0, kv_head * Skv + kv_start_row, mbar_kv);
        }

        // Wait for TMA — phase 0 (re-init resets phase)
        mbar_wait(mbar_kv, 0);
        __syncthreads();

        // ================================================================
        // Q@K^T (same compute as BM128+BN64, K/V NOT swizzled — TMA loads contiguous)
        // ================================================================
        float s_acc[2][BLOCK_N / MMA_N][2];
        for (int rh = 0; rh < 2; rh++)
            for (int nt = 0; nt < BLOCK_N / MMA_N; nt++)
                s_acc[rh][nt][0] = s_acc[rh][nt][1] = 0.0f;

        #pragma unroll
        for (int ki = 0; ki < HEAD_DIM / MMA_K; ki++) {
            uint32_t q_frag[4];
            {
                int r0 = warp * WARP_M + g, r1 = r0 + 8;
                int c0 = ki * MMA_K + t * 2, c1 = c0 + 8;
                q_frag[0] = pack2(q_s[swizzle_idx(r0,c0,SMEM_STRIDE)], q_s[swizzle_idx(r0,c0+1,SMEM_STRIDE)]);
                q_frag[1] = pack2(q_s[swizzle_idx(r1,c0,SMEM_STRIDE)], q_s[swizzle_idx(r1,c0+1,SMEM_STRIDE)]);
                q_frag[2] = pack2(q_s[swizzle_idx(r0,c1,SMEM_STRIDE)], q_s[swizzle_idx(r0,c1+1,SMEM_STRIDE)]);
                q_frag[3] = pack2(q_s[swizzle_idx(r1,c1,SMEM_STRIDE)], q_s[swizzle_idx(r1,c1+1,SMEM_STRIDE)]);
            }

            #pragma unroll
            for (int ni = 0; ni < BLOCK_N / MMA_N; ni++) {
                uint32_t k_frag[2];
                int kn = ni * MMA_N + g;
                int kk0 = ki * MMA_K + t * 2, kk1 = kk0 + 8;
                // K swizzled by TMA hardware — read with matching swizzle
                k_frag[0] = pack2(k_s[swizzle_idx(kn, kk0, SMEM_STRIDE)], k_s[swizzle_idx(kn, kk0+1, SMEM_STRIDE)]);
                k_frag[1] = pack2(k_s[swizzle_idx(kn, kk1, SMEM_STRIDE)], k_s[swizzle_idx(kn, kk1+1, SMEM_STRIDE)]);

                float tile[4] = {s_acc[0][ni][0], s_acc[0][ni][1], s_acc[1][ni][0], s_acc[1][ni][1]};
                mma_m16n8k16(tile, q_frag, k_frag);
                s_acc[0][ni][0] = tile[0]; s_acc[0][ni][1] = tile[1];
                s_acc[1][ni][0] = tile[2]; s_acc[1][ni][1] = tile[3];
            }
        }

        // Scale + mask
        for (int rh = 0; rh < 2; rh++)
            for (int nt = 0; nt < BLOCK_N / MMA_N; nt++) {
                s_acc[rh][nt][0] *= scale;
                s_acc[rh][nt][1] *= scale;
                int kv_idx0 = kv_start_row + nt * MMA_N + t * 2;
                if (kv_idx0 >= Skv) s_acc[rh][nt][0] = -FLT_MAX;
                if (kv_idx0 + 1 >= Skv) s_acc[rh][nt][1] = -FLT_MAX;
            }

        // Softmax
        for (int rh = 0; rh < 2; rh++) {
            float tm = rowmax[rh];
            for (int nt = 0; nt < BLOCK_N / MMA_N; nt++) {
                tm = fmaxf(tm, s_acc[rh][nt][0]);
                tm = fmaxf(tm, s_acc[rh][nt][1]);
            }
            float nm = tm;
            nm = fmaxf(nm, __shfl_xor_sync(0xffffffff, nm, 1));
            nm = fmaxf(nm, __shfl_xor_sync(0xffffffff, nm, 2));
            float rs = __expf(rowmax[rh] - nm);
            rowsum[rh] *= rs;
            for (int nt = 0; nt < 16; nt++) { o_acc[rh][nt][0] *= rs; o_acc[rh][nt][1] *= rs; }
            float ls = 0.0f;
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
        for (int nt = 0; nt < BLOCK_N / MMA_N; nt++) {
            int col0 = nt * MMA_N + t * 2;
            int row0 = warp * WARP_M + g, row1 = row0 + 8;
            p_s[row0 * BLOCK_N + col0]     = __float2bfloat16(s_acc[0][nt][0]);
            p_s[row0 * BLOCK_N + col0 + 1] = __float2bfloat16(s_acc[0][nt][1]);
            p_s[row1 * BLOCK_N + col0]     = __float2bfloat16(s_acc[1][nt][0]);
            p_s[row1 * BLOCK_N + col0 + 1] = __float2bfloat16(s_acc[1][nt][1]);
        }
        __syncwarp();

        // P@V (V NOT swizzled — TMA loads contiguous)
        #pragma unroll 1
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

            #pragma unroll 1
            for (int di = 0; di < HEAD_DIM / MMA_N; di++) {
                uint32_t v_frag[2];
                int vk0 = ki * MMA_K + t * 2, vk1 = vk0 + 8;
                int vn = di * MMA_N + g;
                v_frag[0] = pack2(v_s[swizzle_idx(vk0, vn, SMEM_STRIDE)], v_s[swizzle_idx(vk0+1, vn, SMEM_STRIDE)]);
                v_frag[1] = pack2(v_s[swizzle_idx(vk1, vn, SMEM_STRIDE)], v_s[swizzle_idx(vk1+1, vn, SMEM_STRIDE)]);

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
// Host: create TMA descriptors and launch kernel
// ============================================================================
extern "C" void sm120_flash_attn_forward(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    int batch, int Hq, int Hkv, int Sq, int Skv, int hd, cudaStream_t stream
) {
    float sc = 1.0f / sqrtf((float)hd);
    int total_kv_heads = batch * Hkv;

    // Create TMA descriptors for K and V
    // K shape: [total_kv_heads * Skv, HEAD_DIM] — 2D row-major
    // TMA box: [BLOCK_N, HEAD_DIM] per copy

    CUtensorMap k_desc, v_desc;

    // Tensor dimensions: [total_kv_heads * Skv, HEAD_DIM]
    cuuint64_t dims[2] = {(cuuint64_t)HEAD_DIM, (cuuint64_t)(total_kv_heads * Skv)};
    cuuint64_t strides[1] = {(cuuint64_t)(HEAD_DIM * sizeof(__nv_bfloat16))};  // row stride in bytes
    cuuint32_t box[2] = {(cuuint32_t)HEAD_DIM, (cuuint32_t)BLOCK_N};  // tile size
    cuuint32_t elem_strides[2] = {1, 1};

    CUresult err;

    err = cuTensorMapEncodeTiled(
        &k_desc,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,       // rank
        (void*)K,
        dims,
        strides,
        box,
        elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,        // TMA applies 128B swizzle during copy
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (err != CUDA_SUCCESS) {
        // Fallback to non-TMA kernel
        // For now, just return
        return;
    }

    err = cuTensorMapEncodeTiled(
        &v_desc,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        (void*)V,
        dims,
        strides,
        box,
        elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );
    if (err != CUDA_SUCCESS) return;

    // Copy descriptors to device
    CUtensorMap *d_k_desc, *d_v_desc;
    cudaMalloc(&d_k_desc, sizeof(CUtensorMap));
    cudaMalloc(&d_v_desc, sizeof(CUtensorMap));
    cudaMemcpyAsync(d_k_desc, &k_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_v_desc, &v_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);

    dim3 grid((Sq + BLOCK_M - 1) / BLOCK_M, batch * Hq);
    cudaFuncSetAttribute(sm120_flash_attn_tma2d,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    cudaFuncSetAttribute(sm120_flash_attn_tma2d,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxL1);

    sm120_flash_attn_tma2d<<<grid, BLOCK_SIZE, SMEM_BYTES, stream>>>(
        Q, d_k_desc, d_v_desc, K, V, O, L, Sq, Skv, Hq, Hkv, sc);

    cudaFree(d_k_desc);
    cudaFree(d_v_desc);
}
