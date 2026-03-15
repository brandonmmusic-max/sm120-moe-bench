/**
 * SM120 Flash Attention v4 — Unified double-buffered kernel
 *
 * Changes from v3:
 * 1. DS kernel for ALL sequence lengths (no SS fallback)
 * 2. Per-warp mbarrier arrival for done signal (8 arrivals, not 256)
 * 3. Split Q preload: first 4 fragments preloaded, last 4 reloaded
 *    → saves 16 registers (should help compiler scheduling)
 * 4. Constant-memory TMA descriptors (from v3)
 * 5. Vectorized output writes (from v3)
 *
 * BM128, BN64 double-buffered: Q(32KB) + K[2](32KB) + V[2](32KB) = 96KB
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
#define NUM_STAGES 2
#define WARP_SIZE 32
#define NUM_WARPS 8
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE)
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_M MMA_M

#define Q_STRIDE HEAD_DIM
#define KV_HALF_STRIDE HALF_HD
#define Q_ELEMS (BLOCK_M * Q_STRIDE)
#define KV_HALF_ELEMS (BLOCK_N * KV_HALF_STRIDE)
#define SMEM_BYTES ((Q_ELEMS + NUM_STAGES * 4 * KV_HALF_ELEMS) * 2 + 128)

__device__ __forceinline__ int q_sw(int r, int c, int s) {
    return r * s + ((c*2) ^ ((r&7)<<5)) / 2;
}
__device__ __forceinline__ int tma_sw(int r, int c, int s) {
    return r * s + ((c*2) ^ ((r&7)<<4)) / 2;
}
__device__ __forceinline__ uint32_t pk_f2(float a, float b) {
    __nv_bfloat16 ha = __float2bfloat16(a), hb = __float2bfloat16(b);
    uint32_t r; asm("mov.b32 %0, {%1, %2};" : "=r"(r) : "h"(*(const uint16_t*)&ha), "h"(*(const uint16_t*)&hb)); return r;
}
__device__ __forceinline__ void cp16(void* s, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(s))), "l"(g));
}
__device__ __forceinline__ void mma16(float c[4], uint32_t a[4], uint32_t b[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(c[0]),"=f"(c[1]),"=f"(c[2]),"=f"(c[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
}
__device__ __forceinline__ void tma2d(void* d, const CUtensorMap* desc, int x, int y, uint64_t* m) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(d))), "l"(reinterpret_cast<uint64_t>(desc)),
           "r"(x), "r"(y), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(m))));
}
__device__ __forceinline__ void mb_init(uint64_t* m, int n) {
    asm volatile("mbarrier.init.shared.b64 [%0],%1;\n" :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(m))), "r"(n));
}
__device__ __forceinline__ void mb_expect(uint64_t* m, int b) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _,[%0],%1;\n" :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(m))), "r"(b));
}
__device__ __forceinline__ void mb_wait(uint64_t* m, int p) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(m));
    asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.shared.b64 P,[%0],%1;\n@!P bra W;\n}\n" :: "r"(a), "r"(p));
}
__device__ __forceinline__ void mb_arrive(uint64_t* m) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(m));
    asm volatile("mbarrier.arrive.shared.b64 _, [%0];\n" :: "r"(a));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&r)[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

__constant__ CUtensorMap c_descs[4];

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
sm120_fa_v4(
    const __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ O, float* __restrict__ LSE,
    int Sq, int Skv, int Hq, int Hkv, float scale
) {
    const int bm = blockIdx.x, head = blockIdx.y;
    const int kv_head = head / (Hq / Hkv);
    const int m_start = bm * BLOCK_M;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int g = lane / 4, t = lane % 4;
    const __nv_bfloat16* q_ptr = Q + head * Sq * HEAD_DIM;
    const int num_kv = (Skv + BLOCK_N - 1) / BLOCK_N;

    extern __shared__ char smem[];
    __nv_bfloat16* q_s = (__nv_bfloat16*)smem;
    __nv_bfloat16* kv_base = q_s + Q_ELEMS;
    __nv_bfloat16* kl[2], *kr[2], *vl[2], *vr[2];
    for (int s = 0; s < 2; s++) {
        kl[s] = kv_base + s * 4 * KV_HALF_ELEMS;
        kr[s] = kl[s] + KV_HALF_ELEMS;
        vl[s] = kr[s] + KV_HALF_ELEMS;
        vr[s] = vl[s] + KV_HALF_ELEMS;
    }
    uint64_t* mbar_kv = (uint64_t*)((char*)kv_base + NUM_STAGES * 4 * KV_HALF_ELEMS * 2);
    uint64_t* mbar_done = mbar_kv + 2;

    // Load Q
    for (int i = tid; i < Q_ELEMS; i += BLOCK_SIZE) q_s[i] = __float2bfloat16(0.0f);
    __syncthreads();
    for (int i = tid; i < Q_ELEMS / 8; i += BLOCK_SIZE) {
        int row = (i*8)/HEAD_DIM, col = (i*8)%HEAD_DIM, qr = m_start+row;
        if (qr < Sq) cp16(&q_s[q_sw(row,col,Q_STRIDE)], &q_ptr[qr*HEAD_DIM+col]);
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n");
    __syncthreads();

    if (tid == 0) {
        mb_init(&mbar_kv[0], 1);
        mb_init(&mbar_kv[1], 1);
        mb_init(&mbar_done[0], NUM_WARPS);  // per-warp arrival
        mb_init(&mbar_done[1], NUM_WARPS);
    }
    __syncthreads();

    const int ldm_q_row = warp * WARP_M + (lane & 7) + ((lane >> 3) & 1) * 8;
    const int ldm_q_col_base = (lane >> 4) * 8;
    const int ldm_k_lane_row = lane & 7;
    const int ldm_k_col_off = ((lane >> 3) & 1) * 8;
    const int ldm_v_row_off = (lane & 7) + ((lane >> 3) & 1) * 8;

    // Split Q preload: first 4 ki (cols 0-63, K left half)
    uint32_t qf_pre[4][4];
    #pragma unroll
    for (int ki = 0; ki < 4; ki++) {
        int q_col = ki * MMA_K + ldm_q_col_base;
        uint32_t q_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&q_s[q_sw(ldm_q_row, q_col, Q_STRIDE)]));
        ldmatrix_x4(qf_pre[ki], q_addr);
    }
    // Precompute addresses for second half (reloaded each KV iteration)
    uint32_t q_addrs_hi[4];
    #pragma unroll
    for (int ki = 4; ki < 8; ki++) {
        int q_col = ki * MMA_K + ldm_q_col_base;
        q_addrs_hi[ki-4] = static_cast<uint32_t>(
            __cvta_generic_to_shared(&q_s[q_sw(ldm_q_row, q_col, Q_STRIDE)]));
    }

    // Prefetch stages 0 and 1
    if (tid == 0 && num_kv > 0) {
        int kv_row = kv_head * Skv;
        int hbytes = BLOCK_N * HALF_HD * 2;
        mb_expect(&mbar_kv[0], 4 * hbytes);
        tma2d(kl[0], &c_descs[0], 0, kv_row, &mbar_kv[0]);
        tma2d(kr[0], &c_descs[1], 0, kv_row, &mbar_kv[0]);
        tma2d(vl[0], &c_descs[2], 0, kv_row, &mbar_kv[0]);
        tma2d(vr[0], &c_descs[3], 0, kv_row, &mbar_kv[0]);
    }
    if (tid == 0 && num_kv > 1) {
        int kv_row = kv_head * Skv + BLOCK_N;
        int hbytes = BLOCK_N * HALF_HD * 2;
        mb_expect(&mbar_kv[1], 4 * hbytes);
        tma2d(kl[1], &c_descs[0], 0, kv_row, &mbar_kv[1]);
        tma2d(kr[1], &c_descs[1], 0, kv_row, &mbar_kv[1]);
        tma2d(vl[1], &c_descs[2], 0, kv_row, &mbar_kv[1]);
        tma2d(vr[1], &c_descs[3], 0, kv_row, &mbar_kv[1]);
    }

    float o_acc[2][16][2];
    float rmax[2] = {-FLT_MAX, -FLT_MAX}, rsum[2] = {0, 0};
    #pragma unroll
    for (int rh = 0; rh < 2; rh++)
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) o_acc[rh][nt][0] = o_acc[rh][nt][1] = 0;

    int kv_phase[2] = {0, 0};
    int done_phase[2] = {0, 0};

    for (int kv = 0; kv < num_kv; kv++) {
        int cs = kv & 1;

        mb_wait(&mbar_kv[cs], kv_phase[cs]);
        kv_phase[cs] ^= 1;

        // ============ Q@K^T ============
        float s_acc[2][BLOCK_N/MMA_N][2];
        #pragma unroll
        for (int rh = 0; rh < 2; rh++)
            #pragma unroll
            for (int nt = 0; nt < BLOCK_N/MMA_N; nt++)
                s_acc[rh][nt][0] = s_acc[rh][nt][1] = 0;

        // First 4 ki: preloaded Q, K left half
        #pragma unroll
        for (int ki = 0; ki < 4; ki++) {
            uint32_t (&qf)[4] = qf_pre[ki];
            int lc = ki * MMA_K;
            #pragma unroll
            for (int ni = 0; ni < BLOCK_N/MMA_N; ni++) {
                uint32_t kf[2];
                int kn = ni * MMA_N + ldm_k_lane_row;
                uint32_t k_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&kl[cs][tma_sw(kn, lc + ldm_k_col_off, KV_HALF_STRIDE)]));
                ldmatrix_x2(kf, k_addr);
                float tl[4] = {s_acc[0][ni][0], s_acc[0][ni][1], s_acc[1][ni][0], s_acc[1][ni][1]};
                mma16(tl, qf, kf);
                s_acc[0][ni][0] = tl[0]; s_acc[0][ni][1] = tl[1];
                s_acc[1][ni][0] = tl[2]; s_acc[1][ni][1] = tl[3];
            }
        }

        // Last 4 ki: reload Q from SMEM, K right half
        #pragma unroll
        for (int ki = 4; ki < 8; ki++) {
            uint32_t qf[4];
            ldmatrix_x4(qf, q_addrs_hi[ki-4]);
            int lc = (ki - 4) * MMA_K;
            #pragma unroll
            for (int ni = 0; ni < BLOCK_N/MMA_N; ni++) {
                uint32_t kf[2];
                int kn = ni * MMA_N + ldm_k_lane_row;
                uint32_t k_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&kr[cs][tma_sw(kn, lc + ldm_k_col_off, KV_HALF_STRIDE)]));
                ldmatrix_x2(kf, k_addr);
                float tl[4] = {s_acc[0][ni][0], s_acc[0][ni][1], s_acc[1][ni][0], s_acc[1][ni][1]};
                mma16(tl, qf, kf);
                s_acc[0][ni][0] = tl[0]; s_acc[0][ni][1] = tl[1];
                s_acc[1][ni][0] = tl[2]; s_acc[1][ni][1] = tl[3];
            }
        }

        // Scale + mask
        int kvs = kv * BLOCK_N;
        #pragma unroll
        for (int rh = 0; rh < 2; rh++)
            #pragma unroll
            for (int nt = 0; nt < BLOCK_N/MMA_N; nt++) {
                s_acc[rh][nt][0] *= scale; s_acc[rh][nt][1] *= scale;
                int ki0 = kvs + nt * MMA_N + t * 2;
                if (ki0 >= Skv) s_acc[rh][nt][0] = -FLT_MAX;
                if (ki0 + 1 >= Skv) s_acc[rh][nt][1] = -FLT_MAX;
            }

        // Softmax
        for (int rh = 0; rh < 2; rh++) {
            float tm = rmax[rh];
            #pragma unroll
            for (int nt = 0; nt < BLOCK_N/MMA_N; nt++) { tm = fmaxf(tm, s_acc[rh][nt][0]); tm = fmaxf(tm, s_acc[rh][nt][1]); }
            float nm = tm;
            nm = fmaxf(nm, __shfl_xor_sync(0xffffffff, nm, 1));
            nm = fmaxf(nm, __shfl_xor_sync(0xffffffff, nm, 2));
            float rs = __expf(rmax[rh] - nm); rsum[rh] *= rs;
            #pragma unroll
            for (int nt = 0; nt < 16; nt++) { o_acc[rh][nt][0] *= rs; o_acc[rh][nt][1] *= rs; }
            float ls = 0;
            #pragma unroll
            for (int nt = 0; nt < BLOCK_N/MMA_N; nt++) {
                s_acc[rh][nt][0] = __expf(s_acc[rh][nt][0] - nm);
                s_acc[rh][nt][1] = __expf(s_acc[rh][nt][1] - nm);
                ls += s_acc[rh][nt][0] + s_acc[rh][nt][1];
            }
            ls += __shfl_xor_sync(0xffffffff, ls, 1); ls += __shfl_xor_sync(0xffffffff, ls, 2);
            rsum[rh] += ls; rmax[rh] = nm;
        }

        // P@V
        #pragma unroll
        for (int ki = 0; ki < BLOCK_N/MMA_K; ki++) {
            uint32_t pf[4];
            pf[0] = pk_f2(s_acc[0][ki*2][0], s_acc[0][ki*2][1]);
            pf[1] = pk_f2(s_acc[1][ki*2][0], s_acc[1][ki*2][1]);
            pf[2] = pk_f2(s_acc[0][ki*2+1][0], s_acc[0][ki*2+1][1]);
            pf[3] = pk_f2(s_acc[1][ki*2+1][0], s_acc[1][ki*2+1][1]);

            #pragma unroll
            for (int di = 0; di < HEAD_DIM/MMA_N; di++) {
                uint32_t vf[2];
                int dc = di * MMA_N, vh = dc / HALF_HD, vlc = dc - vh * HALF_HD;
                __nv_bfloat16* vhp = (vh == 0) ? vl[cs] : vr[cs];
                int v_row = ki * MMA_K + ldm_v_row_off;
                uint32_t v_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&vhp[tma_sw(v_row, vlc, KV_HALF_STRIDE)]));
                ldmatrix_x2_trans(vf, v_addr);
                float ot[4] = {o_acc[0][di][0], o_acc[0][di][1], o_acc[1][di][0], o_acc[1][di][1]};
                mma16(ot, pf, vf);
                o_acc[0][di][0] = ot[0]; o_acc[0][di][1] = ot[1];
                o_acc[1][di][0] = ot[2]; o_acc[1][di][1] = ot[3];
            }
        }

        // Per-warp done signal (8 arrivals instead of 256)
        if (lane == 0) mb_arrive(&mbar_done[cs]);

        // Fire TMA for kv+2
        if (kv + 2 < num_kv && tid == 0) {
            mb_wait(&mbar_done[cs], done_phase[cs]);
            done_phase[cs] ^= 1;

            int next_row = kv_head * Skv + (kv + 2) * BLOCK_N;
            int hbytes = BLOCK_N * HALF_HD * 2;
            mb_expect(&mbar_kv[cs], 4 * hbytes);
            tma2d(kl[cs], &c_descs[0], 0, next_row, &mbar_kv[cs]);
            tma2d(kr[cs], &c_descs[1], 0, next_row, &mbar_kv[cs]);
            tma2d(vl[cs], &c_descs[2], 0, next_row, &mbar_kv[cs]);
            tma2d(vr[cs], &c_descs[3], 0, next_row, &mbar_kv[cs]);
        }
    }

    // Vectorized output
    __nv_bfloat16* o_ptr = O + head * Sq * HEAD_DIM;
    for (int rh = 0; rh < 2; rh++) {
        float inv = (rsum[rh] > 0) ? 1.0f / rsum[rh] : 0;
        int row = m_start + warp * WARP_M + g + rh * 8;
        if (row < Sq) {
            #pragma unroll
            for (int di = 0; di < 16; di++) {
                int c0 = di * MMA_N + t * 2;
                __nv_bfloat16 v0 = __float2bfloat16(o_acc[rh][di][0] * inv);
                __nv_bfloat16 v1 = __float2bfloat16(o_acc[rh][di][1] * inv);
                uint32_t packed;
                asm("mov.b32 %0, {%1, %2};" : "=r"(packed) : "h"(*(uint16_t*)&v0), "h"(*(uint16_t*)&v1));
                *reinterpret_cast<uint32_t*>(&o_ptr[row * HEAD_DIM + c0]) = packed;
            }
        }
    }
    if (LSE && t == 0) for (int rh = 0; rh < 2; rh++) {
        int row = m_start + warp * WARP_M + g + rh * 8;
        if (row < Sq) LSE[head * Sq + row] = rmax[rh] + logf(rsum[rh]);
    }
}

extern "C" void sm120_flash_attn_forward(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    int batch, int Hq, int Hkv, int Sq, int Skv, int hd, cudaStream_t stream
) {
    float sc = 1.0f / sqrtf((float)hd);
    int total_kv = batch * Hkv, total_rows = total_kv * Skv;

    CUtensorMap kdl __attribute__((aligned(64))), kdr __attribute__((aligned(64)));
    CUtensorMap vdl __attribute__((aligned(64))), vdr __attribute__((aligned(64)));
    cuuint64_t dims[2] = {(cuuint64_t)HALF_HD, (cuuint64_t)total_rows};
    cuuint64_t strides[1] = {(cuuint64_t)(HEAD_DIM * 2)};
    cuuint32_t box[2] = {(cuuint32_t)HALF_HD, (cuuint32_t)BLOCK_N};
    cuuint32_t es[2] = {1, 1};

    cuTensorMapEncodeTiled(&kdl, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)K, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    cuTensorMapEncodeTiled(&kdr, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)(K + HALF_HD), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    cuTensorMapEncodeTiled(&vdl, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)V, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    cuTensorMapEncodeTiled(&vdr, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)(V + HALF_HD), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    cudaMemcpyToSymbolAsync(c_descs, &kdl, sizeof(CUtensorMap), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(c_descs, &kdr, sizeof(CUtensorMap), sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(c_descs, &vdl, sizeof(CUtensorMap), 2*sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(c_descs, &vdr, sizeof(CUtensorMap), 3*sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);

    dim3 grid((Sq + BLOCK_M - 1) / BLOCK_M, batch * Hq);
    cudaFuncSetAttribute(sm120_fa_v4, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    cudaFuncSetAttribute(sm120_fa_v4, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
    sm120_fa_v4<<<grid, BLOCK_SIZE, SMEM_BYTES, stream>>>(Q, O, L, Sq, Skv, Hq, Hkv, sc);
}
