/**
 * SM120 Flash Attention — ldmatrix + Register-Resident P (BN128 variant)
 *
 * Key optimizations:
 * 1. ldmatrix.x4 for Q loads (1 instruction → 4 regs, was 4× pk calls)
 * 2. ldmatrix.x2 for K loads (1 instruction → 2 regs, was 2× pk calls)
 * 3. ldmatrix.x2 for V loads (same)
 * 4. P stays in registers — NO SMEM staging
 *
 * Reduces SMEM load instructions by ~2-4x, cutting L1/TEX pressure.
 *
 * BM128, BN128, HD128, 8 warps, single-stage TMA SWIZZLE_128B.
 * Arithmetic intensity: 42.7 (vs 32 for BN64). Halves outer loop iterations.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define BLOCK_M 128
#define BLOCK_N 128
#define HEAD_DIM 128
#define HALF_HD 64
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
#define SMEM_BYTES ((Q_ELEMS + 4 * KV_HALF_ELEMS) * 2 + 64)

// Software swizzle for Q: 256-byte rows, XOR bits[7:5]
__device__ __forceinline__ int q_sw(int r, int c, int s) {
    return r * s + ((c*2) ^ ((r&7)<<5)) / 2;
}
// TMA swizzle for K/V: 128-byte rows, XOR bits[6:4]
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

// ldmatrix helpers
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
// Transposed ldmatrix for V: swaps row/col in the output mapping
// Thread (g,t) gets pack(M[2t, g], M[2t+1, g]) instead of pack(M[g, 2t], M[g, 2t+1])
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&r)[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
sm120_fa_ldm(
    const __nv_bfloat16* __restrict__ Q,
    const CUtensorMap* __restrict__ Kdl, const CUtensorMap* __restrict__ Kdr,
    const CUtensorMap* __restrict__ Vdl, const CUtensorMap* __restrict__ Vdr,
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
    __nv_bfloat16* k_left = q_s + Q_ELEMS;
    __nv_bfloat16* k_right = k_left + KV_HALF_ELEMS;
    __nv_bfloat16* v_left = k_right + KV_HALF_ELEMS;
    __nv_bfloat16* v_right = v_left + KV_HALF_ELEMS;
    uint64_t* mbar = (uint64_t*)((char*)v_right + KV_HALF_ELEMS * 2);

    // Load Q with software swizzle
    for (int i = tid; i < Q_ELEMS; i += BLOCK_SIZE) q_s[i] = __float2bfloat16(0.0f);
    __syncthreads();
    for (int i = tid; i < Q_ELEMS / 8; i += BLOCK_SIZE) {
        int row = (i*8)/HEAD_DIM, col = (i*8)%HEAD_DIM, qr = m_start+row;
        if (qr < Sq) cp16(&q_s[q_sw(row,col,Q_STRIDE)], &q_ptr[qr*HEAD_DIM+col]);
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n");
    __syncthreads();

    if (tid == 0) mb_init(mbar, 1);
    __syncthreads();

    // Prefetch KV block 0 before entering the loop
    if (tid == 0 && num_kv > 0) {
        int kv_row = kv_head * Skv;
        int hbytes = BLOCK_N * HALF_HD * 2;
        mb_expect(mbar, 4 * hbytes);
        tma2d(k_left,  Kdl, 0, kv_row, mbar);
        tma2d(k_right, Kdr, 0, kv_row, mbar);
        tma2d(v_left,  Vdl, 0, kv_row, mbar);
        tma2d(v_right, Vdr, 0, kv_row, mbar);
    }

    // Precompute Q ldmatrix row/col for this thread
    // For ldmatrix.x4 loading A operand of m16n8k16:
    // Lane mapping: matrix_row = (lane&7) + ((lane>>3)&1)*8, col_half = lane>>4
    const int ldm_q_row = warp * WARP_M + (lane & 7) + ((lane >> 3) & 1) * 8;
    const int ldm_q_col_base = (lane >> 4) * 8;  // 0 or 8 within MMA_K=16 tile

    // Precompute K ldmatrix offsets (B operand of Q@K^T, no transpose)
    // K[N=8][K=16]: lanes 0-7 load N rows 0-7 at K cols 0-7, lanes 8-15 at K cols 8-15
    const int ldm_k_lane_row = lane & 7;        // N dimension (0..7)
    const int ldm_k_col_off = ((lane >> 3) & 1) * 8;  // K half (0 or 8)

    // Precompute V ldmatrix offsets (B operand of P@V, TRANSPOSED)
    // V[K_p=16][D=8]: lanes 0-7 load K_p rows 0-7, lanes 8-15 load K_p rows 8-15
    const int ldm_v_row_off = (lane & 7) + ((lane >> 3) & 1) * 8;  // K_p dim (0..15)

    // Preload ALL Q fragments into registers — eliminates ALL Q SMEM loads from loop
    uint32_t qf_all[HEAD_DIM/MMA_K][4];  // 8 × 4 = 32 registers
    #pragma unroll
    for (int ki = 0; ki < HEAD_DIM/MMA_K; ki++) {
        int q_col = ki * MMA_K + ldm_q_col_base;
        uint32_t q_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&q_s[q_sw(ldm_q_row, q_col, Q_STRIDE)]));
        ldmatrix_x4(qf_all[ki], q_addr);
    }

    float o_acc[2][16][2];
    float rmax[2] = {-FLT_MAX, -FLT_MAX}, rsum[2] = {0, 0};
    #pragma unroll
    for (int rh = 0; rh < 2; rh++)
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) o_acc[rh][nt][0] = o_acc[rh][nt][1] = 0;

    int phase = 0;
    for (int kv = 0; kv < num_kv; kv++) {
        // Wait for TMA (prefetched before loop or at end of prev iteration)
        // mbarrier phase tracking — NO mb_init in loop, NO extra __syncthreads
        mb_wait(mbar, phase);
        phase ^= 1;

        // ============ Q@K^T with ldmatrix ============
        float s_acc[2][BLOCK_N/MMA_N][2];
        #pragma unroll
        for (int rh = 0; rh < 2; rh++)
            #pragma unroll
            for (int nt = 0; nt < BLOCK_N/MMA_N; nt++)
                s_acc[rh][nt][0] = s_acc[rh][nt][1] = 0;

        #pragma unroll
        for (int ki = 0; ki < HEAD_DIM/MMA_K; ki++) {
            // Q fragment — preloaded in registers, ZERO SMEM loads!
            uint32_t (&qf)[4] = qf_all[ki];

            // K fragment: determine half
            int kcol = ki * MMA_K;
            int half = kcol / HALF_HD;
            int lc = kcol - half * HALF_HD;
            __nv_bfloat16* kh = (half == 0) ? k_left : k_right;

            #pragma unroll
            for (int ni = 0; ni < BLOCK_N/MMA_N; ni++) {
                // K fragment via ldmatrix.x2 — ONE instruction for 2 registers!
                uint32_t kf[2];
                {
                    int kn = ni * MMA_N + ldm_k_lane_row;
                    int k_col = lc + ldm_k_col_off;
                    uint32_t k_addr = static_cast<uint32_t>(
                        __cvta_generic_to_shared(&kh[tma_sw(kn, k_col, KV_HALF_STRIDE)]));
                    ldmatrix_x2(kf, k_addr);
                }

                float tl[4] = {s_acc[0][ni][0], s_acc[0][ni][1], s_acc[1][ni][0], s_acc[1][ni][1]};
                mma16(tl, qf, kf);
                s_acc[0][ni][0] = tl[0]; s_acc[0][ni][1] = tl[1];
                s_acc[1][ni][0] = tl[2]; s_acc[1][ni][1] = tl[3];
            }
        }

        // ============ Scale + Mask ============
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

        // ============ Softmax ============
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

        // ============ P@V — register P + ldmatrix V ============
        #pragma unroll
        for (int ki = 0; ki < BLOCK_N/MMA_K; ki++) {
            // P fragment from registers (s_acc)
            uint32_t pf[4];
            pf[0] = pk_f2(s_acc[0][ki*2][0], s_acc[0][ki*2][1]);
            pf[1] = pk_f2(s_acc[1][ki*2][0], s_acc[1][ki*2][1]);
            pf[2] = pk_f2(s_acc[0][ki*2+1][0], s_acc[0][ki*2+1][1]);
            pf[3] = pk_f2(s_acc[1][ki*2+1][0], s_acc[1][ki*2+1][1]);

            #pragma unroll
            for (int di = 0; di < HEAD_DIM/MMA_N; di++) {
                // V fragment via ldmatrix.x2.trans
                // V[K_p][D]: rows = K_p dim (0..15), cols = D dim (8 elements)
                // .trans maps: thread (g,t) gets pack(V[2t, g], V[2t+1, g])
                uint32_t vf[2];
                {
                    int dc = di * MMA_N;
                    int vh = dc / HALF_HD;
                    int vlc = dc - vh * HALF_HD;
                    __nv_bfloat16* vhp = (vh == 0) ? v_left : v_right;
                    int v_row = ki * MMA_K + ldm_v_row_off;  // 0..15 (K_p dimension)
                    int v_col = vlc;  // D dimension start (NO col offset!)
                    uint32_t v_addr = static_cast<uint32_t>(
                        __cvta_generic_to_shared(&vhp[tma_sw(v_row, v_col, KV_HALF_STRIDE)]));
                    ldmatrix_x2_trans(vf, v_addr);
                }
                float ot[4] = {o_acc[0][di][0], o_acc[0][di][1], o_acc[1][di][0], o_acc[1][di][1]};
                mma16(ot, pf, vf);
                o_acc[0][di][0] = ot[0]; o_acc[0][di][1] = ot[1];
                o_acc[1][di][0] = ot[2]; o_acc[1][di][1] = ot[3];
            }
        }
        // Ensure all threads done reading KV before next TMA overwrites
        __syncthreads();
        // Fire TMA for NEXT iteration (overlaps with next mb_wait spin)
        if (kv + 1 < num_kv && tid == 0) {
            int next_row = kv_head * Skv + (kv + 1) * BLOCK_N;
            int hbytes = BLOCK_N * HALF_HD * 2;
            mb_expect(mbar, 4 * hbytes);
            tma2d(k_left,  Kdl, 0, next_row, mbar);
            tma2d(k_right, Kdr, 0, next_row, mbar);
            tma2d(v_left,  Vdl, 0, next_row, mbar);
            tma2d(v_right, Vdr, 0, next_row, mbar);
        }
    }

    // Output
    __nv_bfloat16* o_ptr = O + head * Sq * HEAD_DIM;
    for (int rh = 0; rh < 2; rh++) {
        float inv = (rsum[rh] > 0) ? 1.0f / rsum[rh] : 0;
        int row = m_start + warp * WARP_M + g + rh * 8;
        if (row < Sq) {
            #pragma unroll
            for (int di = 0; di < 16; di++) {
                int c0 = di * MMA_N + t * 2;
                o_ptr[row * HEAD_DIM + c0] = __float2bfloat16(o_acc[rh][di][0] * inv);
                o_ptr[row * HEAD_DIM + c0 + 1] = __float2bfloat16(o_acc[rh][di][1] * inv);
            }
        }
    }
    if (LSE && t == 0) for (int rh = 0; rh < 2; rh++) { int row = m_start + warp * WARP_M + g + rh * 8; if (row < Sq) LSE[head * Sq + row] = rmax[rh] + logf(rsum[rh]); }
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
    CUtensorMap* dd; cudaMalloc(&dd, 4 * sizeof(CUtensorMap));
    cudaMemcpyAsync(dd, &kdl, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dd + 1, &kdr, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dd + 2, &vdl, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dd + 3, &vdr, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
    dim3 grid((Sq + BLOCK_M - 1) / BLOCK_M, batch * Hq);
    cudaFuncSetAttribute(sm120_fa_ldm, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    cudaFuncSetAttribute(sm120_fa_ldm, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
    sm120_fa_ldm<<<grid, BLOCK_SIZE, SMEM_BYTES, stream>>>(Q, dd, dd + 1, dd + 2, dd + 3, O, L, Sq, Skv, Hq, Hkv, sc);
    cudaFree(dd);
}
