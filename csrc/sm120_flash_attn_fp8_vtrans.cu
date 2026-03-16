/**
 * SM120 Flash Attention — FP8 with Pre-Transposed V
 *
 * Key insight: store V as V^T[d][k] in global memory BEFORE the kernel.
 * TMA loads V^T rows as contiguous data. ldmatrix.x2 gives correct B
 * fragments directly — no transpose, no scalar loads, no split-D.
 *
 * Architecture:
 * - BM=128, BN=128 double-buffered, HD=128, 8 warps (256 threads)
 * - FP8 e4m3 for Q (converted from BF16), K, V^T
 * - SMEM: Q_FP8(16KB) + K[2DS](16KB) + VT[2DS](16KB) = 48KB (fits easily)
 *   Actually with BN=128: K[2DS] = 2*128*64*1 = 16KB, VT[2DS] = 2*128*64*1 = 16KB
 *   Total = 16 + 32 + 32 = 80KB
 * - mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
 * - HD/32 = 4 ki iterations (half of BF16's 8)
 * - Target: 400+ TFLOPS
 *
 * TMA layout for K: [HALF_HD, total_rows] with UINT8, stride=HD bytes
 *   Inner dim = HALF_HD (64 bytes), box = [64, BN]
 * TMA layout for V^T: [Skv, total_heads*HALF_HD] with UINT8, stride=Skv bytes
 *   Inner dim = BN (128 bytes), box = [BN, HALF_HD]
 *   Actually V^T is [d][k] so we tile [BN, HALF_HD] from the [Skv, HD] transposed layout
 *
 * V^T memory layout: V_T[head][d][k] where d=HEAD_DIM, k=Skv
 *   For TMA: treat as 2D [Skv, total_d] where total_d = total_kv_heads * HD
 *   Inner dim = Skv, load [BN] chunk. Outer dim = d, load [HALF_HD] chunk.
 *   Stride between d-rows = Skv bytes.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define BLOCK_M 128
#define BLOCK_N 128
#define HEAD_DIM 128
#define HALF_HD 64
#define NUM_STAGES 2
#define WARP_SIZE 32
#define NUM_WARPS 8
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE)
#define MMA_M 16
#define MMA_N 8
#define MMA_K 32

#define Q_STRIDE HEAD_DIM
#define Q_ELEMS (BLOCK_M * Q_STRIDE)  // 128*128 = 16384 FP8 bytes = 16KB

// K: unsplit [BN, HEAD_DIM] FP8 = 128*128 = 16KB per stage. SWIZZLE_128B.
// FP8 K row = HD = 128 bytes → supports SWIZZLE_128B directly (no KL/KR split).
#define K_STRIDE HEAD_DIM
#define K_ELEMS (BLOCK_N * K_STRIDE)  // 16384 bytes = 16KB

// V^T: unsplit [HEAD_DIM, BLOCK_N] FP8 = 128*128 = 16KB per stage. SWIZZLE_128B.
// V^T row = BN = 128 bytes → supports SWIZZLE_128B directly (no VTL/VTR split).
#define VT_STRIDE BLOCK_N
#define VT_ELEMS (HEAD_DIM * VT_STRIDE)  // 16384 bytes = 16KB

// SMEM: Q(16KB) + 2 stages × (K(16KB) + VT(16KB)) + mbar
// = 16384 + 2 * 32768 + 128 = 82048 bytes ≈ 80KB
#define STAGE_BYTES (K_ELEMS + VT_ELEMS)  // 32768
#define SMEM_BYTES (Q_ELEMS + NUM_STAGES * STAGE_BYTES + 128)

// Swizzle for Q: 256-byte rows (128 FP8 = 128 bytes), XOR bits for bank conflict avoidance
// FP8 Q: 128 elements per row = 128 bytes. 128B swizzle: XOR bits[6:4] with row[2:0]
__device__ __forceinline__ int q_sw_fp8(int r, int c) {
    // FP8: 1 byte per element, row stride = HEAD_DIM = 128 bytes
    // SWIZZLE_128B: XOR byte_offset[6:4] with row[2:0]
    int swizzled = c ^ ((r & 7) << 4);
    return r * HEAD_DIM + swizzled;
}

// Unified swizzle for K and V^T: both have 128-byte rows, TMA SWIZZLE_128B.
// XOR byte_offset[6:4] with row[2:0]. Stride = 128 bytes.
__device__ __forceinline__ int sw128(int r, int c) {
    return r * 128 + (c ^ ((r & 7) << 4));
}

__device__ __forceinline__ void mma_fp8(float c[4], uint32_t a[4], uint32_t b[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
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
__device__ __forceinline__ void cp16(void* s, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(s))), "l"(g));
}

// 2 TMA descriptors: K (unsplit) and VT (unsplit)
__constant__ CUtensorMap c_descs[2];

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
sm120_fa_fp8_vtrans(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_fp8_e4m3* __restrict__ K_fp8,
    const __nv_fp8_e4m3* __restrict__ VT_fp8,  // V^T[head][d][k], pre-transposed
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
    // Layout: Q_FP8(16KB) | K[0](16KB) | VT[0](16KB) | K[1](16KB) | VT[1](16KB) | mbar
    __nv_fp8_e4m3* q_fp8 = (__nv_fp8_e4m3*)smem;
    __nv_fp8_e4m3* stage_base = q_fp8 + Q_ELEMS;
    #define K_S(s) (stage_base + (s) * STAGE_BYTES)
    #define VT_S(s) (K_S(s) + K_ELEMS)
    uint64_t* mbar_kv = (uint64_t*)((char*)stage_base + NUM_STAGES * STAGE_BYTES);
    uint64_t* mbar_done = mbar_kv + 2;

    // ============ Load Q BF16 → convert to FP8 in SMEM ============
    // Phase 1: Load BF16 Q into SMEM with swizzle (reuse q_fp8 area as __nv_bfloat16 temp)
    // We need 128*128*2 = 32KB for BF16 Q but only have 16KB Q area.
    // Solution: convert in-register, write FP8 directly.
    // Load Q BF16 from global → convert to FP8 → store linearly in q_fp8
    for (int i = tid; i < Q_ELEMS; i += BLOCK_SIZE) {
        int row = i / HEAD_DIM, col = i % HEAD_DIM;
        int qr = m_start + row;
        float val = (qr < Sq) ? __bfloat162float(q_ptr[qr * HEAD_DIM + col]) : 0.0f;
        q_fp8[q_sw_fp8(row, col)] = __nv_fp8_e4m3(val);
    }
    __syncthreads();

    if (tid == 0) {
        mb_init(&mbar_kv[0], 1);
        mb_init(&mbar_kv[1], 1);
        mb_init(&mbar_done[0], NUM_WARPS);
        mb_init(&mbar_done[1], NUM_WARPS);
    }
    __syncthreads();

    // ldmatrix lane mapping for FP8 m16n8k32:
    // A (Q/P) fragment: ldmatrix.x4 loads 4 x m8n8 = m16n8k32 A fragment
    //   Each thread holds 4 x uint32_t = 16 bytes = 16 FP8 values
    //   Row = warp*16 + (lane&7) + ((lane>>3)&1)*8, col_base = (lane>>4)*16
    const int ldm_q_row = warp * MMA_M + (lane & 7) + ((lane >> 3) & 1) * 8;
    const int ldm_q_col_base = (lane >> 4) * 16;  // 0 or 16 within the k=32 chunk

    // B (K/V) fragment: ldmatrix.x2 loads m8n8.x2 = 2 matrices for k=32
    //   Row within tile = lane & 7, col = ((lane>>3)&1)*16
    const int ldm_k_lane_row = lane & 7;
    const int ldm_k_col_off = ((lane >> 3) & 1) * 16;  // 0 or 16 FP8 elements

    // For V^T: same pattern as K since V^T rows are contiguous
    // V^T[d][k] loaded as [HALF_HD, BN]. ldmatrix.x2 on V^T tile.
    // Row = d-index within tile, col = k-index within BN chunk

    // TMA bytes per stage: K(16KB) + VT(16KB) = 32KB
    int tma_stage_bytes = K_ELEMS + VT_ELEMS;

    // VT y-coordinate: kv_head * HEAD_DIM (d-offset in unsplit layout)
    int vt_y_base = kv_head * HEAD_DIM;

    // Prefetch stages 0 and 1
    if (tid == 0 && num_kv > 0) {
        int kv_row = kv_head * Skv;
        mb_expect(&mbar_kv[0], tma_stage_bytes);
        tma2d(K_S(0),  &c_descs[0], 0, kv_row, &mbar_kv[0]);          // K: x=0 (inner=HD), y=kv_row
        tma2d(VT_S(0), &c_descs[1], 0, vt_y_base, &mbar_kv[0]);       // VT: x=0 (inner=Skv), y=d_base
    }
    if (tid == 0 && num_kv > 1) {
        int kv_row = kv_head * Skv + BLOCK_N;
        mb_expect(&mbar_kv[1], tma_stage_bytes);
        tma2d(K_S(1),  &c_descs[0], 0, kv_row, &mbar_kv[1]);
        tma2d(VT_S(1), &c_descs[1], BLOCK_N, vt_y_base, &mbar_kv[1]); // VT: x=BN (next k chunk)
    }

    // Output accumulators: 16 di tiles × 2 row-halves × 2 values = 64 floats
    float o_acc[2][16][2];
    float rmax[2] = {-FLT_MAX, -FLT_MAX}, rsum[2] = {0, 0};
    #pragma unroll
    for (int rh = 0; rh < 2; rh++)
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) o_acc[rh][nt][0] = o_acc[rh][nt][1] = 0;

    uint32_t phases = 0;
    #define KV_PHASE(s) ((phases >> (s)) & 1)
    #define DONE_PHASE(s) ((phases >> ((s)+2)) & 1)
    #define FLIP_KV(s) (phases ^= (1u << (s)))
    #define FLIP_DONE(s) (phases ^= (1u << ((s)+2)))

    for (int kv = 0; kv < num_kv; kv++) {
        int cs = kv & 1;

        mb_wait(&mbar_kv[cs], KV_PHASE(cs));
        FLIP_KV(cs);

        // ============ Q@K^T ============
        // FP8 m16n8k32: HD/32 = 4 ki iterations
        // BN=128: 128/8 = 16 ni tiles
        float s_acc[2][BLOCK_N/MMA_N][2];
        #pragma unroll
        for (int rh = 0; rh < 2; rh++)
            #pragma unroll
            for (int nt = 0; nt < BLOCK_N/MMA_N; nt++)
                s_acc[rh][nt][0] = s_acc[rh][nt][1] = 0;

        __nv_fp8_e4m3* k_s = K_S(cs);

        #pragma unroll
        for (int ki = 0; ki < HEAD_DIM/MMA_K; ki++) {
            // Load Q fragment for this ki
            uint32_t qf[4];
            int q_col = ki * MMA_K + ldm_q_col_base;
            uint32_t q_addr = static_cast<uint32_t>(
                __cvta_generic_to_shared(&q_fp8[q_sw_fp8(ldm_q_row, q_col)]));
            ldmatrix_x4(qf, q_addr);

            // K unsplit: column = ki * MMA_K within full HEAD_DIM
            int k_col = ki * MMA_K + ldm_k_col_off;

            #pragma unroll
            for (int ni = 0; ni < BLOCK_N/MMA_N; ni++) {
                uint32_t kf[2];
                int kn = ni * MMA_N + ldm_k_lane_row;
                uint32_t k_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&k_s[sw128(kn, k_col)]));
                ldmatrix_x2(kf, k_addr);
                float tl[4] = {s_acc[0][ni][0], s_acc[0][ni][1], s_acc[1][ni][0], s_acc[1][ni][1]};
                mma_fp8(tl, qf, kf);
                s_acc[0][ni][0]=tl[0]; s_acc[0][ni][1]=tl[1]; s_acc[1][ni][0]=tl[2]; s_acc[1][ni][1]=tl[3];
            }
        }

        // Scale + mask
        int kvs = kv * BLOCK_N;
        #pragma unroll
        for (int rh=0;rh<2;rh++)
            #pragma unroll
            for (int nt=0;nt<BLOCK_N/MMA_N;nt++) {
                s_acc[rh][nt][0]*=scale; s_acc[rh][nt][1]*=scale;
                int ki0=kvs+nt*MMA_N+t*2;
                if(ki0>=Skv) s_acc[rh][nt][0]=-FLT_MAX;
                if(ki0+1>=Skv) s_acc[rh][nt][1]=-FLT_MAX;
            }

        // Online softmax
        for (int rh=0;rh<2;rh++) {
            float tm=rmax[rh];
            #pragma unroll
            for (int nt=0;nt<BLOCK_N/MMA_N;nt++) { tm=fmaxf(tm,s_acc[rh][nt][0]); tm=fmaxf(tm,s_acc[rh][nt][1]); }
            float nm=tm;
            nm=fmaxf(nm,__shfl_xor_sync(0xffffffff,nm,1));
            nm=fmaxf(nm,__shfl_xor_sync(0xffffffff,nm,2));
            float rs=__expf(rmax[rh]-nm); rsum[rh]*=rs;
            #pragma unroll
            for (int nt=0;nt<16;nt++) { o_acc[rh][nt][0]*=rs; o_acc[rh][nt][1]*=rs; }
            float ls=0;
            #pragma unroll
            for (int nt=0;nt<BLOCK_N/MMA_N;nt++) {
                float e0=__expf(s_acc[rh][nt][0]-nm), e1=__expf(s_acc[rh][nt][1]-nm);
                s_acc[rh][nt][0]=e0; s_acc[rh][nt][1]=e1;
                ls=fmaf(1.0f,e0,ls); ls=fmaf(1.0f,e1,ls);
            }
            ls+=__shfl_xor_sync(0xffffffff,ls,1); ls+=__shfl_xor_sync(0xffffffff,ls,2);
            rsum[rh]+=ls; rmax[rh]=nm;
        }

        // ============ P@V^T ============
        // Write P to SMEM for correct A fragment layout via ldmatrix.x4.
        // Reuse K_S(cs) as P buffer (K consumed after Q@K^T). 16KB = 128×128 FP8 ✓
        // P uses sw128 (same SWIZZLE_128B pattern, 128-byte rows).

        // Write P to SMEM (reuse K buffer)
        __nv_fp8_e4m3* p_smem = k_s;  // K_S(cs), 16KB, same 128-byte row stride
        #pragma unroll
        for (int rh = 0; rh < 2; rh++) {
            int p_row = warp * MMA_M + g + rh * 8;
            #pragma unroll
            for (int nt = 0; nt < BLOCK_N/MMA_N; nt++) {
                int p_col0 = nt * MMA_N + t * 2;
                __nv_fp8_e4m3 fp0(s_acc[rh][nt][0]);
                __nv_fp8_e4m3 fp1(s_acc[rh][nt][1]);
                p_smem[sw128(p_row, p_col0)] = fp0;
                p_smem[sw128(p_row, p_col0 + 1)] = fp1;
            }
        }
        __syncwarp();

        // P@V^T: iterate over k-chunks (BN/32 = 4 ki iterations)
        __nv_fp8_e4m3* vt_s = VT_S(cs);
        #pragma unroll
        for (int ki = 0; ki < BLOCK_N/MMA_K; ki++) {
            // Load P fragment via ldmatrix.x4
            uint32_t pf[4];
            int p_ldm_row = warp * MMA_M + (lane & 7) + ((lane >> 3) & 1) * 8;
            int p_ldm_col = ki * MMA_K + (lane >> 4) * 16;
            uint32_t p_addr = static_cast<uint32_t>(
                __cvta_generic_to_shared(&p_smem[sw128(p_ldm_row, p_ldm_col)]));
            ldmatrix_x4(pf, p_addr);

            // Iterate over d tiles (output dimension) — VT unsplit
            int v_k_base = ki * MMA_K;
            #pragma unroll
            for (int di = 0; di < HEAD_DIM/MMA_N; di++) {
                uint32_t vf[2];
                int vt_r = di * MMA_N + ldm_k_lane_row;  // d position in full HEAD_DIM
                int vt_c = v_k_base + ldm_k_col_off;
                uint32_t v_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&vt_s[sw128(vt_r, vt_c)]));
                ldmatrix_x2(vf, v_addr);

                float ot[4] = {o_acc[0][di][0], o_acc[0][di][1], o_acc[1][di][0], o_acc[1][di][1]};
                mma_fp8(ot, pf, vf);
                o_acc[0][di][0] = ot[0]; o_acc[0][di][1] = ot[1];
                o_acc[1][di][0] = ot[2]; o_acc[1][di][1] = ot[3];
            }
        }

        // Per-warp done signal
        if (lane == 0) mb_arrive(&mbar_done[cs]);

        // Fire TMA for kv+2
        if (kv + 2 < num_kv && tid == 0) {
            mb_wait(&mbar_done[cs], DONE_PHASE(cs));
            FLIP_DONE(cs);
            int next_row = kv_head * Skv + (kv+2) * BLOCK_N;
            int next_vt_x = (kv+2) * BLOCK_N;
            mb_expect(&mbar_kv[cs], tma_stage_bytes);
            tma2d(K_S(cs),  &c_descs[0], 0, next_row, &mbar_kv[cs]);
            tma2d(VT_S(cs), &c_descs[1], next_vt_x, vt_y_base, &mbar_kv[cs]);
        }
    }
    #undef K_S
    #undef VT_S
    #undef KV_PHASE
    #undef DONE_PHASE
    #undef FLIP_KV
    #undef FLIP_DONE

    // ============ Write output ============
    __nv_bfloat16* o_ptr = O + head * Sq * HEAD_DIM;
    for (int rh=0;rh<2;rh++) {
        float inv=(rsum[rh]>0)?1.0f/rsum[rh]:0;
        int row=m_start+warp*MMA_M+g+rh*8;
        if(row<Sq) {
            #pragma unroll
            for (int di=0;di<16;di++) {
                int c0=di*MMA_N+t*2;
                __nv_bfloat16 v0=__float2bfloat16(o_acc[rh][di][0]*inv);
                __nv_bfloat16 v1=__float2bfloat16(o_acc[rh][di][1]*inv);
                uint32_t packed;
                asm("mov.b32 %0, {%1, %2};" : "=r"(packed) : "h"(*(uint16_t*)&v0), "h"(*(uint16_t*)&v1));
                *reinterpret_cast<uint32_t*>(&o_ptr[row*HEAD_DIM+c0])=packed;
            }
        }
    }
    if(LSE&&t==0) for(int rh=0;rh<2;rh++) { int row=m_start+warp*MMA_M+g+rh*8; if(row<Sq) LSE[head*Sq+row]=rmax[rh]+logf(rsum[rh]); }
}

// ============================================================================
// Conversion kernels
// ============================================================================

// BF16 → FP8 e4m3 (in-place layout, no transpose)
__global__ void bf16_to_fp8_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_fp8_e4m3* __restrict__ dst,
    int total_elems
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elems) {
        dst[idx] = __nv_fp8_e4m3(__bfloat162float(src[idx]));
    }
}

// BF16 → FP8 e4m3 with transpose: V[head][k][d] → VT[head][d][k]
// Each thread handles one element
__global__ void bf16_to_fp8_transpose_kernel(
    const __nv_bfloat16* __restrict__ V,  // [total_kv, Skv, hd]
    __nv_fp8_e4m3* __restrict__ VT,       // [total_kv, hd, Skv]
    int Skv, int hd
) {
    int head = blockIdx.z;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < hd && k < Skv) {
        float val = __bfloat162float(V[head * Skv * hd + k * hd + d]);
        VT[head * hd * Skv + d * Skv + k] = __nv_fp8_e4m3(val);
    }
}

// ============================================================================
// Host launch
// ============================================================================
extern "C" void sm120_flash_attn_fp8_vtrans_forward(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    int batch, int Hq, int Hkv, int Sq, int Skv, int hd, cudaStream_t stream
) {
    float sc = 1.0f / sqrtf((float)hd);
    int total_kv = batch * Hkv;
    int total_kv_rows = total_kv * Skv;

    // ============ Convert K to FP8 ============
    __nv_fp8_e4m3* K_fp8;
    size_t k_bytes = (size_t)total_kv_rows * hd;
    cudaMalloc(&K_fp8, k_bytes);
    {
        int threads = 256;
        int blocks = ((int)k_bytes + threads - 1) / threads;
        bf16_to_fp8_kernel<<<blocks, threads, 0, stream>>>(K, K_fp8, (int)k_bytes);
    }

    // ============ Pre-transpose V and convert to FP8 ============
    // V is [total_kv, Skv, hd] BF16. We need VT[total_kv, hd, Skv] FP8.
    __nv_fp8_e4m3* VT_fp8;
    size_t vt_bytes = (size_t)total_kv * hd * Skv;
    cudaMalloc(&VT_fp8, vt_bytes);
    {
        dim3 threads(32, 8);  // k, d
        dim3 blocks((Skv + 31) / 32, (hd + 7) / 8, total_kv);
        bf16_to_fp8_transpose_kernel<<<blocks, threads, 0, stream>>>(V, VT_fp8, Skv, hd);
    }

    // ============ TMA descriptors (2 total: K and VT) ============
    // K: 2D [HEAD_DIM, total_kv_rows], SWIZZLE_128B (128-byte rows for FP8)
    CUtensorMap k_desc __attribute__((aligned(64)));
    {
        cuuint64_t dims[2] = {(cuuint64_t)hd, (cuuint64_t)total_kv_rows};
        cuuint64_t strides[1] = {(cuuint64_t)(hd * 1)};
        cuuint32_t box[2] = {(cuuint32_t)hd, (cuuint32_t)BLOCK_N};
        cuuint32_t es[2] = {1, 1};
        cuTensorMapEncodeTiled(&k_desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)K_fp8,
            dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    // VT: 2D [Skv, total_kv * HD], SWIZZLE_128B (128-byte inner tiles)
    // VT[head][d][k]: inner dim = Skv, outer dim = head*HD + d
    // Box: [BLOCK_N, HEAD_DIM] — loads full HEAD_DIM d-positions per tile
    CUtensorMap vt_desc __attribute__((aligned(64)));
    {
        cuuint64_t dims[2] = {(cuuint64_t)Skv, (cuuint64_t)(total_kv * hd)};
        cuuint64_t strides[1] = {(cuuint64_t)(Skv * 1)};
        cuuint32_t box[2] = {(cuuint32_t)BLOCK_N, (cuuint32_t)hd};
        cuuint32_t es[2] = {1, 1};
        cuTensorMapEncodeTiled(&vt_desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)VT_fp8,
            dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    }

    cudaMemcpyToSymbolAsync(c_descs, &k_desc, sizeof(CUtensorMap), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(c_descs, &vt_desc, sizeof(CUtensorMap), sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);

    dim3 grid((Sq + BLOCK_M - 1) / BLOCK_M, batch * Hq);
    cudaFuncSetAttribute(sm120_fa_fp8_vtrans, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    cudaFuncSetAttribute(sm120_fa_fp8_vtrans, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
    sm120_fa_fp8_vtrans<<<grid, BLOCK_SIZE, SMEM_BYTES, stream>>>(Q, K_fp8, VT_fp8, O, L, Sq, Skv, Hq, Hkv, sc);

    cudaFreeAsync(K_fp8, stream);
    cudaFreeAsync(VT_fp8, stream);
}
