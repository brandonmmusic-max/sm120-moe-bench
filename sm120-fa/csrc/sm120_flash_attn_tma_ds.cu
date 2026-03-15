/**
 * SM120 Flash Attention — TMA Double-Stage with SWIZZLE_128B
 *
 * BM128+BN32 with double-buffered TMA loading.
 * While consumers compute on stage N, thread 0 issues TMA for stage N+1.
 * This overlaps memory transfer with MMA compute.
 *
 * SMEM: Q(32KB) + K_halves×2_stages(16KB) + V_halves×2_stages(16KB) + P(8KB) = 72KB
 *
 * Expected: break through 117 TFLOPS via memory/compute overlap.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define BLOCK_M 128
#define BLOCK_N 32
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
#define Q_ELEMS (BLOCK_M * Q_STRIDE)           // 32KB
#define KV_HALF_ELEMS (BLOCK_N * KV_HALF_STRIDE) // 2KB per half (32×64)
// Per stage: 2 halves for K + 2 halves for V = 4 × 2KB = 8KB
// 2 stages = 16KB total for K, 16KB for V
#define P_ELEMS (BLOCK_M * BLOCK_N)            // 8KB
// Total: 32 + 16 + 16 + 8 = 72KB + mbar
#define SMEM_BYTES ((Q_ELEMS + NUM_STAGES * 4 * KV_HALF_ELEMS + P_ELEMS) * 2 + 128)

__device__ __forceinline__ int q_swizzle(int row, int col, int stride) {
    return row * stride + ((col * 2) ^ ((row & 7) << 5)) / 2;
}

__device__ __forceinline__ int tma_swizzle(int row, int col, int stride) {
    return row * stride + ((col * 2) ^ ((row & 7) << 4)) / 2;
}

__device__ __forceinline__ uint32_t pk(const __nv_bfloat16& a, const __nv_bfloat16& b) {
    uint32_t r; asm("mov.b32 %0, {%1, %2};" : "=r"(r) : "h"(*(const uint16_t*)&a), "h"(*(const uint16_t*)&b)); return r;
}

__device__ __forceinline__ void cp16(void* s, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(s))), "l"(g));
}

__device__ __forceinline__ void mma16(float c[4], uint32_t a[4], uint32_t b[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(c[0]),"=f"(c[1]),"=f"(c[2]),"=f"(c[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]));
}

__device__ __forceinline__ void tma2d(void* dst, const CUtensorMap* d, int x, int y, uint64_t* m) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];\n"
        :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))), "l"(reinterpret_cast<uint64_t>(d)),
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

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
sm120_fa_tma_ds(
    const __nv_bfloat16* __restrict__ Q,
    const CUtensorMap* __restrict__ Kdl,
    const CUtensorMap* __restrict__ Kdr,
    const CUtensorMap* __restrict__ Vdl,
    const CUtensorMap* __restrict__ Vdr,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
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
    // Double-buffered KV halves: [stage][half(L/R)][K/V]
    // Layout: kl[0], kr[0], vl[0], vr[0], kl[1], kr[1], vl[1], vr[1]
    __nv_bfloat16* kv_base = q_s + Q_ELEMS;
    __nv_bfloat16* kl[2], *kr[2], *vl[2], *vr[2];
    for (int s = 0; s < 2; s++) {
        kl[s] = kv_base + s * 4 * KV_HALF_ELEMS;
        kr[s] = kl[s] + KV_HALF_ELEMS;
        vl[s] = kr[s] + KV_HALF_ELEMS;
        vr[s] = vl[s] + KV_HALF_ELEMS;
    }
    __nv_bfloat16* p_s = kv_base + NUM_STAGES * 4 * KV_HALF_ELEMS;
    uint64_t* mbar = (uint64_t*)((char*)p_s + P_ELEMS * 2);

    // Load Q
    for (int i = tid; i < Q_ELEMS; i += BLOCK_SIZE) q_s[i] = __float2bfloat16(0.0f);
    __syncthreads();
    for (int i = tid; i < Q_ELEMS / 8; i += BLOCK_SIZE) {
        int row = (i*8)/HEAD_DIM, col = (i*8)%HEAD_DIM, qr = m_start + row;
        if (qr < Sq) cp16(&q_s[q_swizzle(row, col, Q_STRIDE)], &q_ptr[qr*HEAD_DIM+col]);
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n");
    __syncthreads();

    // Init mbarriers for both stages
    if (tid == 0) { mb_init(&mbar[0], 1); mb_init(&mbar[1], 1); }
    __syncthreads();

    // Prefetch stage 0
    if (tid == 0) {
        int row0 = kv_head * Skv;
        int hbytes = BLOCK_N * HALF_HD * 2;  // 4KB per half
        mb_expect(&mbar[0], 4 * hbytes);
        tma2d(kl[0], Kdl, 0, row0, &mbar[0]);
        tma2d(kr[0], Kdr, 0, row0, &mbar[0]);
        tma2d(vl[0], Vdl, 0, row0, &mbar[0]);
        tma2d(vr[0], Vdr, 0, row0, &mbar[0]);
    }

    float o_acc[2][16][2];
    float rmax[2] = {-FLT_MAX, -FLT_MAX}, rsum[2] = {0, 0};
    #pragma unroll
    for (int rh = 0; rh < 2; rh++)
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) o_acc[rh][nt][0] = o_acc[rh][nt][1] = 0;

    for (int kv = 0; kv < num_kv; kv++) {
        int cs = kv % 2;

        // Wait for current stage
        mb_wait(&mbar[cs], 0);
        __syncthreads();

        // Prefetch NEXT stage (overlaps with compute below!)
        if (kv + 1 < num_kv && tid == 0) {
            int ns = 1 - cs;
            mb_init(&mbar[ns], 1);
            int next_row = kv_head * Skv + (kv+1) * BLOCK_N;
            int hbytes = BLOCK_N * HALF_HD * 2;
            mb_expect(&mbar[ns], 4 * hbytes);
            tma2d(kl[ns], Kdl, 0, next_row, &mbar[ns]);
            tma2d(kr[ns], Kdr, 0, next_row, &mbar[ns]);
            tma2d(vl[ns], Vdl, 0, next_row, &mbar[ns]);
            tma2d(vr[ns], Vdr, 0, next_row, &mbar[ns]);
        }

        // Q@K^T
        float s_acc[2][BLOCK_N/MMA_N][2];
        #pragma unroll
        #pragma unroll
        for (int rh=0;rh<2;rh++)
            #pragma unroll
            for (int nt=0;nt<BLOCK_N/MMA_N;nt++) s_acc[rh][nt][0]=s_acc[rh][nt][1]=0;

        #pragma unroll
        for (int ki = 0; ki < HEAD_DIM/MMA_K; ki++) {
            uint32_t qf[4];
            { int r0=warp*WARP_M+g, r1=r0+8, c0=ki*MMA_K+t*2, c1=c0+8;
              qf[0]=pk(q_s[q_swizzle(r0,c0,Q_STRIDE)],q_s[q_swizzle(r0,c0+1,Q_STRIDE)]);
              qf[1]=pk(q_s[q_swizzle(r1,c0,Q_STRIDE)],q_s[q_swizzle(r1,c0+1,Q_STRIDE)]);
              qf[2]=pk(q_s[q_swizzle(r0,c1,Q_STRIDE)],q_s[q_swizzle(r0,c1+1,Q_STRIDE)]);
              qf[3]=pk(q_s[q_swizzle(r1,c1,Q_STRIDE)],q_s[q_swizzle(r1,c1+1,Q_STRIDE)]); }

            int kcol = ki*MMA_K, half = kcol/HALF_HD, lc = kcol - half*HALF_HD;
            __nv_bfloat16* kh = (half==0) ? kl[cs] : kr[cs];

            #pragma unroll
            for (int ni=0; ni<BLOCK_N/MMA_N; ni++) {
                uint32_t kf[2]; int kn=ni*MMA_N+g, k0=lc+t*2, k1=k0+8;
                kf[0]=pk(kh[tma_swizzle(kn,k0,KV_HALF_STRIDE)],kh[tma_swizzle(kn,k0+1,KV_HALF_STRIDE)]);
                kf[1]=pk(kh[tma_swizzle(kn,k1,KV_HALF_STRIDE)],kh[tma_swizzle(kn,k1+1,KV_HALF_STRIDE)]);
                float tl[4]={s_acc[0][ni][0],s_acc[0][ni][1],s_acc[1][ni][0],s_acc[1][ni][1]};
                mma16(tl,qf,kf);
                s_acc[0][ni][0]=tl[0]; s_acc[0][ni][1]=tl[1];
                s_acc[1][ni][0]=tl[2]; s_acc[1][ni][1]=tl[3];
            }
        }

        // Scale+mask
        int kvs = kv*BLOCK_N;
        #pragma unroll
        #pragma unroll
        for (int rh=0;rh<2;rh++)
            #pragma unroll
            for (int nt=0;nt<BLOCK_N/MMA_N;nt++) {
                s_acc[rh][nt][0]*=scale; s_acc[rh][nt][1]*=scale;
                int ki0=kvs+nt*MMA_N+t*2;
                if(ki0>=Skv) s_acc[rh][nt][0]=-FLT_MAX;
                if(ki0+1>=Skv) s_acc[rh][nt][1]=-FLT_MAX;
            }

        // Softmax
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
                s_acc[rh][nt][0]=__expf(s_acc[rh][nt][0]-nm);
                s_acc[rh][nt][1]=__expf(s_acc[rh][nt][1]-nm);
                ls+=s_acc[rh][nt][0]+s_acc[rh][nt][1];
            }
            ls+=__shfl_xor_sync(0xffffffff,ls,1); ls+=__shfl_xor_sync(0xffffffff,ls,2);
            rsum[rh]+=ls; rmax[rh]=nm;
        }

        // P stage
        #pragma unroll
        for (int nt=0;nt<BLOCK_N/MMA_N;nt++) {
            int c0=nt*MMA_N+t*2, r0=warp*WARP_M+g, r1=r0+8;
            p_s[r0*BLOCK_N+c0]=__float2bfloat16(s_acc[0][nt][0]);
            p_s[r0*BLOCK_N+c0+1]=__float2bfloat16(s_acc[0][nt][1]);
            p_s[r1*BLOCK_N+c0]=__float2bfloat16(s_acc[1][nt][0]);
            p_s[r1*BLOCK_N+c0+1]=__float2bfloat16(s_acc[1][nt][1]);
        }
        __syncwarp();

        // P@V
        #pragma unroll
        for (int ki=0;ki<BLOCK_N/MMA_K;ki++) {
            uint32_t pf[4];
            { int pr0=warp*WARP_M+g, pr1=pr0+8, pc0=ki*MMA_K+t*2, pc1=pc0+8;
              pf[0]=pk(p_s[pr0*BLOCK_N+pc0],p_s[pr0*BLOCK_N+pc0+1]);
              pf[1]=pk(p_s[pr1*BLOCK_N+pc0],p_s[pr1*BLOCK_N+pc0+1]);
              pf[2]=pk(p_s[pr0*BLOCK_N+pc1],p_s[pr0*BLOCK_N+pc1+1]);
              pf[3]=pk(p_s[pr1*BLOCK_N+pc1],p_s[pr1*BLOCK_N+pc1+1]); }

            #pragma unroll
            for (int di=0;di<HEAD_DIM/MMA_N;di++) {
                int dc=di*MMA_N, vh=dc/HALF_HD, vlc=dc-vh*HALF_HD;
                __nv_bfloat16* vhp=(vh==0)?vl[cs]:vr[cs];
                uint32_t vf[2]; int vk0=ki*MMA_K+t*2, vk1=vk0+8, vn=vlc+g;
                vf[0]=pk(vhp[tma_swizzle(vk0,vn,KV_HALF_STRIDE)],vhp[tma_swizzle(vk0+1,vn,KV_HALF_STRIDE)]);
                vf[1]=pk(vhp[tma_swizzle(vk1,vn,KV_HALF_STRIDE)],vhp[tma_swizzle(vk1+1,vn,KV_HALF_STRIDE)]);
                float ot[4]={o_acc[0][di][0],o_acc[0][di][1],o_acc[1][di][0],o_acc[1][di][1]};
                mma16(ot,pf,vf);
                o_acc[0][di][0]=ot[0]; o_acc[0][di][1]=ot[1];
                o_acc[1][di][0]=ot[2]; o_acc[1][di][1]=ot[3];
            }
        }
        __syncthreads();
    }

    // Output
    __nv_bfloat16* o_ptr = O + head*Sq*HEAD_DIM;
    for (int rh=0;rh<2;rh++) {
        float inv=(rsum[rh]>0)?1.0f/rsum[rh]:0;
        int row=m_start+warp*WARP_M+g+rh*8;
        if(row<Sq) {
            #pragma unroll
            for (int di=0;di<16;di++) {
                int c0=di*MMA_N+t*2;
                o_ptr[row*HEAD_DIM+c0]=__float2bfloat16(o_acc[rh][di][0]*inv);
                o_ptr[row*HEAD_DIM+c0+1]=__float2bfloat16(o_acc[rh][di][1]*inv);
            }
        }
    }
    if(LSE&&t==0) for(int rh=0;rh<2;rh++) { int row=m_start+warp*WARP_M+g+rh*8; if(row<Sq) LSE[head*Sq+row]=rmax[rh]+logf(rsum[rh]); }
}

extern "C" void sm120_flash_attn_forward(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    int batch, int Hq, int Hkv, int Sq, int Skv, int hd, cudaStream_t stream
) {
    float sc = 1.0f / sqrtf((float)hd);
    int total_kv_heads = batch * Hkv;
    int total_rows = total_kv_heads * Skv;

    CUtensorMap kdl __attribute__((aligned(64))), kdr __attribute__((aligned(64)));
    CUtensorMap vdl __attribute__((aligned(64))), vdr __attribute__((aligned(64)));

    cuuint64_t dims[2] = {(cuuint64_t)HALF_HD, (cuuint64_t)total_rows};
    cuuint64_t strides[1] = {(cuuint64_t)(HEAD_DIM * 2)};
    cuuint32_t box[2] = {(cuuint32_t)HALF_HD, (cuuint32_t)BLOCK_N};
    cuuint32_t es[2] = {1, 1};

    cuTensorMapEncodeTiled(&kdl, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)K, dims, strides, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    cuTensorMapEncodeTiled(&kdr, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)(K+HALF_HD), dims, strides, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    cuTensorMapEncodeTiled(&vdl, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)V, dims, strides, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    cuTensorMapEncodeTiled(&vdr, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)(V+HALF_HD), dims, strides, box, es,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    CUtensorMap* dd; cudaMalloc(&dd, 4*sizeof(CUtensorMap));
    cudaMemcpyAsync(dd, &kdl, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dd+1, &kdr, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dd+2, &vdl, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(dd+3, &vdr, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);

    dim3 grid((Sq+BLOCK_M-1)/BLOCK_M, batch*Hq);
    cudaFuncSetAttribute(sm120_fa_tma_ds, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    cudaFuncSetAttribute(sm120_fa_tma_ds, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
    sm120_fa_tma_ds<<<grid, BLOCK_SIZE, SMEM_BYTES, stream>>>(Q, dd, dd+1, dd+2, dd+3, O, L, Sq, Skv, Hq, Hkv, sc);
    cudaFree(dd);
}
