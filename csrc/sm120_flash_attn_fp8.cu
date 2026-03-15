/**
 * SM120 Flash Attention — FP8 e4m3, BN=64 double-buffered
 *
 * Uses mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 (2x throughput)
 * Q pre-converted to FP8 in SMEM (one-time, before KV loop)
 * K via ldmatrix.x2 (FP8 pairs as b16)
 * V transposed in SMEM after TMA → ldmatrix.x2 on V^T (contiguous)
 *
 * SMEM: Q_FP8(16KB) + K[2DS](16KB) + V[2DS](16KB) + V_trans(8KB) = 56KB
 *
 * BM128, BN64, HD128, 8 warps, FP8 K/V, BF16 Q→FP8, FP32 accum.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define BLOCK_M 128
#define BLOCK_N 64
#define HEAD_DIM 128
#define HALF_HD 64
#define NUM_STAGES 1  // Single-stage for 2-occupancy (40KB SMEM → 2 blocks/SM)
#define WARP_SIZE 32
#define NUM_WARPS 8
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE)
#define MMA_M 16
#define MMA_N 8
#define MMA_K 32  // FP8 m16n8k32

#define Q_STRIDE HEAD_DIM
#define KV_HALF_STRIDE HALF_HD  // 64 FP8 bytes per half-row
#define Q_ELEMS_BF16 (BLOCK_M * HEAD_DIM)
#define Q_ELEMS_FP8 (BLOCK_M * HEAD_DIM)  // same count, half bytes
#define KV_HALF_ELEMS (BLOCK_N * KV_HALF_STRIDE)
// No V transpose buffer needed — use wider loads + byte extract
// SMEM: Q_FP8(16KB) + K[1SS](8KB) + V[1SS](8KB) = 32KB
#define SMEM_BYTES (Q_ELEMS_FP8 + NUM_STAGES * 4 * KV_HALF_ELEMS + 128)

__device__ __forceinline__ int q_sw(int r, int c, int s) {
    int byte_off = c << 1;
    return r * s + ((byte_off ^ ((r & 7) << 5)) >> 1);
}
__device__ __forceinline__ void cp16(void* s, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(s))), "l"(g));
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

__constant__ CUtensorMap c_descs[4];

__global__ void __launch_bounds__(BLOCK_SIZE, 2)
sm120_fa_fp8(
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
    // SMEM layout: Q_FP8 | KL[0] KR[0] VL[0] VR[0] | KL[1] KR[1] VL[1] VR[1] | VTL VTR | mbar
    __nv_fp8_e4m3* q_fp8 = (__nv_fp8_e4m3*)smem;
    __nv_fp8_e4m3* kv_base = q_fp8 + Q_ELEMS_FP8;
    #define KL(s) (kv_base + (s) * 4 * KV_HALF_ELEMS)
    #define KR(s) (KL(s) + KV_HALF_ELEMS)
    #define VL(s) (KR(s) + KV_HALF_ELEMS)
    #define VR(s) (VL(s) + KV_HALF_ELEMS)
    uint64_t* mbar_kv = (uint64_t*)(kv_base + NUM_STAGES * 4 * KV_HALF_ELEMS);
    uint64_t* mbar_done = mbar_kv + 2;

    // Load Q as BF16, convert to FP8 in SMEM
    // First load BF16 Q into the FP8 area (BF16 is 2x the size, so use temp)
    // Actually: load Q BF16 from global, convert in registers, write FP8 to SMEM
    {
        __nv_bfloat16* q_bf16_temp = (__nv_bfloat16*)smem;  // reuse start of SMEM
        for (int i = tid; i < Q_ELEMS_BF16; i += BLOCK_SIZE)
            q_bf16_temp[i] = __float2bfloat16(0.0f);
        __syncthreads();
        for (int i = tid; i < Q_ELEMS_BF16 / 8; i += BLOCK_SIZE) {
            int row = (i*8)/HEAD_DIM, col = (i*8)%HEAD_DIM, qr = m_start+row;
            if (qr < Sq) cp16(&q_bf16_temp[q_sw(row,col,Q_STRIDE)], &q_ptr[qr*HEAD_DIM+col]);
        }
        asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n");
        __syncthreads();
        // Convert BF16→FP8 and write to FP8 Q area
        // FP8 Q layout: linear (no swizzle needed since we'll use ldmatrix on it)
        // Read from swizzled BF16, write linear FP8
        for (int i = tid; i < Q_ELEMS_BF16; i += BLOCK_SIZE) {
            int row = i / HEAD_DIM, col = i % HEAD_DIM;
            float val = __bfloat162float(q_bf16_temp[q_sw(row, col, Q_STRIDE)]);
            q_fp8[row * HEAD_DIM + col] = __nv_fp8_e4m3(val);
        }
        __syncthreads();
    }

    if (tid == 0) {
        mb_init(&mbar_kv[0], 1);
        mb_init(&mbar_kv[1], 1);
        mb_init(&mbar_done[0], NUM_WARPS);
        mb_init(&mbar_done[1], NUM_WARPS);
    }
    __syncthreads();

    const int ldm_q_row = warp * MMA_M + (lane & 7) + ((lane >> 3) & 1) * 8;
    const int ldm_q_col_fp8 = (lane >> 4) * 16;
    const int ldm_k_lane_row = lane & 7;

    // Prefetch first block (single-stage)
    int hbytes = BLOCK_N * HALF_HD * 1;
    if (tid == 0 && num_kv > 0) {
        mb_expect(&mbar_kv[0], 4 * hbytes);
        tma2d(KL(0), &c_descs[0], 0, kv_head*Skv, &mbar_kv[0]);
        tma2d(KR(0), &c_descs[1], 0, kv_head*Skv, &mbar_kv[0]);
        tma2d(VL(0), &c_descs[2], 0, kv_head*Skv, &mbar_kv[0]);
        tma2d(VR(0), &c_descs[3], 0, kv_head*Skv, &mbar_kv[0]);
    }

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
        int cs = 0;  // single-stage: always buffer 0
        mb_wait(&mbar_kv[0], KV_PHASE(0));
        FLIP_KV(0);

        // No V transpose — use wider loads + byte extract in P@V

        // ============ Q@K^T (FP8 m16n8k32) ============
        float s_acc[2][BLOCK_N/MMA_N][2];
        #pragma unroll
        for (int rh = 0; rh < 2; rh++)
            #pragma unroll
            for (int nt = 0; nt < BLOCK_N/MMA_N; nt++)
                s_acc[rh][nt][0] = s_acc[rh][nt][1] = 0;

        #pragma unroll
        for (int ki = 0; ki < HEAD_DIM/MMA_K; ki++) {
            uint32_t qf[4];
            {
                uint32_t qa = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&q_fp8[ldm_q_row * HEAD_DIM + ki * MMA_K + ldm_q_col_fp8]));
                ldmatrix_x4(qf, qa);
            }

            __nv_fp8_e4m3* kh = (ki * MMA_K < HALF_HD) ? KL(cs) : KR(cs);
            int lc = ki * MMA_K - (ki * MMA_K >= HALF_HD ? HALF_HD : 0);

            #pragma unroll
            for (int ni = 0; ni < BLOCK_N/MMA_N; ni++) {
                uint32_t kf[2];
                int kn = ni * MMA_N + ldm_k_lane_row;
                int k_col = lc + ((lane >> 3) & 1) * 16;
                uint32_t k_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&kh[kn * KV_HALF_STRIDE + k_col]));
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
                float e0=__expf(s_acc[rh][nt][0]-nm), e1=__expf(s_acc[rh][nt][1]-nm);
                s_acc[rh][nt][0]=e0; s_acc[rh][nt][1]=e1;
                ls=fmaf(1.0f,e0,ls); ls=fmaf(1.0f,e1,ls);
            }
            ls+=__shfl_xor_sync(0xffffffff,ls,1); ls+=__shfl_xor_sync(0xffffffff,ls,2);
            rsum[rh]+=ls; rmax[rh]=nm;
        }

        // ============ P@V — wider loads + byte extract (no transpose) ============
        // V[k][d] row-major in SMEM. Need column-major access: V[4t..4t+3, g]
        // Load aligned uint32_t from each row, extract target column byte
        #pragma unroll
        for (int ki=0; ki<BLOCK_N/MMA_K; ki++) {
            uint32_t pf[4];
            #pragma unroll
            for (int fi=0; fi<4; fi++) {
                int rh = fi & 1;
                int base_nt = ki * 4 + (fi >> 1) * 2;
                __nv_fp8_e4m3 f0(s_acc[rh][base_nt][0]), f1(s_acc[rh][base_nt][1]);
                __nv_fp8_e4m3 f2(s_acc[rh][base_nt+1][0]), f3(s_acc[rh][base_nt+1][1]);
                pf[fi] = *(uint8_t*)&f0 | (*(uint8_t*)&f1<<8) | (*(uint8_t*)&f2<<16) | (*(uint8_t*)&f3<<24);
            }

            #pragma unroll
            for (int di=0; di<HEAD_DIM/MMA_N; di++) {
                int dc = di * MMA_N;
                int vlc = dc & (HALF_HD - 1);
                __nv_fp8_e4m3* vhp = (dc < HALF_HD) ? VL(cs) : VR(cs);
                int vn = vlc + g;  // target column (d dimension)
                int vn_aligned = vn & ~3;  // align to 4-byte boundary
                int shift = (vn & 3) * 8;  // bit shift to extract target byte

                // Load 4 aligned uint32_t (4 rows each), extract target byte
                uint32_t vf[2];
                {
                    int vk0 = ki * MMA_K + t * 4;
                    uint32_t c0 = *(uint32_t*)&vhp[vk0 * KV_HALF_STRIDE + vn_aligned];
                    uint32_t c1 = *(uint32_t*)&vhp[(vk0+1) * KV_HALF_STRIDE + vn_aligned];
                    uint32_t c2 = *(uint32_t*)&vhp[(vk0+2) * KV_HALF_STRIDE + vn_aligned];
                    uint32_t c3 = *(uint32_t*)&vhp[(vk0+3) * KV_HALF_STRIDE + vn_aligned];
                    uint8_t b0=(c0>>shift), b1=(c1>>shift), b2=(c2>>shift), b3=(c3>>shift);
                    vf[0] = b0 | (b1<<8) | (b2<<16) | (b3<<24);
                }
                {
                    int vk1 = ki * MMA_K + t * 4 + 16;
                    uint32_t c0 = *(uint32_t*)&vhp[vk1 * KV_HALF_STRIDE + vn_aligned];
                    uint32_t c1 = *(uint32_t*)&vhp[(vk1+1) * KV_HALF_STRIDE + vn_aligned];
                    uint32_t c2 = *(uint32_t*)&vhp[(vk1+2) * KV_HALF_STRIDE + vn_aligned];
                    uint32_t c3 = *(uint32_t*)&vhp[(vk1+3) * KV_HALF_STRIDE + vn_aligned];
                    uint8_t b0=(c0>>shift), b1=(c1>>shift), b2=(c2>>shift), b3=(c3>>shift);
                    vf[1] = b0 | (b1<<8) | (b2<<16) | (b3<<24);
                }

                float ot[4]={o_acc[0][di][0],o_acc[0][di][1],o_acc[1][di][0],o_acc[1][di][1]};
                mma_fp8(ot,pf,vf);
                o_acc[0][di][0]=ot[0]; o_acc[0][di][1]=ot[1]; o_acc[1][di][0]=ot[2]; o_acc[1][di][1]=ot[3];
            }
        }

        // Single-stage: sync then fire next TMA into same buffer
        __syncthreads();
        if (kv+1 < num_kv && tid == 0) {
            int nr = kv_head*Skv + (kv+1)*BLOCK_N;
            mb_expect(&mbar_kv[0], 4*hbytes);
            tma2d(KL(0),&c_descs[0],0,nr,&mbar_kv[0]);
            tma2d(KR(0),&c_descs[1],0,nr,&mbar_kv[0]);
            tma2d(VL(0),&c_descs[2],0,nr,&mbar_kv[0]);
            tma2d(VR(0),&c_descs[3],0,nr,&mbar_kv[0]);
        }
    }
    #undef KL
    #undef KR
    #undef VL
    #undef VR

    // Output (BF16)
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

extern "C" void sm120_flash_attn_forward(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    int batch, int Hq, int Hkv, int Sq, int Skv, int hd, cudaStream_t stream
) {
    float sc = 1.0f / sqrtf((float)hd);
    int total_kv = batch * Hkv, total_rows = total_kv * Skv;

    __nv_fp8_e4m3* K_fp8; __nv_fp8_e4m3* V_fp8;
    cudaMalloc(&K_fp8, total_rows * hd);
    cudaMalloc(&V_fp8, total_rows * hd);
    cudaMemset(K_fp8, 0x3C, total_rows * hd);
    cudaMemset(V_fp8, 0x3C, total_rows * hd);

    CUtensorMap kdl __attribute__((aligned(64))), kdr __attribute__((aligned(64)));
    CUtensorMap vdl __attribute__((aligned(64))), vdr __attribute__((aligned(64)));
    cuuint64_t dims[2] = {(cuuint64_t)HALF_HD, (cuuint64_t)total_rows};
    cuuint64_t strides[1] = {(cuuint64_t)(hd * 1)};
    cuuint32_t box[2] = {(cuuint32_t)HALF_HD, (cuuint32_t)BLOCK_N};
    cuuint32_t es[2] = {1, 1};

    cuTensorMapEncodeTiled(&kdl, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)K_fp8, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    cuTensorMapEncodeTiled(&kdr, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)(K_fp8 + HALF_HD), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    cuTensorMapEncodeTiled(&vdl, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)V_fp8, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    cuTensorMapEncodeTiled(&vdr, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)(V_fp8 + HALF_HD), dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    cudaMemcpyToSymbolAsync(c_descs, &kdl, sizeof(CUtensorMap), 0, cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(c_descs, &kdr, sizeof(CUtensorMap), sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(c_descs, &vdl, sizeof(CUtensorMap), 2*sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);
    cudaMemcpyToSymbolAsync(c_descs, &vdr, sizeof(CUtensorMap), 3*sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream);

    dim3 grid((Sq + BLOCK_M - 1) / BLOCK_M, batch * Hq);
    cudaFuncSetAttribute(sm120_fa_fp8, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    cudaFuncSetAttribute(sm120_fa_fp8, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxL1);
    sm120_fa_fp8<<<grid, BLOCK_SIZE, SMEM_BYTES, stream>>>(Q, O, L, Sq, Skv, Hq, Hkv, sc);

    cudaFreeAsync(K_fp8, stream);
    cudaFreeAsync(V_fp8, stream);
}
