/**
 * CORRECT MMA packing based on empirical hardware probes.
 *
 * EMPIRICALLY DETERMINED REGISTER LAYOUT:
 *
 * A operand (4 registers per thread, 8 nibbles each = 32 FP4 values):
 *   Thread group g = tid/4 (0..7):
 *     a[0]: 8 K values at M = 2*g     (K formula: K = t0 + p*8, t0=tid%4, p=nibble)
 *     a[1]: 8 K values at M = 2*g + 1 (same K formula)
 *     a[2]: 8 K values at M = 2*g     (K = t0 + 4 + p*8)
 *     a[3]: 8 K values at M = 2*g + 1 (K = t0 + 4 + p*8)
 *
 * B operand (2 registers per thread, 8 nibbles each = 16 FP4 values):
 *   Thread group g = tid/4 (0..7):
 *     ALL nibbles → N = 4*(g%2) + g/2
 *     b[0]: K = t0 + p*8         (t0=tid%4, p=nibble 0..7)
 *     b[1]: K = t0 + 4 + p*8
 *
 * C output (4 float registers):
 *   d[0] = C[2*(tid/4), tid%4]
 *   d[1] = C[2*(tid/4), tid%4 + 4]
 *   d[2] = C[2*(tid/4) + 1, tid%4]
 *   d[3] = C[2*(tid/4) + 1, tid%4 + 4]
 *
 * Build: nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a -o mma_correct_final mma_correct_final.cu
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static const float E2M1[8]={0,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};

__device__ __forceinline__ void mma_nvf4(
    float(&d)[4],const uint32_t(&a)[4],const uint32_t(&b)[2],
    const float(&c)[4],uint32_t sfa,uint32_t sfb){
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        :"=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),
         "r"(b[0]),"r"(b[1]),
         "f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]),
         "r"(sfa),"h"((uint16_t)0),"h"((uint16_t)0),
         "r"(sfb),"h"((uint16_t)0),"h"((uint16_t)0));
}

__device__ __forceinline__ uint32_t get_nib(const uint8_t*d,int k){
    return (k&1)?((d[k/2]>>4)&0xFu):(d[k/2]&0xFu);
}

// CORRECT A packing for M=1 (only M=0 row has data)
__device__ __forceinline__ void pack_a_m1_v2(
    uint32_t(&a)[4], const uint8_t* row0, int tid)
{
    a[0]=a[1]=a[2]=a[3]=0;
    int g = tid/4;
    if (g != 0) return;  // Only group 0 has M=0
    int t0 = tid%4;

    for(int p=0;p<8;p++){
        int k0 = t0 + p*8;      // a[0] K values
        int k2 = t0 + 4 + p*8;  // a[2] K values
        a[0] |= get_nib(row0, k0) << (p*4);
        a[2] |= get_nib(row0, k2) << (p*4);
    }
    // a[1], a[3] = 0 (M=1 row = 0)
}

// CORRECT B packing
__device__ __forceinline__ void pack_b_v2(
    uint32_t(&b)[2], const uint8_t* s_B, int tid)
{
    int g = tid/4;
    int t0 = tid%4;
    int N = 4*(g&1) + (g>>1);  // N column for this thread group
    const uint8_t* row = s_B + N * 32;  // 32 bytes per N row

    b[0]=b[1]=0;
    for(int p=0;p<8;p++){
        int k0 = t0 + p*8;
        int k1 = t0 + 4 + p*8;
        b[0] |= get_nib(row, k0) << (p*4);
        b[1] |= get_nib(row, k1) << (p*4);
    }
}

void quant(const float*d,int n,uint8_t*p,uint8_t*s){
    memset(p,0,n/2);
    for(int b=0;b<n/32;b++){
        float mx=0;
        for(int i=b*32;i<(b+1)*32;i++)mx=fmaxf(mx,fabsf(d[i]));
        float sc=fmaxf(mx/6.0f,1e-30f);
        int e=127+(int)ceilf(log2f(sc));
        e=e<1?1:e>254?254:e;s[b]=e;
        float as=powf(2.0f,(float)(e-127));
        for(int i=b*32;i<(b+1)*32;i++){
            float v=d[i]/as;int sg=v<0?1:0;float av=fabsf(v);
            int idx=0;float bd=av;
            for(int j=1;j<8;j++){float dd=fabsf(av-E2M1[j]);if(dd<bd){bd=dd;idx=j;}}
            uint8_t f4=(sg<<3)|idx;
            if(i%2==0)p[i/2]=f4;else p[i/2]|=(f4<<4);
        }
    }
}

__global__ void test_correct(
    const uint8_t* g_a, const uint8_t* g_b, float* results)
{
    int lid = threadIdx.x;
    __shared__ __align__(128) uint8_t sB[256];
    for(int i=lid;i<256;i+=32) sB[i]=g_b[i];
    __syncthreads();

    uint32_t a[4], b[2];
    pack_a_m1_v2(a, g_a, lid);
    pack_b_v2(b, sB, lid);

    uint32_t sfa=0x7F7F, sfb=0x7F7F;
    float acc[4]={0,0,0,0};
    mma_nvf4(acc, a, b, acc, sfa, sfb);

    // Store C[0, 0..7]
    // CLayout: d[0]=C[2*(tid/4), tid%4], d[1]=C[2*(tid/4), tid%4+4]
    if(lid<4){
        results[lid]=acc[0];    // C[0, lid]
        results[lid+4]=acc[1];  // C[0, lid+4]
    }
}

int main(){
    printf("=== Final Correct Packing Test ===\n\n");

    srand(42);
    float ha[64],hb[8*64];
    for(int k=0;k<64;k++)ha[k]=((float)rand()/RAND_MAX-0.5f)*4;
    for(int i=0;i<8*64;i++)hb[i]=((float)rand()/RAND_MAX-0.5f)*4;

    uint8_t pa[32],sa[2],pb[256],sb[16];
    quant(ha,64,pa,sa);
    for(int n=0;n<8;n++)quant(&hb[n*64],64,&pb[n*32],&sb[n*2]);

    // No-scale reference
    float ref[8];
    for(int n=0;n<8;n++){
        ref[n]=0;
        for(int k=0;k<64;k++){
            uint8_t na=(k&1)?((pa[k/2]>>4)&0xF):(pa[k/2]&0xF);
            uint8_t nb=(k&1)?((pb[n*32+k/2]>>4)&0xF):(pb[n*32+k/2]&0xF);
            float va=((na>>3)?-1.0f:1.0f)*E2M1[na&7];
            float vb=((nb>>3)?-1.0f:1.0f)*E2M1[nb&7];
            ref[n]+=va*vb;
        }
    }
    printf("Ref (no scale) C[0,0..7]: ");
    for(int n=0;n<8;n++)printf("%.2f ",ref[n]);
    printf("\n");

    uint8_t*da,*db;float*dr;
    cudaMalloc(&da,32);cudaMalloc(&db,256);cudaMalloc(&dr,32);
    cudaMemcpy(da,pa,32,cudaMemcpyHostToDevice);
    cudaMemcpy(db,pb,256,cudaMemcpyHostToDevice);
    cudaMemset(dr,0,32);

    test_correct<<<1,32>>>(da,db,dr);
    cudaDeviceSynchronize();

    float hr[8];
    cudaMemcpy(hr,dr,32,cudaMemcpyDeviceToHost);
    printf("Ker C[0,0..7]:           ");
    for(int n=0;n<8;n++)printf("%.2f ",hr[n]);
    float me=0;
    for(int n=0;n<8;n++)me=fmaxf(me,fabsf(hr[n]-ref[n]));
    printf("\nmax_err=%.3f %s\n",me,me<0.5f?"PASS":"FAIL");

    // Also test with uniform data
    printf("\n--- Uniform test ---\n");
    for(int k=0;k<64;k++)ha[k]=1.0f;
    for(int i=0;i<8*64;i++)hb[i]=1.0f;
    quant(ha,64,pa,sa);
    for(int n=0;n<8;n++)quant(&hb[n*64],64,&pb[n*32],&sb[n*2]);
    float ref_u=0;
    for(int k=0;k<64;k++){
        uint8_t na=(k&1)?((pa[k/2]>>4)&0xF):(pa[k/2]&0xF);
        uint8_t nb=(k&1)?((pb[k/2]>>4)&0xF):(pb[k/2]&0xF);
        ref_u+=E2M1[na&7]*E2M1[nb&7];
    }
    printf("Ref: %.1f\n",ref_u);
    cudaMemcpy(da,pa,32,cudaMemcpyHostToDevice);
    cudaMemcpy(db,pb,256,cudaMemcpyHostToDevice);
    cudaMemset(dr,0,32);
    test_correct<<<1,32>>>(da,db,dr);
    cudaDeviceSynchronize();
    cudaMemcpy(hr,dr,32,cudaMemcpyDeviceToHost);
    printf("Ker: ");for(int n=0;n<8;n++)printf("%.1f ",hr[n]);printf("\n");

    cudaFree(da);cudaFree(db);cudaFree(dr);
    printf("\n=== Done ===\n");
    return 0;
}
