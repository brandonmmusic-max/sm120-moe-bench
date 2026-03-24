/**
 * Isolate A vs B issues.
 * Test 1: A=all 1.0 (E2M1=2, uniform across M), B=random via pack_b.
 *         C[m,n] = sum_k B[n,k] (since A=1.0). Independent of K ordering!
 * Test 2: Same but A=random via manual pack (all M rows = same), B=all 1.0.
 *         C[m,n] = sum_k A[0,k] (since B=1.0). Also independent of K ordering!
 * These tests tell us if pack_b / A_pack are individually correct.
 *
 * Build: nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a -o mma_isolate mma_isolate.cu
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

__device__ __forceinline__ void pack_b(uint32_t(&b)[2],const uint8_t*s,int lid){
    int t0=lid%4,t1=lid/4;
    const uint8_t*r0=s+t0*32,*r4=s+(t0+4)*32;
    b[0]=b[1]=0;
    for(int vi=0;vi<8;vi++){
        int k=t1+vi*8;
        b[0]|=get_nib(r0,k)<<(vi*4);
        b[1]|=get_nib(r4,k)<<(vi*4);
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

// Test 1: A=all E2M1=2 (1.0) uniform, B=random, pack_b
// Expected C[0,n] = sum_k E2M1(B_nib[n,k]) (no scale factor)
__global__ void test_uniform_a(const uint8_t*g_b,float*res){
    int lid=threadIdx.x;
    __shared__ __align__(128) uint8_t sB[256];
    for(int i=lid;i<256;i+=32) sB[i]=g_b[i];
    __syncthreads();

    uint32_t a[4]={0x22222222,0x22222222,0x22222222,0x22222222};
    uint32_t b[2]; pack_b(b,sB,lid);
    uint32_t sfa=0x7F7F,sfb=0x7F7F;
    float acc[4]={0,0,0,0};
    mma_nvf4(acc,a,b,acc,sfa,sfb);
    if(lid<4){res[lid]=acc[0];res[lid+4]=acc[1];}
}

// Test 2: A=random manual pack (ALL M rows same), B=all E2M1=2 (1.0)
// Uses pack_b-style K formula for A: K = tid%4 + p*8
// Expected C[m,n] = sum_k E2M1(A_nib[k]) (since all M rows same)
__global__ void test_uniform_b(const uint8_t*g_a,float*res){
    int lid=threadIdx.x;

    // A: fill all registers with A[0,:] data using stride-8 K
    uint32_t a[4];
    int t0=lid%4;
    // a[0] and a[1]: K = t0 + p*8 (same K for both even/odd M, since data is same)
    uint32_t val=0;
    for(int p=0;p<8;p++){
        int k=t0+p*8;
        val|=get_nib(g_a,k)<<(p*4);
    }
    a[0]=a[1]=val;

    // a[2] and a[3]: K = t0+4 + p*8
    val=0;
    for(int p=0;p<8;p++){
        int k=t0+4+p*8;
        val|=get_nib(g_a,k)<<(p*4);
    }
    a[2]=a[3]=val;

    uint32_t b[2]={0x22222222,0x22222222};
    uint32_t sfa=0x7F7F,sfb=0x7F7F;
    float acc[4]={0,0,0,0};
    mma_nvf4(acc,a,b,acc,sfa,sfb);
    if(lid<4){res[lid]=acc[0];res[lid+4]=acc[1];}
}

int main(){
    srand(42);
    float hb[8*64],ha[64];
    for(int i=0;i<8*64;i++)hb[i]=((float)rand()/RAND_MAX-0.5f)*4;
    for(int k=0;k<64;k++)ha[k]=((float)rand()/RAND_MAX-0.5f)*4;

    uint8_t pb[256],sb[16],pa[32],sa[2];
    for(int n=0;n<8;n++)quant(&hb[n*64],64,&pb[n*32],&sb[n*2]);
    quant(ha,64,pa,sa);

    // References (no scale = raw E2M1 dot products)
    float ref1[8],ref2[8];
    for(int n=0;n<8;n++){
        ref1[n]=0; ref2[n]=0;
        for(int k=0;k<64;k++){
            // Test1 ref: A=1.0, B=random
            uint8_t nb=(k&1)?((pb[n*32+k/2]>>4)&0xF):(pb[n*32+k/2]&0xF);
            float vb=(nb>>3)?-1:1; vb*=E2M1[nb&7];
            ref1[n]+=1.0f*vb;
            // Test2 ref: A=random, B=1.0
            uint8_t na=(k&1)?((pa[k/2]>>4)&0xF):(pa[k/2]&0xF);
            float va=(na>>3)?-1:1; va*=E2M1[na&7];
            ref2[n]+=va*1.0f;
        }
    }

    uint8_t*db;float*dr;
    cudaMalloc(&db,256);cudaMalloc(&dr,32);

    // Test 1
    cudaMemcpy(db,pb,256,cudaMemcpyHostToDevice);
    cudaMemset(dr,0,32);
    test_uniform_a<<<1,32>>>(db,dr);
    cudaDeviceSynchronize();
    float hr[8];
    cudaMemcpy(hr,dr,32,cudaMemcpyDeviceToHost);
    printf("Test 1: A=uniform 1.0, B=random (pack_b)\n");
    printf("  Ref C[0,0..7]: ");for(int n=0;n<8;n++)printf("%.2f ",ref1[n]);
    printf("\n  Ker C[0,0..7]: ");for(int n=0;n<8;n++)printf("%.2f ",hr[n]);
    float me=0;for(int n=0;n<8;n++)me=fmaxf(me,fabsf(hr[n]-ref1[n]));
    printf("\n  max_err=%.3f %s\n\n",me,me<0.5f?"PASS":"FAIL");

    // Test 2
    uint8_t*da;
    cudaMalloc(&da,32);
    cudaMemcpy(da,pa,32,cudaMemcpyHostToDevice);
    cudaMemset(dr,0,32);
    test_uniform_b<<<1,32>>>(da,dr);
    cudaDeviceSynchronize();
    cudaMemcpy(hr,dr,32,cudaMemcpyDeviceToHost);
    printf("Test 2: A=random (stride-8 pack), B=uniform 1.0\n");
    printf("  Ref C[0,0..7]: ");for(int n=0;n<8;n++)printf("%.2f ",ref2[n]);
    printf("\n  Ker C[0,0..7]: ");for(int n=0;n<8;n++)printf("%.2f ",hr[n]);
    me=0;for(int n=0;n<8;n++)me=fmaxf(me,fabsf(hr[n]-ref2[n]));
    printf("\n  max_err=%.3f %s\n\n",me,me<0.5f?"PASS":"FAIL");

    cudaFree(db);cudaFree(da);cudaFree(dr);
    return 0;
}
