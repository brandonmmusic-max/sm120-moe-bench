/**
 * Final M=1 validation: Direct A load (M-interleaved SMEM) + pack_b (BLayout).
 * Tests if the MMA hardware does full K routing across the warp.
 *
 * A SMEM layout: 512 bytes total.
 * Block 0 [0..127]: M=even data, rows at 16-byte intervals
 * Block 1 [128..255]: M=odd data
 * Block 2 [256..383]: M=even (diff K range)
 * Block 3 [384..511]: M=odd (diff K range)
 *
 * For M=1: only bytes [0..15] and [256..271] are non-zero.
 * Within each 4-byte chunk: row-major K (K=0..7 in bytes 0..3 for thread 0, etc.)
 *
 * Build: nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a -o mma_m1_final mma_m1_final.cu
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static const float E2M1[8] = {0,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};

__device__ __forceinline__ void mma_nvf4(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
    const float (&c)[4], uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
          "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

__device__ __forceinline__ uint32_t get_nibble(const uint8_t* d, int k) {
    return (k&1)?((d[k/2]>>4)&0xFu):(d[k/2]&0xFu);
}

// BLayout pack (proven correct for uniform data)
__device__ __forceinline__ void pack_b(
    uint32_t (&b)[2], const uint8_t* s_B, int lid)
{
    int t0=lid%4, t1=lid/4;
    const uint8_t* rn0=s_B+t0*32, *rn4=s_B+(t0+4)*32;
    b[0]=b[1]=0;
    for(int vi=0;vi<8;vi++){
        int k=t1+vi*8;
        b[0]|=get_nibble(rn0,k)<<(vi*4);
        b[1]|=get_nibble(rn4,k)<<(vi*4);
    }
}

void quantize(const float* d, int n, uint8_t* p, uint8_t* s) {
    memset(p,0,n/2);
    for(int b=0;b<n/32;b++){
        float mx=0;
        for(int i=b*32;i<(b+1)*32;i++) mx=fmaxf(mx,fabsf(d[i]));
        float sc=fmaxf(mx/6.0f,1e-30f);
        int e=127+(int)ceilf(log2f(sc));
        e=e<1?1:e>254?254:e; s[b]=e;
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

__global__ void test_m1_combined(
    const uint8_t* g_a_packed,  // A[0,:] packed, 32 bytes (K=64)
    const uint8_t* g_b_packed,  // B[8,32] row-major packed, 256 bytes
    float* results)
{
    int lid = threadIdx.x;

    // A SMEM: M-interleaved layout for direct load
    __shared__ __align__(128) uint8_t s_A[512];
    __shared__ __align__(128) uint8_t s_B[256];

    for (int i = lid; i < 512; i += 32) s_A[i] = 0;
    for (int i = lid; i < 256; i += 32) s_B[i] = g_b_packed[i];
    __syncthreads();

    // Write M=0 K data to A SMEM
    // Block 0, group 0 (threads 0-3): bytes [0..15]
    //   Thread c: bytes [c*4 .. c*4+3] = A[0, K = c*8 .. c*8+7]
    // Block 2, group 0: bytes [256..271]
    //   Thread c: bytes [256+c*4 .. 256+c*4+3] = A[0, K = 32+c*8 .. 32+c*8+7]
    // Direct load: thread c reads from s_A[c*16 .. c*16+15]
    // a[0] = bytes [c*16 .. c*16+3] → M=0 K data
    // a[1] = bytes [c*16+4 .. c*16+7] → must be 0 (M=1)
    // a[2] = bytes [c*16+8 .. c*16+11] → M=0 K data (different K range)
    // a[3] = bytes [c*16+12 .. c*16+15] → must be 0 (M=1)
    //
    // BUT: direct load reads from tid*16, and ldmatrix routing is different!
    // For DIRECT LOAD: a[m] = *(uint32_t*)&s_A[tid*16 + m*4]
    // So thread 0: a[0]=SMEM[0..3], a[1]=SMEM[4..7], a[2]=SMEM[8..11], a[3]=SMEM[12..15]
    // For M=1: SMEM[0..3] and SMEM[8..11] = M=0 K data
    //          SMEM[4..7] and SMEM[12..15] = 0
    // Hypothesis: K = t0 + p*8 for a[0], K = t0 + 4 + p*8 for a[2]
    // Thread c: a[0] K = c,c+8,c+16,...,c+56 (stride 8, 8 values)
    //           a[2] K = c+4,c+12,...,c+60 (stride 8, offset 4)
    // Total: 16 K per thread, 4 threads = 64 ✓
    if (lid == 0) {
        for (int c = 0; c < 4; c++) {
            int base = c * 16;
            uint32_t val0 = 0, val2 = 0;
            for (int p = 0; p < 8; p++) {
                int k0 = c + p * 8;       // stride 8
                int k2 = c + 4 + p * 8;   // stride 8, offset 4
                uint8_t nib0 = (k0 & 1) ? ((g_a_packed[k0/2] >> 4) & 0xF)
                                         : (g_a_packed[k0/2] & 0xF);
                uint8_t nib2 = (k2 & 1) ? ((g_a_packed[k2/2] >> 4) & 0xF)
                                         : (g_a_packed[k2/2] & 0xF);
                val0 |= (uint32_t)nib0 << (p * 4);
                val2 |= (uint32_t)nib2 << (p * 4);
            }
            *(uint32_t*)&s_A[base] = val0;
            *(uint32_t*)&s_A[base + 8] = val2;
        }
    }
    __syncthreads();

    // Direct load A
    uint32_t a[4];
    const uint32_t* ap = (const uint32_t*)&s_A[lid * 16];
    a[0]=ap[0]; a[1]=ap[1]; a[2]=ap[2]; a[3]=ap[3];

    // BLayout pack B
    uint32_t b[2];
    pack_b(b, s_B, lid);

    uint32_t sfa = 0x7F7F, sfb = 0x7F7F;
    float acc[4] = {0,0,0,0};
    mma_nvf4(acc, a, b, acc, sfa, sfb);

    // Store C[0, n] from threads 0-3
    if (lid < 4) {
        results[lid] = acc[0];      // C[0, lid]
        results[lid+4] = acc[1];    // C[0, lid+4]
        results[lid+8] = acc[2];    // C[1, lid]
        results[lid+12] = acc[3];   // C[1, lid+4]
    }
}

int main() {
    printf("=== Final M=1 Test: Direct A + pack_b ===\n\n");

    // Test 1: Uniform data
    {
        float h_a[64], h_b[8*64];
        for(int k=0;k<64;k++) h_a[k]=1.0f;
        for(int i=0;i<8*64;i++) h_b[i]=1.0f;

        uint8_t pa[32],sa[2],pb[256],sb[16];
        quantize(h_a,64,pa,sa);
        for(int n=0;n<8;n++) quantize(&h_b[n*64],64,&pb[n*32],&sb[n*2]);

        // No-scale reference
        float ref_ns = 0;
        for(int k=0;k<64;k++){
            uint8_t na=(k&1)?((pa[k/2]>>4)&0xF):(pa[k/2]&0xF);
            uint8_t nb=(k&1)?((pb[k/2]>>4)&0xF):(pb[k/2]&0xF);
            float va=(na>>3)?-1:1; va*=E2M1[na&7];
            float vb=(nb>>3)?-1:1; vb*=E2M1[nb&7];
            ref_ns += va*vb;
        }
        printf("Uniform ref (no scale): C[0,0]=%.1f (expect 64.0)\n", ref_ns);

        uint8_t *da,*db; float *dr;
        cudaMalloc(&da,32); cudaMalloc(&db,256); cudaMalloc(&dr,64);
        cudaMemcpy(da,pa,32,cudaMemcpyHostToDevice);
        cudaMemcpy(db,pb,256,cudaMemcpyHostToDevice);
        cudaMemset(dr,0,64);

        test_m1_combined<<<1,32>>>(da,db,dr);
        cudaDeviceSynchronize();

        float hr[16];
        cudaMemcpy(hr,dr,64,cudaMemcpyDeviceToHost);
        printf("Kernel C[0,0..7]: ");
        for(int n=0;n<8;n++) printf("%.1f ",hr[n]);
        printf("\nKernel C[1,0..7]: ");
        for(int n=0;n<8;n++) printf("%.1f ",hr[8+n]);
        printf("\n\n");

        cudaFree(da); cudaFree(db); cudaFree(dr);
    }

    // Test 2: Random data
    {
        srand(42);
        float h_a[64], h_b[8*64];
        for(int k=0;k<64;k++) h_a[k]=((float)rand()/RAND_MAX-0.5f)*4;
        for(int i=0;i<8*64;i++) h_b[i]=((float)rand()/RAND_MAX-0.5f)*4;

        uint8_t pa[32],sa[2],pb[256],sb[16];
        quantize(h_a,64,pa,sa);
        for(int n=0;n<8;n++) quantize(&h_b[n*64],64,&pb[n*32],&sb[n*2]);

        float ref[8];
        for(int n=0;n<8;n++){
            ref[n]=0;
            for(int k=0;k<64;k++){
                uint8_t na=(k&1)?((pa[k/2]>>4)&0xF):(pa[k/2]&0xF);
                uint8_t nb=(k&1)?((pb[n*32+k/2]>>4)&0xF):(pb[n*32+k/2]&0xF);
                float va=(na>>3)?-1.0f:1.0f; va*=E2M1[na&7];
                float vb=(nb>>3)?-1.0f:1.0f; vb*=E2M1[nb&7];
                ref[n]+=va*vb;
            }
        }
        printf("Random ref (no scale) C[0,0..7]: ");
        for(int n=0;n<8;n++) printf("%.2f ",ref[n]);
        printf("\n");

        uint8_t *da,*db; float *dr;
        cudaMalloc(&da,32); cudaMalloc(&db,256); cudaMalloc(&dr,64);
        cudaMemcpy(da,pa,32,cudaMemcpyHostToDevice);
        cudaMemcpy(db,pb,256,cudaMemcpyHostToDevice);
        cudaMemset(dr,0,64);

        test_m1_combined<<<1,32>>>(da,db,dr);
        cudaDeviceSynchronize();

        float hr[16];
        cudaMemcpy(hr,dr,64,cudaMemcpyDeviceToHost);
        printf("Kernel C[0,0..7]: ");
        for(int n=0;n<8;n++) printf("%.2f ",hr[n]);
        float me=0;
        for(int n=0;n<8;n++) me=fmaxf(me,fabsf(hr[n]-ref[n]));
        printf("\nmax_err=%.3f %s\n", me, me<0.5f?"PASS":"FAIL");
        printf("Kernel C[1,0..3]: ");
        for(int n=0;n<4;n++) printf("%.2f ",hr[8+n]);
        printf("(should be ~0)\n");

        cudaFree(da); cudaFree(db); cudaFree(dr);
    }

    printf("\n=== Done ===\n");
    return 0;
}
