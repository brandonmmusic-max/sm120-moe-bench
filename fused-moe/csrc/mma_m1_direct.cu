/**
 * Minimal test: direct load with M=1 data arranged so a[0],a[2] get data,
 * a[1],a[3] get zeros. B also uses direct load for consistency.
 *
 * Build: nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a -o mma_m1_direct mma_m1_direct.cu
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static const float E2M1_TABLE[8] = {0,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};

__device__ __forceinline__ void mma_nvf4(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
    const float (&c)[4], uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},"
        "{%4,%5,%6,%7},"
        "{%8,%9},"
        "{%10,%11,%12,%13},"
        "{%14},{%15,%16},"
        "{%17},{%18,%19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
          "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

__global__ void test_m1_direct(float* results) {
    int lid = threadIdx.x;

    // A SMEM: 512 bytes, direct load pattern
    // For M=1: only group 0 (threads 0-3) should have data
    // Direct load: a[m] = SMEM[tid*16 + m*4 .. +3]
    // a[0]=bytes[0..3], a[1]=bytes[4..7], a[2]=bytes[8..11], a[3]=bytes[12..15]
    // We want: a[0],a[2] = M=0 data; a[1],a[3] = 0
    // So: bytes[0..3]=data, bytes[4..7]=0, bytes[8..11]=data, bytes[12..15]=0
    __shared__ __align__(128) uint8_t s_A[512];
    __shared__ __align__(128) uint8_t s_B[256];

    // Zero everything
    for (int i = lid; i < 512; i += 32) s_A[i] = 0;
    for (int i = lid; i < 256; i += 32) s_B[i] = 0x22;  // B = all 1.0
    __syncthreads();

    // Fill A: group 0 (threads 0-3), bytes 0..3 and 8..11 with 0x22
    // Thread 0: SMEM[0..3] = 0x22, SMEM[8..11] = 0x22
    // Thread 1: SMEM[16+0..16+3] = 0x22, SMEM[16+8..16+11] = 0x22
    // Thread 2: SMEM[32+0..32+3] = 0x22, SMEM[32+8..32+11] = 0x22
    // Thread 3: SMEM[48+0..48+3] = 0x22, SMEM[48+8..48+11] = 0x22
    if (lid == 0) {
        for (int t = 0; t < 4; t++) {
            int base = t * 16;
            for (int b = 0; b < 4; b++) {
                s_A[base + b] = 0x22;       // a[0] data
                s_A[base + 8 + b] = 0x22;   // a[2] data
            }
            // bytes 4..7 and 12..15 stay 0 (a[1] and a[3])
        }
    }
    __syncthreads();

    // Direct load A
    uint32_t a[4];
    const uint32_t* ap = (const uint32_t*)&s_A[lid * 16];
    a[0] = ap[0]; a[1] = ap[1]; a[2] = ap[2]; a[3] = ap[3];

    // Direct load B
    uint32_t b[2];
    const uint32_t* bp = (const uint32_t*)&s_B[(lid%16) * 16];
    b[0] = bp[0]; b[1] = bp[1];

    uint32_t sfa = 0x7F7F, sfb = 0x7F7F;
    float acc[4] = {0,0,0,0};
    mma_nvf4(acc, a, b, acc, sfa, sfb);

    // Store from all threads
    results[lid * 4 + 0] = acc[0];
    results[lid * 4 + 1] = acc[1];
    results[lid * 4 + 2] = acc[2];
    results[lid * 4 + 3] = acc[3];
}

int main() {
    float* d_res;
    cudaMalloc(&d_res, 128 * sizeof(float));
    cudaMemset(d_res, 0, 128 * sizeof(float));

    test_m1_direct<<<1, 32>>>(d_res);
    cudaDeviceSynchronize();

    float h[128];
    cudaMemcpy(h, d_res, sizeof(h), cudaMemcpyDeviceToHost);

    printf("=== Direct load M=1 test ===\n");
    printf("A: threads 0-3 have a[0],a[2]=0x22 (1.0), a[1],a[3]=0\n");
    printf("B: all 0x22 (1.0). SFA=SFB=0x7F (1.0)\n");
    printf("Expected: C[0,n]=64, C[1,n]=0, C[m>=2,n]=0\n\n");

    for (int t = 0; t < 8; t++) {
        int m0 = 2*(t/4), m1 = 2*(t/4)+1;
        int n0 = t%4, n1 = t%4+4;
        printf("T%d: d[0]=%.1f (C[%d,%d])  d[1]=%.1f (C[%d,%d])  d[2]=%.1f (C[%d,%d])  d[3]=%.1f (C[%d,%d])\n",
               t, h[t*4], m0, n0, h[t*4+1], m0, n1, h[t*4+2], m1, n0, h[t*4+3], m1, n1);
    }

    // Count total non-zero A nibbles
    printf("\nThread 0 regs: a[0]=0x%08X a[1]=0x%08X a[2]=0x%08X a[3]=0x%08X\n",
           0x22220000, 0, 0x22220000, 0);  // Expected from the setup

    cudaFree(d_res);
    return 0;
}
