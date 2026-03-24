/**
 * Test: direct register load (no ldmatrix) for NVF4 MMA m16n8k64.
 * Load raw packed FP4 bytes directly from SMEM to registers.
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

__device__ __forceinline__ void mma_nvf4(float (&d)[4], const uint32_t (&a)[4],
    const uint32_t (&b)[2], const float (&c)[4], uint32_t sfa, uint32_t sfb) {
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

// b4x16_p64 ldmatrix for comparison
__device__ __forceinline__ void ldmatrix_b4_x4(uint32_t (&d)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a));
}
__device__ __forceinline__ void ldmatrix_b4_x2(uint32_t (&d)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 {%0,%1}, [%2];\n"
        : "=r"(d[0]), "=r"(d[1]) : "r"(a));
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__global__ void test(float* out) {
    __shared__ __align__(128) uint8_t s_A[16 * 32];
    __shared__ __align__(128) uint8_t s_B[8 * 32];
    __shared__ __align__(16)  uint8_t s_SFA[4];
    __shared__ __align__(16)  uint8_t s_SFB[16];

    int tid = threadIdx.x;

    // Fill all with FP4 1.0 (0x22 = both nibbles = E2M1 idx 2)
    if (tid == 0) {
        for (int i = 0; i < 16*32; i++) s_A[i] = 0x22;
        for (int i = 0; i < 8*32; i++) s_B[i] = 0x22;
        s_SFA[0] = s_SFA[1] = 0x7F;
        for (int i = 0; i < 16; i++) s_SFB[i] = 0x7F;
    }
    __syncthreads();

    uint16_t sfa = (uint16_t)s_SFA[0] | ((uint16_t)s_SFA[1] << 8);
    int sf_n = tid / 4;
    uint16_t sfb = (uint16_t)s_SFB[sf_n*2] | ((uint16_t)s_SFB[sf_n*2+1] << 8);

    // === Test 1: b4x16_p64 (known: gives 32) ===
    {
        uint32_t a[4], b[2];
        ldmatrix_b4_x4(a, smem_u32(&s_A[tid * 16]));
        ldmatrix_b4_x2(b, smem_u32(&s_B[(tid%16) * 16]));
        float acc[4] = {0,0,0,0};
        mma_nvf4(acc, a, b, acc, (uint32_t)sfa, (uint32_t)sfb);
        if (tid == 0) { out[0] = acc[0]; out[1] = acc[1]; out[2] = acc[2]; out[3] = acc[3]; }
    }
    __syncthreads();

    // === Test 2: direct load (raw packed bytes) ===
    {
        uint32_t a[4], b[2];
        // Thread tid loads 16 bytes starting at s_A[tid*16]
        const uint32_t* ap = (const uint32_t*)&s_A[tid * 16];
        a[0] = ap[0]; a[1] = ap[1]; a[2] = ap[2]; a[3] = ap[3];
        const uint32_t* bp = (const uint32_t*)&s_B[(tid%16) * 16];
        b[0] = bp[0]; b[1] = bp[1];
        float acc[4] = {0,0,0,0};
        mma_nvf4(acc, a, b, acc, (uint32_t)sfa, (uint32_t)sfb);
        if (tid == 0) { out[4] = acc[0]; out[5] = acc[1]; out[6] = acc[2]; out[7] = acc[3]; }
    }
    __syncthreads();

    // === Test 3: direct load + b4x16_p64 for B only ===
    {
        uint32_t a[4], b[2];
        const uint32_t* ap = (const uint32_t*)&s_A[tid * 16];
        a[0] = ap[0]; a[1] = ap[1]; a[2] = ap[2]; a[3] = ap[3];
        ldmatrix_b4_x2(b, smem_u32(&s_B[(tid%16) * 16]));
        float acc[4] = {0,0,0,0};
        mma_nvf4(acc, a, b, acc, (uint32_t)sfa, (uint32_t)sfb);
        if (tid == 0) { out[8] = acc[0]; out[9] = acc[1]; out[10] = acc[2]; out[11] = acc[3]; }
    }
    __syncthreads();

    // === Test 4: b4x16_p64 A + direct B ===
    {
        uint32_t a[4], b[2];
        ldmatrix_b4_x4(a, smem_u32(&s_A[tid * 16]));
        const uint32_t* bp = (const uint32_t*)&s_B[(tid%16) * 16];
        b[0] = bp[0]; b[1] = bp[1];
        float acc[4] = {0,0,0,0};
        mma_nvf4(acc, a, b, acc, (uint32_t)sfa, (uint32_t)sfb);
        if (tid == 0) { out[12] = acc[0]; out[13] = acc[1]; out[14] = acc[2]; out[15] = acc[3]; }
    }
}

int main() {
    float* d_out;
    cudaMalloc(&d_out, 16 * sizeof(float));
    cudaMemset(d_out, 0, 16 * sizeof(float));

    test<<<1, 32>>>(d_out);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(err)); return 1; }

    float h[16];
    cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost);

    printf("A=B=all 1.0 (0x22), SFA=SFB=0x7F, expected ~64 (bias=127) or ~32 (bias=128)\n\n");
    printf("Test 1 (b4 A + b4 B):     d[0..3] = %.1f %.1f %.1f %.1f\n", h[0],h[1],h[2],h[3]);
    printf("Test 2 (direct A + dir B): d[0..3] = %.1f %.1f %.1f %.1f\n", h[4],h[5],h[6],h[7]);
    printf("Test 3 (direct A + b4 B):  d[0..3] = %.1f %.1f %.1f %.1f\n", h[8],h[9],h[10],h[11]);
    printf("Test 4 (b4 A + direct B):  d[0..3] = %.1f %.1f %.1f %.1f\n", h[12],h[13],h[14],h[15]);

    cudaFree(d_out);
    return 0;
}
