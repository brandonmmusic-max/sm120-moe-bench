/**
 * Test: compare ldmatrix b4x16_p64 vs b8x16 vs manual load for NVF4 MMA.
 * Goal: find the correct SMEM→register loading for full K=64 result.
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

static constexpr int BM = 16, BK = 64, MMA_N = 8, SF_BLOCK = 32;
static constexpr int SMEM_A = BM * (BK / 2);  // 512
static constexpr int SMEM_B = MMA_N * (BK / 2);  // 256

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
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Regular ldmatrix (8-bit elements, no FP4 conversion)
__device__ __forceinline__ void ldmatrix_b8x16_x4(uint32_t (&d)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a));
}
__device__ __forceinline__ void ldmatrix_b8x16_x2(uint32_t (&d)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n16.x2.shared.b8x16 {%0,%1}, [%2];\n"
        : "=r"(d[0]), "=r"(d[1]) : "r"(a));
}
// FP4 ldmatrix (with b4x16_p64 conversion)
__device__ __forceinline__ void ldmatrix_b4_x4(uint32_t (&d)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a));
}
__device__ __forceinline__ void ldmatrix_b4_x2(uint32_t (&d)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 {%0,%1}, [%2];\n"
        : "=r"(d[0]), "=r"(d[1]) : "r"(a));
}

__global__ void test_load_variants(float* out) {
    __shared__ __align__(128) uint8_t s_A[SMEM_A];  // [16, 32]
    __shared__ __align__(128) uint8_t s_B[SMEM_B];  // [8, 32]
    __shared__ __align__(16)  uint8_t s_SFA[4];
    __shared__ __align__(16)  uint8_t s_SFB[MMA_N * 2];

    int tid = threadIdx.x;

    // Fill A: all 0x22 = both nibbles are 2 (FP4 1.0)
    if (tid == 0) {
        for (int i = 0; i < SMEM_A; i++) s_A[i] = 0x22;
        for (int i = 0; i < SMEM_B; i++) s_B[i] = 0x22;
        s_SFA[0] = s_SFA[1] = 0x7F;  // UE8M0 scale
        for (int i = 0; i < MMA_N * 2; i++) s_SFB[i] = 0x7F;
    }
    __syncthreads();

    // Pack SFA/SFB
    uint16_t sfa = (uint16_t)s_SFA[0] | ((uint16_t)s_SFA[1] << 8);
    int sf_n = tid / 4;
    uint16_t sfb = (uint16_t)s_SFB[sf_n * 2] | ((uint16_t)s_SFB[sf_n * 2 + 1] << 8);

    // === Method 1: b4x16_p64 (current, gives half result) ===
    {
        uint32_t a[4], b[2];
        ldmatrix_b4_x4(a, smem_u32(&s_A[tid * 16]));
        ldmatrix_b4_x2(b, smem_u32(&s_B[(tid % 16) * 16]));
        float acc[4] = {0,0,0,0};
        mma_nvf4(acc, a, b, acc, (uint32_t)sfa, (uint32_t)sfb);
        if (tid == 0) out[0] = acc[0];
    }
    __syncthreads();

    // === Method 2: b8x16 (regular, raw packed bytes) ===
    {
        uint32_t a[4], b[2];
        ldmatrix_b8x16_x4(a, smem_u32(&s_A[tid * 16]));
        ldmatrix_b8x16_x2(b, smem_u32(&s_B[(tid % 16) * 16]));
        float acc[4] = {0,0,0,0};
        mma_nvf4(acc, a, b, acc, (uint32_t)sfa, (uint32_t)sfb);
        if (tid == 0) out[1] = acc[0];
    }
    __syncthreads();

    // === Method 3: Direct register load (no ldmatrix) ===
    {
        uint32_t a[4], b[2];
        // Load A: 4 uint32 per thread, 16 bytes starting at tid*16
        const uint32_t* a_ptr = (const uint32_t*)&s_A[tid * 16];
        a[0] = a_ptr[0]; a[1] = a_ptr[1]; a[2] = a_ptr[2]; a[3] = a_ptr[3];
        // Load B: 2 uint32 per thread
        // For b8x16 layout: each thread needs specific bytes
        // Simple linear: first 8 bytes per thread
        const uint32_t* b_ptr = (const uint32_t*)&s_B[(tid % 16) * 16];
        b[0] = b_ptr[0]; b[1] = b_ptr[1];
        float acc[4] = {0,0,0,0};
        mma_nvf4(acc, a, b, acc, (uint32_t)sfa, (uint32_t)sfb);
        if (tid == 0) out[2] = acc[0];
    }

    // === Method 4: b8x16 with halved A stride (8 bytes/addr) ===
    // Use 16B-aligned but read half rows
    // SMEM reorganized: pad each 8-byte chunk to 16-byte boundary
    __shared__ __align__(128) uint8_t s_A_padded[SMEM_A * 2];
    if (tid == 0) {
        memset(s_A_padded, 0, SMEM_A * 2);
        // Copy A data with 16-byte padding between 8-byte chunks
        for (int r = 0; r < 16; r++) {
            for (int c = 0; c < 32; c++) {
                int chunk = c / 8;  // 4 chunks per row
                int offset = c % 8;
                // Padded: chunk at 16-byte boundaries
                s_A_padded[r * 64 + chunk * 16 + offset] = s_A[r * 32 + c];
            }
        }
    }
    __syncthreads();
    {
        uint32_t a[4];
        // Now each 8-byte data chunk is at 16-byte boundaries
        // 64 bytes per padded row × 16 rows = 1024 bytes
        // 32 lanes × 16 bytes = 512 addresses... need 64 addresses for 1024 bytes
        // Doesn't work with x4 (32 addresses)
        // Skip this test
        a[0] = a[1] = a[2] = a[3] = 0;
        uint32_t b_reg[2];
        ldmatrix_b8x16_x2(b_reg, smem_u32(&s_B[(tid % 16) * 16]));
        float acc[4] = {0,0,0,0};
        mma_nvf4(acc, a, b_reg, acc, (uint32_t)sfa, (uint32_t)sfb);
        if (tid == 0) out[3] = acc[0];  // should be 0
    }
}

int main() {
    float* d_out;
    cudaMalloc(&d_out, 16 * sizeof(float));
    cudaMemset(d_out, 0, 16 * sizeof(float));

    test_load_variants<<<1, 32>>>(d_out);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    float h[4];
    cudaMemcpy(h, d_out, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("A=B=all 1.0 (FP4 0x22), SFA=SFB=0x7F\n");
    printf("Expected: 64.0 (if bias=127) or 32.0 (if bias=128, with SFA*SFB=0.5)\n\n");
    printf("Method 1 (b4x16_p64 ldmatrix): %.4f\n", h[0]);
    printf("Method 2 (b8x16 ldmatrix):     %.4f\n", h[1]);
    printf("Method 3 (direct reg load):    %.4f\n", h[2]);
    printf("Method 4 (padded A, zero):     %.4f\n", h[3]);

    cudaFree(d_out);
    return 0;
}
