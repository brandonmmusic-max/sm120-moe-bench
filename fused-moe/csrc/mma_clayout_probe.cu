/**
 * Minimal probe: determine CLayout mapping of NVF4 MMA m16n8k64 on SM120.
 * Sets A and B to known patterns, runs one MMA, dumps all thread accumulators.
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

static constexpr int BM = 16, BK = 64, MMA_N = 8;
static constexpr int SF_BLOCK = 32;
static constexpr int SMEM_A = BM * (BK / 2);  // 512
static constexpr int SMEM_B = MMA_N * (BK / 2);  // 256
static constexpr int SMEM_SFB = MMA_N * (BK / SF_BLOCK); // 16

__device__ __forceinline__ void ldmatrix_b4x16_x4(uint32_t (&d)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a));
}
__device__ __forceinline__ void ldmatrix_b4x16_x2(uint32_t (&d)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 {%0,%1}, [%2];\n"
        : "=r"(d[0]), "=r"(d[1]) : "r"(a));
}
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

// E2M1 values: idx 0..7 → {0, 0.5, 1, 1.5, 2, 3, 4, 6}
// UE8M0 scale = 2^(byte-127), 0x7F=127→1.0

__global__ void probe(float* out_d) {
    // out_d: [32 threads × 4 values]
    __shared__ uint8_t s_A[SMEM_A];
    __shared__ uint8_t s_B[SMEM_B];
    __shared__ uint8_t s_SFA[4];
    __shared__ uint8_t s_SFB[SMEM_SFB];

    int tid = threadIdx.x;

    // Zero everything
    for (int i = tid; i < SMEM_A; i += 32) s_A[i] = 0;
    for (int i = tid; i < SMEM_B; i += 32) s_B[i] = 0;
    if (tid < 4) s_SFA[tid] = 0;
    for (int i = tid; i < SMEM_SFB; i += 32) s_SFB[i] = 0;
    __syncthreads();

    // Set A: ALL rows = 1.0 everywhere (to see full CLayout)
    // Then also test with just row 0
    // FP4 nibble for 1.0: idx=2, sign=0
    // Packed byte 0x22 = two 1.0 values
    if (tid == 0) {
        for (int i = 0; i < SMEM_A; i++) s_A[i] = 0x22; // all 1.0
    }

    // Test: A = all 1.0, B = all 1.0
    // Expected: C[m,n] = sum_{k=0..63} 1.0*1.0 = 64.0 for ALL elements
    if (tid == 0) {
        for (int i = 0; i < SMEM_B; i++) s_B[i] = 0x22;  // all 1.0
    }

    // Test 3 scale configurations
    // Config stored in blockIdx.x (run 3 blocks)
    int cfg = blockIdx.x;
    if (tid == 0) {
        if (cfg == 0) {  // SFA=0x7F, SFB=0x7F
            s_SFA[0] = s_SFA[1] = 0x7F;
            for (int i = 0; i < SMEM_SFB; i++) s_SFB[i] = 0x7F;
        } else if (cfg == 1) {  // SFA=0x80, SFB=0x7F
            s_SFA[0] = s_SFA[1] = 0x80;
            for (int i = 0; i < SMEM_SFB; i++) s_SFB[i] = 0x7F;
        } else {  // SFA=0x7F, SFB=0x80
            s_SFA[0] = s_SFA[1] = 0x7F;
            for (int i = 0; i < SMEM_SFB; i++) s_SFB[i] = 0x80;
        }
    }
    __syncthreads();

    // Load fragments
    uint32_t a_regs[4];
    ldmatrix_b4x16_x4(a_regs, smem_u32(&s_A[tid * 16]));

    uint32_t b_regs[2];
    ldmatrix_b4x16_x2(b_regs, smem_u32(&s_B[(tid % 16) * 16]));

    // SFA
    uint16_t sfa = (uint16_t)s_SFA[0] | ((uint16_t)s_SFA[1] << 8);

    // SFB: thread group determines column
    int sf_n = tid / 4;  // 0..7
    int sfb_base = sf_n * (BK / SF_BLOCK);
    uint16_t sfb = (uint16_t)s_SFB[sfb_base] | ((uint16_t)s_SFB[sfb_base + 1] << 8);

    float acc[4] = {0, 0, 0, 0};
    mma_nvf4(acc, a_regs, b_regs, acc, (uint32_t)sfa, (uint32_t)sfb);

    // Dump all accumulators (offset by block)
    int offset = blockIdx.x * 32 * 4;
    out_d[offset + tid * 4 + 0] = acc[0];
    out_d[offset + tid * 4 + 1] = acc[1];
    out_d[offset + tid * 4 + 2] = acc[2];
    out_d[offset + tid * 4 + 3] = acc[3];
}

int main() {
    float* d_out;
    cudaMalloc(&d_out, 3 * 32 * 4 * sizeof(float));
    cudaMemset(d_out, 0, 3 * 32 * 4 * sizeof(float));

    probe<<<3, 32>>>(d_out);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    float h[3 * 128];
    cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost);

    const char* cfgnames[] = {"SFA=0x7F,SFB=0x7F", "SFA=0x80,SFB=0x7F", "SFA=0x7F,SFB=0x80"};
    for (int cfg = 0; cfg < 3; cfg++) {
        printf("\n--- Config %d: %s ---\n", cfg, cfgnames[cfg]);
        printf("  Thread 0 d[0] = %.4f\n", h[cfg*128 + 0]);
    }

    // Expected: C[m,n] = A[m,k=0] * B[k=0,n]
    // A[0,0] = 1.0, all other A = 0
    // B[0,n] = {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.5}
    // So C[0,n] = B[0,n] for n=0..7, and C[m>0,n] = 0

    printf("Expected C[0,:] = {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.5}\n\n");

    // Find non-zero values to determine CLayout
    printf("Non-zero accumulator values:\n");
    for (int t = 0; t < 32; t++) {
        for (int v = 0; v < 4; v++) {
            float val = h[t * 4 + v];
            if (fabsf(val) > 1e-6f) {
                int g = t / 4, l = t % 4;
                printf("  thread=%2d (g=%d,l=%d) d[%d] = %8.4f\n", t, g, l, v, val);
            }
        }
    }

    printf("\n--- Full dump (threads with any non-zero) ---\n");
    for (int t = 0; t < 32; t++) {
        bool any = false;
        for (int v = 0; v < 4; v++)
            if (fabsf(h[t*4+v]) > 1e-6f) any = true;
        if (any) {
            printf("  t%02d: d[0]=%8.4f  d[1]=%8.4f  d[2]=%8.4f  d[3]=%8.4f\n",
                   t, h[t*4], h[t*4+1], h[t*4+2], h[t*4+3]);
        }
    }

    cudaFree(d_out);
    return 0;
}
