/**
 * Probe the ACTUAL register-to-(M,K) mapping of the mxf4nvf4 MMA.
 * Use direct register loads with strategic data patterns to determine
 * which register nibble positions correspond to which (M,K) elements.
 *
 * Build: nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a -o mma_layout_probe mma_layout_probe.cu
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

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

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Probe 1: Put non-zero data only in SPECIFIC register positions
// and see which output elements (d[0..3]) are non-zero.
// A has E2M1=2 (1.0) at one nibble position, rest 0.
// B has all E2M1=2 (1.0).
// SFA=SFB=0x7F (scale=1.0).
// The non-zero d[] values tell us which M output that A nibble contributes to.
__global__ void probe_a_nibble_to_m(float* results) {
    int lid = threadIdx.x;

    // B: all 1.0 for all threads
    // Direct load: fill SMEM with 0x22, load
    __shared__ __align__(128) uint8_t s_B[8 * 32];
    if (lid == 0) for (int i = 0; i < 256; i++) s_B[i] = 0x22;
    __syncthreads();

    uint32_t b[2];
    const uint32_t* bp = (const uint32_t*)&s_B[(lid%16)*16];
    b[0] = bp[0]; b[1] = bp[1];

    uint32_t sfa = 0x7F7F, sfb = 0x7F7F;

    // For thread 0: test each A register and nibble position independently
    // Test format: set ONE nibble to E2M1=2 (1.0), rest to 0.
    // Run MMA, check which d[] outputs are non-zero.

    // We test: reg 0 nibble 0, reg 0 nibble 4, reg 1 nibble 0, etc.
    // Total 32 tests (4 regs × 8 nibbles), but only thread 0 has non-zero data.
    // Store the 4 d[] values for each test.

    for (int test_reg = 0; test_reg < 4; test_reg++) {
        for (int test_nib = 0; test_nib < 8; test_nib++) {
            uint32_t a[4] = {0, 0, 0, 0};

            if (lid == 0) {
                // Set one nibble to E2M1=2 (value 1.0)
                a[test_reg] = (uint32_t)2 << (test_nib * 4);
            }

            float acc[4] = {0, 0, 0, 0};
            mma_nvf4(acc, a, b, acc, sfa, sfb);

            // Thread 0 stores results
            if (lid == 0) {
                int idx = (test_reg * 8 + test_nib) * 4;
                results[idx + 0] = acc[0];
                results[idx + 1] = acc[1];
                results[idx + 2] = acc[2];
                results[idx + 3] = acc[3];
            }
            __syncthreads();
        }
    }
}

// Probe 2: Direct load with M=1 row-major data
// See if the MMA correctly computes C[0,n] = 64 when A row 0 = 1.0, rest = 0
__global__ void probe_direct_m1(float* results) {
    int lid = threadIdx.x;

    __shared__ __align__(128) uint8_t s_A[16 * 32];
    __shared__ __align__(128) uint8_t s_B[8 * 32];

    // A: row 0 = all 0x22 (E2M1 1.0), rows 1-15 = 0
    // B: all 0x22
    if (lid == 0) {
        for (int i = 0; i < 32; i++) s_A[i] = 0x22;
        for (int i = 32; i < 512; i++) s_A[i] = 0;
        for (int i = 0; i < 256; i++) s_B[i] = 0x22;
    }
    __syncthreads();

    // Direct load
    uint32_t a[4], b[2];
    const uint32_t* ap = (const uint32_t*)&s_A[lid * 16];
    a[0] = ap[0]; a[1] = ap[1]; a[2] = ap[2]; a[3] = ap[3];
    const uint32_t* bp = (const uint32_t*)&s_B[(lid%16)*16];
    b[0] = bp[0]; b[1] = bp[1];

    uint32_t sfa = 0x7F7F, sfb = 0x7F7F;
    float acc[4] = {0, 0, 0, 0};
    mma_nvf4(acc, a, b, acc, sfa, sfb);

    // Store d[0..3] from multiple threads
    if (lid < 8) {
        results[lid * 4 + 0] = acc[0];
        results[lid * 4 + 1] = acc[1];
        results[lid * 4 + 2] = acc[2];
        results[lid * 4 + 3] = acc[3];
    }
}

// Probe 3: CLayout mapping - determine which thread holds which (M,N) output
__global__ void probe_clayout(float* results) {
    int lid = threadIdx.x;

    // A: all 0x22, B: all 0x22, scales = 0x7F
    // Set A reg[0] nibble 0 of thread 0 to E2M1=4 (2.0), rest = E2M1=2 (1.0)
    // Then C[m_target, n_target] = sum over K of (A * B)
    // The cell containing the E2M1=4 nibble will have a slightly higher sum.

    __shared__ __align__(128) uint8_t s_B[256];
    if (lid == 0) for (int i = 0; i < 256; i++) s_B[i] = 0x22;
    __syncthreads();

    uint32_t b[2];
    const uint32_t* bp = (const uint32_t*)&s_B[(lid%16)*16];
    b[0] = bp[0]; b[1] = bp[1];

    // All A = 0x22222222, except thread 0 reg[0] nibble 0 = 4 (E2M1 2.0)
    uint32_t a[4] = {0x22222222, 0x22222222, 0x22222222, 0x22222222};
    if (lid == 0)
        a[0] = (a[0] & ~0xFu) | 4u;  // nibble 0 = 4 (2.0 instead of 1.0)

    uint32_t sfa = 0x7F7F, sfb = 0x7F7F;
    float acc[4] = {0, 0, 0, 0};
    mma_nvf4(acc, a, b, acc, sfa, sfb);

    // All threads store d[0..3]
    results[lid * 4 + 0] = acc[0];
    results[lid * 4 + 1] = acc[1];
    results[lid * 4 + 2] = acc[2];
    results[lid * 4 + 3] = acc[3];
}

int main() {
    printf("=== MMA Layout Probe ===\n\n");

    // ---- Probe 1: A nibble -> M output mapping ----
    {
        float* d_res;
        cudaMalloc(&d_res, 32 * 4 * sizeof(float));
        cudaMemset(d_res, 0, 32 * 4 * sizeof(float));

        probe_a_nibble_to_m<<<1, 32>>>(d_res);
        cudaDeviceSynchronize();

        float h[128];
        cudaMemcpy(h, d_res, sizeof(h), cudaMemcpyDeviceToHost);

        printf("--- Probe 1: Thread 0 A nibble -> d[] non-zero pattern ---\n");
        printf("Format: reg[R] nib[N] -> d[0] d[1] d[2] d[3]\n\n");

        for (int r = 0; r < 4; r++) {
            for (int n = 0; n < 8; n++) {
                int idx = (r * 8 + n) * 4;
                // Only print non-zero results for clarity
                if (h[idx] != 0 || h[idx+1] != 0 || h[idx+2] != 0 || h[idx+3] != 0) {
                    printf("  a[%d] nib[%d]: d = %.1f %.1f %.1f %.1f\n",
                           r, n, h[idx], h[idx+1], h[idx+2], h[idx+3]);
                }
            }
        }

        cudaFree(d_res);
    }
    printf("\n");

    // ---- Probe 2: Direct load M=1 ----
    {
        float* d_res;
        cudaMalloc(&d_res, 32 * sizeof(float));
        cudaMemset(d_res, 0, 32 * sizeof(float));

        probe_direct_m1<<<1, 32>>>(d_res);
        cudaDeviceSynchronize();

        float h[32];
        cudaMemcpy(h, d_res, sizeof(h), cudaMemcpyDeviceToHost);

        printf("--- Probe 2: Direct load, M=1 row-major data ---\n");
        printf("A row 0 = 0x22 (1.0), rows 1-15 = 0. B = all 0x22.\n\n");
        for (int t = 0; t < 8; t++) {
            printf("  T%d: d = %.1f %.1f %.1f %.1f\n",
                   t, h[t*4], h[t*4+1], h[t*4+2], h[t*4+3]);
        }
    }
    printf("\n");

    // ---- Probe 3: CLayout -> (M,N) mapping ----
    {
        float* d_res;
        cudaMalloc(&d_res, 32 * 4 * sizeof(float));

        probe_clayout<<<1, 32>>>(d_res);
        cudaDeviceSynchronize();

        float h[128];
        cudaMemcpy(h, d_res, sizeof(h), cudaMemcpyDeviceToHost);

        printf("--- Probe 3: CLayout (thread, d[]) -> output value ---\n");
        printf("All A=B=1.0 except T0 a[0] nib0 = 2.0\n");
        printf("Normal cells = 64.0, affected cell = 65.0 (extra 1.0)\n\n");

        for (int t = 0; t < 32; t++) {
            // Only show threads with non-64.0 values
            bool interesting = false;
            for (int i = 0; i < 4; i++)
                if (h[t*4+i] != 64.0f) interesting = true;
            if (interesting || t < 8) {
                printf("  T%2d: d = %.1f %.1f %.1f %.1f", t,
                       h[t*4], h[t*4+1], h[t*4+2], h[t*4+3]);
                for (int i = 0; i < 4; i++)
                    if (h[t*4+i] == 65.0f) printf("  <-- d[%d] affected", i);
                printf("\n");
            }
        }
    }

    printf("\n=== Done ===\n");
    return 0;
}
