/**
 * Probe the EXACT K mapping for each register nibble position.
 * Strategy: set B to have value 1.0 at ONE K position and 0.0 elsewhere.
 * Then set A to all 1.0. The output C[m,n] = A[m,k_target] * B[n,k_target].
 * The non-zero d[] values + which thread tells us the K mapping.
 *
 * But B is distributed across threads too. Alternative: set ONE A nibble to a
 * unique value (3.0 = E2M1 idx 5), all others to 1.0. B = all 1.0.
 * C[m,n] = 63 + 3.0 = 66 for the row containing the unique value, 64 for others.
 * The 66 tells us which M row that nibble belongs to.
 *
 * But we already know reg[0] → even M, reg[1] → odd M. We need the K mapping.
 *
 * Better strategy: Use B = all zeros except one specific K position.
 * If B[n, k0] = 1.0 and B[n, k≠k0] = 0: C[m,n] = A[m, k0].
 * If A = all 1.0: C[m,n] = 1.0 for the M rows that have the matching K, 0 otherwise.
 *
 * But B is packed across threads and we need to know B's layout too.
 *
 * Simplest approach: Set ALL A and B to unique values and check full products.
 * Or: systematically test by putting E2M1=4 (2.0) at one A nibble, rest = E2M1=2 (1.0).
 * C[m,n] = 63 + 2.0 = 65 for the (m,n) receiving the K of the modified nibble.
 * Since B is all 1.0 and both the 2.0 and 1.0 contribute to the same (m,n),
 * we can't distinguish K by output value alone.
 *
 * Best approach: Dual sweep.
 * 1. For each thread t and register r, put E2M1=4 (2.0) in nibble 0, rest = E2M1=2.
 *    All d[] where value ≠ 64 reveal the M row.
 * 2. For K: put A[m=all, k=target] to 2.0, rest 1.0. Check which threads' d[] change.
 *    This requires knowing the SMEM-to-K mapping, which is what we want to find!
 *
 * Actually, simplest: just load data through ldmatrix from a KNOWN SMEM layout
 * and dump all registers to determine the mapping. Then compare with ALayout.
 *
 * Build: nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a -o mma_k_mapping_probe mma_k_mapping_probe.cu
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

// Fill all A registers with 1.0 (E2M1=2) except target_thread, target_reg, target_nibble = 2.0 (E2M1=4)
// Then run MMA with B = all 1.0, check which d[] elements get 65.0 instead of 64.0
// This reveals the (M, K) position of that nibble.
__global__ void probe_layout(
    int target_thread, int target_reg, int target_nibble,
    float* results)  // [32 threads * 4 d values]
{
    int lid = threadIdx.x;

    // A: all E2M1=2 (1.0) except modified nibble
    uint32_t a[4] = {0x22222222, 0x22222222, 0x22222222, 0x22222222};
    if (lid == target_thread) {
        uint32_t mask = 0xFu << (target_nibble * 4);
        a[target_reg] = (a[target_reg] & ~mask) | (4u << (target_nibble * 4));  // E2M1=4 = 2.0
    }

    // B: all E2M1=2 (1.0)
    uint32_t b[2] = {0x22222222, 0x22222222};

    uint32_t sfa = 0x7F7F, sfb = 0x7F7F;
    float acc[4] = {0, 0, 0, 0};
    mma_nvf4(acc, a, b, acc, sfa, sfb);

    results[lid * 4 + 0] = acc[0];
    results[lid * 4 + 1] = acc[1];
    results[lid * 4 + 2] = acc[2];
    results[lid * 4 + 3] = acc[3];
}

int main() {
    float* d_res;
    cudaMalloc(&d_res, 32 * 4 * sizeof(float));
    float h[128];

    printf("=== MMA Register Layout Probe ===\n");
    printf("A=B=all 1.0, one nibble set to 2.0\n");
    printf("Affected output = 65.0 (1 extra from 2.0-1.0)\n\n");

    // CLayout mapping (from probes): thread t, d[v]:
    // M = 2*(t/4) + (v >= 2 ? 1 : 0)
    // N = (t%4) + (v%2 == 1 ? 4 : 0)

    printf("CLayout reference:\n");
    printf("  d[0]: M=2*(t/4),   N=t%%4\n");
    printf("  d[1]: M=2*(t/4),   N=t%%4+4\n");
    printf("  d[2]: M=2*(t/4)+1, N=t%%4\n");
    printf("  d[3]: M=2*(t/4)+1, N=t%%4+4\n\n");

    // Test ALL 32 nibbles for thread 0
    printf("--- Thread 0 nibble mapping ---\n");
    printf("Format: reg[R].nib[N] -> affects (M, N_range)\n\n");

    for (int r = 0; r < 4; r++) {
        for (int n = 0; n < 8; n++) {
            cudaMemset(d_res, 0, 32 * 4 * sizeof(float));
            probe_layout<<<1, 32>>>(0, r, n, d_res);
            cudaDeviceSynchronize();
            cudaMemcpy(h, d_res, sizeof(h), cudaMemcpyDeviceToHost);

            // Find which (thread, d) positions show 65.0
            printf("  a[%d] nib[%d]: ", r, n);
            int found_m = -1;
            for (int t = 0; t < 32; t++) {
                for (int v = 0; v < 4; v++) {
                    if (h[t*4+v] > 64.5f) {  // 65.0
                        int m = 2*(t/4) + (v >= 2 ? 1 : 0);
                        int n_out = (t%4) + ((v%2 == 1) ? 4 : 0);
                        if (found_m == -1) {
                            found_m = m;
                            printf("M=%d  N=", m);
                        }
                        printf("%d,", n_out);
                    }
                }
            }
            printf("\n");
        }
    }

    printf("\n--- Thread 4 nibble mapping ---\n");
    for (int r = 0; r < 4; r++) {
        for (int n = 0; n < 8; n++) {
            cudaMemset(d_res, 0, 32 * 4 * sizeof(float));
            probe_layout<<<1, 32>>>(4, r, n, d_res);
            cudaDeviceSynchronize();
            cudaMemcpy(h, d_res, sizeof(h), cudaMemcpyDeviceToHost);

            printf("  a[%d] nib[%d]: ", r, n);
            int found_m = -1;
            for (int t = 0; t < 32; t++) {
                for (int v = 0; v < 4; v++) {
                    if (h[t*4+v] > 64.5f) {
                        int m = 2*(t/4) + (v >= 2 ? 1 : 0);
                        if (found_m == -1) {
                            found_m = m;
                            printf("M=%d  N=all", m);
                        }
                    }
                }
            }
            if (found_m == -1) printf("(no change)");
            printf("\n");
        }
    }

    cudaFree(d_res);
    printf("\n=== Done ===\n");
    return 0;
}
