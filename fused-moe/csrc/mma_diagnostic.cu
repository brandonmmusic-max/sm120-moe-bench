/**
 * Diagnostic: isolate why manual pack gives 4.0 instead of 64.0.
 * Tests multiple data configurations with the same MMA instruction.
 *
 * Build: nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a -o mma_diagnostic mma_diagnostic.cu
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

static constexpr int BK = 64;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

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

__device__ __forceinline__ uint32_t get_nibble(const uint8_t* data, int k) {
    uint8_t byte = data[k / 2];
    return (k & 1) ? ((byte >> 4) & 0xFu) : (byte & 0xFu);
}

// Pack A for M=1 (matches ALayout from mma_traits_sm120.hpp)
__device__ __forceinline__ void pack_a_m1(
    uint32_t (&a)[4], const uint8_t* s_A_row0, int lane_id)
{
    a[0] = a[1] = a[2] = a[3] = 0;
    if (lane_id % 4 == 0) {
        int t1 = lane_id / 4;  // 0..7
        a[0] = get_nibble(s_A_row0, t1)
             | (get_nibble(s_A_row0, t1 + 16) << 4)
             | (get_nibble(s_A_row0, t1 + 32) << 8)
             | (get_nibble(s_A_row0, t1 + 48) << 12);
        a[1] = get_nibble(s_A_row0, t1 + 8)
             | (get_nibble(s_A_row0, t1 + 24) << 4)
             | (get_nibble(s_A_row0, t1 + 40) << 8)
             | (get_nibble(s_A_row0, t1 + 56) << 12);
    }
}

// Pack B (matches BLayout from mma_traits_sm120.hpp)
__device__ __forceinline__ void pack_b(
    uint32_t (&b)[2], const uint8_t* s_B, int lane_id)
{
    int t0 = lane_id % 4;
    int t1 = lane_id / 4;
    const uint8_t* row_n0 = s_B + t0 * (BK / 2);
    const uint8_t* row_n4 = s_B + (t0 + 4) * (BK / 2);
    b[0] = b[1] = 0;
    #pragma unroll
    for (int vi = 0; vi < 8; vi++) {
        int k = t1 + vi * 8;
        b[0] |= get_nibble(row_n0, k) << (vi * 4);
        b[1] |= get_nibble(row_n4, k) << (vi * 4);
    }
}

__global__ void diagnostic_test(float* results) {
    int lid = threadIdx.x;

    __shared__ __align__(128) uint8_t s_A[16 * 32];  // [16, 32] = M16 × K64 packed
    __shared__ __align__(128) uint8_t s_B[8 * 32];   // [8, 32] = N8 × K64 packed

    // ====================================================================
    // Test 1: ALL 0x22 (E2M1=1.0), scale=0x7F (1.0), DIRECT register load
    // Expected: 64.0 (confirmed by mma_direct_test)
    // ====================================================================
    if (lid == 0) {
        for (int i = 0; i < 16*32; i++) s_A[i] = 0x22;
        for (int i = 0; i < 8*32; i++) s_B[i] = 0x22;
    }
    __syncthreads();
    {
        uint32_t a[4], b[2];
        const uint32_t* ap = (const uint32_t*)&s_A[lid * 16];
        a[0] = ap[0]; a[1] = ap[1]; a[2] = ap[2]; a[3] = ap[3];
        const uint32_t* bp = (const uint32_t*)&s_B[(lid%16) * 16];
        b[0] = bp[0]; b[1] = bp[1];
        uint32_t sfa = 0x7F7F, sfb = 0x7F7F;
        float acc[4] = {0,0,0,0};
        mma_nvf4(acc, a, b, acc, sfa, sfb);
        if (lid == 0) {
            results[0] = acc[0]; results[1] = acc[1];
            results[2] = acc[2]; results[3] = acc[3];
        }
    }
    __syncthreads();

    // ====================================================================
    // Test 2: ALL 0x22, scale=0x7F, MANUAL PACK A + manual pack B
    // Expected: should be 64.0 if manual pack is correct
    // BUT: pack_a_m1 only fills nibbles 0-3 (M=0 data),
    //      so nibbles 4-7 (M=1 data) are 0 even though s_A has 0x22.
    //      For M=0 output, this should still give 64.0 since all K values
    //      at M=0 are correctly filled across 8 t0=0 threads.
    // ====================================================================
    {
        uint32_t a[4], b[2];
        pack_a_m1(a, s_A, lid);  // Only fills M=0 data from row 0
        pack_b(b, s_B, lid);
        uint32_t sfa = 0x7F7F, sfb = 0x7F7F;
        float acc[4] = {0,0,0,0};
        mma_nvf4(acc, a, b, acc, sfa, sfb);
        if (lid == 0) {
            results[4] = acc[0]; results[5] = acc[1];
            results[6] = acc[2]; results[7] = acc[3];
        }
    }
    __syncthreads();

    // ====================================================================
    // Test 3: ALL 0x22, scale=0x7F, manual pack A FULL M=16
    // Fill ALL register nibbles (not just M=0)
    // ====================================================================
    {
        uint32_t a[4], b[2];
        a[0] = a[1] = a[2] = a[3] = 0;

        int t0 = lid % 4;
        int t1 = lid / 4;

        // Pack all M rows, not just M=0
        // ALayout: flat = t0*128 + t1 + v0*16 + v1*8 + v2*512
        // reg[m] = v1 + v2*2, nibble p = v0
        for (int m_reg = 0; m_reg < 4; m_reg++) {
            int v1 = m_reg % 2;
            int v2 = m_reg / 2;
            uint32_t reg_val = 0;
            for (int p = 0; p < 8; p++) {
                int v0 = p;
                int flat = t0 * 128 + t1 + v0 * 16 + v1 * 8 + v2 * 512;
                int M = flat / 64;
                int K = flat % 64;
                // s_A[M * 32 + K/2], nibble K%2
                uint8_t byte = s_A[M * 32 + K / 2];
                uint32_t nib = (K & 1) ? ((byte >> 4) & 0xFu) : (byte & 0xFu);
                reg_val |= nib << (p * 4);
            }
            a[m_reg] = reg_val;
        }

        pack_b(b, s_B, lid);
        uint32_t sfa = 0x7F7F, sfb = 0x7F7F;
        float acc[4] = {0,0,0,0};
        mma_nvf4(acc, a, b, acc, sfa, sfb);
        if (lid == 0) {
            results[8] = acc[0]; results[9] = acc[1];
            results[10] = acc[2]; results[11] = acc[3];
        }
    }
    __syncthreads();

    // ====================================================================
    // Test 4: M=1 data (row 0 = 0x22, rest = 0), scale=0x7F
    // Full ALayout packing (all 16 M rows)
    // ====================================================================
    if (lid == 0) {
        for (int i = 0; i < 32; i++) s_A[i] = 0x22;  // row 0
        for (int i = 32; i < 16*32; i++) s_A[i] = 0;  // rows 1-15
    }
    __syncthreads();
    {
        uint32_t a[4], b[2];
        a[0] = a[1] = a[2] = a[3] = 0;

        int t0 = lid % 4;
        int t1 = lid / 4;

        for (int m_reg = 0; m_reg < 4; m_reg++) {
            int v1 = m_reg % 2;
            int v2 = m_reg / 2;
            uint32_t reg_val = 0;
            for (int p = 0; p < 8; p++) {
                int v0 = p;
                int flat = t0 * 128 + t1 + v0 * 16 + v1 * 8 + v2 * 512;
                int M = flat / 64;
                int K = flat % 64;
                uint8_t byte = s_A[M * 32 + K / 2];
                uint32_t nib = (K & 1) ? ((byte >> 4) & 0xFu) : (byte & 0xFu);
                reg_val |= nib << (p * 4);
            }
            a[m_reg] = reg_val;
        }

        pack_b(b, s_B, lid);
        uint32_t sfa = 0x7F7F, sfb = 0x7F7F;
        float acc[4] = {0,0,0,0};
        mma_nvf4(acc, a, b, acc, sfa, sfb);

        // Store d[0..3] for threads 0 AND 4 to see M=0 output from multiple threads
        if (lid == 0) {
            results[12] = acc[0]; results[13] = acc[1];
            results[14] = acc[2]; results[15] = acc[3];
        }
        // Also store from thread that should hold M=0, N=1 output
        if (lid == 4) {
            results[16] = acc[0]; results[17] = acc[1];
            results[18] = acc[2]; results[19] = acc[3];
        }
    }
    __syncthreads();

    // ====================================================================
    // Test 5: M=1 data, pack_a_m1 (original broken function)
    // ====================================================================
    {
        uint32_t a[4], b[2];
        pack_a_m1(a, s_A, lid);
        pack_b(b, s_B, lid);
        uint32_t sfa = 0x7F7F, sfb = 0x7F7F;
        float acc[4] = {0,0,0,0};
        mma_nvf4(acc, a, b, acc, sfa, sfb);
        if (lid == 0) {
            results[20] = acc[0]; results[21] = acc[1];
            results[22] = acc[2]; results[23] = acc[3];
        }
    }
    __syncthreads();

    // ====================================================================
    // Test 6: Register dump for pack_a_m1 vs full ALayout pack
    // Verify register contents match for M=0 positions
    // ====================================================================
    if (lid == 0) {
        uint32_t a_m1[4], a_full[4];
        // pack_a_m1 for thread 0
        a_m1[0] = a_m1[1] = a_m1[2] = a_m1[3] = 0;
        a_m1[0] = get_nibble(s_A, 0) | (get_nibble(s_A, 16) << 4)
                 | (get_nibble(s_A, 32) << 8) | (get_nibble(s_A, 48) << 12);
        a_m1[1] = get_nibble(s_A, 8) | (get_nibble(s_A, 24) << 4)
                 | (get_nibble(s_A, 40) << 8) | (get_nibble(s_A, 56) << 12);

        // Full ALayout for thread 0
        a_full[0] = a_full[1] = a_full[2] = a_full[3] = 0;
        for (int m_reg = 0; m_reg < 4; m_reg++) {
            int v1 = m_reg % 2;
            int v2 = m_reg / 2;
            uint32_t reg_val = 0;
            for (int p = 0; p < 8; p++) {
                int flat = 0 + 0 + p * 16 + v1 * 8 + v2 * 512;
                int M = flat / 64;
                int K = flat % 64;
                uint8_t byte = s_A[M * 32 + K / 2];
                uint32_t nib = (K & 1) ? ((byte >> 4) & 0xFu) : (byte & 0xFu);
                reg_val |= nib << (p * 4);
            }
            a_full[m_reg] = reg_val;
        }

        printf("Thread 0 register comparison (M=1 data):\n");
        printf("  pack_a_m1:  %08X %08X %08X %08X\n", a_m1[0], a_m1[1], a_m1[2], a_m1[3]);
        printf("  full_pack:  %08X %08X %08X %08X\n", a_full[0], a_full[1], a_full[2], a_full[3]);

        // Show which (M,K) each nibble maps to
        printf("  reg[0] nibble mapping:\n");
        for (int p = 0; p < 8; p++) {
            int flat = 0 + 0 + p * 16 + 0 * 8 + 0 * 512;
            printf("    p%d: flat=%d -> M=%d K=%d val_m1=%X val_full=%X\n",
                   p, flat, flat/64, flat%64,
                   (a_m1[0] >> (p*4)) & 0xF,
                   (a_full[0] >> (p*4)) & 0xF);
        }
    }
}

int main() {
    float* d_results;
    cudaMalloc(&d_results, 64 * sizeof(float));
    cudaMemset(d_results, 0, 64 * sizeof(float));

    diagnostic_test<<<1, 32>>>(d_results);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("KERNEL FAILED: %s\n", cudaGetErrorString(err));
        return 1;
    }

    float h[64];
    cudaMemcpy(h, d_results, sizeof(h), cudaMemcpyDeviceToHost);

    printf("\n=== MMA Diagnostic Tests ===\n");
    printf("All tests: A=B=E2M1(2)=1.0, SFA=SFB=0x7F (scale=1.0)\n\n");

    printf("Test 1: Direct load, all 0x22\n");
    printf("  d[0..3] = %.1f %.1f %.1f %.1f  (expect 64.0)\n\n", h[0],h[1],h[2],h[3]);

    printf("Test 2: pack_a_m1 + pack_b, all 0x22 (M=1 func on full data)\n");
    printf("  d[0..3] = %.1f %.1f %.1f %.1f\n\n", h[4],h[5],h[6],h[7]);

    printf("Test 3: Full ALayout pack, all 0x22\n");
    printf("  d[0..3] = %.1f %.1f %.1f %.1f  (expect 64.0)\n\n", h[8],h[9],h[10],h[11]);

    printf("Test 4: Full ALayout pack, M=1 data (row0=0x22, rest=0)\n");
    printf("  T0 d[0..3] = %.1f %.1f %.1f %.1f  (expect d[0]=64.0, rest ~0)\n",
           h[12],h[13],h[14],h[15]);
    printf("  T4 d[0..3] = %.1f %.1f %.1f %.1f  (T4 holds M=0,N=1)\n\n",
           h[16],h[17],h[18],h[19]);

    printf("Test 5: pack_a_m1, M=1 data\n");
    printf("  d[0..3] = %.1f %.1f %.1f %.1f  (same as Test 4?)\n\n",
           h[20],h[21],h[22],h[23]);

    cudaFree(d_results);
    printf("=== Done ===\n");
    return 0;
}
