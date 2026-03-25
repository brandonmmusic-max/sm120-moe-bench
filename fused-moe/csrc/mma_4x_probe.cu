/**
 * Minimal probe: test scale_vec::4X ue4m3 MMA on SM120
 * Compare with scale_vec::2X ue8m0 using equivalent data
 *
 * Build: /usr/local/cuda/bin/nvcc -std=c++17 -O2 \
 *   -gencode=arch=compute_120a,code=sm_120a -o mma_4x_probe mma_4x_probe.cu
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

// E2M1 table
static const float E2M1_TABLE[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// Swizzle<3,4,3>
__device__ __forceinline__ uint32_t swizzle_343(uint32_t off) {
    return off ^ ((off >> 3) & 0x70u);
}

__device__ __forceinline__ uint32_t get_nibble_swz(
    const uint8_t* smem, int row_off, int k)
{
    int addr = row_off + k / 2;
    uint8_t bv = smem[swizzle_343(addr)];
    return (k & 1) ? ((bv >> 4) & 0xFu) : (bv & 0xFu);
}

// MMA with scale_vec::2X + UE8M0 (Sprint 4 validated)
__device__ __forceinline__ void mma_2x_ue8m0(
    float (&d)[4],
    const uint32_t (&a)[4], const uint32_t (&b)[2],
    const float (&c)[4],
    uint32_t sfa, uint32_t sfb)
{
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

// MMA with scale_vec::4X + UE4M3
__device__ __forceinline__ void mma_4x_ue4m3(
    float (&d)[4],
    const uint32_t (&a)[4], const uint32_t (&b)[2],
    const float (&c)[4],
    uint32_t sfa, uint32_t sfb)
{
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
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

__global__ void test_mma_variants(float* out_2x, float* out_4x)
{
    int tid = threadIdx.x;
    int lane_id = tid % 32;

    // SMEM: A[512] + B[2048]
    __shared__ uint8_t s_A[512];   // 16 rows x 32 bytes
    __shared__ uint8_t s_B[2048];  // 64 rows x 32 bytes

    // Fill A row 0 with all-ones (nibble = 0x2 = E2M1 1.0)
    // Each byte = 0x22 (two nibbles of 1.0)
    for (int i = tid; i < 512; i += 256) {
        int row = i / 32;
        uint8_t val = (row == 0) ? 0x22 : 0x00;
        s_A[swizzle_343(i)] = val;
    }

    // Fill B all rows with all-ones (nibble = 0x2 = E2M1 1.0)
    for (int i = tid; i < 2048; i += 256) {
        s_B[swizzle_343(i)] = 0x22;
    }
    __syncthreads();

    // Pack A (group 0 only for M=0)
    uint32_t a[4] = {0, 0, 0, 0};
    if (lane_id / 4 == 0) {
        int t0 = lane_id % 4;
        for (int p = 0; p < 8; p++) {
            a[0] |= get_nibble_swz(s_A, 0, t0 + p * 8) << (p * 4);
            a[2] |= get_nibble_swz(s_A, 0, t0 + 4 + p * 8) << (p * 4);
        }
    }

    // Pack B (warp 0 only for simplicity)
    int warp_id = tid / 32;
    uint32_t b[2] = {0, 0};
    {
        int g = lane_id / 4;
        int t0 = lane_id % 4;
        int N_local = 4 * (g & 1) + (g >> 1);
        int rbo = (warp_id * 8 + N_local) * 32;
        for (int p = 0; p < 8; p++) {
            b[0] |= get_nibble_swz(s_B, rbo, t0 + p * 8) << (p * 4);
            b[1] |= get_nibble_swz(s_B, rbo, t0 + 4 + p * 8) << (p * 4);
        }
    }

    // === Test 1: scale_vec::2X with UE8M0 scale = 1.0 ===
    // UE8M0: 1.0 = 2^(127-127) = 2^0, byte = 0x7F
    // With scale_vec::2X: sfa = 2 bytes, both 0x7F (unity)
    {
        uint32_t sfa_2x = 0x7F | (0x7F << 8);  // 2 UE8M0 bytes
        uint32_t sfb_2x = 0x7F | (0x7F << 8);
        float acc[4] = {0, 0, 0, 0};
        mma_2x_ue8m0(acc, a, b, acc, sfa_2x, sfb_2x);

        if (lane_id < 4 && warp_id == 0) {
            out_2x[lane_id]     = acc[0];
            out_2x[lane_id + 4] = acc[1];
        }
    }

    // === Test 2: scale_vec::4X with UE4M3 scale = 1.0 ===
    // UE4M3 (E4M3FN, unsigned): 1.0 = (1+0/8) * 2^(7-7) = 1.0, byte = 0x38
    // With scale_vec::4X: sfa = 4 bytes, all 0x38 (unity)
    {
        uint32_t sfa_4x = 0x38 | (0x38 << 8) | (0x38 << 16) | (0x38 << 24);
        uint32_t sfb_4x = 0x38 | (0x38 << 8) | (0x38 << 16) | (0x38 << 24);
        float acc[4] = {0, 0, 0, 0};
        mma_4x_ue4m3(acc, a, b, acc, sfa_4x, sfb_4x);

        if (lane_id < 4 && warp_id == 0) {
            out_4x[lane_id]     = acc[0];
            out_4x[lane_id + 4] = acc[1];
        }
    }

    // === Test 3: scale_vec::4X with UE4M3 scale = 2.0 ===
    // UE4M3: 2.0 = (1+0/8) * 2^(8-7) = 2.0, byte = 0x40
    if (warp_id == 1) {
        uint32_t sfa_4x = 0x40404040;  // scale = 2.0 for A
        uint32_t sfb_4x = 0x38383838;  // scale = 1.0 for B
        float acc[4] = {0, 0, 0, 0};
        mma_4x_ue4m3(acc, a, b, acc, sfa_4x, sfb_4x);

        if (lane_id < 4) {
            out_4x[8 + lane_id]     = acc[0];
            out_4x[8 + lane_id + 4] = acc[1];
        }
    }

    // === Test 4: scale_vec::4X with mixed scales ===
    // byte0 = 0.5 (0x30), byte1 = 1.0 (0x38), byte2 = 2.0 (0x40), byte3 = 4.0 (0x48)
    if (warp_id == 2) {
        uint32_t sfa_4x = 0x30 | (0x38 << 8) | (0x40 << 16) | (0x48 << 24);
        uint32_t sfb_4x = 0x38383838;  // B scales all 1.0
        float acc[4] = {0, 0, 0, 0};
        mma_4x_ue4m3(acc, a, b, acc, sfa_4x, sfb_4x);

        if (lane_id < 4) {
            out_4x[16 + lane_id]     = acc[0];
            out_4x[16 + lane_id + 4] = acc[1];
        }
    }
}

// FP32 reference for A=1, B=1, sfa=1, sfb=1:
// dot product over K=64 of (1.0 * sfa) * (1.0 * sfb) = 64 * sfa * sfb
// With sfa=sfb=1.0: expected = 64.0 per element? No...
// Actually MMA m16n8k64: C[m,n] = sum_k A[m,k] * B[n,k] (transposed)
// For M=1: C[0,n] = sum_k=0..63 A[0,k] * B[n,k] * sfa_k * sfb_k
// With all 1s: = sum_k=0..63 1.0 * 1.0 * 1.0 * 1.0 = 64.0

int main() {
    float *d_out_2x, *d_out_4x;
    cudaMalloc(&d_out_2x, 8 * sizeof(float));
    cudaMalloc(&d_out_4x, 24 * sizeof(float));
    cudaMemset(d_out_2x, 0, 8 * sizeof(float));
    cudaMemset(d_out_4x, 0, 24 * sizeof(float));

    test_mma_variants<<<1, 256>>>(d_out_2x, d_out_4x);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    float h_2x[8], h_4x[24];
    cudaMemcpy(h_2x, d_out_2x, 8 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_4x, d_out_4x, 24 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("=== MMA Probe: scale_vec variants on SM120 ===\n\n");

    printf("All FP4 nibbles = 1.0 (E2M1 0x2), all B = 1.0\n");
    printf("Expected: C[0,n] = sum_k (1.0 * sfa) * (1.0 * sfb) = 64 * sfa * sfb\n\n");

    printf("Test 1: scale_vec::2X, UE8M0, sfa=sfb=1.0 (0x7F7F)\n");
    printf("  Expected: 64.0 per element\n");
    printf("  Got: ");
    for (int i = 0; i < 8; i++) printf("%.2f ", h_2x[i]);
    printf("\n\n");

    printf("Test 2: scale_vec::4X, UE4M3, sfa=sfb=1.0 (0x38383838)\n");
    printf("  Expected: 64.0 per element\n");
    printf("  Got: ");
    for (int i = 0; i < 8; i++) printf("%.2f ", h_4x[i]);
    printf("\n\n");

    printf("Test 3: scale_vec::4X, sfa=2.0 (0x40404040), sfb=1.0\n");
    printf("  Expected: 128.0 per element\n");
    printf("  Got: ");
    for (int i = 8; i < 16; i++) printf("%.2f ", h_4x[i]);
    printf("\n\n");

    printf("Test 4: scale_vec::4X, sfa mixed [0.5,1.0,2.0,4.0], sfb=1.0\n");
    printf("  Expected per 16-elem block: 16*0.5=8, 16*1=16, 16*2=32, 16*4=64 → sum=120\n");
    printf("  Got: ");
    for (int i = 16; i < 24; i++) printf("%.2f ", h_4x[i]);
    printf("\n");

    cudaFree(d_out_2x);
    cudaFree(d_out_4x);
    return 0;
}
