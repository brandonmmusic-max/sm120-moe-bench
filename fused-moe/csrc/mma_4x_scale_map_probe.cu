/**
 * Probe: Determine the nibble-to-scale byte mapping for scale_vec::4X
 *
 * Sets A to have nonzero data ONLY in K block 0 (positions 0..15),
 * with different SFA scales per block. This reveals which SFA byte
 * the hardware applies to which nibble positions.
 *
 * Build: /usr/local/cuda/bin/nvcc -std=c++17 -O2 \
 *   -gencode=arch=compute_120a,code=sm_120a -o mma_4x_scale_map_probe mma_4x_scale_map_probe.cu
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

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

__device__ __forceinline__ void mma_4x(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
    const float (&c)[4], uint32_t sfa, uint32_t sfb)
{
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
          "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

__global__ void test_scale_mapping(float* results)
{
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;

    __shared__ uint8_t s_A[16 * 32];   // BM=16, BK/2=32
    __shared__ uint8_t s_B[64 * 32];   // BN=64, BK/2=32

    // ===== Test 1: A has data ONLY in K[0:15] (block 0) =====
    // Nibble 0x2 = 1.0 in E2M1. Each byte has 2 nibbles.
    // K[0:15] = byte indices 0..7 (8 bytes = 16 nibbles)
    // K[16:63] = zero
    for (int i = tid; i < 16 * 32; i += 256) {
        int row = i / 32;
        int col = i % 32;
        uint8_t val = 0;
        if (row == 0 && col < 8) val = 0x22; // K[0:15] = 1.0
        s_A[swizzle_343(i)] = val;
    }

    // B = all 1.0
    for (int i = tid; i < 64 * 32; i += 256)
        s_B[swizzle_343(i)] = 0x22;

    __syncthreads();

    if (warp_id > 0) return;

    // Pack A (group 0 only)
    uint32_t a[4] = {0,0,0,0};
    if (lane_id / 4 == 0) {
        int t0 = lane_id % 4;
        for (int p = 0; p < 8; p++) {
            a[0] |= get_nibble_swz(s_A, 0, t0 + p*8) << (p*4);
            a[2] |= get_nibble_swz(s_A, 0, t0+4 + p*8) << (p*4);
        }
    }

    // Pack B (warp 0)
    uint32_t b[2] = {0,0};
    {
        int g = lane_id/4, t0 = lane_id%4;
        int N_local = 4*(g&1) + (g>>1);
        int rbo = N_local * 32;
        for (int p = 0; p < 8; p++) {
            b[0] |= get_nibble_swz(s_B, rbo, t0+p*8) << (p*4);
            b[1] |= get_nibble_swz(s_B, rbo, t0+4+p*8) << (p*4);
        }
    }

    // Print A register for lane 0
    if (lane_id == 0) {
        printf("A regs lane0: a[0]=%08X a[1]=%08X a[2]=%08X a[3]=%08X\n",
               a[0], a[1], a[2], a[3]);
        // Decode: each nibble 0x2 = 1.0, 0x0 = 0.0
        printf("a[0] nibbles: ");
        for (int p = 0; p < 8; p++) {
            int nib = (a[0] >> (p*4)) & 0xF;
            int mag = nib & 7;
            float vals[8] = {0,0.5,1,1.5,2,3,4,6};
            printf("K=%d:%.1f ", lane_id%4 + p*8, vals[mag]);
        }
        printf("\na[2] nibbles: ");
        for (int p = 0; p < 8; p++) {
            int nib = (a[2] >> (p*4)) & 0xF;
            int mag = nib & 7;
            float vals[8] = {0,0.5,1,1.5,2,3,4,6};
            printf("K=%d:%.1f ", lane_id%4 + 4 + p*8, vals[mag]);
        }
        printf("\n\n");
    }

    // E4M3FN encode: 1.0=0x38, 2.0=0x40, 4.0=0x48, 8.0=0x50

    // Test 1a: SFA=[1.0, 1.0, 1.0, 1.0], SFB=[1.0 x4]
    // Only block 0 has data → expected = 16 * 1.0 * 1.0 = 16
    {
        uint32_t sfa = 0x38383838;
        uint32_t sfb = 0x38383838;
        float acc[4] = {0,0,0,0};
        mma_4x(acc, a, b, acc, sfa, sfb);
        if (lane_id == 0)
            printf("Test1a: SFA=[1,1,1,1] → acc[0]=%.1f (expect 16.0)\n", acc[0]);
    }

    // Test 1b: SFA=[2.0, 1.0, 1.0, 1.0]
    // If byte 0 maps to block 0 → 16*2 = 32
    // If byte 0 maps to block 1 → 16*1 = 16
    {
        uint32_t sfa = 0x40 | (0x38<<8) | (0x38<<16) | (0x38<<24);
        uint32_t sfb = 0x38383838;
        float acc[4] = {0,0,0,0};
        mma_4x(acc, a, b, acc, sfa, sfb);
        if (lane_id == 0)
            printf("Test1b: SFA=[2,1,1,1] → acc[0]=%.1f (expect 32.0 if byte0→block0)\n", acc[0]);
    }

    // Test 1c: SFA=[1.0, 2.0, 1.0, 1.0]
    // If byte 1 maps to block 1 → 16*1 = 16
    // If byte 1 maps to block 0 → 16*2 = 32
    {
        uint32_t sfa = 0x38 | (0x40<<8) | (0x38<<16) | (0x38<<24);
        uint32_t sfb = 0x38383838;
        float acc[4] = {0,0,0,0};
        mma_4x(acc, a, b, acc, sfa, sfb);
        if (lane_id == 0)
            printf("Test1c: SFA=[1,2,1,1] → acc[0]=%.1f (expect 16.0 if byte1→block1)\n", acc[0]);
    }

    // Test 1d: SFA=[1.0, 1.0, 2.0, 1.0]
    {
        uint32_t sfa = 0x38 | (0x38<<8) | (0x40<<16) | (0x38<<24);
        uint32_t sfb = 0x38383838;
        float acc[4] = {0,0,0,0};
        mma_4x(acc, a, b, acc, sfa, sfb);
        if (lane_id == 0)
            printf("Test1d: SFA=[1,1,2,1] → acc[0]=%.1f (expect 16.0 if byte2→block2)\n", acc[0]);
    }

    // Test 1e: SFA=[1.0, 1.0, 1.0, 2.0]
    {
        uint32_t sfa = 0x38 | (0x38<<8) | (0x38<<16) | (0x40<<24);
        uint32_t sfb = 0x38383838;
        float acc[4] = {0,0,0,0};
        mma_4x(acc, a, b, acc, sfa, sfb);
        if (lane_id == 0)
            printf("Test1e: SFA=[1,1,1,2] → acc[0]=%.1f (expect 16.0 if byte3→block3)\n", acc[0]);
    }

    // ===== Test 2: A has data ONLY in K[16:31] (block 1) =====
    __syncthreads();
    for (int i = tid; i < 16 * 32; i += 256) {
        int row = i / 32;
        int col = i % 32;
        uint8_t val = 0;
        if (row == 0 && col >= 8 && col < 16) val = 0x22; // K[16:31] = 1.0
        s_A[swizzle_343(i)] = val;
    }
    __syncthreads();

    // Re-pack A
    a[0] = a[1] = a[2] = a[3] = 0;
    if (lane_id / 4 == 0) {
        int t0 = lane_id % 4;
        for (int p = 0; p < 8; p++) {
            a[0] |= get_nibble_swz(s_A, 0, t0 + p*8) << (p*4);
            a[2] |= get_nibble_swz(s_A, 0, t0+4 + p*8) << (p*4);
        }
    }

    if (lane_id == 0) {
        printf("\nBlock 1 only (K=16..31):\n");
        printf("a[0] nibbles: ");
        for (int p = 0; p < 8; p++) {
            int nib = (a[0] >> (p*4)) & 0xF;
            int mag = nib & 7;
            float vals[8] = {0,0.5,1,1.5,2,3,4,6};
            printf("K=%d:%.1f ", lane_id%4 + p*8, vals[mag]);
        }
        printf("\n");
    }

    // Test 2a: SFA=[1,1,1,1] → expect 16
    {
        uint32_t sfa = 0x38383838;
        uint32_t sfb = 0x38383838;
        float acc[4] = {0,0,0,0};
        mma_4x(acc, a, b, acc, sfa, sfb);
        if (lane_id == 0)
            printf("Test2a: SFA=[1,1,1,1] → acc[0]=%.1f (expect 16.0)\n", acc[0]);
    }

    // Test 2b: SFA=[2,1,1,1] → expect 16 if byte0→block0 (no effect on block1 data)
    {
        uint32_t sfa = 0x40 | (0x38<<8) | (0x38<<16) | (0x38<<24);
        uint32_t sfb = 0x38383838;
        float acc[4] = {0,0,0,0};
        mma_4x(acc, a, b, acc, sfa, sfb);
        if (lane_id == 0)
            printf("Test2b: SFA=[2,1,1,1] → acc[0]=%.1f (expect 16.0: byte0 on block0, not block1)\n", acc[0]);
    }

    // Test 2c: SFA=[1,2,1,1] → expect 32 if byte1→block1
    {
        uint32_t sfa = 0x38 | (0x40<<8) | (0x38<<16) | (0x38<<24);
        uint32_t sfb = 0x38383838;
        float acc[4] = {0,0,0,0};
        mma_4x(acc, a, b, acc, sfa, sfb);
        if (lane_id == 0)
            printf("Test2c: SFA=[1,2,1,1] → acc[0]=%.1f (expect 32.0 if byte1→block1)\n", acc[0]);
    }

    if (lane_id == 0) printf("\nDone.\n");
}

int main() {
    float* d_results;
    cudaMalloc(&d_results, 256 * sizeof(float));
    test_scale_mapping<<<1, 256>>>(d_results);
    cudaDeviceSynchronize();
    cudaFree(d_results);
    return 0;
}
