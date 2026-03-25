/**
 * Minimal probe: Compare strided vs consecutive K packing for scale_vec::4X
 * Uses all-1.0 data (0x22 nibbles) and unity scales to isolate packing effects.
 *
 * Build: nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a -o consec_k_probe consec_k_probe.cu
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

static constexpr int BK = 64;
static constexpr int BN = 8;  // Just 8 N columns = 1 warp

__device__ __forceinline__ uint32_t swizzle_343(uint32_t off) {
    return off ^ ((off >> 3) & 0x70u);
}

__device__ __forceinline__ uint32_t get_nibble_swz(const uint8_t* smem, int rbo, int k) {
    int addr = rbo + k / 2;
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
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},"
        "{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
          "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

__global__ void test_packing(float* out_strided, float* out_consec) {
    int lane_id = threadIdx.x % 32;

    __shared__ uint8_t s_A[16 * 32];  // BM=16, BK/2=32
    __shared__ uint8_t s_B[8 * 32];   // BN=8, BK/2=32

    // Fill ALL data with 0x22 (nibble 0x2 = E2M1 1.0)
    for (int i = threadIdx.x; i < 16 * 32; i += 256)
        s_A[swizzle_343(i)] = (i / 32 == 0) ? 0x22 : 0;  // Only row 0
    for (int i = threadIdx.x; i < 8 * 32; i += 256)
        s_B[swizzle_343(i)] = 0x22;
    __syncthreads();

    // Unity scales: 0x38 = E4M3FN 1.0
    uint32_t sfa = 0x38383838;
    uint32_t sfb = 0x38383838;

    // ===== Test 1: STRIDED packing (validated baseline) =====
    {
        uint32_t a[4] = {0,0,0,0};
        if (lane_id / 4 == 0) {
            int t0 = lane_id % 4;
            for (int p = 0; p < 8; p++) {
                a[0] |= get_nibble_swz(s_A, 0, t0 + p*8) << (p*4);
                a[2] |= get_nibble_swz(s_A, 0, t0+4 + p*8) << (p*4);
            }
        }
        uint32_t b[2] = {0,0};
        { int g = lane_id/4, t0 = lane_id%4, Nl = 4*(g&1) + (g>>1);
          int rbo = Nl * 32;
          for (int p = 0; p < 8; p++) {
              b[0] |= get_nibble_swz(s_B, rbo, t0 + p*8) << (p*4);
              b[1] |= get_nibble_swz(s_B, rbo, t0+4 + p*8) << (p*4);
          }
        }
        float acc[4] = {0,0,0,0};
        mma_4x(acc, a, b, acc, sfa, sfb);
        // Print from all 32 lanes
        if (lane_id < 8) out_strided[lane_id] = acc[0];
        if (lane_id < 4) {
            out_strided[8 + lane_id] = acc[1];
            out_strided[12 + lane_id] = acc[2];
            out_strided[16 + lane_id] = acc[3];
        }
    }

    __syncthreads();

    // ===== Test 2: CONSECUTIVE packing (new approach) =====
    {
        uint32_t a[4] = {0,0,0,0};
        if (lane_id / 4 == 0) {
            int t0 = lane_id % 4;
            for (int p = 0; p < 8; p++) {
                a[0] |= get_nibble_swz(s_A, 0, t0*8 + p) << (p*4);
                a[2] |= get_nibble_swz(s_A, 0, 32 + t0*8 + p) << (p*4);
            }
        }
        uint32_t b[2] = {0,0};
        { int g = lane_id/4, t0 = lane_id%4, Nl = 4*(g&1) + (g>>1);
          int rbo = Nl * 32;
          for (int p = 0; p < 8; p++) {
              b[0] |= get_nibble_swz(s_B, rbo, t0*8 + p) << (p*4);
              b[1] |= get_nibble_swz(s_B, rbo, 32 + t0*8 + p) << (p*4);
          }
        }
        float acc[4] = {0,0,0,0};
        mma_4x(acc, a, b, acc, sfa, sfb);
        if (lane_id < 8) out_consec[lane_id] = acc[0];
        if (lane_id < 4) {
            out_consec[8 + lane_id] = acc[1];
            out_consec[12 + lane_id] = acc[2];
            out_consec[16 + lane_id] = acc[3];
        }
    }

    // Print A register contents for diagnosis
    if (lane_id == 0) {
        // Re-pack to inspect
        uint32_t a_s[4] = {0,0,0,0}, a_c[4] = {0,0,0,0};
        for (int p = 0; p < 8; p++) {
            a_s[0] |= get_nibble_swz(s_A, 0, 0 + p*8) << (p*4);  // strided t0=0
            a_c[0] |= get_nibble_swz(s_A, 0, 0*8 + p) << (p*4);  // consecutive t0=0
        }
        printf("Lane 0 a[0]: strided=%08X consecutive=%08X\n", a_s[0], a_c[0]);
    }
}

int main() {
    float *d_s, *d_c;
    cudaMalloc(&d_s, 20 * sizeof(float));
    cudaMalloc(&d_c, 20 * sizeof(float));
    cudaMemset(d_s, 0, 20 * sizeof(float));
    cudaMemset(d_c, 0, 20 * sizeof(float));

    test_packing<<<1, 256>>>(d_s, d_c);
    cudaDeviceSynchronize();

    float h_s[20], h_c[20];
    cudaMemcpy(h_s, d_s, 20*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_c, 20*sizeof(float), cudaMemcpyDeviceToHost);

    printf("\n=== Strided K packing (baseline) ===\n");
    printf("d[0] lanes 0-7: ");
    for (int i = 0; i < 8; i++) printf("%.1f ", h_s[i]);
    printf("\nd[1] lanes 0-3: ");
    for (int i = 8; i < 12; i++) printf("%.1f ", h_s[i]);
    printf("\nd[2] lanes 0-3: ");
    for (int i = 12; i < 16; i++) printf("%.1f ", h_s[i]);
    printf("\nd[3] lanes 0-3: ");
    for (int i = 16; i < 20; i++) printf("%.1f ", h_s[i]);

    printf("\n\n=== Consecutive K packing (new) ===\n");
    printf("d[0] lanes 0-7: ");
    for (int i = 0; i < 8; i++) printf("%.1f ", h_c[i]);
    printf("\nd[1] lanes 0-3: ");
    for (int i = 8; i < 12; i++) printf("%.1f ", h_c[i]);
    printf("\nd[2] lanes 0-3: ");
    for (int i = 12; i < 16; i++) printf("%.1f ", h_c[i]);
    printf("\nd[3] lanes 0-3: ");
    for (int i = 16; i < 20; i++) printf("%.1f ", h_c[i]);
    printf("\n\nExpected: d[0] at lanes 0-7 = 64.0 each (64 × 1.0 × 1.0 × 1.0 × 1.0)\n");

    cudaFree(d_s); cudaFree(d_c);
    return 0;
}
