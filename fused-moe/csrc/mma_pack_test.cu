/**
 * Validate pack_a_m1 and pack_b register packing for NVF4 MMA m16n8k64.
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

static constexpr int BK = 64, MMA_N = 8;

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

__device__ __forceinline__ uint32_t get_nibble(const uint8_t* smem, int k) {
    uint8_t byte = smem[k / 2];
    return (k & 1) ? ((byte >> 4) & 0xFu) : (byte & 0xFu);
}

__device__ void pack_a_m1(uint32_t (&a)[4], const uint8_t* s_A, int lane_id) {
    a[0] = a[1] = a[2] = a[3] = 0;
    if (lane_id % 4 == 0) {
        int t1 = lane_id / 4;
        a[0] = get_nibble(s_A, t1)
             | (get_nibble(s_A, t1 + 16) << 4)
             | (get_nibble(s_A, t1 + 32) << 8)
             | (get_nibble(s_A, t1 + 48) << 12);
        a[1] = get_nibble(s_A, t1 + 8)
             | (get_nibble(s_A, t1 + 24) << 4)
             | (get_nibble(s_A, t1 + 40) << 8)
             | (get_nibble(s_A, t1 + 56) << 12);
    }
}

__device__ void pack_b(uint32_t (&b)[2], const uint8_t* s_B_warp, int lane_id) {
    int t0 = lane_id % 4, t1 = lane_id / 4;
    const uint8_t* row_n0 = s_B_warp + t0 * (BK / 2);
    const uint8_t* row_n4 = s_B_warp + (t0 + 4) * (BK / 2);
    b[0] = b[1] = 0;
    for (int vi = 0; vi < 8; vi++) {
        int k = t1 + vi * 8;
        b[0] |= get_nibble(row_n0, k) << (vi * 4);
        b[1] |= get_nibble(row_n4, k) << (vi * 4);
    }
}

__global__ void test(float* out) {
    __shared__ __align__(128) uint8_t s_A[16 * 32];
    __shared__ __align__(128) uint8_t s_B[8 * 32];

    int tid = threadIdx.x;

    // A: all 0x22 (1.0), B: all 0x22 (1.0), SFA=SFB=0x7F
    if (tid == 0) {
        for (int i = 0; i < 16*32; i++) s_A[i] = 0x22;
        for (int i = 0; i < 8*32; i++) s_B[i] = 0x22;
    }
    __syncthreads();

    uint16_t sfa = 0x7F7F;  // both K-groups = 1.0
    uint16_t sfb = 0x7F7F;

    // Test 1: pack_a_m1 + pack_b
    {
        uint32_t a[4], b[2];
        pack_a_m1(a, s_A, tid);
        pack_b(b, s_B, tid);
        float acc[4] = {0,0,0,0};
        mma_nvf4(acc, a, b, acc, (uint32_t)sfa, (uint32_t)sfb);
        // CLayout: t%4==0 threads, d[0], N=t/4
        if (tid % 4 == 0) {
            out[tid / 4] = acc[0];  // should be 64.0 for all 8 columns
        }
    }
    __syncthreads();

    // Test 2: direct linear load (known to give 64.0)
    {
        uint32_t a[4], b[2];
        const uint32_t* ap = (const uint32_t*)&s_A[tid * 16];
        a[0] = ap[0]; a[1] = ap[1]; a[2] = ap[2]; a[3] = ap[3];
        const uint32_t* bp = (const uint32_t*)&s_B[(tid%16)*16];
        b[0] = bp[0]; b[1] = bp[1];
        float acc[4] = {0,0,0,0};
        mma_nvf4(acc, a, b, acc, (uint32_t)sfa, (uint32_t)sfb);
        if (tid == 0) out[8] = acc[0];  // known: 64.0
    }
    __syncthreads();

    // Test 3: B distinct columns, pack_a_m1 + pack_b
    if (tid == 0) {
        // Col 0: 0.5, Col 1: 1.0, ..., Col 7: 0.5
        uint8_t vals[8] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x11};
        for (int c = 0; c < 8; c++)
            for (int k = 0; k < 32; k++)
                s_B[c * 32 + k] = vals[c];
    }
    __syncthreads();
    {
        uint32_t a[4], b[2];
        pack_a_m1(a, s_A, tid);
        pack_b(b, s_B, tid);
        float acc[4] = {0,0,0,0};
        mma_nvf4(acc, a, b, acc, (uint32_t)sfa, (uint32_t)sfb);
        if (tid % 4 == 0)
            out[16 + tid/4] = acc[0];
        // Expected: col n = 64 * E2M1[n+1]
        // = 64 * {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.5}
        // = {32, 64, 96, 128, 192, 256, 384, 32}
    }
}

int main() {
    float* d_out;
    cudaMalloc(&d_out, 32 * sizeof(float));
    cudaMemset(d_out, 0, 32 * sizeof(float));

    test<<<1, 32>>>(d_out);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(err)); return 1; }

    float h[32];
    cudaMemcpy(h, d_out, sizeof(h), cudaMemcpyDeviceToHost);

    printf("Test 1: pack_a_m1 + pack_b, A=B=1.0 (expect all 64.0)\n  ");
    for (int i = 0; i < 8; i++) printf("%.1f ", h[i]);
    printf("\n\nTest 2: direct linear load (reference)\n  %.1f\n", h[8]);

    printf("\nTest 3: distinct B columns, pack functions\n");
    printf("  Expected: 32 64 96 128 192 256 384 32\n  Got:      ");
    for (int i = 0; i < 8; i++) printf("%.1f ", h[16+i]);
    printf("\n");

    cudaFree(d_out);
    return 0;
}
