/**
 * B-row → output column mapping probe
 * =====================================
 * A = row 0 all 1.0, rest zero
 * B = single row non-zero, sweep row 0..7
 * Which B-row produces non-zero d[0] in lanes 0-3?
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

static constexpr int MMA_K = 64;
static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int WARP_SIZE = 32;
static constexpr int SMEM_A = MMA_M * MMA_K / 2;
static constexpr int SMEM_B = MMA_N * MMA_K / 2;
static constexpr int SMEM_TOTAL = SMEM_A + SMEM_B + 256;

__device__ __forceinline__ void ldmatrix_b4x16_x4(uint32_t (&d)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0,%1,%2,%3},[%4];\n"
        :"=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3]):"r"(a));
}
__device__ __forceinline__ void ldmatrix_b4x16_x2(uint32_t (&d)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 {%0,%1},[%2];\n"
        :"=r"(d[0]),"=r"(d[1]):"r"(a));
}
__device__ __forceinline__ uint32_t to_smem(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__global__ void brow_probe(float* out, int active_brow) {
    const int lane = threadIdx.x;
    extern __shared__ char smem[];
    uint8_t* s_A = (uint8_t*)smem;
    uint8_t* s_B = s_A + SMEM_A;

    // Zero all
    for (int i = lane; i < SMEM_A + SMEM_B; i += WARP_SIZE) ((uint8_t*)smem)[i] = 0;
    __syncwarp();

    // A: row 0 = all 1.0 (0x22)
    for (int i = lane; i < MMA_K / 2; i += WARP_SIZE) s_A[i] = 0x22;

    // B: only active_brow = all 1.0
    if (active_brow < MMA_N) {
        int base = active_brow * (MMA_K / 2);
        for (int i = lane; i < MMA_K / 2; i += WARP_SIZE) s_B[base + i] = 0x22;
    }
    __syncwarp();

    uint32_t a[4]; ldmatrix_b4x16_x4(a, to_smem(&s_A[lane * 16]));
    uint32_t b[2]; ldmatrix_b4x16_x2(b, to_smem(&s_B[(lane % 16) * 16]));

    float acc[4] = {0,0,0,0};
    uint16_t sf = 0x8080;  // SF = 1.0 (bias 128: 0x80 = 2^0)
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        :"=f"(acc[0]),"=f"(acc[1]),"=f"(acc[2]),"=f"(acc[3])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),
         "r"(b[0]),"r"(b[1]),
         "f"(0.f),"f"(0.f),"f"(0.f),"f"(0.f),
         "r"((uint32_t)sf),"h"((uint16_t)0),"h"((uint16_t)0),
         "r"((uint32_t)sf),"h"((uint16_t)0),"h"((uint16_t)0));

    out[lane * 4 + 0] = acc[0];
    out[lane * 4 + 1] = acc[1];
    out[lane * 4 + 2] = acc[2];
    out[lane * 4 + 3] = acc[3];
}

int main() {
    float* d_out; cudaMalloc(&d_out, 128 * sizeof(float));
    float h[128];

    printf("B-row → output mapping (A row 0 = 1.0, SF = 1.0 via 0x80)\n");
    printf("Expected per-element: 64 * 1.0 * 1.0 * 1.0 * 1.0 = 64\n\n");

    for (int brow = 0; brow < MMA_N; brow++) {
        brow_probe<<<1, WARP_SIZE, SMEM_TOTAL>>>(d_out, brow);
        cudaDeviceSynchronize();
        cudaMemcpy(h, d_out, 128 * sizeof(float), cudaMemcpyDeviceToHost);

        printf("B-row %d: ", brow);
        // Print which (lane, reg) are non-zero
        int count = 0;
        for (int l = 0; l < 32; l++) {
            for (int r = 0; r < 4; r++) {
                if (h[l*4+r] != 0.0f) {
                    if (count < 8) printf("L%02d.d%d=%.0f ", l, r, h[l*4+r]);
                    count++;
                }
            }
        }
        printf(" (%d non-zero)\n", count);
    }

    cudaFree(d_out);
    return 0;
}
