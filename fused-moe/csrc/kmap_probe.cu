#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

static constexpr int SMEM_A = 16 * 64 / 2;
static constexpr int SMEM_B = 8 * 64 / 2;
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

__global__ void kmap(float* out, int start_byte) {
    const int lane = threadIdx.x;
    extern __shared__ char smem[];
    uint8_t* s_A = (uint8_t*)smem;
    uint8_t* s_B = s_A + SMEM_A;
    for (int i = lane; i < SMEM_A + SMEM_B; i += 32) ((uint8_t*)smem)[i] = 0;
    __syncwarp();
    // Set 1 byte in A row 0 to 1.0 (2 FP4 values)
    if (lane == 0) s_A[start_byte] = 0x22;
    // B row 0: all 1.0
    for (int i = lane; i < 32; i += 32) s_B[i] = 0x22;
    __syncwarp();

    uint32_t a[4]; ldmatrix_b4x16_x4(a, to_smem(&s_A[lane * 16]));
    uint32_t b[2]; ldmatrix_b4x16_x2(b, to_smem(&s_B[(lane % 16) * 16]));
    float acc[4] = {0,0,0,0};
    uint16_t sf = 0x8080;
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
    // Store all lanes d[0] and d[2]
    out[lane * 2 + 0] = acc[0];
    out[lane * 2 + 1] = acc[2];
}

int main() {
    float* d_out; cudaMalloc(&d_out, 64 * sizeof(float));
    float h[64];

    printf("A-tile byte → (lane, K-block) mapping\n");
    printf("Each byte = 2 FP4 values = 1.0. Expected dot = 2.0 per byte.\n\n");

    for (int sb = 0; sb < 32; sb++) {
        kmap<<<1, 32, SMEM_TOTAL>>>(d_out, sb);
        cudaDeviceSynchronize();
        cudaMemcpy(h, d_out, 64 * sizeof(float), cudaMemcpyDeviceToHost);

        // Find which lane has non-zero output
        int hit_lane = -1; int hit_reg = -1; float hit_val = 0;
        for (int l = 0; l < 32; l++) {
            if (h[l*2] != 0) { hit_lane = l; hit_reg = 0; hit_val = h[l*2]; break; }
            if (h[l*2+1] != 0) { hit_lane = l; hit_reg = 2; hit_val = h[l*2+1]; break; }
        }
        int m_row = (hit_reg == 0) ? (hit_lane / 8) : (hit_lane / 8 + 4);
        int n_col = hit_lane % 4;  // within quadrant
        printf("  byte %2d → lane %2d d[%d] val=%.1f  (M=%d, N_local=%d)\n",
               sb, hit_lane, hit_reg, hit_val, m_row, n_col);
    }

    cudaFree(d_out);
    return 0;
}
