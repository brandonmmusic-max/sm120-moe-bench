#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

static constexpr int MMA_K = 64;
static constexpr int WARP_SIZE = 32;
static constexpr int SMEM_A = 16 * MMA_K / 2;
static constexpr int SMEM_B = 8 * MMA_K / 2;
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

// Test: A row 0 = 1.0 for K[0:31], 2.0 for K[32:63]
// B row 0 = all 1.0
// With SFA=[sf0, sf1] and SFB=[sf0, sf1]:
// Output = sf0_a * sf0_b * 32*1.0*1.0 + sf1_a * sf1_b * 32*2.0*1.0
// = 32 * sf0a*sf0b + 64 * sf1a*sf1b

__global__ void sf_kblock(float* out, int test_mode) {
    const int lane = threadIdx.x;
    extern __shared__ char smem[];
    uint8_t* s_A = (uint8_t*)smem;
    uint8_t* s_B = s_A + SMEM_A;

    for (int i = lane; i < SMEM_A + SMEM_B; i += WARP_SIZE) ((uint8_t*)smem)[i] = 0;
    __syncwarp();

    // A row 0: K[0:31] = 1.0 (0x22), K[32:63] = 2.0 (0x44)
    // A row 0 is bytes [0..31], bytes 0-15 = K[0:31], bytes 16-31 = K[32:63]
    for (int i = lane; i < 16; i += WARP_SIZE) s_A[i] = 0x22;       // K[0:31] = 1.0
    for (int i = lane; i < 16; i += WARP_SIZE) s_A[16 + i] = 0x44;  // K[32:63] = 2.0

    // B row 0: all 1.0
    for (int i = lane; i < 32; i += WARP_SIZE) s_B[i] = 0x22;
    __syncwarp();

    uint32_t a[4]; ldmatrix_b4x16_x4(a, to_smem(&s_A[lane * 16]));
    uint32_t b[2]; ldmatrix_b4x16_x2(b, to_smem(&s_B[(lane % 16) * 16]));

    float acc[4] = {0,0,0,0};

    // SF packed as uint16: [lo_byte=k_block_0, hi_byte=k_block_1]
    // Test different SF configurations:
    uint16_t sfa, sfb;
    if (test_mode == 0) {
        // Both SF = 1.0 (0x80)
        sfa = 0x8080; sfb = 0x8080;
    } else if (test_mode == 1) {
        // SFA: k0=1.0, k1=2.0 (0x81)
        sfa = 0x8180; sfb = 0x8080;
    } else if (test_mode == 2) {
        // SFA: k0=2.0 (0x81), k1=1.0 (0x80) — SWAPPED
        sfa = 0x8081; sfb = 0x8080;
    } else {
        // SFB: k0=1.0, k1=0.5(0x7F)
        sfa = 0x8080; sfb = 0x7F80;
    }

    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        :"=f"(acc[0]),"=f"(acc[1]),"=f"(acc[2]),"=f"(acc[3])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),
         "r"(b[0]),"r"(b[1]),
         "f"(0.f),"f"(0.f),"f"(0.f),"f"(0.f),
         "r"((uint32_t)sfa),"h"((uint16_t)0),"h"((uint16_t)0),
         "r"((uint32_t)sfb),"h"((uint16_t)0),"h"((uint16_t)0));

    out[lane * 4 + 0] = acc[0];
    out[lane * 4 + 1] = acc[1];
    out[lane * 4 + 2] = acc[2];
    out[lane * 4 + 3] = acc[3];
}

int main() {
    float* d_out; cudaMalloc(&d_out, 128 * sizeof(float));
    float h[128];

    printf("K-block SF alignment probe\n");
    printf("A: K[0:31]=1.0, K[32:63]=2.0. B: all 1.0\n");
    printf("Raw dot: block0=32*1*1=32, block1=32*2*1=64\n\n");

    const char* descs[] = {
        "SF all 1.0:         expect 32*1*1 + 64*1*1 = 96",
        "SFA=[1.0,2.0]:      expect 32*1*1 + 64*2*1 = 160",
        "SFA=[2.0,1.0] SWAP: expect 32*2*1 + 64*1*1 = 128",
        "SFB=[1.0,0.5]:      expect 32*1*1 + 64*1*0.5 = 64"
    };

    for (int mode = 0; mode < 4; mode++) {
        sf_kblock<<<1, WARP_SIZE, SMEM_TOTAL>>>(d_out, mode);
        cudaDeviceSynchronize();
        cudaMemcpy(h, d_out, 128 * sizeof(float), cudaMemcpyDeviceToHost);
        printf("Mode %d: %s\n", mode, descs[mode]);
        printf("  Lane 0 d[0] = %.2f\n\n", h[0]);
    }

    cudaFree(d_out);
    return 0;
}
