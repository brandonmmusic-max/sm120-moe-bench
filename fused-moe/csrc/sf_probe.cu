/**
 * Scale Factor Routing Probe — Empirical discovery of bidA/tidA/bidB/tidB
 * ========================================================================
 *
 * Strategy: all-ones FP4 data, single SF set to 2.0, sweep positions.
 * Whichever outputs double → that SF controls those elements.
 *
 * Runs a SINGLE MMA m16n8k64 block-scaled instruction (no loops, no SwiGLU).
 * Dumps all 4 accumulator values per thread for all 32 lanes.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 64;
static constexpr int WARP_SIZE = 32;
static constexpr int SF_BLOCK = 32;  // NVFP4 block size

// SMEM: A tile [16, 32] bytes + B tile [8, 32] bytes + SF areas
static constexpr int SMEM_A = MMA_M * MMA_K / 2;  // 512 bytes
static constexpr int SMEM_B = MMA_N * MMA_K / 2;   // 256 bytes
static constexpr int SMEM_TOTAL = SMEM_A + SMEM_B + 256;  // + padding for SF

__device__ __forceinline__ void ldmatrix_b4x16_x4(
    uint32_t (&dst)[4], uint32_t addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_b4x16_x2(
    uint32_t (&dst)[2], uint32_t addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 {%0,%1}, [%2];\n"
        : "=r"(dst[0]), "=r"(dst[1])
        : "r"(addr));
}

__device__ __forceinline__ uint32_t to_smem(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Single-warp probe kernel: executes one MMA and dumps all outputs
__global__ void sf_probe(
    float* __restrict__ output,   // [32 lanes × 4 regs] = 128 floats
    int sfa_val,    // UE8M0 byte for SFA (all positions use this)
    int sfb_val,    // UE8M0 byte for SFB (all positions use this)
    int test_bidA, int test_tidA,
    int test_bidB, int test_tidB
) {
    const int lane = threadIdx.x;
    if (lane >= WARP_SIZE) return;  // single warp

    extern __shared__ char smem[];
    uint8_t* s_A = reinterpret_cast<uint8_t*>(smem);
    uint8_t* s_B = s_A + SMEM_A;

    // Fill A: identity-like pattern to distinguish rows
    // E2M1 values: 0=0.0, 1=0.5, 2=1.0, 3=1.5, 4=2.0, 5=3.0, 6=4.0, 7=6.0
    // Row r gets value (r%7 + 1) in all K positions
    // This way different M-rows produce different dot products
    for (int i = lane; i < SMEM_A; i += WARP_SIZE) {
        int row = i / (MMA_K / 2);  // 0..15
        uint8_t val = (uint8_t)((row % 7) + 1);  // 1..7
        s_A[i] = val | (val << 4);  // pack same value twice
    }
    // Fill B: different value per N-row to distinguish columns
    // B is [MMA_N=8, MMA_K=64] FP4 packed = [8, 32] bytes
    // Row n gets value (n+1) → E2M1 values 0.5,1.0,1.5,2.0,3.0,4.0,6.0,0.5
    for (int i = lane; i < SMEM_B; i += WARP_SIZE) {
        int row = i / (MMA_K / 2);  // 0..7 (N dimension)
        uint8_t val = (uint8_t)((row % 7) + 1);
        s_B[i] = val | (val << 4);
    }
    __syncwarp();

    // Load fragments
    uint32_t a_regs[4];
    ldmatrix_b4x16_x4(a_regs, to_smem(&s_A[lane * 16]));

    uint32_t b_regs[2];
    ldmatrix_b4x16_x2(b_regs, to_smem(&s_B[(lane % 16) * 16]));

    // Scale factors
    // Pack 2 UE8M0 values (for 2 K-blocks of 32) into uint16
    uint16_t sfa_packed = (uint16_t)((sfa_val & 0xFF) | ((sfa_val & 0xFF) << 8));
    uint16_t sfb_packed = (uint16_t)((sfb_val & 0xFF) | ((sfb_val & 0xFF) << 8));

    float acc[4] = {0, 0, 0, 0};

    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},"
        "{%4,%5,%6,%7},"
        "{%8,%9},"
        "{%10,%11,%12,%13},"
        "{%14},{%15,%16},"
        "{%17},{%18,%19};\n"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        :  "r"(a_regs[0]),  "r"(a_regs[1]),  "r"(a_regs[2]),  "r"(a_regs[3]),
           "r"(b_regs[0]),  "r"(b_regs[1]),
           "f"(acc[0]),  "f"(acc[1]),  "f"(acc[2]),  "f"(acc[3]),
           "r"((uint32_t)sfa_packed),
           "h"((uint16_t)test_bidA), "h"((uint16_t)test_tidA),
           "r"((uint32_t)sfb_packed),
           "h"((uint16_t)test_bidB), "h"((uint16_t)test_tidB)
    );

    // Store all outputs
    output[lane * 4 + 0] = acc[0];
    output[lane * 4 + 1] = acc[1];
    output[lane * 4 + 2] = acc[2];
    output[lane * 4 + 3] = acc[3];
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("SM%d%d Scale Factor Routing Probe\n", prop.major, prop.minor);
    printf("MMA: mxf4nvf4.block_scale.scale_vec::2X.m16n8k64\n");
    printf("Data: all-ones (E2M1 = 1.0 = 0x22 per byte)\n\n");

    float* d_out;
    cudaMalloc(&d_out, 128 * sizeof(float));
    float h_out[128];

    // === Test 1: Baseline with SF = 1.0 (0x7F), bid=0, tid=0 ===
    printf("=== Test 1: SF=1.0, bid=0, tid=0 ===\n");
    sf_probe<<<1, WARP_SIZE, SMEM_TOTAL>>>(d_out, 0x7F, 0x7F, 0, 0, 0, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 128 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Lane  d[0]       d[1]       d[2]       d[3]\n");
    for (int i = 0; i < 32; i++) {
        printf(" %2d: %10.2f %10.2f %10.2f %10.2f\n",
               i, h_out[i*4], h_out[i*4+1], h_out[i*4+2], h_out[i*4+3]);
    }

    // Count non-zeros
    int nz = 0;
    float sum = 0;
    for (int i = 0; i < 128; i++) { if (h_out[i] != 0) nz++; sum += h_out[i]; }
    printf("Non-zero: %d/128, Sum: %.2f\n\n", nz, sum);

    // === Test 2: SF = 2.0 (0x80 = 2^(128-127) = 2.0), bid=0, tid=0 ===
    printf("=== Test 2: SF=2.0, bid=0, tid=0 ===\n");
    sf_probe<<<1, WARP_SIZE, SMEM_TOTAL>>>(d_out, 0x80, 0x80, 0, 0, 0, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 128 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Lane  d[0]       d[1]       d[2]       d[3]\n");
    for (int i = 0; i < 8; i++) {
        printf(" %2d: %10.2f %10.2f %10.2f %10.2f\n",
               i, h_out[i*4], h_out[i*4+1], h_out[i*4+2], h_out[i*4+3]);
    }
    nz = 0; sum = 0;
    for (int i = 0; i < 128; i++) { if (h_out[i] != 0) nz++; sum += h_out[i]; }
    printf("Non-zero: %d/128, Sum: %.2f\n\n", nz, sum);

    // === Test 3: Sweep tidA from 0 to 15, SF=1.0, see if output changes ===
    printf("=== Test 3: Sweep tidA (0..15), SF=1.0 ===\n");
    float baseline_sum = 0;
    for (int tidA = 0; tidA < 16; tidA++) {
        sf_probe<<<1, WARP_SIZE, SMEM_TOTAL>>>(d_out, 0x7F, 0x7F, 0, tidA, 0, 0);
        cudaDeviceSynchronize();
        cudaMemcpy(h_out, d_out, 128 * sizeof(float), cudaMemcpyDeviceToHost);
        float s = 0;
        for (int i = 0; i < 128; i++) s += fabsf(h_out[i]);
        if (tidA == 0) baseline_sum = s;
        printf("  tidA=%2d: sum=%.2f %s\n", tidA, s, (s != baseline_sum) ? "DIFFERENT" : "same");
    }

    // === Test 4: Sweep tidB from 0 to 7, SF=1.0 ===
    printf("\n=== Test 4: Sweep tidB (0..7), SF=1.0 ===\n");
    for (int tidB = 0; tidB < 8; tidB++) {
        sf_probe<<<1, WARP_SIZE, SMEM_TOTAL>>>(d_out, 0x7F, 0x7F, 0, 0, 0, tidB);
        cudaDeviceSynchronize();
        cudaMemcpy(h_out, d_out, 128 * sizeof(float), cudaMemcpyDeviceToHost);
        float s = 0;
        for (int i = 0; i < 128; i++) s += fabsf(h_out[i]);
        printf("  tidB=%2d: sum=%.2f %s\n", tidB, s, (s != baseline_sum) ? "DIFFERENT" : "same");
    }

    // === Test 5: Set SFA only to 0 (should zero everything), keep SFB=1.0 ===
    printf("\n=== Test 5: SFA=0 (exp=0 → 2^-127 ≈ 0), SFB=1.0 ===\n");
    sf_probe<<<1, WARP_SIZE, SMEM_TOTAL>>>(d_out, 0x00, 0x7F, 0, 0, 0, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 128 * sizeof(float), cudaMemcpyDeviceToHost);
    nz = 0;
    for (int i = 0; i < 128; i++) if (h_out[i] != 0) nz++;
    printf("  Non-zero: %d/128 (expect 0 or near-zero)\n", nz);

    // === Test 6: SFA=1.0, SFB=0 ===
    printf("\n=== Test 6: SFA=1.0, SFB=0 ===\n");
    sf_probe<<<1, WARP_SIZE, SMEM_TOTAL>>>(d_out, 0x7F, 0x00, 0, 0, 0, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 128 * sizeof(float), cudaMemcpyDeviceToHost);
    nz = 0;
    for (int i = 0; i < 128; i++) if (h_out[i] != 0) nz++;
    printf("  Non-zero: %d/128 (expect 0 or near-zero)\n", nz);

    // === Test 7: Each lane uses its own tidA = lane_id ===
    printf("\n=== Test 7: tidA=lane_id (per-thread SF addressing) ===\n");
    // Can't do per-thread tidA in a uniform kernel arg — need to embed in the kernel
    // For now, test tidA=lane_id%16 which is what the SFALayout suggests

    cudaFree(d_out);
    return 0;
}
