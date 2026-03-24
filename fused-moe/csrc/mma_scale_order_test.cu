/**
 * MMA scale byte order test: determines which byte in sfa/sfb uint32
 * maps to which K block (0-31 vs 32-63) for mxf4nvf4 scale_vec::2X.
 *
 * Strategy: A = [all 1.0 for K=0..31, all 0.0 for K=32..63]
 *           B = [all 1.0]
 * Then C = sfa_block0 * sfb_block0 * dot_block0 + sfa_block1 * sfb_block1 * 0
 *        = sfa_block0 * sfb_block0 * dot_block0
 * By checking C vs known sfa values, we determine byte order.
 *
 * Build: nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a -o mma_scale_order_test mma_scale_order_test.cu
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static const float E2M1[8] = {0, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

__device__ __forceinline__ uint32_t swizzle_343(uint32_t o) {
    return o ^ ((o >> 3) & 0x70u);
}
__device__ __forceinline__ uint32_t get_nibble_swz(const uint8_t* s, int rbo, int k) {
    uint8_t v = s[swizzle_343(rbo + k/2)];
    return (k&1) ? ((v>>4)&0xFu) : (v&0xFu);
}
__device__ __forceinline__ void pack_a_m1_v2(uint32_t (&a)[4], const uint8_t* s, int lid) {
    a[0]=a[1]=a[2]=a[3]=0;
    if (lid/4!=0) return;
    int t0=lid%4;
    for(int p=0;p<8;p++){
        a[0]|=get_nibble_swz(s,0,t0+p*8)<<(p*4);
        a[2]|=get_nibble_swz(s,0,t0+4+p*8)<<(p*4);
    }
}
__device__ __forceinline__ void pack_b_v2(uint32_t (&b)[2], const uint8_t* s, int wnb, int lid) {
    int g=lid/4,t0=lid%4,N=4*(g&1)+(g>>1);
    int rbo=(wnb+N)*32;
    b[0]=b[1]=0;
    for(int p=0;p<8;p++){
        b[0]|=get_nibble_swz(s,rbo,t0+p*8)<<(p*4);
        b[1]|=get_nibble_swz(s,rbo,t0+4+p*8)<<(p*4);
    }
}
__device__ __forceinline__ void mma_nvf4(
    float(&d)[4],const uint32_t(&a)[4],const uint32_t(&b)[2],
    const float(&c)[4],uint32_t sfa,uint32_t sfb){
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        :"=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),
         "r"(b[0]),"r"(b[1]),
         "f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]),
         "r"(sfa),"h"((uint16_t)0),"h"((uint16_t)0),
         "r"(sfb),"h"((uint16_t)0),"h"((uint16_t)0));
}

__global__ void test_scale_order(float* results)
{
    int lid = threadIdx.x;

    // A: K=0..31 all 1.0, K=32..63 all 0.0
    // FP4(1.0) = E2M1 index 2 = 0b0010 = nibble 2
    // Block scale for K[0:31]: max=1.0, scale=1.0/6.0, exp=125, actual_scale=0.25
    // Quantized: 1.0/0.25 = 4.0, E2M1 index 6 = nibble 6
    // So nibbles 0-31: 0x6, nibbles 32-63: 0x0
    // Packed bytes (pairs): 0x66 for first 16 bytes, 0x00 for last 16 bytes
    __shared__ __align__(128) uint8_t sA[512];
    for (int i = lid; i < 512; i += 32) {
        int row = i / 32, col = i % 32;
        uint8_t val = 0;
        if (row == 0 && col < 16) val = 0x66;  // Two nibbles of 6 (E2M1 value 4.0)
        sA[swizzle_343(i)] = val;
    }

    // B: all 1.0 = nibble 6 with same scale (exp=125)
    __shared__ __align__(128) uint8_t sB[256];
    for (int i = lid; i < 256; i += 32)
        sB[swizzle_343(i)] = 0x66;
    __syncthreads();

    uint32_t a[4], b[2];
    pack_a_m1_v2(a, sA, lid);
    pack_b_v2(b, sB, 0, lid);

    // Test 1: sfa = [exp_a0=125, exp_a1=127], sfb = [127, 127] (scale=1.0)
    // A block 0 scale = 2^(125-127) = 0.25
    // A block 1 scale = 2^(127-127) = 1.0
    // Since A block 1 data = 0, C depends only on block 0:
    // C = 0.25 * 1.0 * sum_{k=0..31}(4.0 * 4.0) = 0.25 * 32 * 16 = 128
    // ... wait, the MMA applies sfa * sfb to each pair, so:
    // C = sfa_b0 * sfb_b0 * sum_{k=0..31}(E2M1_raw[a_nib] * E2M1_raw[b_nib])
    // = sfa_b0 * sfb_b0 * 32 * 4.0 * 4.0 = sfa_b0 * sfb_b0 * 512
    // With sfb=1.0: C = sfa_b0 * 512

    // Test with sfa_byte0=125(0x7D), sfa_byte1=130(0x82) → very different
    // If byte0=K block 0: C = 2^(125-127) * 512 = 0.25*512 = 128
    // If byte0=K block 1: C = 2^(130-127) * 512 = 8*512 = 4096 (but block 1 data=0)
    //                       C = 0  (since A block 1 = 0)
    // Wait, we need to be careful. If byte0 maps to K block 0:
    //   C = 2^(0x7D-127)*512 + 2^(0x82-127)*0 = 0.25*512 = 128
    // If byte0 maps to K block 1:
    //   C = 2^(0x82-127)*512 + 2^(0x7D-127)*0 = 8*512 = 4096?
    //   No! Block 1 data is 0, so:
    //   C = sfa_for_K0_block * sfb_K0 * 512 + sfa_for_K1_block * sfb_K1 * 0
    //   If byte0=K1: sfa_K0 = byte1 = 0x82 → 2^(130-127) = 8.0
    //   C = 8.0 * 1.0 * 512 = 4096
    //   If byte0=K0: sfa_K0 = byte0 = 0x7D → 2^(125-127) = 0.25
    //   C = 0.25 * 1.0 * 512 = 128

    uint32_t sfa_test = 0x7Du | (0x82u << 8);  // byte0=0x7D, byte1=0x82
    uint32_t sfb_test = 0x7F7Fu;                 // both = 1.0

    float acc1[4] = {0,0,0,0};
    mma_nvf4(acc1, a, b, acc1, sfa_test, sfb_test);

    // Test 2: swap bytes
    uint32_t sfa_swap = 0x82u | (0x7Du << 8);  // byte0=0x82, byte1=0x7D
    float acc2[4] = {0,0,0,0};
    mma_nvf4(acc2, a, b, acc2, sfa_swap, sfb_test);

    // Test 3: uniform (both = 0x7D)
    float acc3[4] = {0,0,0,0};
    mma_nvf4(acc3, a, b, acc3, 0x7D7Du, sfb_test);

    // Test 4: uniform (both = 0x82)
    float acc4[4] = {0,0,0,0};
    mma_nvf4(acc4, a, b, acc4, 0x8282u, sfb_test);

    // Test 5: SFB byte order test
    // Use sfa=0x7F7F (scale 1.0), sfb varies per group
    // sfb_byte0=0x7D(0.25), sfb_byte1=0x82(8.0)
    int g = lid / 4;
    int N_local = 4*(g&1)+(g>>1);
    // All N columns use same SFB for this test
    uint32_t sfb_test2 = 0x7Du | (0x82u << 8);  // byte0=0x7D, byte1=0x82
    float acc5[4] = {0,0,0,0};
    mma_nvf4(acc5, a, b, acc5, 0x7F7Fu, sfb_test2);

    // Test 6: wider sfa register test - what if bytes 2,3 matter?
    // Pack both scales into bytes 0 and 2
    uint32_t sfa_wide = 0x7Du | (0x82u << 16);  // byte0=0x7D, byte2=0x82
    float acc6[4] = {0,0,0,0};
    mma_nvf4(acc6, a, b, acc6, sfa_wide, sfb_test);

    // Test 7: Pack into bytes 0 and 2 with 0x7F in bytes 1,3
    uint32_t sfa_alt = 0x7Du | (0x7Fu << 8) | (0x82u << 16) | (0x7Fu << 24);
    float acc7[4] = {0,0,0,0};
    mma_nvf4(acc7, a, b, acc7, sfa_alt, sfb_test);

    if (lid < 4) {
        results[0 + lid] = acc1[0]; results[4 + lid] = acc1[1];
        results[8 + lid] = acc2[0]; results[12 + lid] = acc2[1];
        results[16 + lid] = acc3[0]; results[20 + lid] = acc3[1];
        results[24 + lid] = acc4[0]; results[28 + lid] = acc4[1];
        results[32 + lid] = acc5[0]; results[36 + lid] = acc5[1];
        results[40 + lid] = acc6[0]; results[44 + lid] = acc6[1];
        results[48 + lid] = acc7[0]; results[52 + lid] = acc7[1];
    }

    if (lid == 0) {
        printf("a_regs: %08x %08x %08x %08x\n", a[0], a[1], a[2], a[3]);
        printf("b_regs: %08x %08x\n", b[0], b[1]);
    }
}

int main() {
    printf("=== MMA Scale Byte Order Test ===\n\n");
    printf("Setup: A=[4.0 for K=0..31, 0 for K=32..63] with block_scale=[0.25,1.0]\n");
    printf("       B=[4.0 for all K] with block_scale=[1.0,1.0]\n");
    printf("Expected unscaled dot (block 0): 32 * 4.0 * 4.0 = 512\n");
    printf("Expected unscaled dot (block 1): 0\n\n");

    float* dr;
    cudaMalloc(&dr, 56 * sizeof(float));
    cudaMemset(dr, 0, 56 * sizeof(float));

    test_scale_order<<<1, 32>>>(dr);
    cudaDeviceSynchronize();

    float hr[56];
    cudaMemcpy(hr, dr, 56 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Test 1: sfa=[0x7D,0x82,0,0] sfb=[0x7F,0x7F]\n");
    printf("  If byte0=K0: C=0.25*512=128. If byte1=K0: C=8*512=4096\n");
    printf("  C[0,0..7]: "); for(int i=0;i<8;i++) printf("%.1f ",hr[i]); printf("\n\n");

    printf("Test 2: sfa=[0x82,0x7D,0,0] (swapped) sfb=[0x7F,0x7F]\n");
    printf("  C[0,0..7]: "); for(int i=8;i<16;i++) printf("%.1f ",hr[i]); printf("\n\n");

    printf("Test 3: sfa=[0x7D,0x7D,0,0] (uniform 0.25) sfb=[0x7F,0x7F]\n");
    printf("  Expected: 0.25*512=128\n");
    printf("  C[0,0..7]: "); for(int i=16;i<24;i++) printf("%.1f ",hr[i]); printf("\n\n");

    printf("Test 4: sfa=[0x82,0x82,0,0] (uniform 8.0) sfb=[0x7F,0x7F]\n");
    printf("  Expected: 8*512=4096\n");
    printf("  C[0,0..7]: "); for(int i=24;i<32;i++) printf("%.1f ",hr[i]); printf("\n\n");

    printf("Test 5: sfa=[0x7F,0x7F] sfb=[0x7D,0x82,0,0]\n");
    printf("  If sfb byte0=K0: C=0.25*512=128. If sfb byte1=K0: C=8*512=4096\n");
    printf("  C[0,0..7]: "); for(int i=32;i<40;i++) printf("%.1f ",hr[i]); printf("\n\n");

    printf("Test 6: sfa=byte0=0x7D,byte2=0x82 (skip byte1)\n");
    printf("  C[0,0..7]: "); for(int i=40;i<48;i++) printf("%.1f ",hr[i]); printf("\n\n");

    printf("Test 7: sfa=[0x7D,0x7F,0x82,0x7F]\n");
    printf("  C[0,0..7]: "); for(int i=48;i<56;i++) printf("%.1f ",hr[i]); printf("\n\n");

    cudaFree(dr);
    return 0;
}
