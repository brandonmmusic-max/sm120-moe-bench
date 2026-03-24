/**
 * Definitive SFA/SFB byte-to-K-block mapping probe.
 * Uses A=B=all 4.0 (nonzero everywhere) to detect which bytes affect which K values.
 *
 * Build: nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a -o mma_scale_probe mma_scale_probe.cu
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

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

__global__ void probe(float* results)
{
    int lid = threadIdx.x;

    // A = all nibble 6 (E2M1 = 4.0) for ALL K=0..63
    __shared__ __align__(128) uint8_t sA[512];
    for (int i = lid; i < 512; i += 32) {
        int row = i / 32;
        sA[swizzle_343(i)] = (row == 0) ? 0x66 : 0;
    }

    // B = all nibble 6 (E2M1 = 4.0)
    __shared__ __align__(128) uint8_t sB[256];
    for (int i = lid; i < 256; i += 32)
        sB[swizzle_343(i)] = 0x66;
    __syncthreads();

    uint32_t a[4], b[2];
    pack_a_m1_v2(a, sA, lid);
    pack_b_v2(b, sB, 0, lid);

    // Unscaled dot: 64 pairs of 4.0*4.0 = 64*16 = 1024
    // With sfa=s, sfb=1.0: C = s * 1024 (if single block)
    // Or split: C = s0*256 + s1*256 + s2*256 + s3*256 (if 4 blocks of 16)

    int idx = 0;
    auto run = [&](uint32_t sfa, uint32_t sfb) {
        float acc[4] = {0,0,0,0};
        mma_nvf4(acc, a, b, acc, sfa, sfb);
        if (lid < 4) {
            results[idx*8 + lid] = acc[0];
            results[idx*8 + lid + 4] = acc[1];
        }
        idx++;
    };

    // sfb = 0x7F7F7F7F (all 1.0) for all tests unless noted
    uint32_t sfb_uniform = 0x7F7F7F7Fu;

    // T0: sfa all 1.0 → expected 1024
    run(0x7F7F7F7Fu, sfb_uniform);

    // T1: sfa all 0.25 → expected 256
    run(0x7D7D7D7Du, sfb_uniform);

    // T2: sfa = [0x7F, 0x7F, 0x00, 0x00] → if 4 blocks: 512; if 2 blocks: 1024
    run(0x00007F7Fu, sfb_uniform);

    // T3: sfa = [0x7D, 0x7F, 0x7F, 0x7F] → if byte0=K0..15: 64+256+256+256=832
    run(0x7F7F7F7Du, sfb_uniform);

    // T4: sfa = [0x7F, 0x7D, 0x7F, 0x7F] → if byte1=K16..31: 256+64+256+256=832
    run(0x7F7F7D7Fu, sfb_uniform);

    // T5: sfa = [0x7F, 0x7F, 0x7D, 0x7F] → if byte2=K32..47: 256+256+64+256=832
    run(0x7F7D7F7Fu, sfb_uniform);

    // T6: sfa = [0x7F, 0x7F, 0x7F, 0x7D] → if byte3=K48..63: 256+256+256+64=832
    run(0x7D7F7F7Fu, sfb_uniform);

    // T7: sfa = [0x00, 0x00, 0x7F, 0x7F]
    run(0x7F7F0000u, sfb_uniform);

    // T8: SFB test - sfa=1.0, sfb=[0x7D,0x7D,0x7D,0x7D]
    run(0x7F7F7F7Fu, 0x7D7D7D7Du);

    // T9: SFB test - sfa=1.0, sfb=[0x7D,0x7F,0x7F,0x7F]
    run(0x7F7F7F7Fu, 0x7F7F7F7Du);
}

int main() {
    printf("=== Definitive Scale Byte Probe ===\n");
    printf("A=B=all 4.0 (nibble 6), all 64 K values nonzero.\n");
    printf("Unscaled dot = 64*16 = 1024. Each 16-element block dot = 256.\n\n");

    float* dr;
    cudaMalloc(&dr, 80 * sizeof(float));
    cudaMemset(dr, 0, 80 * sizeof(float));
    probe<<<1,32>>>(dr);
    cudaDeviceSynchronize();

    float hr[80];
    cudaMemcpy(hr, dr, 80 * sizeof(float), cudaMemcpyDeviceToHost);

    auto print = [&](int t, const char* label) {
        printf("T%d: %s\n  C[0,0]=%.1f\n", t, label, hr[t*8]);
    };

    print(0, "sfa=0x7F7F7F7F (all 1.0). Expect 1024");
    print(1, "sfa=0x7D7D7D7D (all 0.25). Expect 256");
    print(2, "sfa=0x00007F7F (bytes[0,1]=0x7F, bytes[2,3]=0x00). Expect 512 if 4-block, 1024 if 2-block");
    print(3, "sfa=0x7F7F7F7D (byte0=0x7D, rest 0x7F). If byte0=16 elems: 832");
    print(4, "sfa=0x7F7F7D7F (byte1=0x7D, rest 0x7F). If byte1=16 elems: 832");
    print(5, "sfa=0x7F7D7F7F (byte2=0x7D, rest 0x7F). If byte2=16 elems: 832");
    print(6, "sfa=0x7D7F7F7F (byte3=0x7D, rest 0x7F). If byte3=16 elems: 832");
    print(7, "sfa=0x7F7F0000 (bytes[2,3]=0x7F, bytes[0,1]=0x00). Expect 512 if 4-block");
    print(8, "sfb=0x7D7D7D7D (all 0.25). Expect 256");
    print(9, "sfb=0x7F7F7F7D (byte0=0x7D, rest 0x7F). If byte0=16 elems: 832");

    cudaFree(dr);
    return 0;
}
