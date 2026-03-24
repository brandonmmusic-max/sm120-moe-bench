/**
 * M=1 validation with random data.
 * Tests the correct SMEM layout for direct load to produce accurate MMA results.
 *
 * Key discovery: for direct load, M=0 data goes to a[0] and a[2] (bytes 0..3 and 8..11 of
 * each thread's 16-byte chunk). a[1] and a[3] (bytes 4..7 and 12..15) must be zero for M=1.
 *
 * K ordering hypothesis: row-major within each register's 4-byte chunk.
 * Thread 0: a[0] = K=0..7, a[2] = K=8..15
 * Thread 1: a[0] = K=16..23, a[2] = K=24..31
 * Thread 2: a[0] = K=32..39, a[2] = K=40..47
 * Thread 3: a[0] = K=48..55, a[2] = K=56..63
 *
 * Build: nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a -o mma_m1_random mma_m1_random.cu
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static const float E2M1[8] = {0,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};

__device__ __forceinline__ void mma_nvf4(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
    const float (&c)[4], uint32_t sfa, uint32_t sfb) {
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

void quantize(const float* d, int n, uint8_t* p, uint8_t* s) {
    int nblk = n / 32;
    memset(p, 0, n / 2);
    for (int b = 0; b < nblk; b++) {
        float mx = 0;
        for (int i = b*32; i < (b+1)*32; i++) mx = fmaxf(mx, fabsf(d[i]));
        float sc = fmaxf(mx / 6.0f, 1e-30f);
        int e = 127 + (int)ceilf(log2f(sc));
        e = e<1?1:e>254?254:e; s[b] = e;
        float as = powf(2.0f, (float)(e-127));
        for (int i = b*32; i < (b+1)*32; i++) {
            float v = d[i]/as; int sg=v<0?1:0; float av=fabsf(v);
            int idx=0; float bd=av;
            for (int j=1;j<8;j++){float dd=fabsf(av-E2M1[j]);if(dd<bd){bd=dd;idx=j;}}
            uint8_t f4=(sg<<3)|idx;
            if(i%2==0)p[i/2]=f4;else p[i/2]|=(f4<<4);
        }
    }
}

float dequant(const uint8_t* p, const uint8_t* s, int idx) {
    uint8_t nib = (idx&1) ? (p[idx/2]>>4) : (p[idx/2]&0xF);
    int sign=(nib>>3)&1; int mag=nib&7;
    float scale = powf(2.0f, (float)((int)s[idx/32]-127));
    return (sign?-1.0f:1.0f)*E2M1[mag]*scale;
}

// Pack A[0,:] into SMEM for direct load M=1 pattern
// Hypothesis: thread t (0..3) covers K = t*16 .. t*16+15
// a[0] (bytes 0..3) = K[t*16 .. t*16+7], a[2] (bytes 8..11) = K[t*16+8 .. t*16+15]
void pack_a_smem_hypothesis1(const uint8_t* a_packed, uint8_t* smem) {
    memset(smem, 0, 512);
    for (int t = 0; t < 4; t++) {
        int k_base = t * 16;
        // bytes 0..3: K=k_base .. k_base+7 (8 nibbles)
        for (int i = 0; i < 4; i++)
            smem[t*16 + i] = a_packed[(k_base + i*2)/2 * 1]; // pack 2 consecutive K nibbles
        // Actually, just copy the packed bytes directly
        // K=k_base to k_base+7: nibbles at packed byte k_base/2 .. (k_base+7)/2
        for (int i = 0; i < 4; i++)
            smem[t*16 + i] = a_packed[k_base/2 + i];
        // bytes 4..7: zero (M=1)
        // bytes 8..11: K=k_base+8 .. k_base+15
        for (int i = 0; i < 4; i++)
            smem[t*16 + 8 + i] = a_packed[(k_base+8)/2 + i];
        // bytes 12..15: zero (M=1)
    }
}

// Alternative: a[0] = K block 0 part, a[2] = K block 1 part
// Thread 0: a[0] = K=0..7, a[2] = K=32..39
// Thread 1: a[0] = K=8..15, a[2] = K=40..47
// Thread 2: a[0] = K=16..23, a[2] = K=48..55
// Thread 3: a[0] = K=24..31, a[2] = K=56..63
void pack_a_smem_hypothesis2(const uint8_t* a_packed, uint8_t* smem) {
    memset(smem, 0, 512);
    for (int t = 0; t < 4; t++) {
        int k0 = t * 8;
        int k1 = 32 + t * 8;
        for (int i = 0; i < 4; i++) smem[t*16 + i] = a_packed[k0/2 + i];
        for (int i = 0; i < 4; i++) smem[t*16 + 8 + i] = a_packed[k1/2 + i];
    }
}

// Hypothesis 3: match the original BLayout stride pattern
// BLayout K stride: t1 + vi*8 where t1 = tid/4, vi = 0..7
// For A: maybe same pattern?
// Thread t%4 = 0..3, K values from a[0]: t%4 + vi*4 for vi=0..7?
// That gives K=0,4,8,...,28 for t%4=0
// No, let me try: a[0] nibble p = K at position p*stride + offset
// With BLayout: b[0] nibble vi has K = t1 + vi*8 (stride=8, offset=t1)
// Maybe A uses: a[0] nibble p has K = ???
// Let's try: nibble p in a[0] of thread t%4: K = t%4 + p*4 (stride=4)
// Thread 0: K=0,4,8,12,16,20,24,28 (stride 4, 8 values, block 0)
// Thread 1: K=1,5,9,...,29
// Thread 2: K=2,6,10,...,30
// Thread 3: K=3,7,11,...,31
// a[2]: K = 32 + t%4 + p*4
void pack_a_smem_hypothesis3(const uint8_t* a_packed, uint8_t* smem) {
    memset(smem, 0, 512);
    for (int t = 0; t < 4; t++) {
        // a[0]: 8 nibbles, K = t + p*4 for p=0..7
        uint32_t val0 = 0, val2 = 0;
        for (int p = 0; p < 8; p++) {
            int k0 = t + p * 4;
            int k1 = 32 + t + p * 4;
            uint8_t nib0 = (k0&1) ? ((a_packed[k0/2]>>4)&0xF) : (a_packed[k0/2]&0xF);
            uint8_t nib1 = (k1&1) ? ((a_packed[k1/2]>>4)&0xF) : (a_packed[k1/2]&0xF);
            val0 |= (uint32_t)nib0 << (p*4);
            val2 |= (uint32_t)nib1 << (p*4);
        }
        // Write as uint32
        *(uint32_t*)&smem[t*16] = val0;
        *(uint32_t*)&smem[t*16+8] = val2;
    }
}

__global__ void test_random(const uint8_t* smem_a, const uint8_t* b_packed,
                            const uint8_t* b_sf, float* results)
{
    int lid = threadIdx.x;
    __shared__ __align__(128) uint8_t s_A[512];
    __shared__ __align__(128) uint8_t s_B[256];

    for (int i = lid; i < 512; i += 32) s_A[i] = smem_a[i];
    for (int i = lid; i < 256; i += 32) s_B[i] = b_packed[i];
    __syncthreads();

    // Direct load A
    uint32_t a[4];
    const uint32_t* ap = (const uint32_t*)&s_A[lid * 16];
    a[0] = ap[0]; a[1] = ap[1]; a[2] = ap[2]; a[3] = ap[3];

    // Direct load B
    uint32_t b[2];
    const uint32_t* bp = (const uint32_t*)&s_B[(lid%16)*16];
    b[0] = bp[0]; b[1] = bp[1];

    // SFA from first arg (passed via results trick)
    // For simplicity, use uniform SFA=SFB=0x7F7F
    uint32_t sfa = 0x7F7F, sfb = 0x7F7F;

    float acc[4] = {0,0,0,0};
    mma_nvf4(acc, a, b, acc, sfa, sfb);

    if (lid < 4) {
        results[lid] = acc[0];      // C[0, lid]
        results[lid+4] = acc[1];    // C[0, lid+4]
    }
}

int main() {
    srand(42);
    float h_a[64], h_b[8*64];
    for (int k=0;k<64;k++) h_a[k] = ((float)rand()/RAND_MAX-0.5f)*4;
    for (int i=0;i<8*64;i++) h_b[i] = ((float)rand()/RAND_MAX-0.5f)*4;

    uint8_t pa[32], sa[2], pb[256], sb[16];
    quantize(h_a, 64, pa, sa);
    for (int n=0;n<8;n++) quantize(&h_b[n*64], 64, &pb[n*32], &sb[n*2]);

    // Reference
    float ref[8];
    for (int n=0;n<8;n++) {
        ref[n] = 0;
        for (int k=0;k<64;k++)
            ref[n] += dequant(pa, sa, k) * dequant(&pb[n*32], &sb[n*2], k);
    }
    printf("Reference C[0,0..7]: ");
    for (int n=0;n<8;n++) printf("%.2f ",ref[n]);
    printf("\n\nNote: SFA=SFB=0x7F (1.0) used in kernel, so results will differ\n");
    printf("      from reference which uses actual scales.\n\n");

    // Reference with SFA=SFB=1.0 (what the kernel computes)
    float ref_noscale[8];
    for (int n=0;n<8;n++) {
        ref_noscale[n] = 0;
        for (int k=0;k<64;k++) {
            uint8_t nib_a = (k&1) ? ((pa[k/2]>>4)&0xF) : (pa[k/2]&0xF);
            uint8_t nib_b = (k&1) ? ((pb[n*32+k/2]>>4)&0xF) : (pb[n*32+k/2]&0xF);
            int sa_a = (nib_a>>3)&1, ma_a = nib_a&7;
            int sa_b = (nib_b>>3)&1, ma_b = nib_b&7;
            float va = (sa_a?-1.0f:1.0f) * E2M1[ma_a];
            float vb = (sa_b?-1.0f:1.0f) * E2M1[ma_b];
            ref_noscale[n] += va * vb;
        }
    }
    printf("Reference (no scale): ");
    for (int n=0;n<8;n++) printf("%.2f ", ref_noscale[n]);
    printf("\n\n");

    // B SMEM: for direct load, B is stored as N-interleaved
    // 16 threads cover B: thread t reads SMEM[(t%16)*16 .. +15]
    // For 8 N rows × 32 bytes: total = 256 bytes, (t%16)*16 with 16 threads = 256
    // thread 0 reads bytes 0..15 = B[0, K=0..31] first 16 bytes
    // thread 1 reads bytes 16..31 = B[0, K=32..63] next 16 bytes
    // thread 2 reads bytes 32..47 = B[1, K=0..31]... no that's 8*32=256
    // Actually: row-major B, 8 rows × 32 bytes, thread 0 gets bytes 0..15 = row 0 first half
    // thread 1 gets bytes 16..31 = row 0 second half
    // thread 2 gets bytes 32..47 = row 1 first half, etc.
    // With 16 threads: covers 16*16 = 256 bytes = all of B ✓

    uint8_t *da, *db, *dsb; float *dr;
    cudaMalloc(&da, 512); cudaMalloc(&db, 256); cudaMalloc(&dsb, 16); cudaMalloc(&dr, 64);

    // Test each hypothesis
    const char* names[] = {"H1: t*16 contiguous", "H2: a0=blk0 a2=blk1", "H3: stride-4 interleave"};
    void (*packers[])(const uint8_t*, uint8_t*) = {pack_a_smem_hypothesis1, pack_a_smem_hypothesis2, pack_a_smem_hypothesis3};

    for (int h = 0; h < 3; h++) {
        uint8_t smem_a[512];
        packers[h](pa, smem_a);

        cudaMemcpy(da, smem_a, 512, cudaMemcpyHostToDevice);
        cudaMemcpy(db, pb, 256, cudaMemcpyHostToDevice);
        cudaMemset(dr, 0, 64);

        test_random<<<1, 32>>>(da, db, dsb, dr);
        cudaDeviceSynchronize();

        float hr[8];
        cudaMemcpy(hr, dr, 32, cudaMemcpyDeviceToHost);

        printf("%s:\n  C[0,0..7] = ", names[h]);
        for (int n=0;n<8;n++) printf("%.2f ", hr[n]);
        float me = 0;
        for (int n=0;n<8;n++) me = fmaxf(me, fabsf(hr[n]-ref_noscale[n]));
        printf("\n  max_err=%.3f %s\n\n", me, me<0.5f?"PASS":"FAIL");
    }

    cudaFree(da); cudaFree(db); cudaFree(dsb); cudaFree(dr);
    printf("=== Done ===\n");
    return 0;
}
