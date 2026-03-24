/**
 * Correct M=1 packing based on empirical hardware register layout probe.
 *
 * DISCOVERED LAYOUT (differs from CuTe ALayout!):
 *   Thread group g = tid/4 (0..7):
 *     a[0]: 8 nibbles at M = 2*g,    K range "A" (covers 8 K values)
 *     a[1]: 8 nibbles at M = 2*g+1,  K range "A"
 *     a[2]: 8 nibbles at M = 2*g,    K range "B"
 *     a[3]: 8 nibbles at M = 2*g+1,  K range "B"
 *
 *   4 threads per group × 16 K nibbles per thread = 64 K values per M row ✓
 *   8 groups × 2 M rows per group = 16 M rows ✓
 *
 * ldmatrix m8n8.x4.b16 routing:
 *   reg[m] for thread t = SMEM[m*128 + (t/4)*16 + (t%4)*4 .. +3]
 *
 * SMEM layout (512 bytes total):
 *   Block 0 [0..127]:   M=0,2,4,6,8,10,12,14 K-block-A data
 *   Block 1 [128..255]: M=1,3,5,7,9,11,13,15 K-block-A data
 *   Block 2 [256..383]: M=0,2,...,14          K-block-B data
 *   Block 3 [384..511]: M=1,3,...,15          K-block-B data
 *
 * Within each block: thread group g gets bytes at offset g*16..g*16+15
 *   = 16 bytes = 32 nibbles = 32 K values for that M row
 *   But K=64 total: block A has K=0..31, block B has K=32..63 (hypothesis)
 *
 * Build: nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a -o mma_correct_pack mma_correct_pack.cu
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static constexpr int BM = 16, BN = 8, BK = 64, SF_BLOCK = 32;
static const float E2M1_TABLE[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

__device__ __forceinline__ void mma_nvf4(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
    const float (&c)[4], uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},"
        "{%4,%5,%6,%7},"
        "{%8,%9},"
        "{%10,%11,%12,%13},"
        "{%14},{%15,%16},"
        "{%17},{%18,%19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
          "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&d)[4], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(addr));
}

__device__ __forceinline__ uint32_t get_nibble(const uint8_t* data, int k) {
    return (k & 1) ? ((data[k/2] >> 4) & 0xFu) : (data[k/2] & 0xFu);
}

// ============================================================================
// CORRECT M=1 packing: put ALL M=0 K data in a[0] and a[2]
// ============================================================================
// a[0]: 8 nibbles covering 8 K values at M=0 (K block A)
// a[2]: 8 nibbles covering 8 more K values at M=0 (K block B)
// a[1], a[3]: zero (M=1 = 0)
//
// K distribution across threads 0-3 in group:
// Thread t%4 holds 16 K values out of 64.
// Hypothesis: simple interleaving by thread:
//   Thread 0: K=0,1,...,15 (first 16 of 64)? Or interleaved?
// Need to test empirically.
//
// For now: try simple layout — each 4-byte chunk = 8 consecutive K nibbles
__device__ __forceinline__ void pack_a_m1_correct(
    uint32_t (&a)[4], const uint8_t* row0_packed, int tid)
{
    a[0] = a[1] = a[2] = a[3] = 0;

    int group = tid / 4;
    if (group != 0) return;  // Only group 0 has M=0 data

    int t_in_group = tid % 4;

    // Hypothesis 1: K values are contiguous per thread
    // Thread t_in_group gets K = t_in_group*16 .. t_in_group*16+15
    // a[0] = K[t*16 .. t*16+7], a[2] = K[t*16+8 .. t*16+15]
    // But this puts both SF blocks in a[0] for t=0,1 and in a[2] for t=2,3

    // Hypothesis 2: reg[0] = K block 0 (K=0..31), reg[2] = K block 1 (K=32..63)
    // Thread t_in_group gets K = t*8 .. t*8+7 from block 0 in a[0]
    //                    and K = 32+t*8 .. 32+t*8+7 from block 1 in a[2]
    int k_base_0 = t_in_group * 8;       // K=0,8,16,24 for t=0,1,2,3
    int k_base_1 = 32 + t_in_group * 8;  // K=32,40,48,56

    for (int p = 0; p < 8; p++) {
        a[0] |= get_nibble(row0_packed, k_base_0 + p) << (p * 4);
        a[2] |= get_nibble(row0_packed, k_base_1 + p) << (p * 4);
    }
}

// B packing — same as before (proven correct with uniform data)
__device__ __forceinline__ void pack_b(
    uint32_t (&b)[2], const uint8_t* s_B, int lane_id)
{
    int t0 = lane_id % 4;
    int t1 = lane_id / 4;
    const uint8_t* row_n0 = s_B + t0 * (BK / 2);
    const uint8_t* row_n4 = s_B + (t0 + 4) * (BK / 2);
    b[0] = b[1] = 0;
    for (int vi = 0; vi < 8; vi++) {
        int k = t1 + vi * 8;
        b[0] |= get_nibble(row_n0, k) << (vi * 4);
        b[1] |= get_nibble(row_n4, k) << (vi * 4);
    }
}

// ============================================================================
// Test: Compare old (wrong) vs new (correct) M=1 packing
// ============================================================================
__global__ void test_packing(
    const uint8_t* g_a_fp4,   // A row 0 packed (32 bytes)
    const uint8_t* g_b_fp4,   // B [8, 32] packed
    float* results)
{
    int lid = threadIdx.x;

    __shared__ __align__(128) uint8_t s_A_row0[32];
    __shared__ __align__(128) uint8_t s_B[8 * 32];

    if (lid < 32) s_A_row0[lid] = g_a_fp4[lid];
    for (int i = lid; i < 256; i += 32) s_B[i] = g_b_fp4[i];
    __syncthreads();

    uint32_t b[2];
    pack_b(b, s_B, lid);
    uint32_t sfa = 0x7F7F, sfb = 0x7F7F;

    // Test 1: CORRECT packing
    {
        uint32_t a[4];
        pack_a_m1_correct(a, s_A_row0, lid);
        float acc[4] = {0,0,0,0};
        mma_nvf4(acc, a, b, acc, sfa, sfb);
        // CLayout: d[0] = C[M=2*(lid/4), N=lid%4]
        //          d[1] = C[M=2*(lid/4), N=lid%4+4]
        if (lid < 4) {
            results[lid] = acc[0];      // C[0, lid]
            results[lid + 4] = acc[1];  // C[0, lid+4]
        }
    }
    __syncthreads();

    // Test 2: SMEM layout + ldmatrix (hypothesis: row-major K in each 16-byte block)
    {
        // Fill SMEM for ldmatrix:
        // Block 0 [0..127]: M-even K-block-A
        // Block 2 [256..383]: M-even K-block-B
        // Others = 0
        __shared__ __align__(1024) uint8_t s_A_smem[512];
        for (int i = lid; i < 512; i += 32) s_A_smem[i] = 0;
        __syncthreads();

        // Only M=0 (group 0, bytes 0-15 in block 0, bytes 256-271 in block 2)
        if (lid == 0) {
            // Block 0, group 0: bytes 0-15 = K=0..31 packed
            for (int i = 0; i < 16; i++) s_A_smem[i] = s_A_row0[i];
            // Block 2, group 0: bytes 256-271 = K=32..63 packed
            for (int i = 0; i < 16; i++) s_A_smem[256 + i] = s_A_row0[16 + i];
        }
        __syncthreads();

        uint32_t a[4];
        ldmatrix_x4(a, smem_u32(&s_A_smem[lid * 16]));
        float acc[4] = {0,0,0,0};
        mma_nvf4(acc, a, b, acc, sfa, sfb);
        if (lid < 4) {
            results[8 + lid] = acc[0];
            results[12 + lid] = acc[1];
        }
    }
    __syncthreads();

    // Test 3: old WRONG packing (for comparison)
    {
        uint32_t a[4] = {0,0,0,0};
        if (lid % 4 == 0) {
            int t1 = lid / 4;
            a[0] = get_nibble(s_A_row0, t1)
                 | (get_nibble(s_A_row0, t1+16) << 4)
                 | (get_nibble(s_A_row0, t1+32) << 8)
                 | (get_nibble(s_A_row0, t1+48) << 12);
            a[1] = get_nibble(s_A_row0, t1+8)
                 | (get_nibble(s_A_row0, t1+24) << 4)
                 | (get_nibble(s_A_row0, t1+40) << 8)
                 | (get_nibble(s_A_row0, t1+56) << 12);
        }
        float acc[4] = {0,0,0,0};
        mma_nvf4(acc, a, b, acc, sfa, sfb);
        if (lid < 4) {
            results[16 + lid] = acc[0];
            results[20 + lid] = acc[1];
        }
    }
}

// Host reference
void host_quantize(const float* data, int n, uint8_t* packed, uint8_t* sf) {
    int nblk = n / SF_BLOCK;
    memset(packed, 0, n / 2);
    for (int b = 0; b < nblk; b++) {
        float mx = 0;
        for (int i = b*SF_BLOCK; i < (b+1)*SF_BLOCK; i++)
            mx = fmaxf(mx, fabsf(data[i]));
        float sc = fmaxf(mx / 6.0f, 1e-30f);
        int e = 127 + (int)ceilf(log2f(sc));
        e = e < 1 ? 1 : e > 254 ? 254 : e;
        sf[b] = e;
        float as = powf(2.0f, (float)(e - 127));
        for (int i = b*SF_BLOCK; i < (b+1)*SF_BLOCK; i++) {
            float v = data[i] / as;
            int sg = v < 0 ? 1 : 0;
            float av = fabsf(v);
            int idx = 0; float bd = av;
            for (int j = 1; j < 8; j++) {
                float d = fabsf(av - E2M1_TABLE[j]);
                if (d < bd) { bd = d; idx = j; }
            }
            uint8_t fp4 = (sg << 3) | idx;
            if (i % 2 == 0) packed[i/2] = fp4; else packed[i/2] |= (fp4 << 4);
        }
    }
}

float host_dequant(const uint8_t* p, const uint8_t* s, int idx) {
    uint8_t nib = (idx & 1) ? (p[idx/2] >> 4) : (p[idx/2] & 0xF);
    int sign = (nib >> 3) & 1; int mag = nib & 7;
    float scale = powf(2.0f, (float)((int)s[idx/SF_BLOCK] - 127));
    return (sign ? -1.0f : 1.0f) * E2M1_TABLE[mag] * scale;
}

int main() {
    printf("=== Correct M=1 Packing Test ===\n\n");

    // Test 1: all 1.0 data
    {
        float h_a[64], h_b[8 * 64];
        for (int k = 0; k < 64; k++) h_a[k] = 1.0f;
        for (int i = 0; i < 8*64; i++) h_b[i] = 1.0f;

        uint8_t pa[32], sa[2], pb[256], sb[16];
        host_quantize(h_a, 64, pa, sa);
        for (int n = 0; n < 8; n++) host_quantize(&h_b[n*64], 64, &pb[n*32], &sb[n*2]);

        // Reference
        float ref = 0;
        for (int k = 0; k < 64; k++)
            ref += host_dequant(pa, sa, k) * host_dequant(pb, sb, k);
        printf("Reference C[0,0] = %.2f (expect 64.0)\n\n", ref);

        uint8_t *da, *db; float *dr;
        cudaMalloc(&da, 32); cudaMalloc(&db, 256); cudaMalloc(&dr, 64*4);
        cudaMemcpy(da, pa, 32, cudaMemcpyHostToDevice);
        cudaMemcpy(db, pb, 256, cudaMemcpyHostToDevice);
        cudaMemset(dr, 0, 64*4);

        test_packing<<<1, 32>>>(da, db, dr);
        cudaDeviceSynchronize();

        float hr[64];
        cudaMemcpy(hr, dr, 64*4, cudaMemcpyDeviceToHost);

        printf("Test 1 (correct pack): C[0,0..7] = ");
        for (int n = 0; n < 8; n++) printf("%.1f ", hr[n]);
        printf("\nTest 2 (ldmatrix):     C[0,0..7] = ");
        for (int n = 0; n < 8; n++) printf("%.1f ", hr[8+n]);
        printf("\nTest 3 (old wrong):    C[0,0..7] = ");
        for (int n = 0; n < 8; n++) printf("%.1f ", hr[16+n]);
        printf("\n\n");

        cudaFree(da); cudaFree(db); cudaFree(dr);
    }

    // Test 2: random data
    {
        srand(42);
        float h_a[64], h_b[8 * 64];
        for (int k = 0; k < 64; k++) h_a[k] = ((float)rand()/RAND_MAX - 0.5f) * 4;
        for (int i = 0; i < 8*64; i++) h_b[i] = ((float)rand()/RAND_MAX - 0.5f) * 4;

        uint8_t pa[32], sa[2], pb[256], sb[16];
        host_quantize(h_a, 64, pa, sa);
        for (int n = 0; n < 8; n++) host_quantize(&h_b[n*64], 64, &pb[n*32], &sb[n*2]);

        float ref[8];
        for (int n = 0; n < 8; n++) {
            ref[n] = 0;
            for (int k = 0; k < 64; k++)
                ref[n] += host_dequant(pa, sa, k) * host_dequant(&pb[n*32], &sb[n*2], k);
        }

        printf("Random data reference C[0,0..7]: ");
        for (int n = 0; n < 8; n++) printf("%.2f ", ref[n]);
        printf("\n\n");

        uint8_t *da, *db; float *dr;
        cudaMalloc(&da, 32); cudaMalloc(&db, 256); cudaMalloc(&dr, 64*4);
        cudaMemcpy(da, pa, 32, cudaMemcpyHostToDevice);
        cudaMemcpy(db, pb, 256, cudaMemcpyHostToDevice);
        cudaMemset(dr, 0, 64*4);

        test_packing<<<1, 32>>>(da, db, dr);
        cudaDeviceSynchronize();

        float hr[64];
        cudaMemcpy(hr, dr, 64*4, cudaMemcpyDeviceToHost);

        printf("Test 1 (correct pack): C[0,0..7] = ");
        for (int n = 0; n < 8; n++) printf("%.2f ", hr[n]);
        float max_err1 = 0;
        for (int n = 0; n < 8; n++) max_err1 = fmaxf(max_err1, fabsf(hr[n]-ref[n]));
        printf(" max_err=%.3f %s\n", max_err1, max_err1 < 0.5f ? "PASS" : "FAIL");

        printf("Test 2 (ldmatrix):     C[0,0..7] = ");
        for (int n = 0; n < 8; n++) printf("%.2f ", hr[8+n]);
        float max_err2 = 0;
        for (int n = 0; n < 8; n++) max_err2 = fmaxf(max_err2, fabsf(hr[8+n]-ref[n]));
        printf(" max_err=%.3f %s\n", max_err2, max_err2 < 0.5f ? "PASS" : "FAIL");

        printf("Test 3 (old wrong):    C[0,0..7] = ");
        for (int n = 0; n < 8; n++) printf("%.2f ", hr[16+n]);
        float max_err3 = 0;
        for (int n = 0; n < 8; n++) max_err3 = fmaxf(max_err3, fabsf(hr[16+n]-ref[n]));
        printf(" max_err=%.3f %s\n", max_err3, max_err3 < 0.5f ? "PASS" : "FAIL");

        cudaFree(da); cudaFree(db); cudaFree(dr);
    }

    printf("\n=== Done ===\n");
    return 0;
}
