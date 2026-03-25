/**
 * Probe: single-expert GEMM1 with scale_vec::4X to isolate correctness issue
 * Compare MMA output vs host dequant reference for one K=64 tile, one N=64 pass
 *
 * Build: /usr/local/cuda/bin/nvcc -std=c++17 -O2 \
 *   -gencode=arch=compute_120a,code=sm_120a -o mma_4x_gemm1_probe mma_4x_gemm1_probe.cu
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static constexpr int BK = 64;
static constexpr int BN = 64;
static constexpr int BM = 16;
static constexpr int SF_BLOCK = 16;
static constexpr int SF_PER_K = BK / SF_BLOCK;  // 4

static const float E2M1_TABLE[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

__device__ __forceinline__ uint32_t swizzle_343(uint32_t off) {
    return off ^ ((off >> 3) & 0x70u);
}

__device__ __forceinline__ uint32_t get_nibble_swz(
    const uint8_t* smem, int row_off, int k)
{
    int addr = row_off + k / 2;
    uint8_t bv = smem[swizzle_343(addr)];
    return (k & 1) ? ((bv >> 4) & 0xFu) : (bv & 0xFu);
}

__device__ __forceinline__ void mma_4x_ue4m3(
    float (&d)[4],
    const uint32_t (&a)[4], const uint32_t (&b)[2],
    const float (&c)[4],
    uint32_t sfa, uint32_t sfb)
{
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
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

// Single tile GEMM1 test kernel
// A[1, BK/2] + SFA[SF_PER_K], B[BN, BK/2] + SFB[BN, SF_PER_K]
__global__ void test_single_tile(
    const uint8_t* __restrict__ a_fp4,  // [BK/2] = 32 bytes
    const uint8_t* __restrict__ a_sf,   // [SF_PER_K] = 4 bytes
    const uint8_t* __restrict__ b_fp4,  // [BN, BK/2] = 64×32 bytes
    const uint8_t* __restrict__ b_sf,   // [BN, SF_PER_K] = 64×4 bytes
    float* output)                      // [BN] = 64 floats
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    __shared__ uint8_t s_A[BM * (BK/2)];    // 512
    __shared__ uint8_t s_B[BN * (BK/2)];    // 2048
    __shared__ uint8_t s_SFA[16];             // 4 bytes used
    __shared__ uint8_t s_SFB[BN * SF_PER_K]; // 256

    // Load A: input FP4 [16, 32] — row 0 only
    for (int i = tid; i < BM * (BK/2); i += 256) {
        int row = i / (BK/2);
        int col = i % (BK/2);
        uint8_t val = (row == 0) ? a_fp4[col] : 0;
        s_A[swizzle_343(i)] = val;
    }

    // Load B: weight FP4 [64, 32]
    for (int i = tid; i < BN * (BK/2); i += 256) {
        int row = i / (BK/2);
        int col = i % (BK/2);
        s_B[swizzle_343(i)] = b_fp4[row * (BK/2) + col];
    }

    // Load SFA: 4 E4M3FN bytes
    if (tid < SF_PER_K)
        s_SFA[tid] = a_sf[tid];

    // Load SFB: [64, 4]
    for (int i = tid; i < BN * SF_PER_K; i += 256) {
        int row = i / SF_PER_K;
        int col = i % SF_PER_K;
        s_SFB[i] = b_sf[row * SF_PER_K + col];
    }

    __syncthreads();

    // Pack A
    uint32_t a_regs[4] = {0, 0, 0, 0};
    if (lane_id / 4 == 0) {
        int t0 = lane_id % 4;
        for (int p = 0; p < 8; p++) {
            a_regs[0] |= get_nibble_swz(s_A, 0, t0 + p * 8) << (p * 4);
            a_regs[2] |= get_nibble_swz(s_A, 0, t0 + 4 + p * 8) << (p * 4);
        }
    }

    // Pack B
    uint32_t b_regs[2] = {0, 0};
    {
        int g = lane_id / 4;
        int t0 = lane_id % 4;
        int N_local = 4 * (g & 1) + (g >> 1);
        int rbo = (warp_id * 8 + N_local) * (BK/2);
        for (int p = 0; p < 8; p++) {
            b_regs[0] |= get_nibble_swz(s_B, rbo, t0 + p * 8) << (p * 4);
            b_regs[1] |= get_nibble_swz(s_B, rbo, t0 + 4 + p * 8) << (p * 4);
        }
    }

    // Pack SFA
    uint32_t sfa = (uint32_t)s_SFA[0]
                 | ((uint32_t)s_SFA[1] << 8)
                 | ((uint32_t)s_SFA[2] << 16)
                 | ((uint32_t)s_SFA[3] << 24);

    // Pack SFB
    int g_sfb = lane_id / 4;
    int N_loc_sfb = 4 * (g_sfb & 1) + (g_sfb >> 1);
    int sfb_n = warp_id * 8 + N_loc_sfb;
    uint32_t sfb = (uint32_t)s_SFB[sfb_n * SF_PER_K]
                 | ((uint32_t)s_SFB[sfb_n * SF_PER_K + 1] << 8)
                 | ((uint32_t)s_SFB[sfb_n * SF_PER_K + 2] << 16)
                 | ((uint32_t)s_SFB[sfb_n * SF_PER_K + 3] << 24);

    // MMA
    float acc[4] = {0, 0, 0, 0};
    mma_4x_ue4m3(acc, a_regs, b_regs, acc, sfa, sfb);

    // Output: only group 0 (lanes 0-3) has M=0 data
    if (lane_id < 4) {
        output[warp_id * 8 + lane_id]     = acc[0];
        output[warp_id * 8 + lane_id + 4] = acc[1];
    }
}

// Host E4M3FN
float h_e4m3fn_decode(uint8_t x) {
    int e = (x >> 3) & 0xF;
    int m = x & 7;
    if (e == 15 && m == 7) return 0;
    if (e == 0) return ldexpf((float)m / 8.0f, -6);
    return ldexpf(1.0f + (float)m / 8.0f, e - 7);
}

uint8_t h_e4m3fn_ceil(float v) {
    if (v <= 0) return 0x01;
    if (v > 448.0f) return 0x7E;
    uint8_t best = 0x7E;
    float bval = 448.0f;
    for (int e = 0; e <= 15; e++)
        for (int m = 0; m <= 7; m++) {
            if (e == 15 && m == 7) continue;
            float repr = (e == 0) ? ldexpf((float)m / 8.0f, -6)
                                  : ldexpf(1.0f + (float)m / 8.0f, e - 7);
            if (repr >= v && repr < bval) { bval = repr; best = (uint8_t)((e << 3) | m); }
        }
    return best;
}

void quantize_nvfp4_e4m3(const float* data, int n, uint8_t* packed, uint8_t* sf) {
    int nblk = (n + SF_BLOCK - 1) / SF_BLOCK;
    memset(packed, 0, n / 2);
    for (int b = 0; b < nblk; b++) {
        int s = b * SF_BLOCK, e = (s + SF_BLOCK < n) ? s + SF_BLOCK : n;
        float mx = 0;
        for (int i = s; i < e; i++) mx = fmaxf(mx, fabsf(data[i]));
        float des = mx / 6.0f;
        if (des < 1e-30f) des = 1e-30f;
        sf[b] = h_e4m3fn_ceil(des);
        float sc = h_e4m3fn_decode(sf[b]);
        if (sc < 1e-30f) sc = 1e-30f;
        for (int i = s; i < e; i++) {
            float sv = data[i] / sc;
            float av = fabsf(sv);
            int sign = (sv < 0) ? 1 : 0;
            int idx = 0; float bd = av;
            for (int j = 1; j < 8; j++) {
                float d = fabsf(av - E2M1_TABLE[j]);
                if (d < bd) { bd = d; idx = j; }
            }
            uint8_t fp4 = (uint8_t)((sign << 3) | idx);
            int bi = i / 2;
            if (i % 2 == 0) packed[bi] = fp4; else packed[bi] |= (fp4 << 4);
        }
    }
}

float dequant_e4m3(const uint8_t* p, const uint8_t* sf, int i) {
    uint8_t bv = p[i/2];
    uint8_t nib = (i & 1) ? (bv >> 4) : (bv & 0xF);
    int sign = (nib >> 3) & 1;
    int mag = nib & 7;
    float val = E2M1_TABLE[mag] * h_e4m3fn_decode(sf[i / SF_BLOCK]);
    return sign ? -val : val;
}

int main() {
    srand(42);
    auto randf = []() { return ((float)rand() / RAND_MAX - 0.5f) * 2.0f; };

    // Generate random data for one tile
    float h_a[BK], h_b[BN * BK];
    for (int i = 0; i < BK; i++) h_a[i] = randf();
    for (int i = 0; i < BN * BK; i++) h_b[i] = randf() * 0.1f;

    // Quantize
    uint8_t a_fp4[BK/2], a_sf[SF_PER_K];
    quantize_nvfp4_e4m3(h_a, BK, a_fp4, a_sf);

    uint8_t b_fp4[BN * (BK/2)], b_sf[BN * SF_PER_K];
    for (int n = 0; n < BN; n++)
        quantize_nvfp4_e4m3(&h_b[n * BK], BK, &b_fp4[n * (BK/2)], &b_sf[n * SF_PER_K]);

    // Host reference: dequant dot product
    float h_ref[BN];
    for (int n = 0; n < BN; n++) {
        float sum = 0;
        for (int k = 0; k < BK; k++)
            sum += dequant_e4m3(a_fp4, a_sf, k) * dequant_e4m3(&b_fp4[n*(BK/2)], &b_sf[n*SF_PER_K], k);
        h_ref[n] = sum;
    }

    // GPU
    uint8_t *d_a_fp4, *d_a_sf, *d_b_fp4, *d_b_sf;
    float *d_out;
    cudaMalloc(&d_a_fp4, BK/2);
    cudaMalloc(&d_a_sf, SF_PER_K);
    cudaMalloc(&d_b_fp4, BN * (BK/2));
    cudaMalloc(&d_b_sf, BN * SF_PER_K);
    cudaMalloc(&d_out, BN * sizeof(float));
    cudaMemset(d_out, 0, BN * sizeof(float));

    cudaMemcpy(d_a_fp4, a_fp4, BK/2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_sf, a_sf, SF_PER_K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_fp4, b_fp4, BN * (BK/2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_sf, b_sf, BN * SF_PER_K, cudaMemcpyHostToDevice);

    test_single_tile<<<1, 256>>>(d_a_fp4, d_a_sf, d_b_fp4, d_b_sf, d_out);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    float h_out[BN];
    cudaMemcpy(h_out, d_out, BN * sizeof(float), cudaMemcpyDeviceToHost);

    printf("=== Single-tile GEMM1 probe (scale_vec::4X, E4M3FN scales) ===\n\n");
    printf("A scales (E4M3FN): [0x%02X=%.4f, 0x%02X=%.4f, 0x%02X=%.4f, 0x%02X=%.4f]\n",
           a_sf[0], h_e4m3fn_decode(a_sf[0]),
           a_sf[1], h_e4m3fn_decode(a_sf[1]),
           a_sf[2], h_e4m3fn_decode(a_sf[2]),
           a_sf[3], h_e4m3fn_decode(a_sf[3]));

    printf("\nFirst 8 N columns:\n");
    printf("  N   GPU        Ref        Ratio      B_sf[0:4]\n");
    for (int n = 0; n < 8; n++) {
        float ratio = (h_ref[n] != 0) ? h_out[n] / h_ref[n] : 0;
        printf("  %2d  %10.4f %10.4f %10.4f   [%02X %02X %02X %02X]\n",
               n, h_out[n], h_ref[n], ratio,
               b_sf[n*4], b_sf[n*4+1], b_sf[n*4+2], b_sf[n*4+3]);
    }

    // Overall error
    double sum_err = 0, sum_ref = 0;
    int nan_cnt = 0;
    for (int n = 0; n < BN; n++) {
        if (isnan(h_out[n])) { nan_cnt++; continue; }
        sum_err += (double)(h_out[n] - h_ref[n]) * (h_out[n] - h_ref[n]);
        sum_ref += (double)h_ref[n] * h_ref[n];
    }
    double rel_err = (sum_ref > 0) ? sqrt(sum_err / sum_ref) * 100.0 : 0;
    printf("\nOverall: RelErr=%.4f%%, NaN=%d\n", rel_err, nan_cnt);
    printf("VERDICT: %s\n", (rel_err < 5.0 && nan_cnt == 0) ? "PASS" : "FAIL");

    cudaFree(d_a_fp4); cudaFree(d_a_sf);
    cudaFree(d_b_fp4); cudaFree(d_b_sf);
    cudaFree(d_out);
    return 0;
}
