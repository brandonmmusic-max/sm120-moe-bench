/**
 * Minimal test: Does Swizzle<1,4,3> fix the A-tile loading?
 * Just GEMM1 gate (no SwiGLU), compare raw dot products.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define HIDDEN 4096
#define BK 64
#define BN 64
#define BM 16
#define MMA_N 8
#define SF_BLOCK 32
#define WARP_SIZE 32
#define NUM_WARPS 8
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE)
#define SMEM_A (BM * BK / 2)
#define SMEM_B (BN * BK / 2)
#define SMEM_SFB (BN * (BK / SF_BLOCK))
#define SMEM_TOTAL (2 * SMEM_A + SMEM_B + 64 + SMEM_SFB + 256)

__device__ __forceinline__ void ldm4(uint32_t (&d)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0,%1,%2,%3},[%4];\n"
        :"=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3]):"r"(a));
}
__device__ __forceinline__ void ldm2(uint32_t (&d)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 {%0,%1},[%2];\n"
        :"=r"(d[0]),"=r"(d[1]):"r"(a));
}
__device__ __forceinline__ uint32_t smem_ptr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mma_fp4(float (&d)[4], const uint32_t (&a)[4],
    const uint32_t (&b)[2], const float (&c)[4], uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        :"=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),
         "r"(b[0]),"r"(b[1]),"f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]),
         "r"(sfa),"h"((uint16_t)0),"h"((uint16_t)0),
         "r"(sfb),"h"((uint16_t)0),"h"((uint16_t)0));
}

// Swizzle<1,4,3>: byte_col ^= ((row & 1) << 3)
__device__ __host__ __forceinline__ int sw(int row, int bcol) {
    return row * (BK/2) + (bcol ^ ((row & 1) << 3));
}

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
test_gate(const uint8_t* in_fp4, const uint8_t* in_sf,
          const uint8_t* w_fp4, const uint8_t* w_sf,
          float* gate_out) {
    const int tid = threadIdx.x;
    const int wid = tid / 32, lid = tid % 32;

    extern __shared__ char smem[];
    uint8_t* sA1 = (uint8_t*)smem;
    uint8_t* sA2 = sA1 + SMEM_A;
    uint8_t* sB  = sA2 + SMEM_A;
    uint8_t* sSFA = sB + SMEM_B;
    uint8_t* sSFB = sSFA + 64;

    // Only first N-pass (gate cols 0-63)
    float aq1[4]={0,0,0,0}, aq2[4]={0,0,0,0};

    for (int ki = 0; ki < HIDDEN/BK; ki++) {
        int koff = ki * BK;

        // Zero A tiles
        for (int i = tid; i < SMEM_A; i += BLOCK_SIZE) { sA1[i] = 0; sA2[i] = 0; }
        __syncthreads();

        // SWIZZLED A write: row 0 for q1, row 8 for q2
        for (int bc = tid; bc < BK/2; bc += BLOCK_SIZE) {
            uint8_t v = in_fp4[koff/2 + bc];
            sA1[sw(0, bc)] = v;
            sA2[sw(8, bc)] = v;
        }
        // SWIZZLED B write
        for (int i = tid; i < SMEM_B; i += BLOCK_SIZE) {
            int row = i / (BK/2), bc = i % (BK/2);
            sB[sw(row, bc)] = w_fp4[row * (HIDDEN/2) + koff/2 + bc];
        }
        // SF
        if (tid < BK/SF_BLOCK) sSFA[tid] = in_sf[koff/SF_BLOCK + tid];
        for (int i = tid; i < BN*(BK/SF_BLOCK); i += BLOCK_SIZE) {
            int sn = i/(BK/SF_BLOCK), sk = i%(BK/SF_BLOCK);
            sSFB[i] = w_sf[sn*(HIDDEN/SF_BLOCK) + koff/SF_BLOCK + sk];
        }
        __syncthreads();

        if (wid < BN/MMA_N) {
            uint32_t b[2];
            ldm2(b, smem_ptr(&sB[wid*MMA_N*(BK/2) + (lid%16)*16]));
            uint16_t sfa = (uint16_t)sSFA[0] | ((uint16_t)sSFA[1] << 8);
            int sfn1 = wid*MMA_N + lid%4;
            int sfn2 = wid*MMA_N + 4 + lid%4;
            uint16_t sfb1 = (uint16_t)sSFB[sfn1*2] | ((uint16_t)sSFB[sfn1*2+1] << 8);
            uint16_t sfb2 = (uint16_t)sSFB[sfn2*2] | ((uint16_t)sSFB[sfn2*2+1] << 8);

            uint32_t a1[4]; ldm4(a1, smem_ptr(&sA1[lid*16]));
            mma_fp4(aq1, a1, b, aq1, (uint32_t)sfa, (uint32_t)sfb1);
            uint32_t a2[4]; ldm4(a2, smem_ptr(&sA2[lid*16]));
            mma_fp4(aq2, a2, b, aq2, (uint32_t)sfa, (uint32_t)sfb2);
        }
        __syncthreads();
    }

    // Store gate output for first 64 cols
    if (wid < BN/MMA_N && lid/8 == 0) {
        int c = lid % 4;
        int cq1 = wid*8 + c;
        int cq2 = wid*8 + 4 + c;
        if (cq1 < 256) gate_out[cq1] = aq1[0];
        if (cq2 < 256) gate_out[cq2] = aq2[0];
    }
}

static const float E2M1[8] = {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
void quant(const float* d, int n, uint8_t* p, uint8_t* s) {
    for (int b = 0; b < n/SF_BLOCK; b++) {
        float mx = 0;
        for (int i = b*SF_BLOCK; i < (b+1)*SF_BLOCK && i < n; i++) mx = fmaxf(mx, fabsf(d[i]));
        float sc = mx / 6.0f; if (sc < 1e-30f) sc = 1e-30f;
        int e = 128 + (int)ceilf(log2f(sc)); e = e<1?1:e>254?254:e;
        s[b] = e; float as = powf(2.0f, (float)(e-128));
        for (int i = b*SF_BLOCK; i < (b+1)*SF_BLOCK && i < n; i++) {
            float v = d[i]/as; int sg = v<0?1:0; float av = fabsf(v);
            int bi=0; float bd=av;
            for (int j=1;j<8;j++){float dd=fabsf(av-E2M1[j]);if(dd<bd){bd=dd;bi=j;}}
            uint8_t f4=(sg<<3)|bi;
            if(i%2==0)p[i/2]=f4;else p[i/2]|=(f4<<4);
        }
    }
}

int main() {
    srand(42);
    float* hi = new float[HIDDEN];
    float* hw = new float[BN*HIDDEN]; // first 64 weight rows only
    for (int i = 0; i < HIDDEN; i++) hi[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < BN*HIDDEN; i++) hw[i] = ((float)rand()/RAND_MAX - 0.5f) * 2.0f;

    // FP32 gate reference
    float gref[64];
    for (int c = 0; c < 64; c++) {
        float s = 0;
        for (int k = 0; k < HIDDEN; k++) s += hi[k] * hw[c*HIDDEN+k];
        gref[c] = s;
    }

    // Quantize
    uint8_t* pi = new uint8_t[HIDDEN/2]();
    uint8_t* si = new uint8_t[HIDDEN/SF_BLOCK]();
    uint8_t* pw = new uint8_t[BN*HIDDEN/2]();
    uint8_t* sw_h = new uint8_t[BN*HIDDEN/SF_BLOCK]();
    quant(hi, HIDDEN, pi, si);
    for (int n = 0; n < BN; n++) quant(&hw[n*HIDDEN], HIDDEN, &pw[n*HIDDEN/2], &sw_h[n*(HIDDEN/SF_BLOCK)]);

    uint8_t *di, *dsi, *dw, *dsw; float* dgo;
    cudaMalloc(&di, HIDDEN/2); cudaMalloc(&dsi, HIDDEN/SF_BLOCK);
    cudaMalloc(&dw, BN*HIDDEN/2); cudaMalloc(&dsw, BN*HIDDEN/SF_BLOCK);
    cudaMalloc(&dgo, 256*4); cudaMemset(dgo, 0, 256*4);
    cudaMemcpy(di, pi, HIDDEN/2, cudaMemcpyHostToDevice);
    cudaMemcpy(dsi, si, HIDDEN/SF_BLOCK, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, pw, BN*HIDDEN/2, cudaMemcpyHostToDevice);
    cudaMemcpy(dsw, sw_h, BN*HIDDEN/SF_BLOCK, cudaMemcpyHostToDevice);

    cudaFuncSetAttribute(test_gate, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
    test_gate<<<1, BLOCK_SIZE, SMEM_TOTAL>>>(di, dsi, dw, dsw, dgo);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("ERR: %s\n", cudaGetErrorString(err)); return 1; }

    float hgo[256];
    cudaMemcpy(hgo, dgo, 256*4, cudaMemcpyDeviceToHost);

    printf("Gate col comparison (Swizzle<1,4,3> A+B loading):\n");
    float sum_err = 0, sum_rms = 0;
    for (int i = 0; i < 8; i++)
        printf("  [%d] ref=%8.2f  kern=%8.2f  ratio=%.3f\n",
            i, gref[i], hgo[i], gref[i]!=0 ? hgo[i]/gref[i] : 0);
    for (int i = 0; i < 64; i++) {
        sum_err += fabsf(gref[i] - hgo[i]);
        sum_rms += gref[i] * gref[i];
    }
    sum_rms = sqrtf(sum_rms / 64);
    printf("\nGate (64 cols): avg_err=%.2f rms=%.2f rel=%.1f%%\n",
        sum_err/64, sum_rms, sum_rms > 0 ? 100*(sum_err/64)/sum_rms : 0);

    delete[] hi; delete[] hw; delete[] pi; delete[] si; delete[] pw; delete[] sw_h;
    return 0;
}
