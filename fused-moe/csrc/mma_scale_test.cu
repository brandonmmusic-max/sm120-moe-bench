/**
 * MMA scale test: validates rescaled packing with varying scales.
 * Tests the fix for the scale_vec::2X per-register-group scale application.
 *
 * Build: nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a -o mma_scale_test mma_scale_test.cu
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

__device__ __constant__ float E2M1[8] = {0, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
static const float h_E2M1[8] = {0, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

__device__ __forceinline__ uint32_t swizzle_343(uint32_t o) {
    return o ^ ((o >> 3) & 0x70u);
}
__device__ __forceinline__ uint32_t get_nib_swz(const uint8_t* s, int rbo, int k) {
    uint8_t v = s[swizzle_343(rbo + k/2)];
    return (k&1) ? ((v>>4)&0xFu) : (v&0xFu);
}

// Rescale a nibble from sf_old to sf_new (sf_new >= sf_old)
__device__ __forceinline__ uint32_t rescale_nib(uint32_t nib, int sf_old, int sf_new) {
    if (sf_old == sf_new) return nib;
    int sign = (nib >> 3) & 1;
    float val = E2M1[nib & 7] * exp2f((float)(sf_old - sf_new));
    int idx;
    if      (val < 0.25f)  idx = 0;
    else if (val < 0.75f)  idx = 1;
    else if (val < 1.25f)  idx = 2;
    else if (val < 1.75f)  idx = 3;
    else if (val < 2.5f)   idx = 4;
    else if (val < 3.5f)   idx = 5;
    else if (val < 5.0f)   idx = 6;
    else                    idx = 7;
    return (sign << 3) | idx;
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

__global__ void test_rescaled(
    const uint8_t* g_a_fp4, const uint8_t* g_a_sf,
    const uint8_t* g_b_fp4, const uint8_t* g_b_sf,
    float* results)
{
    int lid = threadIdx.x;

    __shared__ __align__(128) uint8_t sA[512];
    for (int i = lid; i < 512; i += 32) {
        int row = i / 32, col = i % 32;
        sA[swizzle_343(i)] = (row == 0) ? g_a_fp4[col] : 0;
    }
    __shared__ __align__(128) uint8_t sB[256];
    for (int i = lid; i < 256; i += 32)
        sB[swizzle_343(i)] = g_b_fp4[i];
    __syncthreads();

    // Pack A with rescaling
    uint32_t a[4] = {0,0,0,0};
    uint8_t sfa0 = g_a_sf[0], sfa1 = g_a_sf[1];
    uint8_t sfam = (sfa0 > sfa1) ? sfa0 : sfa1;
    if (lid / 4 == 0) {
        int t0 = lid % 4;
        for (int p = 0; p < 8; p++) {
            int k0 = t0 + p * 8, k2 = t0 + 4 + p * 8;
            uint32_t n0 = get_nib_swz(sA, 0, k0);
            uint32_t n2 = get_nib_swz(sA, 0, k2);
            n0 = rescale_nib(n0, (k0 < 32) ? (int)sfa0 : (int)sfa1, (int)sfam);
            n2 = rescale_nib(n2, (k2 < 32) ? (int)sfa0 : (int)sfa1, (int)sfam);
            a[0] |= n0 << (p * 4);
            a[2] |= n2 << (p * 4);
        }
    }

    // Pack B with rescaling
    uint32_t b[2] = {0, 0};
    int g = lid / 4, t0 = lid % 4;
    int N = 4 * (g & 1) + (g >> 1);
    uint8_t sfb0 = g_b_sf[N * 2], sfb1 = g_b_sf[N * 2 + 1];
    uint8_t sfbm = (sfb0 > sfb1) ? sfb0 : sfb1;
    int rbo = N * 32;
    for (int p = 0; p < 8; p++) {
        int k0 = t0 + p * 8, k2 = t0 + 4 + p * 8;
        uint32_t n0 = get_nib_swz(sB, rbo, k0);
        uint32_t n2 = get_nib_swz(sB, rbo, k2);
        n0 = rescale_nib(n0, (k0 < 32) ? (int)sfb0 : (int)sfb1, (int)sfbm);
        n2 = rescale_nib(n2, (k2 < 32) ? (int)sfb0 : (int)sfb1, (int)sfbm);
        b[0] |= n0 << (p * 4);
        b[1] |= n2 << (p * 4);
    }

    // Unified scales
    uint32_t sfa_pk = (uint32_t)sfam | ((uint32_t)sfam << 8);
    uint32_t sfb_pk = (uint32_t)sfbm | ((uint32_t)sfbm << 8);

    float acc[4] = {0,0,0,0};
    mma_nvf4(acc, a, b, acc, sfa_pk, sfb_pk);

    if (lid < 4) {
        results[lid] = acc[0];
        results[lid + 4] = acc[1];
    }
}

void quant(const float* d, int n, uint8_t* p, uint8_t* s) {
    memset(p, 0, n / 2);
    for (int b = 0; b < n / 32; b++) {
        float mx = 0;
        for (int i = b*32; i < (b+1)*32; i++) mx = fmaxf(mx, fabsf(d[i]));
        float sc = fmaxf(mx/6.0f, 1e-30f);
        int e = 127 + (int)ceilf(log2f(sc));
        e = e<1?1:e>254?254:e; s[b]=e;
        float as = powf(2.0f, (float)(e-127));
        for (int i = b*32; i < (b+1)*32; i++) {
            float v = d[i]/as; int sg=v<0?1:0; float av=fabsf(v);
            int idx=0; float bd=av;
            for (int j=1;j<8;j++){float dd=fabsf(av-h_E2M1[j]);if(dd<bd){bd=dd;idx=j;}}
            uint8_t f4=(sg<<3)|idx;
            if(i%2==0)p[i/2]=f4;else p[i/2]|=(f4<<4);
        }
    }
}

int main() {
    printf("=== MMA Rescaled Packing Test ===\n\n");
    srand(42);

    // Deliberately varying magnitudes
    float ha[64], hb[8*64];
    for (int k = 0; k < 32; k++) ha[k] = ((float)rand()/RAND_MAX-0.5f) * 0.1f;
    for (int k = 32; k < 64; k++) ha[k] = ((float)rand()/RAND_MAX-0.5f) * 6.0f;
    for (int n = 0; n < 8; n++) {
        float scale = 0.01f * powf(4.0f, (float)n);
        for (int k = 0; k < 64; k++)
            hb[n*64+k] = ((float)rand()/RAND_MAX-0.5f) * scale;
    }

    uint8_t pa[32],sa[2],pb[256],sb[16];
    quant(ha,64,pa,sa);
    for (int n=0;n<8;n++) quant(&hb[n*64],64,&pb[n*32],&sb[n*2]);

    printf("SFA: [%02x,%02x] = [%.6f, %.6f]\n", sa[0],sa[1],
           powf(2.0f,sa[0]-127.f), powf(2.0f,sa[1]-127.f));

    // Compute reference (dequantized FP4 dot product with block scales)
    float ref[8];
    for (int n = 0; n < 8; n++) {
        ref[n] = 0;
        for (int k = 0; k < 64; k++) {
            uint8_t na = (k&1)?((pa[k/2]>>4)&0xF):(pa[k/2]&0xF);
            uint8_t nb = (k&1)?((pb[n*32+k/2]>>4)&0xF):(pb[n*32+k/2]&0xF);
            float va = ((na>>3)?-1.0f:1.0f)*h_E2M1[na&7];
            float vb = ((nb>>3)?-1.0f:1.0f)*h_E2M1[nb&7];
            float sas = powf(2.0f,(float)(sa[k/32]-127));
            float sbs = powf(2.0f,(float)(sb[n*2+k/32]-127));
            ref[n] += va*sas*vb*sbs;
        }
    }

    printf("Ref: "); for(int n=0;n<8;n++) printf("%.4f ",ref[n]); printf("\n");

    uint8_t *da,*db,*dsa,*dsb; float *dr;
    cudaMalloc(&da,32);cudaMalloc(&dsa,2);
    cudaMalloc(&db,256);cudaMalloc(&dsb,16);
    cudaMalloc(&dr,8*sizeof(float));
    cudaMemcpy(da,pa,32,cudaMemcpyHostToDevice);
    cudaMemcpy(dsa,sa,2,cudaMemcpyHostToDevice);
    cudaMemcpy(db,pb,256,cudaMemcpyHostToDevice);
    cudaMemcpy(dsb,sb,16,cudaMemcpyHostToDevice);
    cudaMemset(dr,0,8*sizeof(float));

    test_rescaled<<<1,32>>>(da,dsa,db,dsb,dr);
    cudaDeviceSynchronize();

    float hr[8];
    cudaMemcpy(hr,dr,8*sizeof(float),cudaMemcpyDeviceToHost);
    printf("MMA: "); for(int n=0;n<8;n++) printf("%.4f ",hr[n]); printf("\n");

    float max_err = 0, max_rel = 0;
    for (int n = 0; n < 8; n++) {
        float err = fabsf(hr[n]-ref[n]);
        float rel = (fabsf(ref[n])>1e-6f) ? err/fabsf(ref[n]) : err;
        max_err = fmaxf(max_err, err);
        max_rel = fmaxf(max_rel, rel);
    }
    printf("max_abs_err=%.6f max_rel_err=%.4f%% %s\n",
           max_err, max_rel*100, max_rel < 0.5f ? "PASS" : "FAIL");

    cudaFree(da);cudaFree(dsa);cudaFree(db);cudaFree(dsb);cudaFree(dr);
    return 0;
}
