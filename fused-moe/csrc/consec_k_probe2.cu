/**
 * Probe: consecutive vs strided K packing with NON-UNIFORM random data.
 * Compares MMA output against host dequant reference to check correctness.
 *
 * Build: nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a -o consec_k_probe2 consec_k_probe2.cu
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static constexpr int BK = 64;
static constexpr int SF_BLOCK = 16;
static constexpr int SF_PER_K = 4;
static const float E2M1[8] = {0.0f,0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f};

float h_e4m3_decode(uint8_t x) {
    int e = (x>>3)&0xF, m = x&7;
    if (e==15&&m==7) return 0;
    if (e==0) return ldexpf((float)m/8.0f,-6);
    return ldexpf(1.0f+(float)m/8.0f, e-7);
}
uint8_t h_e4m3_ceil(float v) {
    if (v<=0) return 0x01; if (v>448) return 0x7E;
    uint8_t b=0x7E; float bv=448;
    for(int e=0;e<=15;e++) for(int m=0;m<=7;m++) {
        if(e==15&&m==7) continue;
        float r=(e==0)?ldexpf((float)m/8.0f,-6):ldexpf(1.0f+(float)m/8.0f,e-7);
        if(r>=v&&r<bv){bv=r;b=(e<<3)|m;}
    }
    return b;
}
void quantize(const float* d, int n, uint8_t* pk, uint8_t* sf) {
    memset(pk,0,n/2);
    for(int b=0;b<n/SF_BLOCK;b++) {
        int s=b*SF_BLOCK; float mx=0;
        for(int i=s;i<s+SF_BLOCK;i++) mx=fmaxf(mx,fabsf(d[i]));
        sf[b]=h_e4m3_ceil(fmaxf(mx/6.0f,1e-30f));
        float sc=h_e4m3_decode(sf[b]); if(sc<1e-30f)sc=1e-30f;
        for(int i=s;i<s+SF_BLOCK;i++) {
            float sv=d[i]/sc, av=fabsf(sv); int sign=(sv<0)?1:0, idx=0; float bd=av;
            for(int j=1;j<8;j++){float dd=fabsf(av-E2M1[j]);if(dd<bd){bd=dd;idx=j;}}
            uint8_t fp4=(sign<<3)|idx; int bi=i/2;
            if(i%2==0)pk[bi]=fp4;else pk[bi]|=(fp4<<4);
        }
    }
}
float dequant(const uint8_t* pk, const uint8_t* sf, int i) {
    uint8_t bv=pk[i/2], nib=(i&1)?(bv>>4):(bv&0xF);
    int sign=(nib>>3)&1,mag=nib&7;
    return (sign?-1.0f:1.0f)*E2M1[mag]*h_e4m3_decode(sf[i/SF_BLOCK]);
}

__device__ __forceinline__ uint32_t swizzle_343(uint32_t off) {
    return off ^ ((off >> 3) & 0x70u);
}
__device__ __forceinline__ uint32_t get_nibble_swz(const uint8_t* smem, int rbo, int k) {
    int addr = rbo + k / 2;
    uint8_t bv = smem[swizzle_343(addr)];
    return (k & 1) ? ((bv >> 4) & 0xFu) : (bv & 0xFu);
}
__device__ __forceinline__ void mma_4x(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
    const float (&c)[4], uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},"
        "{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
          "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

// Test STRIDED packing
__global__ void test_strided(
    const uint8_t* a_fp4, const uint8_t* a_sf,
    const uint8_t* b_fp4, const uint8_t* b_sf,
    float* output) {
    int tid=threadIdx.x, warp_id=tid/32, lane_id=tid%32;
    __shared__ uint8_t s_A[16*32], s_B[64*32];
    __shared__ uint8_t s_SFA[16], s_SFB[64*4];
    for(int i=tid;i<16*32;i+=256) {
        int r=i/32,c=i%32;
        s_A[swizzle_343(i)]=(r==0)?a_fp4[c]:0;
    }
    for(int i=tid;i<64*32;i+=256) {
        int r=i/32,c=i%32;
        s_B[swizzle_343(i)]=b_fp4[r*32+c];
    }
    if(tid<4) s_SFA[tid]=a_sf[tid];
    for(int i=tid;i<64*4;i+=256) {
        int r=i/4,c=i%4;
        s_SFB[i]=b_sf[r*4+c];
    }
    __syncthreads();
    // STRIDED A
    uint32_t a[4]={0,0,0,0};
    if(lane_id/4==0) { int t0=lane_id%4;
        for(int p=0;p<8;p++) {
            a[0]|=get_nibble_swz(s_A,0,t0+p*8)<<(p*4);
            a[2]|=get_nibble_swz(s_A,0,t0+4+p*8)<<(p*4);
        }
    }
    // STRIDED B
    uint32_t b[2]={0,0};
    { int g=lane_id/4,t0=lane_id%4,Nl=4*(g&1)+(g>>1);
      int rbo=(warp_id*8+Nl)*32;
      for(int p=0;p<8;p++) {
          b[0]|=get_nibble_swz(s_B,rbo,t0+p*8)<<(p*4);
          b[1]|=get_nibble_swz(s_B,rbo,t0+4+p*8)<<(p*4);
      }
    }
    // SFA/SFB raw
    uint32_t sfa=(uint32_t)s_SFA[0]|((uint32_t)s_SFA[1]<<8)|((uint32_t)s_SFA[2]<<16)|((uint32_t)s_SFA[3]<<24);
    { int g=lane_id/4,Nl=4*(g&1)+(g>>1),sn=warp_id*8+Nl,sb=sn*4;
      uint32_t sfb=(uint32_t)s_SFB[sb]|((uint32_t)s_SFB[sb+1]<<8)|((uint32_t)s_SFB[sb+2]<<16)|((uint32_t)s_SFB[sb+3]<<24);
      float acc[4]={0,0,0,0};
      mma_4x(acc,a,b,acc,sfa,sfb);
      if(lane_id<4) {
          output[warp_id*8+lane_id]=acc[0];
          output[warp_id*8+lane_id+4]=acc[1];
      }
    }
}

// Test CONSECUTIVE packing
__global__ void test_consec(
    const uint8_t* a_fp4, const uint8_t* a_sf,
    const uint8_t* b_fp4, const uint8_t* b_sf,
    float* output) {
    int tid=threadIdx.x, warp_id=tid/32, lane_id=tid%32;
    __shared__ uint8_t s_A[16*32], s_B[64*32];
    __shared__ uint8_t s_SFA[16], s_SFB[64*4];
    for(int i=tid;i<16*32;i+=256) {
        int r=i/32,c=i%32;
        s_A[swizzle_343(i)]=(r==0)?a_fp4[c]:0;
    }
    for(int i=tid;i<64*32;i+=256) {
        int r=i/32,c=i%32;
        s_B[swizzle_343(i)]=b_fp4[r*32+c];
    }
    if(tid<4) s_SFA[tid]=a_sf[tid];
    for(int i=tid;i<64*4;i+=256) {
        int r=i/4,c=i%4;
        s_SFB[i]=b_sf[r*4+c];
    }
    __syncthreads();
    // CONSECUTIVE A
    uint32_t a[4]={0,0,0,0};
    if(lane_id/4==0) { int t0=lane_id%4;
        for(int p=0;p<8;p++) {
            a[0]|=get_nibble_swz(s_A,0,t0*8+p)<<(p*4);
            a[2]|=get_nibble_swz(s_A,0,32+t0*8+p)<<(p*4);
        }
    }
    // CONSECUTIVE B
    uint32_t b[2]={0,0};
    { int g=lane_id/4,t0=lane_id%4,Nl=4*(g&1)+(g>>1);
      int rbo=(warp_id*8+Nl)*32;
      for(int p=0;p<8;p++) {
          b[0]|=get_nibble_swz(s_B,rbo,t0*8+p)<<(p*4);
          b[1]|=get_nibble_swz(s_B,rbo,32+t0*8+p)<<(p*4);
      }
    }
    // SFA/SFB raw
    uint32_t sfa=(uint32_t)s_SFA[0]|((uint32_t)s_SFA[1]<<8)|((uint32_t)s_SFA[2]<<16)|((uint32_t)s_SFA[3]<<24);
    { int g=lane_id/4,Nl=4*(g&1)+(g>>1),sn=warp_id*8+Nl,sb=sn*4;
      uint32_t sfb=(uint32_t)s_SFB[sb]|((uint32_t)s_SFB[sb+1]<<8)|((uint32_t)s_SFB[sb+2]<<16)|((uint32_t)s_SFB[sb+3]<<24);
      float acc[4]={0,0,0,0};
      mma_4x(acc,a,b,acc,sfa,sfb);
      if(lane_id<4) {
          output[warp_id*8+lane_id]=acc[0];
          output[warp_id*8+lane_id+4]=acc[1];
      }
    }
}

int main() {
    srand(42);
    auto rf=[]{return((float)rand()/RAND_MAX-0.5f)*2.0f;};
    // Generate random data for 1 tile
    float h_a[64], h_b[64*64];
    for(int i=0;i<64;i++) h_a[i]=rf();
    for(int i=0;i<64*64;i++) h_b[i]=rf()*0.1f;
    // Quantize
    uint8_t a_fp4[32],a_sf[4];
    quantize(h_a,64,a_fp4,a_sf);
    uint8_t b_fp4[64*32],b_sf[64*4];
    for(int n=0;n<64;n++) quantize(&h_b[n*64],64,&b_fp4[n*32],&b_sf[n*4]);
    // Host reference
    float h_ref[64];
    for(int n=0;n<64;n++) {
        float s=0;
        for(int k=0;k<64;k++) s+=dequant(a_fp4,a_sf,k)*dequant(&b_fp4[n*32],&b_sf[n*4],k);
        h_ref[n]=s;
    }
    // GPU
    uint8_t *d_af,*d_as,*d_bf,*d_bs; float *d_os,*d_oc;
    cudaMalloc(&d_af,32); cudaMalloc(&d_as,4);
    cudaMalloc(&d_bf,64*32); cudaMalloc(&d_bs,64*4);
    cudaMalloc(&d_os,64*sizeof(float)); cudaMalloc(&d_oc,64*sizeof(float));
    cudaMemcpy(d_af,a_fp4,32,cudaMemcpyHostToDevice);
    cudaMemcpy(d_as,a_sf,4,cudaMemcpyHostToDevice);
    cudaMemcpy(d_bf,b_fp4,64*32,cudaMemcpyHostToDevice);
    cudaMemcpy(d_bs,b_sf,64*4,cudaMemcpyHostToDevice);
    cudaMemset(d_os,0,64*sizeof(float)); cudaMemset(d_oc,0,64*sizeof(float));

    test_strided<<<1,256>>>(d_af,d_as,d_bf,d_bs,d_os);
    test_consec<<<1,256>>>(d_af,d_as,d_bf,d_bs,d_oc);
    cudaDeviceSynchronize();

    float h_s[64],h_c[64];
    cudaMemcpy(h_s,d_os,64*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c,d_oc,64*sizeof(float),cudaMemcpyDeviceToHost);

    printf("A scales: [0x%02X=%.4f, 0x%02X=%.4f, 0x%02X=%.4f, 0x%02X=%.4f]\n",
           a_sf[0],h_e4m3_decode(a_sf[0]),a_sf[1],h_e4m3_decode(a_sf[1]),
           a_sf[2],h_e4m3_decode(a_sf[2]),a_sf[3],h_e4m3_decode(a_sf[3]));

    printf("\nFirst 8 N columns:\n");
    printf("  N   Strided    Consec     Ref        S_ratio    C_ratio    B_sf\n");
    for(int n=0;n<8;n++) {
        float sr=(h_ref[n]!=0)?h_s[n]/h_ref[n]:0;
        float cr=(h_ref[n]!=0)?h_c[n]/h_ref[n]:0;
        printf("  %2d  %10.4f %10.4f %10.4f %10.4f %10.4f   [%02X %02X %02X %02X]\n",
               n,h_s[n],h_c[n],h_ref[n],sr,cr,b_sf[n*4],b_sf[n*4+1],b_sf[n*4+2],b_sf[n*4+3]);
    }
    // Errors
    double se_s=0,se_c=0,se_r=0;
    for(int n=0;n<64;n++) {
        se_s+=(h_s[n]-h_ref[n])*(h_s[n]-h_ref[n]);
        se_c+=(h_c[n]-h_ref[n])*(h_c[n]-h_ref[n]);
        se_r+=h_ref[n]*h_ref[n];
    }
    printf("\nRelErr strided:    %.4f%%\n",sqrt(se_s/se_r)*100);
    printf("RelErr consecutive: %.4f%%\n",sqrt(se_c/se_r)*100);
    printf("Strided PASS: %s\n",(sqrt(se_s/se_r)<0.05)?"YES":"NO");
    printf("Consec  PASS: %s\n",(sqrt(se_c/se_r)<0.05)?"YES":"NO");

    cudaFree(d_af); cudaFree(d_as); cudaFree(d_bf); cudaFree(d_bs);
    cudaFree(d_os); cudaFree(d_oc);
    return 0;
}
