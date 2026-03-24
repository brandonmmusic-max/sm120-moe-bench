/**
 * Probe B register-to-(N,K) mapping.
 * Set A=all 1.0, modify ONE B nibble to 2.0, check which outputs change.
 * Affected d values reveal the N assignment for that B nibble.
 *
 * Build: nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a -o mma_b_probe mma_b_probe.cu
 */
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

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

__global__ void probe_b(int target_thread, int target_reg, int target_nibble, float* results) {
    int lid = threadIdx.x;
    uint32_t a[4]={0x22222222,0x22222222,0x22222222,0x22222222}; // A=all 1.0
    uint32_t b[2]={0x22222222,0x22222222}; // B=all 1.0

    if (lid == target_thread) {
        uint32_t mask = 0xFu << (target_nibble * 4);
        b[target_reg] = (b[target_reg] & ~mask) | (4u << (target_nibble * 4)); // E2M1=4=2.0
    }

    uint32_t sfa=0x7F7F,sfb=0x7F7F;
    float acc[4]={0,0,0,0};
    mma_nvf4(acc,a,b,acc,sfa,sfb);
    results[lid*4+0]=acc[0];
    results[lid*4+1]=acc[1];
    results[lid*4+2]=acc[2];
    results[lid*4+3]=acc[3];
}

int main() {
    float*dr; cudaMalloc(&dr,128*sizeof(float)); float h[128];

    printf("=== B Register Layout Probe ===\n");
    printf("A=B=all 1.0, one B nibble set to 2.0\n");
    printf("Normal=64.0, affected=65.0\n\n");

    // CLayout: d[0]=C[2*(t/4), t%4], d[1]=C[2*(t/4), t%4+4]
    //          d[2]=C[2*(t/4)+1, t%4], d[3]=C[2*(t/4)+1, t%4+4]

    // Probe thread 0 b[0] and b[1]
    for (int target_t = 0; target_t < 8; target_t++) {
        for (int reg = 0; reg < 2; reg++) {
            cudaMemset(dr,0,128*sizeof(float));
            probe_b<<<1,32>>>(target_t, reg, 0, dr);
            cudaDeviceSynchronize();
            cudaMemcpy(h,dr,128*sizeof(float),cudaMemcpyDeviceToHost);

            printf("T%d b[%d] nib0: ", target_t, reg);
            // Find affected outputs
            for (int t=0;t<32;t++)
                for (int v=0;v<4;v++)
                    if (h[t*4+v] > 64.5f) {
                        int m=2*(t/4)+(v>=2?1:0);
                        int n=(t%4)+((v%2==1)?4:0);
                        printf("C[%d,%d] ",m,n);
                    }
            printf("\n");
        }
    }

    printf("\n--- Full B nibble scan for thread 0 ---\n");
    for (int reg=0;reg<2;reg++){
        for (int nib=0;nib<8;nib++){
            cudaMemset(dr,0,128*sizeof(float));
            probe_b<<<1,32>>>(0,reg,nib,dr);
            cudaDeviceSynchronize();
            cudaMemcpy(h,dr,sizeof(h),cudaMemcpyDeviceToHost);

            printf("  b[%d] nib[%d]: N=",reg,nib);
            int found_n=-1;
            for(int t=0;t<32;t++)
                for(int v=0;v<4;v++)
                    if(h[t*4+v]>64.5f){
                        int n=(t%4)+((v%2==1)?4:0);
                        if(found_n==-1){found_n=n;printf("%d ",n);}
                        else if(n!=found_n) printf("%d ",n);
                    }
            printf("(all M rows)\n");
        }
    }

    cudaFree(dr);
    printf("\n=== Done ===\n");
    return 0;
}
