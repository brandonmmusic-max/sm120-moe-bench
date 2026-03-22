#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

static constexpr int HIDDEN = 4096;
static constexpr int GATE_UP = 512;
static constexpr int INTERMEDIATE = 256;
static constexpr int BK = 64;
static constexpr int BN = 64;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 64;
static constexpr int MMA_M = 16;
static constexpr int WARP_SIZE = 32;
static constexpr int NUM_WARPS = 8;
static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;
static constexpr int SF_BLOCK = 32;
static constexpr int N_MMA_PER_BN = BN / MMA_N;

static constexpr int SMEM_A = MMA_M * BK / 2;
static constexpr int SMEM_B = BN * BK / 2;
static constexpr int SMEM_SFA = 64;
static constexpr int SMEM_SFB = BN * (BK / SF_BLOCK);
static constexpr int SMEM_TOTAL = 2 * SMEM_A + SMEM_B + SMEM_SFA + SMEM_SFB + 256;

__device__ __forceinline__ void ldmatrix_b4x16_x4(uint32_t (&d)[4], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0,%1,%2,%3},[%4];\n"
        :"=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3]):"r"(a));
}
__device__ __forceinline__ void ldmatrix_b4x16_x2(uint32_t (&d)[2], uint32_t a) {
    asm volatile("ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 {%0,%1},[%2];\n"
        :"=r"(d[0]),"=r"(d[1]):"r"(a));
}
__device__ __forceinline__ void mma_fp4(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
    const float (&c)[4], uint32_t sfa, uint32_t sfb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        :"=f"(d[0]),"=f"(d[1]),"=f"(d[2]),"=f"(d[3])
        :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),
         "r"(b[0]),"r"(b[1]),
         "f"(c[0]),"f"(c[1]),"f"(c[2]),"f"(c[3]),
         "r"(sfa),"h"((uint16_t)0),"h"((uint16_t)0),
         "r"(sfb),"h"((uint16_t)0),"h"((uint16_t)0));
}
__device__ __forceinline__ uint32_t to_smem(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// E2M1 table: idx 0-7 = {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
// Weight col n gets value E2M1[n%7+1] = {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.5, ...}
// Input = all 1.0
// Expected gate output[col] = K * 1.0 * E2M1[col%7+1] * SFA * SFB
// With SFA=SFB=1.0(0x80), K=4096: output[col] = 4096 * E2M1[col%7+1]

__global__ void col_perm_kernel(float* __restrict__ gate_out, int M) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    extern __shared__ char smem[];
    uint8_t* s_A1 = (uint8_t*)smem;
    uint8_t* s_A2 = s_A1 + SMEM_A;
    uint8_t* s_B  = s_A2 + SMEM_A;
    uint8_t* s_SFA = s_B + SMEM_B;
    uint8_t* s_SFB = s_SFA + SMEM_SFA;

    // Only do first N-pass (gate cols 0-63, warp 0 = cols 0-7)
    const int n_off = 0;

    float acc_q1[4] = {0,0,0,0};
    float acc_q2[4] = {0,0,0,0};

    for (int ki = 0; ki < HIDDEN / BK; ki++) {
        const int k_off = ki * BK;

        // A: input = all 1.0, row 0 and row 8
        for (int i = tid; i < SMEM_A; i += BLOCK_SIZE) {
            int row = i / (BK / 2);
            s_A1[i] = (row == 0) ? 0x22 : 0;  // row 0 = 1.0
            s_A2[i] = (row == 8) ? 0x22 : 0;  // row 8 = 1.0
        }

        // B: col n gets E2M1 value (n%7+1)
        // B layout in SMEM: [BN=64 rows, BK/2=32 bytes/row]
        // Row n, all K positions get the same FP4 value
        for (int i = tid; i < SMEM_B; i += BLOCK_SIZE) {
            int row = i / (BK / 2);
            int n_global = n_off + row;
            uint8_t val = (uint8_t)((n_global % 7) + 1);  // E2M1 index
            s_B[i] = val | (val << 4);  // pack same value twice
        }

        // SF = 1.0 = 0x80
        for (int i = tid; i < SMEM_SFA + SMEM_SFB; i += BLOCK_SIZE) {
            if (i < SMEM_SFA) s_SFA[i] = 0x80;
            else s_SFB[i - SMEM_SFA] = 0x80;
        }
        __syncthreads();

        if (warp_id < N_MMA_PER_BN) {
            uint32_t b[2];
            int b_base = warp_id * MMA_N * (BK / 2);
            ldmatrix_b4x16_x2(b, to_smem(&s_B[b_base + (lane_id % 16) * 16]));

            uint16_t sf = 0x8080;

            uint32_t a1[4];
            ldmatrix_b4x16_x4(a1, to_smem(&s_A1[lane_id * 16]));
            mma_fp4(acc_q1, a1, b, acc_q1, (uint32_t)sf, (uint32_t)sf);

            uint32_t a2[4];
            ldmatrix_b4x16_x4(a2, to_smem(&s_A2[lane_id * 16]));
            mma_fp4(acc_q2, a2, b, acc_q2, (uint32_t)sf, (uint32_t)sf);
        }
        __syncthreads();
    }

    // Store gate output: warp w, lane c gives output col
    if (warp_id < N_MMA_PER_BN) {
        int c = lane_id % 4;
        int m_group = lane_id / 8;
        int col_q1 = n_off + warp_id * 8 + c;
        int col_q2 = n_off + warp_id * 8 + 4 + c;
        if (m_group == 0) {
            if (col_q1 < INTERMEDIATE) gate_out[col_q1] = acc_q1[0];
            if (col_q2 < INTERMEDIATE) gate_out[col_q2] = acc_q2[0];
        }
    }
}

int main() {
    float E2M1[] = {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};

    float* d_out; cudaMalloc(&d_out, INTERMEDIATE * sizeof(float));
    cudaMemset(d_out, 0, INTERMEDIATE * sizeof(float));

    cudaFuncSetAttribute(col_perm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
    col_perm_kernel<<<1, BLOCK_SIZE, SMEM_TOTAL>>>(d_out, 1);
    cudaDeviceSynchronize();

    float h[INTERMEDIATE];
    cudaMemcpy(h, d_out, INTERMEDIATE * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Column permutation probe (first 16 gate cols)\n");
    printf("Weight col n has E2M1 value %.1f (idx %d)\n\n", E2M1[0 % 7 + 1], 0 % 7 + 1);

    printf("Expected: output[n] = 4096 * E2M1[n%%7+1]\n");
    for (int n = 0; n < 8; n++)
        printf("  col %d: expected %8.1f (val=%.1f)\n", n, 4096.0 * E2M1[n % 7 + 1], E2M1[n % 7 + 1]);
    printf("\n");

    printf("Actual kernel output (first 16 cols):\n");
    for (int n = 0; n < 16; n++) {
        float expected_val = 4096.0f * E2M1[n % 7 + 1];
        // Find which E2M1 value this output corresponds to
        float ratio = h[n] / 4096.0f;
        int match = -1;
        for (int j = 0; j < 8; j++) {
            if (fabsf(ratio - E2M1[j]) < 0.01f) { match = j; break; }
        }
        printf("  col %2d: output=%8.1f  expected=%8.1f  ratio=%.3f  -> weight_col=%s\n",
               n, h[n], expected_val, ratio,
               match >= 0 ? "?" : "???");
        if (match >= 0) {
            // Find which original column has this E2M1 value
            for (int orig = 0; orig < 64; orig++) {
                if ((orig % 7 + 1) == match) {
                    printf("         (matches weight col %d, E2M1[%d]=%.1f)\n", orig, match, E2M1[match]);
                    break;
                }
            }
        }
    }

    cudaFree(d_out);
    return 0;
}
