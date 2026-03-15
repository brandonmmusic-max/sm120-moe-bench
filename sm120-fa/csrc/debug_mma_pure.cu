/**
 * Pure MMA test: manually load A/B fragments (no ldmatrix).
 * Validates MMA m16n8k16 output layout independently.
 */
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

__device__ __forceinline__ uint32_t pack2(const __nv_bfloat16& a, const __nv_bfloat16& b) {
    uint32_t r;
    asm("mov.b32 %0, {%1, %2};" : "=r"(r) : "h"(*(const uint16_t*)&a), "h"(*(const uint16_t*)&b));
    return r;
}

__global__ void pure_mma_test(
    const __nv_bfloat16* __restrict__ A_in,  // [16, 16] row-major
    const __nv_bfloat16* __restrict__ B_in,  // [16, 8] row-major (= K^T)
    float* __restrict__ C_out                 // [16, 8]
) {
    int lane = threadIdx.x;
    if (lane >= WARP_SIZE) return;

    // A fragment: m16n8k16, A is row-major [M=16, K=16]
    // Thread lane owns:
    //   frag[0]: pack(A[lane/4, 2*(lane%4)], A[lane/4, 2*(lane%4)+1])
    //   frag[1]: pack(A[lane/4, 2*(lane%4)+8], A[lane/4, 2*(lane%4)+9])
    //   frag[2]: pack(A[lane/4+8, 2*(lane%4)], A[lane/4+8, 2*(lane%4)+1])
    //   frag[3]: pack(A[lane/4+8, 2*(lane%4)+8], A[lane/4+8, 2*(lane%4)+9])
    int ar0 = lane / 4;
    int ar1 = lane / 4 + 8;
    int ac = 2 * (lane % 4);

    uint32_t a_frag[4];
    a_frag[0] = pack2(A_in[ar0 * MMA_K + ac],     A_in[ar0 * MMA_K + ac + 1]);
    a_frag[1] = pack2(A_in[ar0 * MMA_K + ac + 8],  A_in[ar0 * MMA_K + ac + 9]);
    a_frag[2] = pack2(A_in[ar1 * MMA_K + ac],     A_in[ar1 * MMA_K + ac + 1]);
    a_frag[3] = pack2(A_in[ar1 * MMA_K + ac + 8],  A_in[ar1 * MMA_K + ac + 9]);

    // B fragment: m16n8k16, B is col-major [K=16, N=8]
    // B_in is actually K^T stored as [K=16, N=8] ROW-MAJOR
    // For col-major B in MMA: element [k, n] = B_colmaj[k + n*K]
    // But B_in is row-major [K, N]: element [k, n] = B_in[k * N + n]
    //
    // Thread lane owns:
    //   frag[0]: pack(B[2*(lane%4), lane/4], B[2*(lane%4)+1, lane/4])
    //   frag[1]: pack(B[2*(lane%4)+8, lane/4], B[2*(lane%4)+9, lane/4])
    //
    // B_in is row-major [K=16, N=8]:
    //   B_in[k, n] = B_in[k * 8 + n]
    int bn = lane / 4;      // 0..7
    int bk = 2 * (lane % 4);

    uint32_t b_frag[2];
    if (bn < MMA_N) {
        b_frag[0] = pack2(B_in[bk * MMA_N + bn],       B_in[(bk + 1) * MMA_N + bn]);
        b_frag[1] = pack2(B_in[(bk + 8) * MMA_N + bn],  B_in[(bk + 9) * MMA_N + bn]);
    } else {
        b_frag[0] = 0;
        b_frag[1] = 0;
    }

    // MMA
    float c[4];
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\n"
        "    {%0, %1, %2, %3},\n"
        "    {%4, %5, %6, %7},\n"
        "    {%8, %9},\n"
        "    {%10, %11, %12, %13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
          "r"(b_frag[0]), "r"(b_frag[1]),
          "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f)
    );

    // Output: thread lane owns:
    //   c[0]: C[lane/4, (lane%4)*2]
    //   c[1]: C[lane/4, (lane%4)*2+1]
    //   c[2]: C[lane/4+8, (lane%4)*2]
    //   c[3]: C[lane/4+8, (lane%4)*2+1]
    int r0 = lane / 4;
    int r1 = lane / 4 + 8;
    int c0 = (lane % 4) * 2;

    C_out[r0 * MMA_N + c0]     = c[0];
    C_out[r0 * MMA_N + c0 + 1] = c[1];
    C_out[r1 * MMA_N + c0]     = c[2];
    C_out[r1 * MMA_N + c0 + 1] = c[3];
}

extern "C" void run_pure_mma(
    const __nv_bfloat16* A, const __nv_bfloat16* B, float* C, cudaStream_t s
) {
    pure_mma_test<<<1, WARP_SIZE, 0, s>>>(A, B, C);
}
