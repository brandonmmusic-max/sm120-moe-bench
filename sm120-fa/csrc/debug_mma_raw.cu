/**
 * Raw MMA fragment dump — no writeback mapping assumptions.
 *
 * Test 1: all-ones → C should be 16.0 everywhere
 * Test 2: basis-vector Q → C selects one K row
 *
 * Output: raw debug_frag[lane][reg] = c_frag[reg] for all 32 lanes × 4 regs
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

__global__ void raw_mma_dump(
    const __nv_bfloat16* __restrict__ A_in,  // [16, 16] row-major
    const __nv_bfloat16* __restrict__ B_in,  // [16, 8] row-major (= K^T)
    float* __restrict__ raw_out               // [32, 4] — raw per-lane fragments
) {
    int lane = threadIdx.x;
    if (lane >= WARP_SIZE) return;

    // ========================================================================
    // A fragment: row-major [M=16, K=16]
    // ========================================================================
    // Correct PTX fragment layout for m16n8k16 A matrix:
    //   Ra0 = [A[g,   2t],   A[g,   2t+1]]     row g, k=2t,2t+1
    //   Ra1 = [A[g+8, 2t],   A[g+8, 2t+1]]     row g+8, k=2t,2t+1
    //   Ra2 = [A[g,   2t+8], A[g,   2t+9]]     row g, k=2t+8,2t+9
    //   Ra3 = [A[g+8, 2t+8], A[g+8, 2t+9]]     row g+8, k=2t+8,2t+9
    int g = lane / 4;      // groupID: 0..7
    int t = lane % 4;      // threadID: 0..3
    int r0 = g;            // row 0..7
    int r1 = g + 8;        // row 8..15
    int c0 = t * 2;        // col 0,2,4,6
    int c1 = t * 2 + 8;    // col 8,10,12,14

    uint32_t a_frag[4];
    a_frag[0] = pack2(A_in[r0 * MMA_K + c0],     A_in[r0 * MMA_K + c0 + 1]);
    a_frag[1] = pack2(A_in[r1 * MMA_K + c0],     A_in[r1 * MMA_K + c0 + 1]);
    a_frag[2] = pack2(A_in[r0 * MMA_K + c1],     A_in[r0 * MMA_K + c1 + 1]);
    a_frag[3] = pack2(A_in[r1 * MMA_K + c1],     A_in[r1 * MMA_K + c1 + 1]);

    // ========================================================================
    // B fragment: B_in is K stored as [N, K] row-major
    // This IS col-major [K, N] storage! K[n, k] at index n*K + k = col-major B[k, n].
    //
    // MMA m16n8k16 expects B col-major [K=16, N=8]:
    //   frag[0] = pack(B_cm[2*(lane%4), lane/4], B_cm[2*(lane%4)+1, lane/4])
    //   frag[1] = pack(B_cm[2*(lane%4)+8, lane/4], B_cm[2*(lane%4)+9, lane/4])
    //
    // B_cm[k, n] is at storage index n*K + k (col-major)
    // Since B_in = K[N, K] row-major: K[n, k] at index n*K + k
    // So B_cm[k, n] = K[n, k] = B_in[n * K + k]
    //
    // Fragment load:
    //   frag[0] = pack(B_in[bn * K + bk], B_in[bn * K + bk + 1])
    //   frag[1] = pack(B_in[bn * K + bk + 8], B_in[bn * K + bk + 9])
    //   where bn = lane/4, bk = 2*(lane%4)
    // ========================================================================
    int bn = lane / 4;      // N index (0..7)
    int bk = 2 * (lane % 4); // K index (0,2,4,6)

    uint32_t b_frag[2];
    if (bn < MMA_N) {
        b_frag[0] = pack2(B_in[bn * MMA_K + bk],       B_in[bn * MMA_K + bk + 1]);
        b_frag[1] = pack2(B_in[bn * MMA_K + bk + 8],    B_in[bn * MMA_K + bk + 9]);
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

    // Dump raw fragments — no writeback mapping
    raw_out[lane * 4 + 0] = c[0];
    raw_out[lane * 4 + 1] = c[1];
    raw_out[lane * 4 + 2] = c[2];
    raw_out[lane * 4 + 3] = c[3];
}

extern "C" void run_raw_mma(
    const __nv_bfloat16* A, const __nv_bfloat16* B, float* out, cudaStream_t s
) {
    raw_mma_dump<<<1, WARP_SIZE, 0, s>>>(A, B, out);
}
