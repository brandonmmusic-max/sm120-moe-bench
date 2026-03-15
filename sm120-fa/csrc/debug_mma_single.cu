/**
 * SM120 FA Debug — Single MMA tile: Q[16×16] @ K^T[16×8] = S[16×8]
 *
 * Validates one mma.sync.m16n8k16 operation using ldmatrix-loaded fragments.
 * No online softmax, no P@V — just Q@K^T for one tile.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define WARP_SIZE 32
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

// Row stride for SMEM: padded to avoid bank conflicts
// 16 bf16 = 32 bytes per MMA_K row. Pad to 32 bytes (no padding needed).
// For HEAD_DIM=128: stride = 128 * 2 = 256 bytes
// For this test: stride = MMA_K * 2 = 32 bytes (tight packing)

// For Q: m16 × k16, row-major, stride = MMA_K * sizeof(bf16) = 32 bytes
#define Q_STRIDE 16   // in bf16 elements (= MMA_K)
// For K: n8 × k16 (stored as k16 × n8 for col-major access), stride = MMA_N
// Actually K is [N, K] in memory. K^T is [K, N].
// For mma row.col: A is row-major [M,K], B is col-major [K,N]
// So K stored as [N_block, K] needs ldmatrix.trans to load as col-major B.
// K in SMEM: [BLOCK_N, HEAD_DIM], row-major. ldmatrix.trans reads columns.
#define K_STRIDE 16   // K is [N, K], stride in K dimension = MMA_K elements

// SMEM: Q[16×16] + K[8×16] = (256 + 128) * 2 = 768 bytes
#define Q_SMEM_ELEMS (MMA_M * Q_STRIDE)  // 256 bf16
#define K_SMEM_ELEMS (MMA_N * K_STRIDE)  // 128 bf16
#define KT_SMEM_ELEMS (MMA_K * MMA_N)  // 128 bf16 for K^T staging
#define TOTAL_SMEM_BYTES ((Q_SMEM_ELEMS + K_SMEM_ELEMS + KT_SMEM_ELEMS) * 2)

__global__ void debug_single_mma(
    const __nv_bfloat16* __restrict__ Q_in,  // [16, 16] row-major
    const __nv_bfloat16* __restrict__ K_in,  // [8, 16] row-major (K not transposed)
    float* __restrict__ S_out                 // [16, 8] output scores
) {
    const int lane = threadIdx.x;
    if (lane >= WARP_SIZE) return;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* k_smem = q_smem + Q_SMEM_ELEMS;

    // ========================================================================
    // Load Q and K to SMEM (simple element-wise, all 32 threads)
    // ========================================================================
    // Q: 16×16 = 256 elements, 32 threads → 8 elements each
    for (int i = lane; i < Q_SMEM_ELEMS; i += WARP_SIZE) {
        q_smem[i] = Q_in[i];
    }
    // K: 8×16 = 128 elements
    for (int i = lane; i < K_SMEM_ELEMS; i += WARP_SIZE) {
        k_smem[i] = K_in[i];
    }
    __syncwarp();

    // ========================================================================
    // ldmatrix.x4 for Q (A matrix): load m16×k16
    //
    // Thread-to-address mapping for A (row-major, 16 rows × 16 cols of bf16):
    //   Thread t provides address for:
    //     sub_matrix = t / 8 (0..3)
    //     sub_row = t % 8
    //     SMEM row = (sub_matrix / 2) * 8 + sub_row  → rows [0..7] or [8..15]
    //     SMEM col = (sub_matrix % 2) * 8            → cols [0..7] or [8..15]
    //     address = q_smem + row * Q_STRIDE + col
    // ========================================================================
    int q_sub = lane / 8;
    int q_sublane = lane % 8;
    int q_row = (q_sub / 2) * 8 + q_sublane;
    int q_col = (q_sub % 2) * 8;

    uint32_t q_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&q_smem[q_row * Q_STRIDE + q_col]));

    uint32_t q_frag[4];
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(q_frag[0]), "=r"(q_frag[1]), "=r"(q_frag[2]), "=r"(q_frag[3])
        : "r"(q_addr)
    );

    // ========================================================================
    // Load K (B matrix) for mma.m16n8k16.row.col
    //
    // B operand is [K=16, N=8] col-major.
    // K in SMEM is [N=8, K=16] row-major.
    //
    // For ldmatrix.x2 (non-transposed): reads 2 × 8×8 sub-matrices.
    // We need to store K transposed in SMEM first: K^T is [K=16, N=8].
    //
    // Alternative: store K as [N=8, K=16] and use ldmatrix.x2.trans.
    // ldmatrix.trans reads columns instead of rows from SMEM.
    //
    // For .trans with x2:
    //   Each thread provides an address. The instruction reads 8 elements
    //   along the COLUMN (stride between elements = row_stride, not 1).
    //   Thread t addresses: column = t % 8
    //   The instruction gathers from k_smem[row * K_STRIDE + col] for row=0..7
    //
    // So thread t should provide: &k_smem[0 * K_STRIDE + (t % 8)]
    // for sub-matrix 0 (k cols 0..7)
    // and: &k_smem[0 * K_STRIDE + 8 + (t % 8)]
    // for sub-matrix 1 (k cols 8..15)
    //
    // But K is [N=8, K=16]. We want B = K^T = [K=16, N=8].
    // If we store K^T in SMEM as [16, 8] row-major, then ldmatrix (non-trans)
    // can load it directly.
    // ========================================================================

    // Transpose K into a staging buffer in SMEM
    // K^T: [K=16, N=8] from K: [N=8, K=16]
    __nv_bfloat16* kt_smem = k_smem + K_SMEM_ELEMS;  // Use space after K

    for (int i = lane; i < MMA_K * MMA_N; i += WARP_SIZE) {
        int kt_row = i / MMA_N;  // K dimension (0..15)
        int kt_col = i % MMA_N;  // N dimension (0..7)
        // K[n, k] → K^T[k, n]
        kt_smem[kt_row * MMA_N + kt_col] = k_smem[kt_col * K_STRIDE + kt_row];
    }
    __syncwarp();

    // Now ldmatrix.x2 (non-transposed) on K^T [16, 8] row-major
    // K^T stride = MMA_N = 8 bf16 = 16 bytes per row
    // Thread mapping for x2: 2 sub-matrices of 8×8
    //   sub 0: rows [0..7], threads [0..7]
    //   sub 1: rows [8..15], threads [8..15]
    int kt_sub = (lane / 8) % 2;
    int kt_sublane = lane % 8;
    int kt_row = kt_sub * 8 + kt_sublane;

    uint32_t k_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&kt_smem[kt_row * MMA_N]));

    uint32_t k_frag[2];
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(k_frag[0]), "=r"(k_frag[1])
        : "r"(k_addr)
    );

    // ========================================================================
    // MMA: S = Q @ K^T → m16n8k16
    // ========================================================================
    float s_frag[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\n"
        "    {%0, %1, %2, %3},\n"
        "    {%4, %5, %6, %7},\n"
        "    {%8, %9},\n"
        "    {%10, %11, %12, %13};\n"
        : "=f"(s_frag[0]), "=f"(s_frag[1]), "=f"(s_frag[2]), "=f"(s_frag[3])
        : "r"(q_frag[0]), "r"(q_frag[1]), "r"(q_frag[2]), "r"(q_frag[3]),
          "r"(k_frag[0]), "r"(k_frag[1]),
          "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f)
    );

    // ========================================================================
    // Write output: MMA output layout for m16n8k16
    //   Thread t owns:
    //     s_frag[0]: (row=t/4,     col=(t%4)*2)
    //     s_frag[1]: (row=t/4,     col=(t%4)*2+1)
    //     s_frag[2]: (row=t/4+8,   col=(t%4)*2)
    //     s_frag[3]: (row=t/4+8,   col=(t%4)*2+1)
    // ========================================================================
    int out_r0 = lane / 4;
    int out_r1 = lane / 4 + 8;
    int out_c0 = (lane % 4) * 2;
    int out_c1 = out_c0 + 1;

    S_out[out_r0 * MMA_N + out_c0] = s_frag[0];
    S_out[out_r0 * MMA_N + out_c1] = s_frag[1];
    S_out[out_r1 * MMA_N + out_c0] = s_frag[2];
    S_out[out_r1 * MMA_N + out_c1] = s_frag[3];
}

extern "C" void run_debug_single_mma(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, float* S, cudaStream_t s
) {
    debug_single_mma<<<1, WARP_SIZE, TOTAL_SMEM_BYTES, s>>>(Q, K, S);
}
