/**
 * ldmatrix + MMA combined test.
 * Load A via ldmatrix.x4, transpose K→K^T in SMEM, load B via ldmatrix.x2.
 */
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

__global__ void ldm_mma_test(
    const __nv_bfloat16* __restrict__ A_in,
    const __nv_bfloat16* __restrict__ K_in,
    float* __restrict__ C_out
) {
    int lane = threadIdx.x;
    if (lane >= WARP_SIZE) return;

    __shared__ __nv_bfloat16 a_smem[16 * 16];   // A[16, 16] row-major
    __shared__ __nv_bfloat16 kt_smem[16 * 8];   // K^T[16, 8] row-major

    // Load A
    for (int i = lane; i < 256; i += WARP_SIZE)
        a_smem[i] = A_in[i];

    // Transpose K[8,16] → K^T[16,8]
    for (int i = lane; i < 128; i += WARP_SIZE) {
        int k = i / 8;
        int n = i % 8;
        kt_smem[k * 8 + n] = K_in[n * 16 + k];
    }
    __syncwarp();

    // === ldmatrix.x4 for A ===
    // A is [16, 16] row-major, stride = 16 bf16 = 32 bytes per row
    // Thread mapping for x4: 4 sub-matrices of 8x8
    int a_sub = lane / 8;
    int a_sublane = lane % 8;
    int a_row = (a_sub / 2) * 8 + a_sublane;
    int a_col = (a_sub % 2) * 8;

    uint32_t a_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&a_smem[a_row * 16 + a_col]));

    uint32_t a_frag[4];
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
        : "r"(a_addr)
    );

    // === ldmatrix.x2.trans for B ===
    // We store K[8,16] directly in SMEM (not transposed).
    // ldmatrix.trans reads COLUMNS from row-major storage → produces col-major B.
    //
    // K is [N=8, K=16] row-major in SMEM, stride=16 bf16=32 bytes.
    // For .trans x2: thread provides address for its COLUMN.
    // Sub0: threads [0..7] provide addresses for 8 rows, reading column (lane%8)
    // Sub1: threads [8..15] provide addresses for 8 rows, reading column 8+(lane%8)
    //
    // Thread t addresses: K_smem[row=(t%8), col_start=col]
    // where col = (t/8)*8 selects the column group
    // The instruction reads K[0..7, col] and K[0..7, col+8] transposed

    // Actually we need K in SMEM, not K^T
    // Overwrite kt_smem with K directly
    for (int i = lane; i < 128; i += WARP_SIZE)
        kt_smem[i] = K_In_shared[i];  // This won't work, need separate buffer

    // Simpler: just load K directly into a shared buffer
    __shared__ __nv_bfloat16 k_smem[8 * 16];
    for (int i = lane; i < 128; i += WARP_SIZE)
        k_smem[i] = K_in[i];
    __syncwarp();

    // ldmatrix.x2.trans on K[8, 16] row-major
    // K stride = 16 bf16 = 32 bytes
    // Thread addressing for .trans:
    //   .trans reads along columns (stride = row_stride_bytes)
    //   Thread t provides base address for column (t % 8) in sub-matrix
    //   sub0: threads [0..7], sub1: threads [8..15]
    int k_sublane = lane % 8;
    int k_sub = (lane / 8) % 2;
    // Address: row 0 of K, column = k_sub*8 + k_sublane
    // But wait — K is [8, 16]. ldmatrix.trans reads 8 elements along a column.
    // Column j of K: K[0,j], K[1,j], ..., K[7,j] — that's N=8 elements.
    // The .trans instruction needs addresses with row stride.
    // Thread provides: &K_smem[k_sublane * 16 + k_sub * 8]
    // This points to K[k_sublane, k_sub*8], and .trans reads K[k_sublane, k_sub*8]
    // across the column (varying k_sublane with stride 16*2=32 bytes).

    // Actually for .trans, each thread provides an address and the instruction
    // reads from that address + multiples of the implicit row stride.
    // The implicit stride is inferred from the addresses provided by threads.

    // For x2.trans with [8,16] matrix (8 rows, 16 cols):
    // We want B col-major = K transposed.
    // .trans with K stored [N=8, K=16]:
    //   Reads K[:, col_idx] for each sub-matrix
    //   Sub0 reads cols 0-7 (of K's 16-wide rows)
    //   Sub1 reads cols 8-15

    // Thread address: &k_smem[k_sublane * 16 + k_sub * 8]
    uint32_t k_addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&k_smem[k_sublane * 16 + k_sub * 8]));

    uint32_t b_frag[2];
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(b_frag[0]), "=r"(b_frag[1])
        : "r"(k_addr)
    );

    // === MMA ===
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

    // Output: PTX spec mapping
    int gid = lane / 4;
    int tid = lane % 4;
    C_out[(gid) * MMA_N + tid * 2]       = c[0];
    C_out[(gid) * MMA_N + tid * 2 + 1]   = c[1];
    C_out[(gid + 8) * MMA_N + tid * 2]   = c[2];
    C_out[(gid + 8) * MMA_N + tid * 2 + 1] = c[3];
}

extern "C" void run_ldm_mma(
    const __nv_bfloat16* A, const __nv_bfloat16* K, float* C, cudaStream_t s
) {
    ldm_mma_test<<<1, WARP_SIZE, 0, s>>>(A, K, C);
}
