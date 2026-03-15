/**
 * SM120 FA Debug Kernel — ldmatrix address validation
 *
 * Isolates ldmatrix loads from MMA to validate:
 *   1. SMEM layout and alignment
 *   2. Per-lane address mapping
 *   3. XOR swizzle pattern
 *   4. Correct data after ldmatrix.x1, x2, x4
 *
 * Single warp, single tile, no MMA.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define WARP_SIZE 32
#define HEAD_DIM 128
#define TILE_M 16   // MMA m16n8k16 tile
#define TILE_K 16
#define TILE_N 8

// ============================================================================
// CUTLASS-style XOR swizzle for bank-conflict-free SMEM access
//
// For 16-byte (8 bf16) access pattern:
//   - SMEM row stride = HEAD_DIM * 2 bytes = 256 bytes
//   - Bank = (byte_offset / 16) % 32
//   - XOR: bank ^= (row % B) where B is swizzle factor
//
// CUTLASS uses Swizzle<3, 3, 3> for 128-byte rows:
//   addr = base + row * stride + col_bytes
//   swizzled_addr = addr ^ ((row & 0x7) << 4)  // XOR bits [6:4] with row[2:0]
// ============================================================================

// Row stride in bytes for HEAD_DIM=128 bf16 elements
#define ROW_STRIDE_BYTES (HEAD_DIM * 2)  // 256 bytes

// No swizzle for correctness validation.
// Swizzle will be added after ldmatrix+MMA are proven correct.
// Simple row-major layout with 256-byte row stride.
__device__ __forceinline__
int swizzle_smem_addr(int row, int col_bytes) {
    return row * ROW_STRIDE_BYTES + col_bytes;
}

// ============================================================================
// Debug kernel: load TILE_M × TILE_K from SMEM using ldmatrix.x1
// ============================================================================

__global__ void debug_ldmatrix_x1(
    const __nv_bfloat16* __restrict__ input,  // [TILE_M, HEAD_DIM]
    float* __restrict__ output,                // [TILE_M, TILE_K] for validation
    int* __restrict__ addr_log                 // [32 * 4] address debug log
) {
    const int lane = threadIdx.x;

    // Only one warp
    if (threadIdx.x >= WARP_SIZE) return;

    // Shared memory: TILE_M rows × HEAD_DIM cols, bf16, with swizzle
    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    // ========================================================================
    // Step 1: Load input to SMEM using normal (non-swizzled) layout first
    // Each of 32 threads loads TILE_M * HEAD_DIM / 32 = 16*128/32 = 64 bf16 = 128 bytes
    // ========================================================================
    for (int i = lane; i < TILE_M * HEAD_DIM; i += WARP_SIZE) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        // Apply swizzle for SMEM write
        int byte_offset = swizzle_smem_addr(row, col * 2);
        __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(smem_raw + byte_offset);
        *dst = input[row * HEAD_DIM + col];
    }
    __syncwarp();

    // ========================================================================
    // Step 2: ldmatrix.x1 — loads one 8×8 bf16 matrix fragment (16 bytes)
    //
    // ldmatrix.sync.aligned.m8n8.x1.shared.b16 loads from SMEM.
    // Each of 32 threads provides an address; the instruction loads
    // 8 rows × 8 cols of 16-bit values (128 bytes total).
    //
    // Thread mapping for m8n8:
    //   Thread t provides address for row (t % 8) within the 8×8 tile.
    //   Threads [0..7] → rows 0..7 of first 8×8
    //   Threads [8..15] → rows 0..7 of second 8×8 (if x2/x4)
    //   etc.
    //
    // For m16n8k16:
    //   A matrix (m16×k16): each thread loads from SMEM
    //   Thread t address → row = (t % 16), col offset = (t / 16) * 8
    //   But actually for ldmatrix.x4 loading A:
    //     Thread t addresses row (t % 8) for each of 4 sub-matrices
    // ========================================================================

    // For ldmatrix.x1: load first 8×8 sub-tile
    // Thread t provides address for row (t % 8) of the 8x8 matrix
    // The address should point to the start of that row (8 bf16 = 16 bytes)

    int mat_row = lane % 8;  // which row within 8×8
    int mat_col_start = 0;   // first 8 columns (bytes 0..15)

    // Compute swizzled SMEM address for this thread's row
    int byte_addr = swizzle_smem_addr(mat_row, mat_col_start * 2);

    // Log address info for debugging
    addr_log[lane * 4 + 0] = mat_row;
    addr_log[lane * 4 + 1] = mat_col_start;
    addr_log[lane * 4 + 2] = byte_addr;
    addr_log[lane * 4 + 3] = byte_addr % 16;  // alignment check (must be 0)

    // ldmatrix.x1: load 16 bytes (8 bf16) into one 32-bit register
    uint32_t frag;
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw + byte_addr));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
        : "=r"(frag)
        : "r"(smem_addr)
    );

    // Extract bf16 pair from register and write to output
    __nv_bfloat16* frag_bf16 = reinterpret_cast<__nv_bfloat16*>(&frag);
    // ldmatrix.x1 gives each thread 2 bf16 values
    // Thread t gets elements for column positions based on the m8n8 layout:
    //   Thread t gets (row=t%8, col=2*(t/8)) and (row=t%8, col=2*(t/8)+1)
    int out_row = lane % 8;
    int out_col0 = 2 * (lane / 8);
    int out_col1 = out_col0 + 1;

    output[out_row * TILE_K + out_col0] = __bfloat162float(frag_bf16[0]);
    output[out_row * TILE_K + out_col1] = __bfloat162float(frag_bf16[1]);
}


// ============================================================================
// Debug kernel: load using ldmatrix.x4 for full m16n8k16 A matrix
// ============================================================================

__global__ void debug_ldmatrix_x4_A(
    const __nv_bfloat16* __restrict__ input,  // [TILE_M, HEAD_DIM]
    float* __restrict__ output,                // [TILE_M, TILE_K] for validation
    int* __restrict__ addr_log                 // [32 * 6] debug
) {
    const int lane = threadIdx.x;
    if (threadIdx.x >= WARP_SIZE) return;

    extern __shared__ char smem_raw[];
    __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    // Load to SMEM with swizzle
    for (int i = lane; i < TILE_M * HEAD_DIM; i += WARP_SIZE) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        int byte_offset = swizzle_smem_addr(row, col * 2);
        *reinterpret_cast<__nv_bfloat16*>(smem_raw + byte_offset) = input[row * HEAD_DIM + col];
    }
    __syncwarp();

    // ========================================================================
    // ldmatrix.x4 for A matrix of m16n8k16
    //
    // Loads 4 × 8×8 sub-matrices = m16×k16 matrix
    // Layout: rows [0..7] and [8..15], cols [0..7] and [8..15]
    //
    // Thread address mapping (from PTX spec):
    //   Thread t provides address for:
    //     x1: row = t % 8  (sub-matrix 0)
    //     x2: additionally row = t % 8 (sub-matrix 1)
    //     x3: additionally row = t % 8 (sub-matrix 2)
    //     x4: additionally row = t % 8 (sub-matrix 3)
    //
    //   For m16n8k16 A matrix (row-major, 16 rows × 16 cols):
    //     Sub-matrix 0: rows [0..7], cols [0..7]   → threads [0..7] provide addrs
    //     Sub-matrix 1: rows [0..7], cols [8..15]  → threads [8..15] provide addrs
    //     Sub-matrix 2: rows [8..15], cols [0..7]  → threads [16..23] provide addrs
    //     Sub-matrix 3: rows [8..15], cols [8..15] → threads [24..31] provide addrs
    //
    //   Each thread provides: addr = &smem[sub_row * stride + sub_col_start]
    //     where sub_row = lane % 8
    //           sub_col_start depends on which sub-matrix (0, 8, 0, 8)
    //           sub_row_offset depends on which sub-matrix (0, 0, 8, 8)
    // ========================================================================

    int sub_matrix = lane / 8;        // 0, 1, 2, 3
    int sub_lane = lane % 8;          // 0..7

    int smem_row = (sub_matrix / 2) * 8 + sub_lane;  // row within 16-row tile
    int smem_col = (sub_matrix % 2) * 8;              // col start (0 or 8)

    int byte_addr = swizzle_smem_addr(smem_row, smem_col * 2);

    // Log
    addr_log[lane * 6 + 0] = sub_matrix;
    addr_log[lane * 6 + 1] = smem_row;
    addr_log[lane * 6 + 2] = smem_col;
    addr_log[lane * 6 + 3] = byte_addr;
    addr_log[lane * 6 + 4] = byte_addr % 16;  // alignment
    addr_log[lane * 6 + 5] = (byte_addr / 16) % 32;  // bank

    uint32_t smem_a = static_cast<uint32_t>(__cvta_generic_to_shared(smem_raw + byte_addr));

    uint32_t frag[4];
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
        : "r"(smem_a)
    );

    // Each thread gets 4 registers = 8 bf16 values
    // For A matrix of m16n8k16:
    //   frag[0]: (row = lane/4, col = 2*(lane%4))    and col+1   — first k8
    //   frag[1]: (row = lane/4, col = 2*(lane%4)+8)  and col+9   — second k8
    //   frag[2]: (row = lane/4+8, col = 2*(lane%4))  and col+1   — first k8, row+8
    //   frag[3]: (row = lane/4+8, col = 2*(lane%4)+8) and col+9  — second k8, row+8

    int out_r0 = lane / 4;
    int out_r1 = lane / 4 + 8;
    int out_c_base = 2 * (lane % 4);

    __nv_bfloat16* f0 = reinterpret_cast<__nv_bfloat16*>(&frag[0]);
    __nv_bfloat16* f1 = reinterpret_cast<__nv_bfloat16*>(&frag[1]);
    __nv_bfloat16* f2 = reinterpret_cast<__nv_bfloat16*>(&frag[2]);
    __nv_bfloat16* f3 = reinterpret_cast<__nv_bfloat16*>(&frag[3]);

    // Write to output for validation
    output[out_r0 * TILE_K + out_c_base]     = __bfloat162float(f0[0]);
    output[out_r0 * TILE_K + out_c_base + 1] = __bfloat162float(f0[1]);
    output[out_r0 * TILE_K + out_c_base + 8] = __bfloat162float(f1[0]);
    output[out_r0 * TILE_K + out_c_base + 9] = __bfloat162float(f1[1]);
    output[out_r1 * TILE_K + out_c_base]     = __bfloat162float(f2[0]);
    output[out_r1 * TILE_K + out_c_base + 1] = __bfloat162float(f2[1]);
    output[out_r1 * TILE_K + out_c_base + 8] = __bfloat162float(f3[0]);
    output[out_r1 * TILE_K + out_c_base + 9] = __bfloat162float(f3[1]);
}


// ============================================================================
// Host-side test harness
// ============================================================================

extern "C" {

void run_debug_ldmatrix_x1(
    const __nv_bfloat16* input, float* output, int* addr_log, cudaStream_t s
) {
    int smem = TILE_M * ROW_STRIDE_BYTES;  // 16 * 256 = 4KB
    debug_ldmatrix_x1<<<1, WARP_SIZE, smem, s>>>(input, output, addr_log);
}

void run_debug_ldmatrix_x4(
    const __nv_bfloat16* input, float* output, int* addr_log, cudaStream_t s
) {
    int smem = TILE_M * ROW_STRIDE_BYTES;
    debug_ldmatrix_x4_A<<<1, WARP_SIZE, smem, s>>>(input, output, addr_log);
}

}  // extern "C"
