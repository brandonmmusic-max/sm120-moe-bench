/**
 * Dump the exact SMEM layout CUTLASS produces for FP4 A operand on SM120.
 * Uses TiledCopy with SmemLayoutAtom to write known data to SMEM,
 * then dumps raw bytes to compare against row-major.
 */

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/algorithm/copy.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

using namespace cute;

// SM120 FP4 block-scaled MMA atom
using MmaAtom = SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cutlass::float_e2m1_t, cutlass::float_e2m1_t, float,
    cutlass::float_ue8m0_t, 32>;

using MmaTraits = MMA_Traits<MmaAtom>;

// SmemLayoutAtom: from sm120_rr_smem_selector<uint4_t, _64>
// For K=64 uint4_t: Layout_K_SW32_Atom<uint4_t>
// = upcast<4>(ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag, Layout<Shape<_8,_256>,Stride<_256,_1>>>)
// = ComposedLayout<Swizzle<1,4,3>, smem_ptr_flag, Layout<Shape<_8,_64>,Stride<_64,_1>>>
using SmemLayoutAtom = decltype(
    GMMA::Layout_K_SW32_Atom<cute::uint4_t>{}
);

// SmemCopyAtom: SM100_SU4_DU8x16_x4_LDSM_N (for A operand, FP4)
using SmemCopyAtom = Copy_Atom<SM100_SU4_DU8x16_x4_LDSM_N, cute::uint4_t>;

// Tile shape: [BM=16, BK=64]
using TileShape = Shape<_16, _64>;

__global__ void dump_layout(
    const cute::uint4_t* __restrict__ gmem_data,  // [16, 64] FP4 elements = [16, 32] bytes
    uint8_t* __restrict__ smem_dump,                   // Output: raw SMEM bytes
    int dump_size
) {
    extern __shared__ char smem[];

    const int tid = threadIdx.x;

    // Create SMEM tensor with the CUTLASS layout
    // Tile the SmemLayoutAtom to cover [16, 64]
    auto smem_layout = tile_to_shape(SmemLayoutAtom{}, TileShape{});

    // Print layout info from thread 0
    if (tid == 0) {
        print("SmemLayoutAtom: "); print(SmemLayoutAtom{}); print("\n");
        print("Tiled smem_layout: "); print(smem_layout); print("\n");
        printf("smem_layout size: %d elements\n", (int)size(smem_layout));
    }
    __syncthreads();

    // Create SMEM tensor
    auto sTensor = make_tensor(make_smem_ptr((cute::uint4_t*)smem), smem_layout);

    // Create GMEM tensor [16, 64] row-major
    auto gTensor = make_tensor(make_gmem_ptr(gmem_data),
                               make_layout(TileShape{}, GenRowMajor{}));

    // Write each logical (row, col) element through the CUTLASS layout
    // This applies SmemLayoutAtom's swizzle automatically
    for (int idx = tid; idx < 16 * 64; idx += blockDim.x) {
        int row = idx / 64;
        int col = idx % 64;
        sTensor(row, col) = gTensor(row, col);
    }
    __syncthreads();

    // Dump raw SMEM bytes
    uint8_t* raw = (uint8_t*)smem;
    for (int i = tid; i < dump_size; i += blockDim.x) {
        smem_dump[i] = raw[i];
    }
}

// Also dump what plain row-major looks like for comparison
__global__ void dump_rowmajor(
    const uint8_t* __restrict__ gmem_bytes,  // [16, 32] bytes
    uint8_t* __restrict__ smem_dump,
    int dump_size
) {
    extern __shared__ char smem[];
    const int tid = threadIdx.x;

    // Plain row-major copy
    for (int i = tid; i < dump_size; i += blockDim.x) {
        ((uint8_t*)smem)[i] = gmem_bytes[i];
    }
    __syncthreads();

    for (int i = tid; i < dump_size; i += blockDim.x) {
        smem_dump[i] = ((uint8_t*)smem)[i];
    }
}

int main() {
    printf("=== CUTLASS FP4 SMEM Layout Dump ===\n\n");

    // Create test data: row r, col c gets a unique value
    // FP4 value = (r*64 + c) % 15 + 1 mapped to E2M1 index
    // Pack as uint4_t pairs into bytes
    const int ROWS = 16, COLS = 64;
    const int BYTES = ROWS * COLS / 2;  // 512 bytes

    uint8_t* h_data = new uint8_t[BYTES];
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c += 2) {
            int byte_idx = r * (COLS/2) + c/2;
            uint8_t lo = ((r * COLS + c) % 7 + 1);      // E2M1 index 1-7
            uint8_t hi = ((r * COLS + c + 1) % 7 + 1);
            h_data[byte_idx] = lo | (hi << 4);
        }
    }

    uint8_t* d_data;
    cudaMalloc(&d_data, BYTES);
    cudaMemcpy(d_data, h_data, BYTES, cudaMemcpyHostToDevice);

    uint8_t* d_smem_cutlass;
    uint8_t* d_smem_rowmajor;
    cudaMalloc(&d_smem_cutlass, BYTES);
    cudaMalloc(&d_smem_rowmajor, BYTES);

    // Run CUTLASS layout dump
    dump_layout<<<1, 256, BYTES + 256>>>(
        (const cute::uint4_t*)d_data, d_smem_cutlass, BYTES);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUTLASS dump error: %s\n", cudaGetErrorString(err));
    }

    // Run row-major dump
    dump_rowmajor<<<1, 256, BYTES + 256>>>(
        d_data, d_smem_rowmajor, BYTES);
    cudaDeviceSynchronize();

    uint8_t* h_cutlass = new uint8_t[BYTES];
    uint8_t* h_rowmajor = new uint8_t[BYTES];
    cudaMemcpy(h_cutlass, d_smem_cutlass, BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rowmajor, d_smem_rowmajor, BYTES, cudaMemcpyDeviceToHost);

    // Compare first 64 bytes (rows 0-1)
    printf("\nFirst 64 bytes comparison (rows 0-1):\n");
    printf("Byte  RowMaj  CUTLASS  Match?\n");
    int diffs = 0;
    for (int i = 0; i < 64; i++) {
        int match = (h_cutlass[i] == h_rowmajor[i]);
        if (!match) diffs++;
        if (i < 64)
            printf(" %3d:  0x%02X    0x%02X     %s\n",
                   i, h_rowmajor[i], h_cutlass[i], match ? "=" : "DIFF");
    }

    printf("\nTotal diffs in first 64 bytes: %d\n", diffs);

    // Count total diffs
    int total_diffs = 0;
    for (int i = 0; i < BYTES; i++)
        if (h_cutlass[i] != h_rowmajor[i]) total_diffs++;
    printf("Total diffs in all %d bytes: %d\n", BYTES, total_diffs);

    if (total_diffs == 0) {
        printf("\nSMEM layouts are IDENTICAL — no transformation needed!\n");
    } else {
        printf("\nSMEM layouts DIFFER — need to replicate CUTLASS layout.\n");
        // Find the permutation pattern
        printf("\nPermutation (CUTLASS byte position -> row-major byte position):\n");
        for (int i = 0; i < 64 && i < BYTES; i++) {
            // Find where h_cutlass[i] appears in h_rowmajor
            for (int j = 0; j < BYTES; j++) {
                if (h_cutlass[i] == h_rowmajor[j]) {
                    printf("  CUTLASS[%3d] = RowMaj[%3d] (row %d, byte_col %d)\n",
                           i, j, j/32, j%32);
                    break;
                }
            }
        }
    }

    delete[] h_data; delete[] h_cutlass; delete[] h_rowmajor;
    cudaFree(d_data); cudaFree(d_smem_cutlass); cudaFree(d_smem_rowmajor);
    return 0;
}
