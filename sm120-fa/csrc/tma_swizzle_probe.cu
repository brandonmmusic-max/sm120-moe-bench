/**
 * Probe TMA SWIZZLE_128B pattern by loading known data and
 * reading it back to determine the exact XOR mapping.
 *
 * Strategy: load a tile where element [row, col] = row * 1000 + col.
 * Then read back from SMEM using various address formulas until
 * we find the one that produces the correct values.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define HEAD_DIM 128
#define BLOCK_N 16  // Small tile for easy debugging
#define SMEM_STRIDE HEAD_DIM

// TMA 2D load
__device__ __forceinline__ void tma_load_2d(
    void* smem_dst, const CUtensorMap* desc,
    int coord_x, int coord_y, uint64_t* mbar
) {
    uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
    uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
    uint64_t desc_addr = reinterpret_cast<uint64_t>(desc);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(dst), "l"(desc_addr), "r"(coord_x), "r"(coord_y), "r"(mbar_addr)
    );
}

__global__ void probe_swizzle(
    const CUtensorMap* __restrict__ desc,
    float* __restrict__ output  // [BLOCK_N, HEAD_DIM] — read-back values
) {
    // SMEM: BLOCK_N × HEAD_DIM bf16
    extern __shared__ char smem[];
    __nv_bfloat16* s = reinterpret_cast<__nv_bfloat16*>(smem);
    __shared__ __align__(8) uint64_t mbar;

    int tid = threadIdx.x;

    if (tid == 0) {
        uint32_t ma = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(ma), "r"(1));
    }
    __syncthreads();

    if (tid == 0) {
        uint32_t ma = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));
        int nbytes = BLOCK_N * HEAD_DIM * 2;
        asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(ma), "r"(nbytes));
        tma_load_2d(s, desc, 0, 0, &mbar);
    }

    // Wait
    {
        uint32_t ma = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));
        asm volatile(
            "{\n"
            ".reg .pred P;\n"
            "W:\n"
            "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
            "@!P bra W;\n"
            "}\n"
            :: "r"(ma), "r"(0)
        );
    }
    __syncthreads();

    // Read back ALL SMEM linearly and write to output
    // This tells us exactly how TMA laid out the data
    for (int i = tid; i < BLOCK_N * HEAD_DIM; i += blockDim.x) {
        output[i] = __bfloat162float(s[i]);
    }
}

extern "C" void run_swizzle_probe(
    const CUtensorMap* desc, float* output, cudaStream_t stream
) {
    int smem = BLOCK_N * HEAD_DIM * 2;
    probe_swizzle<<<1, 128, smem, stream>>>(desc, output);
}
