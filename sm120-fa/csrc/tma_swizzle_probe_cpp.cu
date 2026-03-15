/**
 * TMA swizzle probe — C++ host-side descriptor creation
 *
 * Creates a known-data tensor, loads via TMA SWIZZLE_128B,
 * reads SMEM linearly, writes to output for analysis.
 */
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define HEAD_DIM 128
#define BLOCK_N 8  // Small for easy analysis

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

__global__ void probe_kernel(
    const CUtensorMap* __restrict__ desc,
    float* __restrict__ output
) {
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

    // Read ALL SMEM linearly
    for (int i = tid; i < BLOCK_N * HEAD_DIM; i += blockDim.x) {
        output[i] = __bfloat162float(s[i]);
    }
}

extern "C" void run_tma_swizzle_probe(
    float* output,  // [BLOCK_N * HEAD_DIM] on device
    cudaStream_t stream
) {
    // Create known data: element [row, col] = row * 256 + col
    __nv_bfloat16* d_data;
    cudaMalloc(&d_data, BLOCK_N * HEAD_DIM * sizeof(__nv_bfloat16));

    __nv_bfloat16 h_data[BLOCK_N * HEAD_DIM];
    for (int r = 0; r < BLOCK_N; r++)
        for (int c = 0; c < HEAD_DIM; c++)
            h_data[r * HEAD_DIM + c] = __float2bfloat16((float)(r * 256 + c));

    cudaMemcpy(d_data, h_data, BLOCK_N * HEAD_DIM * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    // Create TMA descriptor with SWIZZLE_128B
    CUtensorMap desc __attribute__((aligned(64)));

    cuuint64_t dims[2] = {(cuuint64_t)HEAD_DIM, (cuuint64_t)BLOCK_N};
    cuuint64_t strides[1] = {(cuuint64_t)(HEAD_DIM * sizeof(__nv_bfloat16))};
    cuuint32_t box[2] = {(cuuint32_t)HEAD_DIM, (cuuint32_t)BLOCK_N};
    cuuint32_t elem_strides[2] = {1, 1};

    CUresult err = cuTensorMapEncodeTiled(
        &desc,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        (void*)d_data,
        dims, strides, box, elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    if (err != CUDA_SUCCESS) {
        printf("cuTensorMapEncodeTiled failed: %d\n", err);

        // Try without swizzle as fallback
        err = cuTensorMapEncodeTiled(
            &desc,
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2,
            (void*)d_data,
            dims, strides, box, elem_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        );
        printf("Without swizzle: %d\n", err);
    }

    // Copy descriptor to device
    CUtensorMap* d_desc;
    cudaMalloc(&d_desc, sizeof(CUtensorMap));
    cudaMemcpy(d_desc, &desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

    int smem = BLOCK_N * HEAD_DIM * 2 + 64;
    probe_kernel<<<1, 128, smem, stream>>>(d_desc, output);

    cudaFree(d_data);
    cudaFree(d_desc);
}
