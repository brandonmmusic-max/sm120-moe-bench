/**
 * TMA bulk copy test for SM120 — correct PTX syntax
 */
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__global__ void tma_bulk_v2(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in
) {
    __shared__ __nv_bfloat16 s[512];
    __shared__ __align__(8) uint64_t mbar;

    if (threadIdx.x == 0) {
        uint32_t ma = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));
        // Initialize mbarrier with expected arrival count = 1
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(ma), "r"(1));
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(s));
        uint32_t ma = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));
        int nbytes = 1024;  // 512 bf16 = 1024 bytes

        // cp.async.bulk with mbarrier completion mechanism
        // SM120 (CC 12.0) syntax:
        asm volatile(
            "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
            :: "r"(dst), "l"(in), "r"(nbytes), "r"(ma)
        );
    }
    __syncthreads();

    // Wait via mbarrier
    if (threadIdx.x == 0) {
        uint32_t ma = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));
        asm volatile(
            "{\n"
            ".reg .pred P;\n"
            "WAIT:\n"
            "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\n"
            "@!P bra WAIT;\n"
            "}\n"
            :: "r"(ma), "r"(0)
        );
    }
    __syncthreads();

    // Verify
    if (threadIdx.x < 4) {
        out[threadIdx.x] = s[threadIdx.x];
    }
}

extern "C" void run_tma_test(
    __nv_bfloat16* out, const __nv_bfloat16* in, cudaStream_t stream
) {
    tma_bulk_v2<<<1, 128, 0, stream>>>(out, in);
}
