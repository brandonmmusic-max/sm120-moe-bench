/**
 * TMA bulk copy test v3 — fixed mbarrier protocol
 *
 * The mbarrier protocol for cp.async.bulk:
 * 1. init mbarrier with arrival count (e.g., 1)
 * 2. Issue cp.async.bulk with mbarrier::complete_tx::bytes
 *    - This automatically signals the mbarrier when the copy completes
 *    - The tx_count is automatically tracked
 * 3. One thread does mbarrier.arrive (to account for the arrival count)
 * 4. All threads wait via mbarrier.try_wait
 *
 * Actually for cp.async.bulk with mbarrier::complete_tx::bytes,
 * the hardware automatically does the arrive when bytes are delivered.
 * We just need to set expect_tx to tell mbarrier how many bytes to expect.
 */
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__global__ void tma_bulk_v3(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    int nbytes
) {
    __shared__ __nv_bfloat16 s[4096];  // up to 8KB
    __shared__ __align__(8) uint64_t mbar;

    uint32_t ma = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar));

    if (threadIdx.x == 0) {
        // Init mbarrier: expect 1 arrival (from the thread issuing arrive)
        asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(ma), "r"(1));
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(s));

        // Tell mbarrier to expect nbytes from async bulk copy
        asm volatile(
            "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n"
            :: "r"(ma), "r"(nbytes)
        );

        // Issue bulk copy — mbarrier tracks completion automatically
        asm volatile(
            "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
            :: "r"(dst), "l"(in), "r"(nbytes), "r"(ma)
        );
    }
    __syncthreads();

    // All threads wait for mbarrier (phase 0)
    {
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

    // Write result
    if (threadIdx.x < 8) {
        out[threadIdx.x] = s[threadIdx.x];
    }
}

extern "C" void run_tma_v3(
    __nv_bfloat16* out, const __nv_bfloat16* in, int nbytes, cudaStream_t stream
) {
    tma_bulk_v3<<<1, 128, 0, stream>>>(out, in, nbytes);
}
