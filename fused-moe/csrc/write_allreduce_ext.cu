/**
 * write_allreduce_ext.cu — Write-Based P2P AllReduce for vLLM TP=4
 *
 * Replaces NCCL AllReduce for small payloads (<64KB) with direct PCIe
 * posted writes. Each GPU writes its partial result to all peers' staging
 * buffers, then each GPU locally reduces its own staging buffers.
 *
 * Algorithm (write-based, 4 GPUs):
 *   Phase 1 (WRITE): Each GPU writes its data to staging[rank] on every peer
 *   Phase 2 (REDUCE): Each GPU reduces staging[0..3] locally → output
 *
 * Key properties:
 *   - CUDA graph compatible: fixed buffer pointers, no host sync
 *   - Uses __threadfence_system() for cross-GPU PCIe visibility
 *   - BF16 native with vectorized (bf162) loads/stores
 *   - Flag-based synchronization with monotonic generation counters
 *   - Auto-fallback: caller checks size, falls back to NCCL for >64KB
 *
 * This is a PyTorch C++ extension (torch::Tensor interface).
 *
 * Build: see setup.py in write_allreduce_ext/
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// ============================================================================
// Constants
// ============================================================================
static constexpr int MAX_GPUS = 8;
static constexpr int BLOCK_SIZE = 256;
// Maximum elements for P2P path (64KB / 2 bytes per bf16 = 32768 elements)
static constexpr int MAX_P2P_ELEMENTS = 32768;

// ============================================================================
// Write-based AllReduce kernel (fused write + fence + flag + wait + reduce)
//
// Each GPU runs this kernel on its own stream. The kernel:
//   1. Writes local data to staging[my_rank] on EVERY GPU (including self)
//   2. Signals ready via flag write to all peers
//   3. Waits for all peers' flags
//   4. Reduces staging[0..world_size-1] → output (local reduction)
//
// This is CUDA-graph safe because:
//   - All pointers are fixed at capture time
//   - No host-side synchronization
//   - Generation counter prevents ABA problems across replays
// ============================================================================
__global__ void write_allreduce_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,      // local input [n_elements]
    __nv_bfloat16* __restrict__ output,            // local output [n_elements]
    // Staging buffers: staging_ptrs[dst_gpu][src_rank] — pointer to
    // the staging slot on dst_gpu where src_rank writes its data.
    // For the LOCAL gpu, staging_ptrs[my_rank][*] are local pointers.
    // For REMOTE gpus, staging_ptrs[peer][my_rank] are P2P-mapped pointers.
    __nv_bfloat16** __restrict__ peer_staging_my_slot, // [world_size] ptrs to staging[peer][my_rank]
    __nv_bfloat16** __restrict__ local_staging,        // [world_size] ptrs to staging[my_rank][src]
    volatile int** __restrict__ peer_flag_ptrs,        // [world_size] ptrs to each GPU's flag[my_rank]
    volatile int* __restrict__ local_flag_ptrs_base,   // local flags[world_size] — peers write here
    int n_elements,
    int my_rank,
    int world_size,
    int generation
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Phase 1: Write local data to staging[peer][my_rank] on every GPU
    // This uses PCIe posted writes for remote GPUs (fast, no read-back)
    for (int i = tid; i < n_elements; i += stride) {
        __nv_bfloat16 val = input[i];
        // Write to all peers' staging buffers (including our own local copy)
        for (int p = 0; p < world_size; p++) {
            peer_staging_my_slot[p][i] = val;
        }
    }

    // Ensure all writes are visible across PCIe before signaling
    __threadfence_system();
    __syncthreads();

    // Phase 2: Signal readiness to all peers by writing to their flag arrays
    // peer_flag_ptrs[p] points to GPU p's flag[my_rank]
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int p = 0; p < world_size; p++) {
            if (p == my_rank) continue;
            // Write our generation to peer's flag slot for us
            *(peer_flag_ptrs[p]) = generation;
        }
        // Also mark our own flag as ready (for the local reduce to proceed)
        local_flag_ptrs_base[my_rank] = generation;
    }

    // Phase 3: Wait for all peers to signal they've written their data
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int p = 0; p < world_size; p++) {
            if (p == my_rank) continue;
            // Spin-wait on our local flag[p] — peer p writes here
            while (local_flag_ptrs_base[p] < generation) {
                // PCIe read spin — ~1μs per iteration
            }
        }
    }
    __syncthreads();

    // Phase 4: Local reduction — all staging[my_rank][*] buffers are local memory
    // Vectorized BF16 reduction (2x BF16 per load with __nv_bfloat162)
    int n_vec = n_elements / 2;
    const __nv_bfloat162* staging_vec[MAX_GPUS];
    for (int p = 0; p < world_size; p++) {
        staging_vec[p] = reinterpret_cast<const __nv_bfloat162*>(local_staging[p]);
    }
    __nv_bfloat162* out_vec = reinterpret_cast<__nv_bfloat162*>(output);

    for (int i = tid; i < n_vec; i += stride) {
        float2 acc = {0.0f, 0.0f};
        #pragma unroll 4
        for (int p = 0; p < world_size; p++) {
            __nv_bfloat162 v = staging_vec[p][i];
            acc.x += __bfloat162float(v.x);
            acc.y += __bfloat162float(v.y);
        }
        out_vec[i] = __nv_bfloat162(__float2bfloat16(acc.x), __float2bfloat16(acc.y));
    }

    // Handle odd element if n_elements is odd
    if (n_elements & 1) {
        int last = n_elements - 1;
        if (tid == 0) {
            float acc = 0.0f;
            for (int p = 0; p < world_size; p++) {
                acc += __bfloat162float(local_staging[p][last]);
            }
            output[last] = __float2bfloat16(acc);
        }
    }
}

// ============================================================================
// Simpler variant: One-shot read-reduce-write (runs on ONE GPU, reads peers)
//
// This variant is simpler and more CUDA-graph friendly — it runs a single
// kernel on GPU 0 (or whichever GPU owns the AllReduce), reads from all
// peer buffers via P2P, reduces locally, then writes results back to peers.
//
// No flags needed — the CUDA stream ordering guarantees the peer data is
// ready (the preceding GEMM wrote it before this kernel launches).
//
// This is the preferred variant for vLLM integration because:
//   - vLLM already serializes AllReduce after the GEMM on the same stream
//   - All 4 GPUs run identical model layers, AllReduce is a sync point
//   - The CUDA graph captures the entire sequence (GEMM → AllReduce → next layer)
// ============================================================================
__global__ void oneshot_read_reduce_bf16_kernel(
    const __nv_bfloat16* __restrict__ buf0,
    const __nv_bfloat16* __restrict__ buf1,
    const __nv_bfloat16* __restrict__ buf2,
    const __nv_bfloat16* __restrict__ buf3,
    __nv_bfloat16* __restrict__ out0,
    __nv_bfloat16* __restrict__ out1,
    __nv_bfloat16* __restrict__ out2,
    __nv_bfloat16* __restrict__ out3,
    int n_elements
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Vectorized: process 2 BF16 elements per thread
    int n_vec = n_elements / 2;

    const __nv_bfloat162* v0 = reinterpret_cast<const __nv_bfloat162*>(buf0);
    const __nv_bfloat162* v1 = reinterpret_cast<const __nv_bfloat162*>(buf1);
    const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(buf2);
    const __nv_bfloat162* v3 = reinterpret_cast<const __nv_bfloat162*>(buf3);
    __nv_bfloat162* o0 = reinterpret_cast<__nv_bfloat162*>(out0);
    __nv_bfloat162* o1 = reinterpret_cast<__nv_bfloat162*>(out1);
    __nv_bfloat162* o2 = reinterpret_cast<__nv_bfloat162*>(out2);
    __nv_bfloat162* o3 = reinterpret_cast<__nv_bfloat162*>(out3);

    for (int i = tid; i < n_vec; i += blockDim.x * gridDim.x) {
        __nv_bfloat162 a = v0[i];
        __nv_bfloat162 b = v1[i];
        __nv_bfloat162 c = v2[i];
        __nv_bfloat162 d = v3[i];

        // Reduce in FP32 for accuracy
        float2 sum;
        sum.x = __bfloat162float(a.x) + __bfloat162float(b.x)
              + __bfloat162float(c.x) + __bfloat162float(d.x);
        sum.y = __bfloat162float(a.y) + __bfloat162float(b.y)
              + __bfloat162float(c.y) + __bfloat162float(d.y);

        __nv_bfloat162 result = __nv_bfloat162(
            __float2bfloat16(sum.x), __float2bfloat16(sum.y));

        // Write result to ALL GPUs (P2P posted writes for remote)
        o0[i] = result;
        o1[i] = result;
        o2[i] = result;
        o3[i] = result;
    }

    // Handle odd trailing element
    if ((n_elements & 1) && tid == 0) {
        int last = n_elements - 1;
        float s = __bfloat162float(buf0[last]) + __bfloat162float(buf1[last])
                + __bfloat162float(buf2[last]) + __bfloat162float(buf3[last]);
        __nv_bfloat16 r = __float2bfloat16(s);
        out0[last] = r;
        out1[last] = r;
        out2[last] = r;
        out3[last] = r;
    }

    // Ensure all P2P writes are posted before kernel completes
    __threadfence_system();
}

// ============================================================================
// Variable world_size variant (2-8 GPUs)
// Uses indirection arrays instead of hardcoded 4 pointers
// ============================================================================
__global__ void oneshot_read_reduce_bf16_varws_kernel(
    const __nv_bfloat16* const* __restrict__ input_ptrs,   // [world_size]
    __nv_bfloat16* const* __restrict__ output_ptrs,         // [world_size]
    int n_elements,
    int world_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_vec = n_elements / 2;

    for (int i = tid; i < n_vec; i += blockDim.x * gridDim.x) {
        float2 sum = {0.0f, 0.0f};
        for (int p = 0; p < world_size; p++) {
            const __nv_bfloat162* vp = reinterpret_cast<const __nv_bfloat162*>(input_ptrs[p]);
            __nv_bfloat162 v = vp[i];
            sum.x += __bfloat162float(v.x);
            sum.y += __bfloat162float(v.y);
        }
        __nv_bfloat162 result = __nv_bfloat162(
            __float2bfloat16(sum.x), __float2bfloat16(sum.y));
        for (int p = 0; p < world_size; p++) {
            __nv_bfloat162* op = reinterpret_cast<__nv_bfloat162*>(output_ptrs[p]);
            op[i] = result;
        }
    }

    if ((n_elements & 1) && tid == 0) {
        int last = n_elements - 1;
        float s = 0.0f;
        for (int p = 0; p < world_size; p++) {
            s += __bfloat162float(input_ptrs[p][last]);
        }
        __nv_bfloat16 r = __float2bfloat16(s);
        for (int p = 0; p < world_size; p++) {
            output_ptrs[p][last] = r;
        }
    }

    __threadfence_system();
}

// ============================================================================
// State management: pre-allocated buffers for the extension lifetime
// ============================================================================
struct WriteAllReduceState {
    bool initialized = false;
    int world_size = 0;
    int my_rank = 0;
    int max_elements = 0;

    // For oneshot variant: device-side pointer arrays
    // input_ptrs_dev[i] = pointer to GPU i's input buffer (P2P accessible)
    // output_ptrs_dev[i] = pointer to GPU i's output buffer (P2P accessible)
    // These are allocated on the local GPU and contain cross-GPU pointers.
    void* input_ptrs_dev = nullptr;   // __nv_bfloat16*[world_size] on device
    void* output_ptrs_dev = nullptr;  // __nv_bfloat16*[world_size] on device

    // Generation counter for the flag-based variant
    int generation = 1;
};

static WriteAllReduceState g_state;

// ============================================================================
// Python-facing functions
// ============================================================================

/**
 * enable_p2p_access() — Enable peer access between all CUDA devices.
 * Must be called once before any P2P operations.
 */
void enable_p2p_access() {
    int n_devices;
    cudaGetDeviceCount(&n_devices);
    for (int i = 0; i < n_devices; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < n_devices; j++) {
            if (i != j) {
                int can_access = 0;
                cudaDeviceCanAccessPeer(&can_access, i, j);
                if (can_access) {
                    cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                        printf("[WriteAR] WARNING: P2P %d->%d failed: %s\n",
                               i, j, cudaGetErrorString(err));
                    }
                }
            }
        }
    }
}

/**
 * oneshot_allreduce(buf0, buf1, buf2, buf3) — In-place AllReduce for 4 GPUs.
 *
 * Each buf_i is a BF16 tensor on cuda:i. After the call, all 4 tensors
 * contain the element-wise sum. Runs the reduction kernel on GPU 0's stream.
 *
 * CUDA graph compatible: uses fixed device pointers, no host sync.
 *
 * Returns buf0 (the tensor on the calling GPU, for chaining).
 */
torch::Tensor oneshot_allreduce_4gpu(
    torch::Tensor buf0,
    torch::Tensor buf1,
    torch::Tensor buf2,
    torch::Tensor buf3
) {
    TORCH_CHECK(buf0.dtype() == torch::kBFloat16, "Expected BF16 tensors");
    TORCH_CHECK(buf0.numel() == buf1.numel() && buf0.numel() == buf2.numel()
                && buf0.numel() == buf3.numel(), "All buffers must have same size");
    TORCH_CHECK(buf0.is_contiguous() && buf1.is_contiguous()
                && buf2.is_contiguous() && buf3.is_contiguous(),
                "All buffers must be contiguous");

    int n = buf0.numel();
    TORCH_CHECK(n <= MAX_P2P_ELEMENTS,
                "Payload too large for P2P AllReduce (", n * 2, " bytes > 64KB)");

    int threads = BLOCK_SIZE;
    // Use enough blocks to saturate but not too many (small payload)
    int blocks = std::min((n / 2 + threads - 1) / threads, 16);
    blocks = std::max(blocks, 1);

    auto* p0 = reinterpret_cast<__nv_bfloat16*>(buf0.data_ptr());
    auto* p1 = reinterpret_cast<__nv_bfloat16*>(buf1.data_ptr());
    auto* p2 = reinterpret_cast<__nv_bfloat16*>(buf2.data_ptr());
    auto* p3 = reinterpret_cast<__nv_bfloat16*>(buf3.data_ptr());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(buf0.device().index()).stream();

    // In-place: read from buf_i, write result back to buf_i
    oneshot_read_reduce_bf16_kernel<<<blocks, threads, 0, stream>>>(
        p0, p1, p2, p3,   // read from these
        p0, p1, p2, p3,   // write result back (in-place)
        n
    );

    return buf0;
}

/**
 * oneshot_allreduce_outofplace(buf0, buf1, buf2, buf3, out0, out1, out2, out3)
 *
 * Out-of-place variant: reads from buf_i, writes reduced result to out_i.
 * Useful when the input buffer must not be modified (vLLM sometimes needs this).
 */
torch::Tensor oneshot_allreduce_outofplace(
    torch::Tensor buf0, torch::Tensor buf1,
    torch::Tensor buf2, torch::Tensor buf3,
    torch::Tensor out0, torch::Tensor out1,
    torch::Tensor out2, torch::Tensor out3
) {
    TORCH_CHECK(buf0.dtype() == torch::kBFloat16, "Expected BF16 tensors");
    int n = buf0.numel();
    TORCH_CHECK(n <= MAX_P2P_ELEMENTS, "Payload too large for P2P path");

    int threads = BLOCK_SIZE;
    int blocks = std::min((n / 2 + threads - 1) / threads, 16);
    blocks = std::max(blocks, 1);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(buf0.device().index()).stream();

    oneshot_read_reduce_bf16_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(buf0.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(buf1.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(buf2.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(buf3.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out0.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out1.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out2.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(out3.data_ptr()),
        n
    );

    return out0;
}

/**
 * setup_varws_ptrs(input_tensors, output_tensors) — Pre-register buffer
 * pointers for the variable-world-size kernel. Call once during init.
 *
 * input_tensors: list of BF16 tensors, one per GPU rank [cuda:0, cuda:1, ...]
 * output_tensors: list of BF16 tensors, one per GPU rank (can be same as input for in-place)
 *
 * Copies the device pointers into a device-side array on the calling GPU.
 */
void setup_varws_ptrs(
    std::vector<torch::Tensor> input_tensors,
    std::vector<torch::Tensor> output_tensors
) {
    int ws = input_tensors.size();
    TORCH_CHECK(ws >= 2 && ws <= MAX_GPUS, "world_size must be 2-8");
    TORCH_CHECK(output_tensors.size() == (size_t)ws);

    // Collect raw pointers
    __nv_bfloat16* in_ptrs[MAX_GPUS];
    __nv_bfloat16* out_ptrs[MAX_GPUS];
    for (int i = 0; i < ws; i++) {
        in_ptrs[i] = reinterpret_cast<__nv_bfloat16*>(input_tensors[i].data_ptr());
        out_ptrs[i] = reinterpret_cast<__nv_bfloat16*>(output_tensors[i].data_ptr());
    }

    // Allocate device arrays on current GPU
    if (g_state.input_ptrs_dev) cudaFree(g_state.input_ptrs_dev);
    if (g_state.output_ptrs_dev) cudaFree(g_state.output_ptrs_dev);

    cudaMalloc(&g_state.input_ptrs_dev, ws * sizeof(__nv_bfloat16*));
    cudaMalloc(&g_state.output_ptrs_dev, ws * sizeof(__nv_bfloat16*));

    cudaMemcpy(g_state.input_ptrs_dev, in_ptrs,
               ws * sizeof(__nv_bfloat16*), cudaMemcpyHostToDevice);
    cudaMemcpy(g_state.output_ptrs_dev, out_ptrs,
               ws * sizeof(__nv_bfloat16*), cudaMemcpyHostToDevice);

    g_state.world_size = ws;
    g_state.max_elements = input_tensors[0].numel();
    g_state.initialized = true;
}

/**
 * varws_allreduce(n_elements) — AllReduce using pre-registered pointers.
 * Call setup_varws_ptrs() first. n_elements must be <= the tensor size
 * registered during setup.
 */
void varws_allreduce(int n_elements) {
    TORCH_CHECK(g_state.initialized, "Call setup_varws_ptrs() first");
    TORCH_CHECK(n_elements <= g_state.max_elements);
    TORCH_CHECK(n_elements <= MAX_P2P_ELEMENTS);

    int threads = BLOCK_SIZE;
    int blocks = std::min((n_elements / 2 + threads - 1) / threads, 16);
    blocks = std::max(blocks, 1);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    oneshot_read_reduce_bf16_varws_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16* const*>(g_state.input_ptrs_dev),
        reinterpret_cast<__nv_bfloat16* const*>(g_state.output_ptrs_dev),
        n_elements,
        g_state.world_size
    );
}

// ============================================================================
// Binding
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Write-based P2P AllReduce for vLLM TP inference";
    m.def("enable_p2p_access", &enable_p2p_access,
          "Enable P2P access between all CUDA devices");
    m.def("oneshot_allreduce_4gpu", &oneshot_allreduce_4gpu,
          "In-place BF16 AllReduce for exactly 4 GPUs (CUDA graph safe)");
    m.def("oneshot_allreduce_outofplace", &oneshot_allreduce_outofplace,
          "Out-of-place BF16 AllReduce for 4 GPUs");
    m.def("setup_varws_ptrs", &setup_varws_ptrs,
          "Pre-register buffer pointers for variable world-size AllReduce");
    m.def("varws_allreduce", &varws_allreduce,
          "AllReduce using pre-registered pointers (variable world size)");
}
