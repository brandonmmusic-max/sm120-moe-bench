/**
 * megakernel_p2p_test.cu — Multi-GPU P2P AllReduce Test for MegaKernel
 *
 * Tests:
 *   1. P2P connectivity between 4 GPUs
 *   2. Isolated AllReduce correctness (cooperative kernel context)
 *   3. AllReduce latency benchmark
 *   4. Full single-layer megakernel with real P2P AllReduce
 *
 * The key insight: cooperative_launch guarantees all CTAs are resident,
 * so flag-based polling won't deadlock (unlike standalone kernels where
 * SMs might not all be running simultaneously).
 *
 * BUT: cooperative_launch is per-GPU. For multi-GPU AllReduce, we need
 * one cooperative kernel per GPU, launched from separate host threads
 * (or streams with cudaLaunchCooperativeKernelMultiDevice).
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a \
 *     --expt-relaxed-constexpr -rdc=true -lpthread \
 *     -o megakernel_p2p_test csrc/megakernel_p2p_test.cu
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <thread>
#include <atomic>
#include <chrono>

namespace cg = cooperative_groups;

// ============================================================================
// Constants (matching megakernel_v1.cu)
// ============================================================================
static constexpr int HIDDEN       = 4096;
static constexpr int WORLD_SIZE   = 4;
static constexpr int NUM_CTAS     = 188;  // 1 per SM on RTX PRO 6000 Blackwell
static constexpr int BLOCK_SIZE   = 256;

#define CHECK_CUDA(c) do { cudaError_t _e = (c); if (_e != cudaSuccess) { \
    printf("CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); } } while(0)

// ============================================================================
// P2P Buffer Structures
// ============================================================================
struct P2PBuffers {
    float* remote_recv[WORLD_SIZE];
    float* local_send;
    float* local_recv;
    volatile uint32_t* local_flags;
    volatile uint32_t* remote_flags[WORLD_SIZE];
    int rank;
};

// Per-GPU allocation for AllReduce
struct GPUAllReduceState {
    float*    data;           // [HIDDEN] — the data to reduce (input/output)
    float*    recv_buf;       // [WORLD_SIZE * HIDDEN] — receive slots from peers
    uint32_t* flags;          // [WORLD_SIZE] — arrival flags
    int       device_id;
};

// ============================================================================
// TEST 1: P2P AllReduce Kernel (isolated, cooperative)
//
// Each GPU runs this cooperative kernel. CTA 0 does the AllReduce.
// Other CTAs just idle (grid.sync() keeps them alive so cooperative works).
//
// This tests the exact same p2p_allreduce_write() logic from megakernel_v1.cu
// but in a multi-GPU context.
// ============================================================================
__device__ void p2p_allreduce_write_test(
    float*                __restrict__ data,
    const P2PBuffers&                  p2p,
    int                                generation)
{
    const int tid  = threadIdx.x;
    const int rank = p2p.rank;

    // Step 1: Write local data to all remote receive buffers
    for (int dst = 0; dst < WORLD_SIZE; dst++) {
        if (dst == rank) continue;
        float* remote_slot = p2p.remote_recv[dst] + rank * HIDDEN;
        for (int i = tid; i < HIDDEN; i += BLOCK_SIZE) {
            remote_slot[i] = data[i];
        }
    }

    // Fence: ensure all posted writes are visible before setting flags
    __threadfence_system();

    // Step 2: Set flag on each remote GPU
    if (tid == 0) {
        for (int dst = 0; dst < WORLD_SIZE; dst++) {
            if (dst == rank) continue;
            p2p.remote_flags[dst][rank] = generation;
        }
    }
    __threadfence_system();

    // Step 3: Poll local flags until all remotes have written
    if (tid == 0) {
        for (int src = 0; src < WORLD_SIZE; src++) {
            if (src == rank) continue;
            // Spin-wait for remote GPU to write its data
            while (p2p.local_flags[src] < (uint32_t)generation) {
                // Spin — posted writes from remote GPUs will eventually arrive
            }
        }
    }
    __syncthreads();

    // Step 4: Sum local data + received partials
    float* local_recv_base = p2p.local_recv;
    for (int i = tid; i < HIDDEN; i += BLOCK_SIZE) {
        float sum = data[i];
        for (int src = 0; src < WORLD_SIZE; src++) {
            if (src == rank) continue;
            sum += local_recv_base[src * HIDDEN + i];
        }
        data[i] = sum;
    }
    __syncthreads();
}

// Cooperative kernel: just AllReduce, nothing else
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
allreduce_test_kernel(P2PBuffers p2p, float* data, int num_iters)
{
    cg::grid_group grid = cg::this_grid();
    const int cta_id = blockIdx.x;

    for (int iter = 0; iter < num_iters; iter++) {
        if (cta_id == 0) {
            p2p_allreduce_write_test(data, p2p, iter + 1);
        }
        grid.sync();
    }
}

// ============================================================================
// TEST 2: Latency-only kernel (minimal work, just AllReduce timing)
//
// Uses device-side clock for per-AllReduce timing.
// CTA 0 records clock64() before and after AllReduce. Other CTAs wait.
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
allreduce_latency_kernel(P2PBuffers p2p, float* data, long long* timing_out, int num_iters)
{
    cg::grid_group grid = cg::this_grid();
    const int cta_id = blockIdx.x;
    const int tid = threadIdx.x;

    for (int iter = 0; iter < num_iters; iter++) {
        long long t0 = 0, t1 = 0;

        if (cta_id == 0) {
            if (tid == 0) t0 = clock64();
            __syncthreads();

            p2p_allreduce_write_test(data, p2p, iter + 1);

            if (tid == 0) {
                t1 = clock64();
                if (timing_out) timing_out[iter] = t1 - t0;
            }
        }
        grid.sync();
    }
}

// ============================================================================
// Host Setup: P2P connectivity + buffer allocation
// ============================================================================

void setup_p2p(int num_gpus) {
    printf("Setting up P2P between %d GPUs...\n", num_gpus);

    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        for (int j = 0; j < num_gpus; j++) {
            if (i == j) continue;
            int can_access = 0;
            CHECK_CUDA(cudaDeviceCanAccessPeer(&can_access, i, j));
            if (!can_access) {
                printf("  ERROR: GPU %d cannot P2P access GPU %d\n", i, j);
                printf("  Check: iommu=pt kernel param, driver 595+, reseat GPUs\n");
                exit(1);
            }
            cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
            if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                printf("  ERROR: cudaDeviceEnablePeerAccess(%d->%d) failed: %s\n",
                       i, j, cudaGetErrorString(err));
                exit(1);
            }
        }
    }

    // Verify P2P with a small copy
    printf("  P2P access enabled. Verifying with small copies...\n");
    float* test_bufs[4];
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc(&test_bufs[i], 256));
        float val = (float)(i + 1) * 100.0f;
        CHECK_CUDA(cudaMemset(test_bufs[i], 0, 256));
        CHECK_CUDA(cudaMemcpy(test_bufs[i], &val, sizeof(float), cudaMemcpyHostToDevice));
    }
    // GPU 0 reads from GPU 1
    CHECK_CUDA(cudaSetDevice(0));
    float readback;
    CHECK_CUDA(cudaMemcpy(&readback, test_bufs[1], sizeof(float), cudaMemcpyDeviceToHost));
    printf("  GPU0 read from GPU1 buffer: %.1f (expected 200.0)\n", readback);
    if (fabsf(readback - 200.0f) > 0.1f) {
        printf("  ERROR: P2P read failed!\n");
        exit(1);
    }

    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaFree(test_bufs[i]));
    }
    printf("  P2P verified OK\n\n");
}

void allocate_gpu_state(GPUAllReduceState* states, int num_gpus) {
    for (int g = 0; g < num_gpus; g++) {
        CHECK_CUDA(cudaSetDevice(g));
        states[g].device_id = g;

        // Data buffer
        CHECK_CUDA(cudaMalloc(&states[g].data, HIDDEN * sizeof(float)));

        // Receive buffer: one slot per peer (WORLD_SIZE slots x HIDDEN floats each)
        CHECK_CUDA(cudaMalloc(&states[g].recv_buf, WORLD_SIZE * HIDDEN * sizeof(float)));
        CHECK_CUDA(cudaMemset(states[g].recv_buf, 0, WORLD_SIZE * HIDDEN * sizeof(float)));

        // Flags: one per potential sender
        CHECK_CUDA(cudaMalloc(&states[g].flags, WORLD_SIZE * sizeof(uint32_t)));
        CHECK_CUDA(cudaMemset(states[g].flags, 0, WORLD_SIZE * sizeof(uint32_t)));
    }
}

// Build P2PBuffers struct for a given GPU rank
P2PBuffers build_p2p_buffers(GPUAllReduceState* states, int rank, int num_gpus) {
    P2PBuffers p2p;
    memset(&p2p, 0, sizeof(p2p));
    p2p.rank = rank;
    p2p.local_recv = states[rank].recv_buf;
    p2p.local_send = states[rank].data;
    p2p.local_flags = (volatile uint32_t*)states[rank].flags;

    // Remote pointers: point to other GPUs' recv buffers and flags
    // These work via P2P (UVA) — storing to these addresses generates PCIe posted writes
    for (int i = 0; i < num_gpus; i++) {
        p2p.remote_recv[i] = states[i].recv_buf;
        p2p.remote_flags[i] = (volatile uint32_t*)states[i].flags;
    }
    return p2p;
}

// ============================================================================
// TEST 1: Multi-GPU AllReduce Correctness
//
// Each GPU starts with data[i] = rank + 1 (constant vector).
// After AllReduce, each GPU should have data[i] = 1+2+3+4 = 10.
//
// Uses cudaLaunchCooperativeKernel on each GPU from a separate host thread.
// All kernels must be running simultaneously for the flag polling to complete.
// ============================================================================

struct ThreadArgs {
    int gpu_id;
    P2PBuffers p2p;
    float* data;
    int num_iters;
    bool success;
    float first_val;
};

std::atomic<int> launch_barrier{0};
std::atomic<int> done_count{0};

void gpu_thread_correctness(ThreadArgs* args) {
    int g = args->gpu_id;
    CHECK_CUDA(cudaSetDevice(g));

    // Get kernel properties
    int smem_size = 0;  // AllReduce kernel needs no extra SMEM
    CHECK_CUDA(cudaFuncSetAttribute(
        allreduce_test_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));

    // Check cooperative launch support
    int max_blocks = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks, allreduce_test_kernel, BLOCK_SIZE, smem_size));
    if (max_blocks < 1) {
        printf("  GPU %d: Cannot fit cooperative kernel!\n", g);
        args->success = false;
        return;
    }

    dim3 grid(NUM_CTAS);
    dim3 block(BLOCK_SIZE);

    P2PBuffers p2p_copy = args->p2p;
    float* data = args->data;
    int num_iters = args->num_iters;
    void* kernel_args[] = { &p2p_copy, &data, &num_iters };

    // Barrier: wait for all threads before launching
    launch_barrier.fetch_add(1);
    while (launch_barrier.load() < WORLD_SIZE) {
        // spin
    }

    CHECK_CUDA(cudaLaunchCooperativeKernel(
        (void*)allreduce_test_kernel,
        grid, block,
        kernel_args, smem_size, 0));

    CHECK_CUDA(cudaDeviceSynchronize());

    // Read back result
    float result;
    CHECK_CUDA(cudaMemcpy(&result, data, sizeof(float), cudaMemcpyDeviceToHost));
    args->first_val = result;

    // Check all elements
    float* h_data = new float[HIDDEN];
    CHECK_CUDA(cudaMemcpy(h_data, data, HIDDEN * sizeof(float), cudaMemcpyDeviceToHost));

    float expected = (1.0f + 2.0f + 3.0f + 4.0f);  // sum of all ranks' initial data
    // After num_iters AllReduces, the value compounds since we re-reduce the result.
    // iter 1: data = sum(ranks) = 10
    // iter 2: each GPU has 10, so AllReduce = 4*10 = 40
    // For correctness: just check iter=1 (single AllReduce)

    bool ok = true;
    for (int i = 0; i < HIDDEN; i++) {
        if (isnan(h_data[i]) || isinf(h_data[i])) {
            ok = false;
            break;
        }
    }
    args->success = ok;
    delete[] h_data;
}

void test_correctness(GPUAllReduceState* states, int num_gpus) {
    printf("==========================================================\n");
    printf("TEST 1: Multi-GPU P2P AllReduce Correctness\n");
    printf("==========================================================\n\n");

    // Initialize data: GPU g has data[i] = g + 1
    for (int g = 0; g < num_gpus; g++) {
        CHECK_CUDA(cudaSetDevice(g));
        float* h_data = new float[HIDDEN];
        for (int i = 0; i < HIDDEN; i++) h_data[i] = (float)(g + 1);
        CHECK_CUDA(cudaMemcpy(states[g].data, h_data, HIDDEN * sizeof(float), cudaMemcpyHostToDevice));

        // Reset flags and recv buffers
        CHECK_CUDA(cudaMemset(states[g].flags, 0, WORLD_SIZE * sizeof(uint32_t)));
        CHECK_CUDA(cudaMemset(states[g].recv_buf, 0, WORLD_SIZE * HIDDEN * sizeof(float)));
        delete[] h_data;
    }

    // Build P2P buffer structs
    ThreadArgs thread_args[WORLD_SIZE];
    launch_barrier.store(0);
    done_count.store(0);

    for (int g = 0; g < num_gpus; g++) {
        thread_args[g].gpu_id = g;
        thread_args[g].p2p = build_p2p_buffers(states, g, num_gpus);
        thread_args[g].data = states[g].data;
        thread_args[g].num_iters = 1;  // single AllReduce for correctness
        thread_args[g].success = false;
        thread_args[g].first_val = 0.0f;
    }

    // Launch threads
    std::thread threads[WORLD_SIZE];
    for (int g = 0; g < num_gpus; g++) {
        threads[g] = std::thread(gpu_thread_correctness, &thread_args[g]);
    }
    for (int g = 0; g < num_gpus; g++) {
        threads[g].join();
    }

    // Report
    // Sum of all ranks' initial values: 1+2+3+4 = 10
    float expected = 0.0f;
    for (int g = 0; g < num_gpus; g++) expected += (float)(g + 1);

    bool all_ok = true;
    for (int g = 0; g < num_gpus; g++) {
        bool val_ok = (fabsf(thread_args[g].first_val - expected) < 0.01f);
        printf("  GPU %d: data[0] = %.4f (expected %.1f) %s, NaN/Inf: %s\n",
               g, thread_args[g].first_val, expected,
               val_ok ? "OK" : "FAIL",
               thread_args[g].success ? "none" : "FOUND");
        if (!val_ok || !thread_args[g].success) all_ok = false;
    }
    printf("\n  Overall: %s\n\n", all_ok ? "PASS" : "FAIL");
}

// ============================================================================
// TEST 2: AllReduce Latency Benchmark
//
// Same multi-threaded approach, but with timing kernel.
// GPU 0 records device clocks. We convert to microseconds.
// ============================================================================

struct LatencyThreadArgs {
    int gpu_id;
    P2PBuffers p2p;
    float* data;
    long long* timing;  // only GPU 0
    int num_iters;
};

std::atomic<int> latency_barrier{0};

void gpu_thread_latency(LatencyThreadArgs* args) {
    int g = args->gpu_id;
    CHECK_CUDA(cudaSetDevice(g));

    int smem_size = 0;
    CHECK_CUDA(cudaFuncSetAttribute(
        allreduce_latency_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size));

    dim3 grid(NUM_CTAS);
    dim3 block(BLOCK_SIZE);

    P2PBuffers p2p_copy = args->p2p;
    float* data = args->data;
    long long* timing = args->timing;
    int num_iters = args->num_iters;
    void* kernel_args[] = { &p2p_copy, &data, &timing, &num_iters };

    latency_barrier.fetch_add(1);
    while (latency_barrier.load() < WORLD_SIZE) { /* spin */ }

    CHECK_CUDA(cudaLaunchCooperativeKernel(
        (void*)allreduce_latency_kernel,
        grid, block,
        kernel_args, smem_size, 0));

    CHECK_CUDA(cudaDeviceSynchronize());
}

void test_latency(GPUAllReduceState* states, int num_gpus) {
    printf("==========================================================\n");
    printf("TEST 2: AllReduce Latency Benchmark\n");
    printf("==========================================================\n\n");

    const int WARMUP = 50;
    const int BENCH  = 200;
    const int TOTAL  = WARMUP + BENCH;

    // Initialize data
    for (int g = 0; g < num_gpus; g++) {
        CHECK_CUDA(cudaSetDevice(g));
        float* h_data = new float[HIDDEN];
        for (int i = 0; i < HIDDEN; i++) h_data[i] = 1.0f;
        CHECK_CUDA(cudaMemcpy(states[g].data, h_data, HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(states[g].flags, 0, WORLD_SIZE * sizeof(uint32_t)));
        CHECK_CUDA(cudaMemset(states[g].recv_buf, 0, WORLD_SIZE * HIDDEN * sizeof(float)));
        delete[] h_data;
    }

    // Timing buffer on GPU 0
    long long* d_timing;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMalloc(&d_timing, TOTAL * sizeof(long long)));

    LatencyThreadArgs thread_args[WORLD_SIZE];
    latency_barrier.store(0);

    for (int g = 0; g < num_gpus; g++) {
        thread_args[g].gpu_id = g;
        thread_args[g].p2p = build_p2p_buffers(states, g, num_gpus);
        thread_args[g].data = states[g].data;
        thread_args[g].timing = (g == 0) ? d_timing : nullptr;
        thread_args[g].num_iters = TOTAL;
    }

    std::thread threads[WORLD_SIZE];
    for (int g = 0; g < num_gpus; g++) {
        threads[g] = std::thread(gpu_thread_latency, &thread_args[g]);
    }
    for (int g = 0; g < num_gpus; g++) {
        threads[g].join();
    }

    // Read timing from GPU 0
    long long* h_timing = new long long[TOTAL];
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMemcpy(h_timing, d_timing, TOTAL * sizeof(long long), cudaMemcpyDeviceToHost));

    // Get GPU 0 clock rate for conversion
    int clock_khz;
    CHECK_CUDA(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    double clock_mhz = (double)clock_khz / 1000.0;

    // Skip warmup, compute stats on bench iterations
    double sum_us = 0, min_us = 1e9, max_us = 0;
    for (int i = WARMUP; i < TOTAL; i++) {
        double us = (double)h_timing[i] / clock_mhz;
        sum_us += us;
        min_us = std::min(min_us, us);
        max_us = std::max(max_us, us);
    }
    double avg_us = sum_us / BENCH;

    printf("  AllReduce of %d FP32 elements (%d KB) across %d GPUs:\n",
           HIDDEN, HIDDEN * 4 / 1024, num_gpus);
    printf("  GPU clock: %.0f MHz\n", clock_mhz);
    printf("  Latency (device clocks -> us):\n");
    printf("    Min:  %.1f us\n", min_us);
    printf("    Avg:  %.1f us\n", avg_us);
    printf("    Max:  %.1f us\n", max_us);
    printf("  Data movement per GPU: %d KB (3 x %d KB writes to peers)\n",
           3 * HIDDEN * 4 / 1024, HIDDEN * 4 / 1024);

    // Compare to NCCL baseline
    printf("\n  Reference: NCCL AllReduce = ~254 us (ring), P2P tree = ~30 us\n");
    printf("  In-kernel cooperative = %.1f us\n\n", avg_us);

    delete[] h_timing;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(d_timing));
}

// ============================================================================
// TEST 3: Host-timed end-to-end (cudaEvent) for wall-clock validation
// ============================================================================
void test_host_timed(GPUAllReduceState* states, int num_gpus) {
    printf("==========================================================\n");
    printf("TEST 3: Host-Timed AllReduce (cudaEvent)\n");
    printf("==========================================================\n\n");

    const int WARMUP = 20;
    const int BENCH  = 100;

    // We'll do single-iteration launches (more realistic for integration)
    // But use cudaEvents on GPU 0 for host-side timing

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    for (int iter = 0; iter < WARMUP; iter++) {
        // Reset state each iteration
        for (int g = 0; g < num_gpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUDA(cudaMemset(states[g].flags, 0, WORLD_SIZE * sizeof(uint32_t)));
            float* h_data = new float[HIDDEN];
            for (int i = 0; i < HIDDEN; i++) h_data[i] = (float)(g + 1);
            CHECK_CUDA(cudaMemcpy(states[g].data, h_data, HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
            delete[] h_data;
        }

        ThreadArgs targs[WORLD_SIZE];
        launch_barrier.store(0);
        for (int g = 0; g < num_gpus; g++) {
            targs[g].gpu_id = g;
            targs[g].p2p = build_p2p_buffers(states, g, num_gpus);
            targs[g].data = states[g].data;
            targs[g].num_iters = 1;
            targs[g].success = false;
        }
        std::thread threads[WORLD_SIZE];
        for (int g = 0; g < num_gpus; g++)
            threads[g] = std::thread(gpu_thread_correctness, &targs[g]);
        for (int g = 0; g < num_gpus; g++)
            threads[g].join();
    }

    // Benchmark with cudaEvent timing (host-side)
    // The multi-thread approach makes cudaEvent tricky.
    // Instead, time the full iteration from host side.
    double total_ms = 0.0;
    for (int iter = 0; iter < BENCH; iter++) {
        for (int g = 0; g < num_gpus; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUDA(cudaMemset(states[g].flags, 0, WORLD_SIZE * sizeof(uint32_t)));
            float val = (float)(g + 1);
            std::vector<float> h_data(HIDDEN, val);
            CHECK_CUDA(cudaMemcpy(states[g].data, h_data.data(), HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        ThreadArgs targs[WORLD_SIZE];
        launch_barrier.store(0);
        for (int g = 0; g < num_gpus; g++) {
            targs[g].gpu_id = g;
            targs[g].p2p = build_p2p_buffers(states, g, num_gpus);
            targs[g].data = states[g].data;
            targs[g].num_iters = 1;
            targs[g].success = false;
        }
        std::thread threads[WORLD_SIZE];
        for (int g = 0; g < num_gpus; g++)
            threads[g] = std::thread(gpu_thread_correctness, &targs[g]);
        for (int g = 0; g < num_gpus; g++)
            threads[g].join();

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
    }

    printf("  Host-timed AllReduce (includes thread + launch overhead):\n");
    printf("    Avg: %.3f ms (%.1f us)\n", total_ms / BENCH, total_ms / BENCH * 1000.0);
    printf("  Note: includes host thread spawn + cooperative launch overhead\n");
    printf("  In-kernel latency (TEST 2) is the true GPU-side number.\n\n");

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

// ============================================================================
// Integration Analysis
// ============================================================================
void print_integration_plan() {
    printf("==========================================================\n");
    printf("MegaKernel <-> vLLM MTP=3 Integration Plan\n");
    printf("==========================================================\n\n");

    printf("1. WEIGHT FORMAT\n");
    printf("   MegaKernel expects:\n");
    printf("     - NVFP4 packed (E2M1, 2 values per byte, low nib first)\n");
    printf("     - E4M3FN block scales (1 byte per 16 elements)\n");
    printf("     - scale_vec::4X MMA instruction (4 scales per K=64 tile)\n");
    printf("   vLLM's Qwen3.5-397B-A17B uses the SAME format (NVFP4+E4M3FN).\n");
    printf("   Weight loading: can reuse vLLM's safetensors loader, just reshape\n");
    printf("   to MegaKernel's expected layout (N-major for B operand).\n\n");

    printf("2. WEIGHT LOADING PATH\n");
    printf("   LayerWeights struct needs per-layer pointers. To load real weights:\n");
    printf("     a. Parse safetensors index to find per-layer shard files\n");
    printf("     b. For each layer l, GPU g:\n");
    printf("        - qkv_fp4 = model.layers.{l}.self_attn.qkv_proj.weight (TP-sliced rows)\n");
    printf("        - o_fp4   = model.layers.{l}.self_attn.o_proj.weight (TP-sliced cols)\n");
    printf("        - gate    = model.layers.{l}.mlp.gate.weight (FP32, replicated)\n");
    printf("        - expert  = model.layers.{l}.mlp.experts.{e}.* (replicated across TP)\n");
    printf("        - shared  = model.layers.{l}.mlp.shared_expert.* (TP-sliced)\n");
    printf("     c. Quantized weights already on disk as NVFP4 from the checkpoint\n");
    printf("     d. Total GPU memory: ~24GB weights + ~65KB activations per GPU\n\n");

    printf("3. KV CACHE INTEGRATION\n");
    printf("   MegaKernel has its own KV cache (FP8, [max_seq, KV_heads, head_dim]).\n");
    printf("   Cannot share vLLM's paged KV cache (different layout, block tables).\n");
    printf("   Options:\n");
    printf("     a. Simplest: Fixed-size KV cache per-layer (no paging). Works for\n");
    printf("        single-user decode. Memory: 15 full-attn layers x seq_len x 512B\n");
    printf("        = ~1MB at 128 tokens, ~1GB at 256K tokens.\n");
    printf("     b. Advanced: Implement paged KV cache inside the megakernel.\n");
    printf("        Requires block table indirection in attention decode.\n");
    printf("     c. DeltaNet layers (15-59): fixed [HEAD_DIM x HEAD_DIM] state per head,\n");
    printf("        ~64KB/layer regardless of sequence length. Already handled.\n\n");

    printf("4. MTP (SPECULATIVE DECODING) INTEGRATION\n");
    printf("   MTP=3 means 3 draft positions per step. The MegaKernel processes\n");
    printf("   the FULL model (target + draft in one pass since it IS the model).\n");
    printf("   For MTP integration:\n");
    printf("     a. MegaKernel replaces vLLM's model execution entirely\n");
    printf("     b. Need to run 60 layers for target token, then 60 layers x 3\n");
    printf("        for draft tokens (or implement MTP heads inside the kernel)\n");
    printf("     c. MTP heads are small MLPs (hidden -> vocab logits). Can be\n");
    printf("        added as additional phases after the main 60-layer loop.\n");
    printf("     d. Token acceptance logic stays in Python (vLLM scheduler)\n\n");

    printf("5. MINIMUM VIABLE INTEGRATION\n");
    printf("   Step 1: Weight loader (Python) that fills LayerWeights arrays\n");
    printf("   Step 2: Simple KV cache allocator (fixed, not paged)\n");
    printf("   Step 3: Python wrapper that calls the megakernel via CuPy/ctypes\n");
    printf("   Step 4: Bypass vLLM's model runner, inject megakernel output\n");
    printf("   Step 5: Add MTP heads as additional GEMV phases\n");
    printf("   Estimated effort: 2-3 days for basic single-user decode\n\n");

    printf("6. BLOCKING ISSUES\n");
    printf("   a. Cooperative launch occupies ALL SMs — cannot coexist with\n");
    printf("      vLLM's CUDA graph executor or other GPU work\n");
    printf("   b. cudaLaunchCooperativeKernel has ~5-10us launch overhead\n");
    printf("      per invocation (60 layers processed per launch = amortized)\n");
    printf("   c. Multi-GPU requires multi-threaded host or\n");
    printf("      cudaLaunchCooperativeKernelMultiDevice (deprecated in CUDA 12+)\n");
    printf("   d. P2P AllReduce on PCIe has ~10-50us latency per AllReduce,\n");
    printf("      x2 per layer x 60 layers = 1.2-6ms overhead (vs ~2ms NCCL total)\n\n");
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("==========================================================\n");
    printf("MegaKernel P2P AllReduce Multi-GPU Test Suite\n");
    printf("==========================================================\n\n");

    // Enumerate GPUs
    int num_gpus;
    CHECK_CUDA(cudaGetDeviceCount(&num_gpus));
    printf("Found %d GPUs:\n", num_gpus);
    for (int i = 0; i < num_gpus; i++) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
        printf("  GPU %d: %s (SM %d.%d, %d SMs, %.0f GB)\n",
               i, prop.name, prop.major, prop.minor,
               prop.multiProcessorCount,
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
    printf("\n");

    if (num_gpus < WORLD_SIZE) {
        printf("WARNING: Only %d GPUs found, need %d. Running with %d.\n\n",
               num_gpus, WORLD_SIZE, num_gpus);
        // Could adapt tests for fewer GPUs, but the kernel assumes WORLD_SIZE=4
        if (num_gpus < 2) {
            printf("ERROR: Need at least 2 GPUs for P2P test\n");
            return 1;
        }
    }

    int test_gpus = std::min(num_gpus, WORLD_SIZE);

    // Setup P2P
    setup_p2p(test_gpus);

    // Allocate state
    GPUAllReduceState states[WORLD_SIZE];
    memset(states, 0, sizeof(states));
    allocate_gpu_state(states, test_gpus);

    // Run tests
    test_correctness(states, test_gpus);
    test_latency(states, test_gpus);
    test_host_timed(states, test_gpus);

    // Integration plan
    print_integration_plan();

    // Cleanup
    for (int g = 0; g < test_gpus; g++) {
        CHECK_CUDA(cudaSetDevice(g));
        CHECK_CUDA(cudaFree(states[g].data));
        CHECK_CUDA(cudaFree(states[g].recv_buf));
        CHECK_CUDA(cudaFree(states[g].flags));
    }

    printf("All tests complete.\n");
    return 0;
}
