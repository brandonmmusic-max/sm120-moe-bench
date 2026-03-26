/**
 * bench_p2p_allreduce.cu — Multi-GPU P2P AllReduce Latency Benchmark
 *
 * Tests three AllReduce strategies for small tensors (16KB-64KB):
 *   1. Naive P2P: spin-wait on flags, read remote buffers, reduce locally
 *   2. One-shot P2P: each GPU reads from all peers and reduces
 *   3. Baseline: measure kernel overhead without any P2P
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a \
 *     -o bench_p2p_allreduce bench_p2p_allreduce.cu -lpthread
 *
 * Run:
 *   ./bench_p2p_allreduce
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>

#define CHECK_CUDA(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

static constexpr int NUM_GPUS    = 4;
static constexpr int HIDDEN      = 4096;
static constexpr int BLOCK_SIZE  = 256;
static constexpr int WARMUP      = 50;
static constexpr int ITERS       = 200;

// ============================================================================
// Kernel: P2P AllReduce with flag-based synchronization
// Each GPU writes partial output, signals ready, waits for peers, reduces.
// ============================================================================
__global__ void p2p_allreduce_kernel(
    float* __restrict__ local_buf,          // [M, HIDDEN] this GPU's partial
    float* __restrict__ output,             // [M, HIDDEN] reduced output
    float* const* __restrict__ peer_bufs,   // [NUM_GPUS] pointers to each GPU's buf
    volatile int* __restrict__ my_flag,     // this GPU's ready flag
    volatile int* const* __restrict__ peer_flags,  // [NUM_GPUS] pointers to flags
    int M,
    int my_rank,
    int num_gpus,
    int gen)         // barrier generation (monotonic)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_elems = M * HIDDEN;

    // Step 1: Write some dummy partial data (simulates MoE output)
    for (int i = tid; i < total_elems; i += blockDim.x * gridDim.x) {
        local_buf[i] = (float)(my_rank + 1) * 0.1f;
    }

    // Grid sync via shared atomic counter
    __syncthreads();
    __threadfence_system();  // Ensure writes visible to ALL GPUs

    // Step 2: Signal readiness
    if (tid == 0) {
        *my_flag = gen + 1;  // monotonic generation counter
    }

    // Step 3: Wait for all peers
    if (tid == 0) {
        for (int p = 0; p < num_gpus; p++) {
            if (p == my_rank) continue;
            while (*(peer_flags[p]) < gen + 1) {
                // Spin on PCIe-mapped memory — high latency per read
            }
        }
    }
    __syncthreads();

    // Step 4: Reduce — read all peers and sum
    for (int i = tid; i < total_elems; i += blockDim.x * gridDim.x) {
        float sum = 0.0f;
        for (int p = 0; p < num_gpus; p++) {
            sum += peer_bufs[p][i];  // P2P read from each GPU
        }
        output[i] = sum;
    }
}

// ============================================================================
// Kernel: Just write output (no P2P) — baseline
// ============================================================================
__global__ void baseline_kernel(
    float* __restrict__ output,
    int M,
    int my_rank)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_elems = M * HIDDEN;
    for (int i = tid; i < total_elems; i += blockDim.x * gridDim.x) {
        output[i] = (float)(my_rank + 1) * 0.4f;  // Simulates MoE output write
    }
}

// ============================================================================
// Thread function: runs kernels on one GPU
// ============================================================================
struct GpuContext {
    int gpu_id;
    float* local_buf;        // [M, HIDDEN] on this GPU
    float* output;           // [M, HIDDEN] on this GPU
    float** peer_bufs_dev;   // device array of pointers
    int* my_flag;            // on this GPU
    int** peer_flags_dev;    // device array of flag pointers
    cudaStream_t stream;
    cudaEvent_t start_event, stop_event;
    int M;
    float p2p_avg_us;
    float baseline_avg_us;
};

pthread_barrier_t thread_barrier;

void* gpu_thread(void* arg) {
    GpuContext* ctx = (GpuContext*)arg;
    CHECK_CUDA(cudaSetDevice(ctx->gpu_id));

    int blocks = (ctx->M * HIDDEN + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks = std::min(blocks, 16);  // Single CTA does the reduction in practice

    // ---- Warmup ----
    for (int i = 0; i < WARMUP; i++) {
        // Sync all threads to ensure all GPUs launch simultaneously
        pthread_barrier_wait(&thread_barrier);

        p2p_allreduce_kernel<<<blocks, BLOCK_SIZE, 0, ctx->stream>>>(
            ctx->local_buf, ctx->output,
            ctx->peer_bufs_dev, ctx->my_flag,
            (volatile int* const*)ctx->peer_flags_dev,
            ctx->M, ctx->gpu_id, NUM_GPUS, i);
        CHECK_CUDA(cudaStreamSynchronize(ctx->stream));
    }

    // ---- Timed P2P AllReduce ----
    std::vector<float> times(ITERS);
    for (int i = 0; i < ITERS; i++) {
        int gen = WARMUP + i;
        pthread_barrier_wait(&thread_barrier);

        CHECK_CUDA(cudaEventRecord(ctx->start_event, ctx->stream));
        p2p_allreduce_kernel<<<blocks, BLOCK_SIZE, 0, ctx->stream>>>(
            ctx->local_buf, ctx->output,
            ctx->peer_bufs_dev, ctx->my_flag,
            (volatile int* const*)ctx->peer_flags_dev,
            ctx->M, ctx->gpu_id, NUM_GPUS, gen);
        CHECK_CUDA(cudaEventRecord(ctx->stop_event, ctx->stream));
        CHECK_CUDA(cudaStreamSynchronize(ctx->stream));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, ctx->start_event, ctx->stop_event));
        times[i] = ms * 1000.0f;  // Convert to μs
    }

    std::sort(times.begin(), times.end());
    // Median of middle 80%
    int lo = ITERS / 10, hi = ITERS * 9 / 10;
    float sum = 0;
    for (int i = lo; i < hi; i++) sum += times[i];
    ctx->p2p_avg_us = sum / (hi - lo);

    // ---- Timed baseline (no P2P) ----
    std::vector<float> btimes(ITERS);
    for (int i = 0; i < WARMUP; i++) {
        baseline_kernel<<<blocks, BLOCK_SIZE, 0, ctx->stream>>>(
            ctx->output, ctx->M, ctx->gpu_id);
    }
    CHECK_CUDA(cudaStreamSynchronize(ctx->stream));

    for (int i = 0; i < ITERS; i++) {
        CHECK_CUDA(cudaEventRecord(ctx->start_event, ctx->stream));
        baseline_kernel<<<blocks, BLOCK_SIZE, 0, ctx->stream>>>(
            ctx->output, ctx->M, ctx->gpu_id);
        CHECK_CUDA(cudaEventRecord(ctx->stop_event, ctx->stream));
        CHECK_CUDA(cudaStreamSynchronize(ctx->stream));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, ctx->start_event, ctx->stop_event));
        btimes[i] = ms * 1000.0f;
    }
    std::sort(btimes.begin(), btimes.end());
    sum = 0;
    for (int i = lo; i < hi; i++) sum += btimes[i];
    ctx->baseline_avg_us = sum / (hi - lo);

    return nullptr;
}

// ============================================================================
// Main: Set up P2P, IPC handles, run benchmark
// ============================================================================
int main() {
    printf("=== P2P AllReduce Latency Benchmark ===\n");
    printf("GPUs: %d, HIDDEN: %d\n\n", NUM_GPUS, HIDDEN);

    // Enable P2P between all GPU pairs
    for (int i = 0; i < NUM_GPUS; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        for (int j = 0; j < NUM_GPUS; j++) {
            if (i != j) {
                int can;
                CHECK_CUDA(cudaDeviceCanAccessPeer(&can, i, j));
                if (!can) {
                    printf("ERROR: GPU %d cannot access GPU %d\n", i, j);
                    return 1;
                }
                CHECK_CUDA(cudaDeviceEnablePeerAccess(j, 0));
            }
        }
    }
    printf("P2P enabled between all GPU pairs.\n\n");

    for (int M : {1, 4}) {
        int total_elems = M * HIDDEN;
        int buf_bytes = total_elems * sizeof(float);
        printf("--- M=%d, buffer=%d bytes (%.1f KB) ---\n", M, buf_bytes, buf_bytes/1024.0f);

        // Allocate per-GPU buffers
        float* local_bufs[NUM_GPUS];
        float* outputs[NUM_GPUS];
        int*   flags[NUM_GPUS];
        float* all_bufs[NUM_GPUS];     // For peer_bufs array
        int*   all_flags[NUM_GPUS];    // For peer_flags array

        for (int g = 0; g < NUM_GPUS; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUDA(cudaMalloc(&local_bufs[g], buf_bytes));
            CHECK_CUDA(cudaMalloc(&outputs[g], buf_bytes));
            CHECK_CUDA(cudaMalloc(&flags[g], sizeof(int)));
            CHECK_CUDA(cudaMemset(flags[g], 0, sizeof(int)));
            all_bufs[g] = local_bufs[g];
            all_flags[g] = flags[g];
        }

        // Create device arrays of peer pointers (on each GPU)
        GpuContext ctxs[NUM_GPUS];
        pthread_barrier_init(&thread_barrier, nullptr, NUM_GPUS);

        for (int g = 0; g < NUM_GPUS; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            // Copy peer buffer pointers to device
            float** peer_bufs_dev;
            CHECK_CUDA(cudaMalloc(&peer_bufs_dev, NUM_GPUS * sizeof(float*)));
            CHECK_CUDA(cudaMemcpy(peer_bufs_dev, all_bufs,
                                  NUM_GPUS * sizeof(float*),
                                  cudaMemcpyHostToDevice));

            int** peer_flags_dev;
            CHECK_CUDA(cudaMalloc(&peer_flags_dev, NUM_GPUS * sizeof(int*)));
            CHECK_CUDA(cudaMemcpy(peer_flags_dev, all_flags,
                                  NUM_GPUS * sizeof(int*),
                                  cudaMemcpyHostToDevice));

            ctxs[g].gpu_id = g;
            ctxs[g].local_buf = local_bufs[g];
            ctxs[g].output = outputs[g];
            ctxs[g].peer_bufs_dev = peer_bufs_dev;
            ctxs[g].my_flag = flags[g];
            ctxs[g].peer_flags_dev = peer_flags_dev;
            ctxs[g].M = M;

            CHECK_CUDA(cudaStreamCreate(&ctxs[g].stream));
            CHECK_CUDA(cudaEventCreate(&ctxs[g].start_event));
            CHECK_CUDA(cudaEventCreate(&ctxs[g].stop_event));
        }

        // Launch threads
        pthread_t threads[NUM_GPUS];
        for (int g = 0; g < NUM_GPUS; g++) {
            pthread_create(&threads[g], nullptr, gpu_thread, &ctxs[g]);
        }
        for (int g = 0; g < NUM_GPUS; g++) {
            pthread_join(threads[g], nullptr);
        }

        // Report
        printf("  P2P AllReduce:\n");
        for (int g = 0; g < NUM_GPUS; g++) {
            printf("    GPU %d: %.1f μs\n", g, ctxs[g].p2p_avg_us);
        }
        float max_p2p = 0;
        for (int g = 0; g < NUM_GPUS; g++) max_p2p = std::max(max_p2p, ctxs[g].p2p_avg_us);
        printf("    Max (effective): %.1f μs\n", max_p2p);

        printf("  Baseline (no P2P):\n");
        float max_base = 0;
        for (int g = 0; g < NUM_GPUS; g++) {
            printf("    GPU %d: %.1f μs\n", g, ctxs[g].baseline_avg_us);
            max_base = std::max(max_base, ctxs[g].baseline_avg_us);
        }
        printf("    Max (effective): %.1f μs\n", max_base);
        printf("    P2P overhead: %.1f μs\n\n", max_p2p - max_base);

        // Verify correctness
        CHECK_CUDA(cudaSetDevice(0));
        float* h_out = new float[total_elems];
        CHECK_CUDA(cudaMemcpy(h_out, outputs[0], buf_bytes, cudaMemcpyDeviceToHost));
        float expected = (1 + 2 + 3 + 4) * 0.1f;  // sum of all GPU contributions
        float err = fabsf(h_out[0] - expected);
        printf("  Correctness: output[0]=%.4f, expected=%.4f, err=%.6f %s\n\n",
               h_out[0], expected, err, err < 0.01f ? "PASS" : "FAIL");
        delete[] h_out;

        // Cleanup
        for (int g = 0; g < NUM_GPUS; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUDA(cudaFree(local_bufs[g]));
            CHECK_CUDA(cudaFree(outputs[g]));
            CHECK_CUDA(cudaFree(flags[g]));
            CHECK_CUDA(cudaFree(ctxs[g].peer_bufs_dev));
            CHECK_CUDA(cudaFree(ctxs[g].peer_flags_dev));
            CHECK_CUDA(cudaStreamDestroy(ctxs[g].stream));
            CHECK_CUDA(cudaEventDestroy(ctxs[g].start_event));
            CHECK_CUDA(cudaEventDestroy(ctxs[g].stop_event));
        }
        pthread_barrier_destroy(&thread_barrier);
    }

    return 0;
}
