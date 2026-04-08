#include <cstdint>
/**
 * Write-Based P2P AllReduce for 4 PCIe GPUs (SM 12.0 Blackwell)
 *
 * KEY INSIGHT: PCIe reads are 300-500ns per cacheline (request-response round-trip).
 * PCIe writes are POSTED transactions (fire-and-forget, pipelined at link bandwidth).
 * NCCL's LL protocol achieves ~14us by using ALL WRITES + LOCAL flag polling.
 *
 * This kernel implements the same pattern:
 *   - Cross-GPU data transfer: float4 stores to remote BAR-mapped memory (posted writes)
 *   - Synchronization: write flags to remote memory, poll LOCAL flags
 *   - __threadfence_system() for cross-PCIe write ordering
 *   - NO cross-GPU reads anywhere in the critical path
 *
 * Algorithm (ReduceScatter + AllGather):
 *   Phase 1 - ReduceScatter: Each GPU writes its chunk to every peer's staging buffer.
 *             Peer polls local flag, reduces all arrived chunks locally.
 *   Phase 2 - AllGather: Each GPU writes its reduced chunk to every peer's output.
 *             Peer polls local flag to know the result is ready.
 *
 * For 4 GPUs reducing N BF16 elements (N divisible by 4):
 *   Each GPU owns chunk of N/4 elements.
 *   Phase 1: 3 remote writes per GPU (12 total, all parallel). Local reduce after arrival.
 *   Phase 2: 3 remote writes per GPU (12 total, all parallel). No reduce needed.
 *
 * Launch barrier: pure posted-write barrier (no system-scope atomics, which are
 * broken on SM 12.0 Blackwell for cross-GPU targets). Each GPU writes its arrival
 * flag to every peer's local flag array, then polls its own local array.
 *
 * Compile: nvcc -O2 -arch=sm_120a -std=c++17 write_based_allreduce.cu -o write_based_allreduce
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>

#define NUM_GPUS 4
#define WARMUP_ITERS 200
#define BENCH_ITERS 1000

// Check CUDA errors
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ============================================================================
// Buffer layout per GPU for the allreduce protocol
// ============================================================================
// For each GPU g, we allocate a single contiguous region:
//   [staging0 | staging1 | staging2 | flags_recv(3) | flags_done(3) | output(N)]
//
// staging: 3 receive slots (one per peer), each chunk_size BF16 elements
// flags_recv: 3 uint32 arrival flags (written by remote peer, polled locally)
// flags_done: 3 uint32 completion flags for allgather phase
// output: N BF16 elements, the final allreduced result
//
// With P2P enabled, each GPU can directly store to any other GPU's buffer
// via CUDA UVA. Stores to remote memory become PCIe posted writes.

// Compute total allocation size for one GPU
static size_t compute_alloc_size(int N) {
    int chunk = N / NUM_GPUS;
    size_t staging_bytes = 3 * chunk * sizeof(__nv_bfloat16);
    size_t flags_bytes = 2 * 3 * sizeof(uint32_t);  // recv + done flags
    size_t output_bytes = N * sizeof(__nv_bfloat16);
    // Align each section to 256 bytes
    staging_bytes = (staging_bytes + 255) & ~255ULL;
    flags_bytes = (flags_bytes + 255) & ~255ULL;
    output_bytes = (output_bytes + 255) & ~255ULL;
    return staging_bytes + flags_bytes + output_bytes;
}

// Helper: get staging pointer for a given base address and slot index
__device__ __forceinline__
__nv_bfloat16* get_staging(void* base, int slot, int chunk) {
    return (__nv_bfloat16*)((char*)base + (size_t)slot * chunk * sizeof(__nv_bfloat16));
}

// Helper: get flags_recv pointer for a given base address
__device__ __forceinline__
volatile uint32_t* get_flags_recv(void* base, int N) {
    int chunk = N / NUM_GPUS;
    size_t staging_bytes = 3 * chunk * sizeof(__nv_bfloat16);
    staging_bytes = (staging_bytes + 255) & ~255ULL;
    return (volatile uint32_t*)((char*)base + staging_bytes);
}

// Helper: get flags_done pointer for a given base address
__device__ __forceinline__
volatile uint32_t* get_flags_done(void* base, int N) {
    int chunk = N / NUM_GPUS;
    size_t staging_bytes = 3 * chunk * sizeof(__nv_bfloat16);
    staging_bytes = (staging_bytes + 255) & ~255ULL;
    return (volatile uint32_t*)((char*)base + staging_bytes + 3 * sizeof(uint32_t));
}

// Helper: get output pointer for a given base address
__device__ __forceinline__
__nv_bfloat16* get_output(void* base, int N) {
    int chunk = N / NUM_GPUS;
    size_t staging_bytes = 3 * chunk * sizeof(__nv_bfloat16);
    staging_bytes = (staging_bytes + 255) & ~255ULL;
    size_t flags_bytes = (2 * 3 * sizeof(uint32_t) + 255) & ~255ULL;
    return (__nv_bfloat16*)((char*)base + staging_bytes + flags_bytes);
}

// ============================================================================
// Write-based AllReduce kernel
// ============================================================================
// Each GPU runs this kernel with 1 block (to avoid inter-CTA sync complexity).
// The block handles the full reduce-scatter + allgather for this GPU's rank.

__global__ void write_allreduce_kernel(
    const __nv_bfloat16* __restrict__ input,
    void* local_base,
    void* remote_base_0,
    void* remote_base_1,
    void* remote_base_2,
    void* remote_base_3,
    int rank,
    int N,
    uint32_t iteration,
    // Per-GPU barrier flag arrays: barrier_flags_X points to GPU X's local
    // array of NUM_GPUS uint32s. GPU i writes barrier_flags_j[i] = iteration
    // to tell GPU j "I'm here" (posted write). GPU j polls barrier_flags_j[*]
    // locally until all entries match iteration.
    volatile uint32_t* barrier_flags_0,
    volatile uint32_t* barrier_flags_1,
    volatile uint32_t* barrier_flags_2,
    volatile uint32_t* barrier_flags_3)
{
    // Pack remote bases into array for indexing
    void* remote_bases[4];
    remote_bases[0] = remote_base_0;
    remote_bases[1] = remote_base_1;
    remote_bases[2] = remote_base_2;
    remote_bases[3] = remote_base_3;

    // Pack barrier flag arrays for indexing
    volatile uint32_t* barrier_flags[4];
    barrier_flags[0] = barrier_flags_0;
    barrier_flags[1] = barrier_flags_1;
    barrier_flags[2] = barrier_flags_2;
    barrier_flags[3] = barrier_flags_3;

    const int chunk = N / NUM_GPUS;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Number of float4 (16 bytes = 8 BF16) elements per chunk
    const int chunk_f4 = chunk / 8;  // chunk in float4 units

    // ========================================================================
    // Launch barrier: pure posted-write, no atomics
    // ========================================================================
    // Each GPU writes its arrival to ALL peers' local flag arrays (posted writes),
    // then polls its OWN local flag array until all NUM_GPUS entries match iteration.
    // This avoids system-scope atomics which are broken on SM 12.0 Blackwell
    // for cross-GPU memory targets.
    if (tid == 0) {
        // Signal arrival to all GPUs (including self)
        #pragma unroll
        for (int g = 0; g < NUM_GPUS; g++) {
            barrier_flags[g][rank] = iteration;
        }
        // Poll our own local flags until all GPUs have arrived
        #pragma unroll
        for (int g = 0; g < NUM_GPUS; g++) {
            while (barrier_flags[rank][g] != iteration) {
                // spin on LOCAL memory — no PCIe traffic
            }
        }
    }
    __syncthreads();

    // ========================================================================
    // Phase 1: ReduceScatter via posted writes
    // ========================================================================
    // Each GPU writes its data to every peer's staging buffer.
    // For peer p, write input[p*chunk .. (p+1)*chunk-1] to peer's staging[slot]
    // where slot = rank < p ? rank : rank - 1

    for (int peer = 0; peer < NUM_GPUS; peer++) {
        if (peer == rank) continue;

        int slot = rank < peer ? rank : rank - 1;
        __nv_bfloat16* dst = get_staging(remote_bases[peer], slot, chunk);
        const __nv_bfloat16* src = input + peer * chunk;

        // Copy via float4 posted writes to remote BAR memory
        const float4* src4 = (const float4*)src;
        float4* dst4 = (float4*)dst;
        for (int i = tid; i < chunk_f4; i += nthreads) {
            dst4[i] = src4[i];
        }
    }

    // Ensure all writes are visible across PCIe before signaling
    __threadfence_system();

    // Write arrival flags to each peer (posted write)
    if (tid == 0) {
        for (int peer = 0; peer < NUM_GPUS; peer++) {
            if (peer == rank) continue;
            int slot = rank < peer ? rank : rank - 1;
            volatile uint32_t* peer_flags = get_flags_recv(remote_bases[peer], N);
            // Posted write: signal arrival
            peer_flags[slot] = iteration;
        }
    }

    // Wait for all 3 peers to write their data to OUR staging buffers
    // Poll LOCAL flags (no remote reads!)
    if (tid == 0) {
        volatile uint32_t* my_flags = get_flags_recv(local_base, N);
        for (int slot = 0; slot < 3; slot++) {
            while (my_flags[slot] != iteration) {
                // Spin on local memory - no PCIe traffic
            }
        }
    }
    __syncthreads();

    // ========================================================================
    // Local reduce: our chunk = input[rank*chunk..] + staging[0..2]
    // ========================================================================

    __nv_bfloat16* my_output = get_output(local_base, N);
    const __nv_bfloat16* my_input_chunk = input + rank * chunk;

    for (int i = tid; i < chunk; i += nthreads) {
        float acc = __bfloat162float(my_input_chunk[i]);
        #pragma unroll 3
        for (int s = 0; s < 3; s++) {
            __nv_bfloat16* stg = get_staging(local_base, s, chunk);
            acc += __bfloat162float(stg[i]);
        }
        my_output[rank * chunk + i] = __float2bfloat16(acc);
    }
    __syncthreads();

    // ========================================================================
    // Phase 2: AllGather via posted writes
    // ========================================================================

    __nv_bfloat16* my_reduced_chunk = my_output + rank * chunk;

    for (int peer = 0; peer < NUM_GPUS; peer++) {
        if (peer == rank) continue;

        __nv_bfloat16* dst = get_output(remote_bases[peer], N) + rank * chunk;
        const float4* src4 = (const float4*)my_reduced_chunk;
        float4* dst4 = (float4*)dst;

        for (int i = tid; i < chunk_f4; i += nthreads) {
            dst4[i] = src4[i];
        }
    }

    // Ensure allgather writes are visible
    __threadfence_system();

    // Signal allgather completion to peers
    if (tid == 0) {
        for (int peer = 0; peer < NUM_GPUS; peer++) {
            if (peer == rank) continue;
            int slot = rank < peer ? rank : rank - 1;
            volatile uint32_t* peer_done = get_flags_done(remote_bases[peer], N);
            peer_done[slot] = iteration;
        }
    }

    // Wait for all peers' allgather chunks to arrive
    if (tid == 0) {
        volatile uint32_t* my_done = get_flags_done(local_base, N);
        for (int slot = 0; slot < 3; slot++) {
            while (my_done[slot] != iteration) {
                // Spin on local memory
            }
        }
    }
    __syncthreads();

    // Output is now complete: my_output[0..N-1] has the full reduced result
}


// ============================================================================
// Host-side setup and benchmark
// ============================================================================

struct AllReduceContext {
    int N;  // total BF16 elements
    size_t alloc_size;

    // Per-GPU allocations (on each GPU's own memory)
    void* gpu_alloc[NUM_GPUS];          // contiguous allreduce buffer per GPU
    __nv_bfloat16* input_buf[NUM_GPUS]; // input data per GPU
    cudaStream_t streams[NUM_GPUS];

    // Per-GPU barrier flag arrays: barrier_flags[g] points to an array of
    // NUM_GPUS uint32s allocated on GPU g's memory.
    uint32_t* barrier_flags[NUM_GPUS];

    void init(int n) {
        N = n;
        alloc_size = compute_alloc_size(N);

        // Enable P2P and allocate
        for (int i = 0; i < NUM_GPUS; i++) {
            CHECK_CUDA(cudaSetDevice(i));
            for (int j = 0; j < NUM_GPUS; j++) {
                if (i != j) {
                    cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                    if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                        fprintf(stderr, "P2P access failed %d->%d: %s\n",
                                i, j, cudaGetErrorString(err));
                        exit(1);
                    }
                }
            }

            // Alloc contiguous buffer for allreduce protocol
            CHECK_CUDA(cudaMalloc(&gpu_alloc[i], alloc_size));
            CHECK_CUDA(cudaMemset(gpu_alloc[i], 0, alloc_size));

            // Alloc input buffer
            CHECK_CUDA(cudaMalloc(&input_buf[i], N * sizeof(__nv_bfloat16)));

            // Alloc barrier flag array (NUM_GPUS uint32s, on this GPU's memory)
            CHECK_CUDA(cudaMalloc(&barrier_flags[i], NUM_GPUS * sizeof(uint32_t)));
            CHECK_CUDA(cudaMemset(barrier_flags[i], 0, NUM_GPUS * sizeof(uint32_t)));

            CHECK_CUDA(cudaStreamCreate(&streams[i]));
        }

        printf("AllReduce context initialized:\n");
        printf("  N = %d BF16 elements (%d bytes = %.1f KB)\n",
               N, N * 2, N * 2 / 1024.0);
        printf("  Alloc per GPU: %zu bytes (%.1f KB)\n", alloc_size, alloc_size / 1024.0);
        printf("  Chunk: %d elements (%d bytes)\n", N / NUM_GPUS, N / NUM_GPUS * 2);
        printf("\n");
    }

    void fill_input(float* rank_values) {
        for (int g = 0; g < NUM_GPUS; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            std::vector<__nv_bfloat16> host_data(N);
            for (int i = 0; i < N; i++) {
                host_data[i] = __float2bfloat16(rank_values[g]);
            }
            CHECK_CUDA(cudaMemcpy(input_buf[g], host_data.data(),
                                  N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        }
    }

    void reset_flags() {
        // Zero out all flags and barrier arrays
        for (int g = 0; g < NUM_GPUS; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            int chunk = N / NUM_GPUS;
            size_t staging_bytes = 3 * chunk * sizeof(__nv_bfloat16);
            staging_bytes = (staging_bytes + 255) & ~255ULL;
            CHECK_CUDA(cudaMemset((char*)gpu_alloc[g] + staging_bytes, 0, 256));
            // Reset barrier flags (no need for cross-GPU sync — each is on own GPU)
            CHECK_CUDA(cudaMemset(barrier_flags[g], 0, NUM_GPUS * sizeof(uint32_t)));
        }
        // Synchronize all GPUs to ensure resets are complete before kernel launch
        for (int g = 0; g < NUM_GPUS; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUDA(cudaDeviceSynchronize());
        }
    }

    void launch_allreduce(uint32_t iteration) {
        int threads = 256;

        for (int g = 0; g < NUM_GPUS; g++) {
            CHECK_CUDA(cudaSetDevice(g));

            write_allreduce_kernel<<<1, threads, 0, streams[g]>>>(
                input_buf[g],
                gpu_alloc[g],
                gpu_alloc[0], gpu_alloc[1], gpu_alloc[2], gpu_alloc[3],
                g,
                N,
                iteration,
                (volatile uint32_t*)barrier_flags[0],
                (volatile uint32_t*)barrier_flags[1],
                (volatile uint32_t*)barrier_flags[2],
                (volatile uint32_t*)barrier_flags[3]
            );
        }
    }

    void sync_all() {
        for (int g = 0; g < NUM_GPUS; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUDA(cudaStreamSynchronize(streams[g]));
        }
    }

    __nv_bfloat16* host_get_output(void* base) {
        int chunk = N / NUM_GPUS;
        size_t staging_bytes = 3 * chunk * sizeof(__nv_bfloat16);
        staging_bytes = (staging_bytes + 255) & ~255ULL;
        size_t flags_bytes = (2 * 3 * sizeof(uint32_t) + 255) & ~255ULL;
        return (__nv_bfloat16*)((char*)base + staging_bytes + flags_bytes);
    }

    void read_output(int g, __nv_bfloat16* host_out) {
        CHECK_CUDA(cudaSetDevice(g));
        __nv_bfloat16* dev_output = host_get_output(gpu_alloc[g]);
        CHECK_CUDA(cudaMemcpy(host_out, dev_output, N * sizeof(__nv_bfloat16),
                              cudaMemcpyDeviceToHost));
    }

    void cleanup() {
        for (int g = 0; g < NUM_GPUS; g++) {
            CHECK_CUDA(cudaSetDevice(g));
            CHECK_CUDA(cudaFree(gpu_alloc[g]));
            CHECK_CUDA(cudaFree(input_buf[g]));
            CHECK_CUDA(cudaFree(barrier_flags[g]));
            CHECK_CUDA(cudaStreamDestroy(streams[g]));
        }
    }
};


// ============================================================================
// Correctness validation
// ============================================================================

bool validate(AllReduceContext& ctx, float expected_sum) {
    bool pass = true;
    std::vector<__nv_bfloat16> host_out(ctx.N);

    for (int g = 0; g < NUM_GPUS; g++) {
        ctx.read_output(g, host_out.data());

        int errors = 0;
        for (int i = 0; i < ctx.N; i++) {
            float val = __bfloat162float(host_out[i]);
            if (fabsf(val - expected_sum) > 0.5f) {
                if (errors < 5) {
                    printf("  GPU %d, elem %d: got %.2f, expected %.2f\n",
                           g, i, val, expected_sum);
                }
                errors++;
            }
        }
        if (errors > 0) {
            printf("  GPU %d: %d/%d errors\n", g, errors, ctx.N);
            pass = false;
        }
    }
    return pass;
}


// ============================================================================
// Benchmark
// ============================================================================

void benchmark(AllReduceContext& ctx, int N, const char* label) {
    printf("--- %s (N=%d, %d bytes, %.1f KB) ---\n", label, N, N * 2, N * 2 / 1024.0);

    float rank_vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float expected = 10.0f;  // 1+2+3+4

    // Correctness test
    ctx.fill_input(rank_vals);
    ctx.reset_flags();
    ctx.launch_allreduce(1);
    ctx.sync_all();

    bool ok = validate(ctx, expected);
    printf("  Correctness: %s\n", ok ? "PASS" : "FAIL");
    if (!ok) {
        printf("  Skipping benchmark due to correctness failure.\n\n");
        return;
    }

    // Warmup — no need to reset flags between iterations since iteration
    // counter is monotonically increasing and flags are checked with ==.
    // Input data is the same every time (rank values don't change).
    for (uint32_t i = 2; i < 2 + WARMUP_ITERS; i++) {
        ctx.launch_allreduce(i);
        ctx.sync_all();
    }

    // Timed runs
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float times[BENCH_ITERS];
    uint32_t iter_base = 2 + WARMUP_ITERS;

    for (int i = 0; i < BENCH_ITERS; i++) {
        // No reset needed — monotonic iteration counter handles flag matching.
        // No fill_input needed — same data every time.
        ctx.sync_all();

        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaEventRecord(start, ctx.streams[0]));

        ctx.launch_allreduce(iter_base + i);

        CHECK_CUDA(cudaSetDevice(0));
        CHECK_CUDA(cudaEventRecord(stop, ctx.streams[0]));

        ctx.sync_all();

        CHECK_CUDA(cudaEventElapsedTime(&times[i], start, stop));
    }

    // Statistics (skip first 50 of bench for further warmup)
    int skip = 50;
    float sum = 0;
    float min_t = 1e9, max_t = 0;
    for (int i = skip; i < BENCH_ITERS; i++) {
        sum += times[i];
        if (times[i] < min_t) min_t = times[i];
        if (times[i] > max_t) max_t = times[i];
    }
    int count = BENCH_ITERS - skip;
    float avg_us = (sum / count) * 1000.0f;

    // Sort for percentiles
    for (int i = skip; i < BENCH_ITERS; i++) {
        for (int j = i + 1; j < BENCH_ITERS; j++) {
            if (times[j] < times[i]) {
                float t = times[i];
                times[i] = times[j];
                times[j] = t;
            }
        }
    }
    float p50 = times[skip + count / 2] * 1000.0f;
    float p1  = times[skip + count / 100] * 1000.0f;
    float p99 = times[skip + count * 99 / 100] * 1000.0f;

    printf("  Latency (us):  avg=%.1f  p1=%.1f  p50=%.1f  p99=%.1f  min=%.1f  max=%.1f\n",
           avg_us, p1, p50, p99, min_t * 1000, max_t * 1000);
    printf("  vs NCCL p50 ~14us:  %.2fx %s\n",
           p50 < 14.0f ? 14.0f / p50 : p50 / 14.0f,
           p50 < 14.0f ? "faster" : "slower");
    printf("\n");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}


// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    printf("=================================================================\n");
    printf("Write-Based P2P AllReduce Benchmark (4x RTX PRO 6000 Blackwell)\n");
    printf("=================================================================\n");
    printf("Protocol: ReduceScatter + AllGather, all posted writes, local polling\n");
    printf("Target: <10us for 8KB (hidden_size=4096 BF16)\n\n");

    int dev_count;
    CHECK_CUDA(cudaGetDeviceCount(&dev_count));
    if (dev_count < NUM_GPUS) {
        fprintf(stderr, "Need %d GPUs, found %d\n", NUM_GPUS, dev_count);
        return 1;
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
        printf("GPU %d: %s (SM %d.%d, %d SMs)\n",
               i, prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    }

    printf("\nP2P access matrix:\n");
    for (int i = 0; i < NUM_GPUS; i++) {
        printf("  GPU %d:", i);
        for (int j = 0; j < NUM_GPUS; j++) {
            if (i == j) { printf(" self"); continue; }
            int can;
            CHECK_CUDA(cudaDeviceCanAccessPeer(&can, i, j));
            printf(" %d->%d:%s", i, j, can ? "OK" : "NO");
            if (!can) {
                fprintf(stderr, "\nFATAL: P2P not available %d->%d\n", i, j);
                return 1;
            }
        }
        printf("\n");
    }
    printf("\n");

    // Benchmark various sizes
    int sizes[] = {4096, 8192, 16384, 32768};
    const char* labels[] = {"4096 BF16 = 8KB", "8192 BF16 = 16KB",
                            "16384 BF16 = 32KB", "32768 BF16 = 64KB"};

    for (int s = 0; s < 4; s++) {
        AllReduceContext ctx;
        ctx.init(sizes[s]);
        benchmark(ctx, sizes[s], labels[s]);
        ctx.cleanup();
    }

    printf("=================================================================\n");
    printf("NCCL baseline reference (from Sprint 11 measurements):\n");
    printf("  8KB:  p50=13.9us  p99=8779us\n");
    printf("  32KB: p50=15.4us  p99=340us\n");
    printf("  64KB: p50=15.7us  p99=19.7us\n");
    printf("=================================================================\n");

    return 0;
}
