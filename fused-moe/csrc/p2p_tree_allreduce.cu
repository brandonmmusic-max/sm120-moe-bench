/**
 * P2P Tree AllReduce for 4 PCIe GPUs
 *
 * Replaces NCCL's ring AllReduce (254μs) with a tree-based P2P pattern (~30μs).
 *
 * Algorithm (4 GPUs, BF16, 4096 elements = 8KB):
 *   Round 1 (parallel): GPU0 += GPU1, GPU2 += GPU3  (P2P read + local add)
 *   Round 2:            GPU0 += GPU2                 (P2P read + local add)
 *   Round 3 (parallel): GPU1 = GPU0, GPU3 = GPU2    (P2P copy broadcast)
 *   Round 4 (parallel): GPU2 = GPU0                  (broadcast to remaining)
 *
 * Uses cudaMemcpyPeerAsync + a small reduction kernel.
 * Designed for small messages (<64KB) where NCCL's ring protocol is overkill.
 *
 * Compile: nvcc -O2 -arch=sm_120a p2p_tree_allreduce.cu -o p2p_tree_allreduce
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>

// Reduction kernel: out[i] = a[i] + b[i]
__global__ void bf16_add_kernel(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __bfloat162float(a[idx]);
        float vb = __bfloat162float(b[idx]);
        out[idx] = __float2bfloat16(va + vb);
    }
}

// Inplace reduction: a[i] += b[i]
__global__ void bf16_add_inplace_kernel(
    __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __bfloat162float(a[idx]);
        float vb = __bfloat162float(b[idx]);
        a[idx] = __float2bfloat16(va + vb);
    }
}


struct P2PTreeAllreduce {
    int world_size;
    int hidden_size;  // number of BF16 elements

    // Staging buffers on each GPU (for receiving P2P data)
    __nv_bfloat16* staging[4];
    cudaStream_t streams[4];

    void init(int ws, int hs) {
        world_size = ws;
        hidden_size = hs;

        // Enable P2P access
        for (int i = 0; i < world_size; i++) {
            cudaSetDevice(i);
            for (int j = 0; j < world_size; j++) {
                if (i != j) {
                    cudaDeviceEnablePeerAccess(j, 0);
                }
            }
            cudaMalloc(&staging[i], hidden_size * sizeof(__nv_bfloat16));
            cudaStreamCreate(&streams[i]);
        }
    }

    void allreduce(__nv_bfloat16** bufs) {
        // bufs[i] = pointer on GPU i, each with hidden_size BF16 elements
        // Result: all bufs[i] contain the sum

        int threads = 256;
        int blocks = (hidden_size + threads - 1) / threads;

        // Round 1: GPU0 += GPU1, GPU2 += GPU3 (parallel)
        // Copy GPU1 data to GPU0's staging buffer
        cudaMemcpyPeerAsync(staging[0], 0, bufs[1], 1,
                            hidden_size * sizeof(__nv_bfloat16), streams[0]);
        cudaMemcpyPeerAsync(staging[2], 2, bufs[3], 3,
                            hidden_size * sizeof(__nv_bfloat16), streams[2]);

        // Wait for copies, then reduce
        cudaStreamSynchronize(streams[0]);
        cudaSetDevice(0);
        bf16_add_inplace_kernel<<<blocks, threads, 0, streams[0]>>>(
            bufs[0], staging[0], hidden_size);

        cudaStreamSynchronize(streams[2]);
        cudaSetDevice(2);
        bf16_add_inplace_kernel<<<blocks, threads, 0, streams[2]>>>(
            bufs[2], staging[2], hidden_size);

        // Round 2: GPU0 += GPU2
        cudaMemcpyPeerAsync(staging[0], 0, bufs[2], 2,
                            hidden_size * sizeof(__nv_bfloat16), streams[0]);
        cudaStreamSynchronize(streams[0]);
        cudaSetDevice(0);
        bf16_add_inplace_kernel<<<blocks, threads, 0, streams[0]>>>(
            bufs[0], staging[0], hidden_size);
        cudaStreamSynchronize(streams[0]);

        // Round 3: Broadcast GPU0 result to all
        // GPU0 → GPU1, GPU0 → GPU2, GPU0 → GPU3 (parallel)
        cudaMemcpyPeerAsync(bufs[1], 1, bufs[0], 0,
                            hidden_size * sizeof(__nv_bfloat16), streams[0]);
        cudaMemcpyPeerAsync(bufs[2], 2, bufs[0], 0,
                            hidden_size * sizeof(__nv_bfloat16), streams[0]);
        cudaMemcpyPeerAsync(bufs[3], 3, bufs[0], 0,
                            hidden_size * sizeof(__nv_bfloat16), streams[0]);
        cudaStreamSynchronize(streams[0]);
    }

    void cleanup() {
        for (int i = 0; i < world_size; i++) {
            cudaSetDevice(i);
            cudaFree(staging[i]);
            cudaStreamDestroy(streams[i]);
        }
    }
};


int main() {
    printf("P2P Tree AllReduce Benchmark (4 GPUs, 4096 BF16 = 8KB)\n\n");

    int N = 4096;  // hidden_size
    int world_size = 4;

    P2PTreeAllreduce ar;
    ar.init(world_size, N);

    // Allocate test buffers on each GPU
    __nv_bfloat16* bufs[4];
    for (int i = 0; i < world_size; i++) {
        cudaSetDevice(i);
        cudaMalloc(&bufs[i], N * sizeof(__nv_bfloat16));
        // Fill with rank value for verification
        __nv_bfloat16 val = __float2bfloat16((float)(i + 1));
        cudaMemset(bufs[i], 0, N * sizeof(__nv_bfloat16));
        // Set first element to rank+1 for verification
        cudaMemcpy(bufs[i], &val, sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    }

    // Warmup
    for (int i = 0; i < 50; i++) {
        // Reset values
        for (int g = 0; g < world_size; g++) {
            cudaSetDevice(g);
            __nv_bfloat16 val = __float2bfloat16((float)(g + 1));
            cudaMemcpy(bufs[g], &val, sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        }
        ar.allreduce(bufs);
    }
    cudaDeviceSynchronize();

    // Verify: sum should be 1+2+3+4=10
    __nv_bfloat16 result;
    cudaSetDevice(0);
    cudaMemcpy(&result, bufs[0], sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    float fresult = __bfloat162float(result);
    printf("Verification: sum = %.1f (expected 10.0) %s\n\n",
           fresult, fabsf(fresult - 10.0f) < 0.1f ? "OK" : "FAIL");

    // Benchmark
    cudaEvent_t start, stop;
    cudaSetDevice(0);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iters = 500;
    float total_ms = 0;
    float times[500];

    for (int i = 0; i < iters; i++) {
        // Reset
        for (int g = 0; g < world_size; g++) {
            cudaSetDevice(g);
            __nv_bfloat16 val = __float2bfloat16(1.0f);
            cudaMemset(bufs[g], 0, N * sizeof(__nv_bfloat16));
        }
        cudaDeviceSynchronize();

        cudaSetDevice(0);
        cudaEventRecord(start);
        ar.allreduce(bufs);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&times[i], start, stop);
    }

    // Skip warmup
    int skip = 50;
    float sum = 0;
    for (int i = skip; i < iters; i++) sum += times[i];
    float avg = sum / (iters - skip) * 1000;

    // Sort for median
    for (int i = skip; i < iters; i++)
        for (int j = i+1; j < iters; j++)
            if (times[j] < times[i]) { float t = times[i]; times[i] = times[j]; times[j] = t; }
    float med = times[skip + (iters-skip)/2] * 1000;

    printf("P2P Tree AllReduce (4 GPUs, %d BF16 = %dKB):\n", N, N*2/1024);
    printf("  avg = %.1f μs\n", avg);
    printf("  med = %.1f μs\n", med);
    printf("  vs NCCL AllReduce: 254 μs\n");
    printf("  speedup: %.1fx\n", 254.0f / med);
    printf("\n");
    printf("  120 calls/step × %.0fμs savings = %.1fms per step\n",
           254.0f - med, 120 * (254.0f - med) / 1000);
    printf("  At 4 tok/step: %.1fms per token saved\n",
           120 * (254.0f - med) / 1000 / 4);

    // Cleanup
    ar.cleanup();
    for (int i = 0; i < world_size; i++) {
        cudaSetDevice(i);
        cudaFree(bufs[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
