#!/usr/bin/env python3
"""
P2P All-Reduce CUDA kernel for 4 PCIe GPUs — CUDA graph compatible.

Uses a custom CUDA kernel that reads directly from peer GPU memory
via P2P mapped addresses. This IS graph-capturable because the kernel
runs on one GPU with fixed pointer arguments.

The key insight: after cudaDeviceEnablePeerAccess, GPU 0 can read GPU 1's
memory using GPU 1's device pointer. A kernel doing this is just a normal
kernel from the graph capture perspective.
"""

import torch
import torch.cuda
import time
from torch.utils.cpp_extension import load_inline

# CUDA kernel source
cuda_src = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Enable P2P access between all GPU pairs
void enable_peer_access() {
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
                        printf("WARNING: P2P %d->%d failed: %s\n", i, j, cudaGetErrorString(err));
                    }
                }
            }
        }
    }
}

// Kernel: read from 4 buffers (potentially on different GPUs), sum, write output
__global__ void p2p_allreduce_bf16_kernel(
    const __nv_bfloat16* __restrict__ buf0,
    const __nv_bfloat16* __restrict__ buf1,
    const __nv_bfloat16* __restrict__ buf2,
    const __nv_bfloat16* __restrict__ buf3,
    __nv_bfloat16* __restrict__ output,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        float v0 = __bfloat162float(buf0[idx]);
        float v1 = __bfloat162float(buf1[idx]);
        float v2 = __bfloat162float(buf2[idx]);
        float v3 = __bfloat162float(buf3[idx]);
        output[idx] = __float2bfloat16(v0 + v1 + v2 + v3);
    }
}

// Kernel: copy output to peer GPU buffer (broadcast via P2P write)
__global__ void p2p_broadcast_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        dst[idx] = src[idx];
    }
}

// Python-callable: allreduce on device 0, reading from peer buffers
// buf0..buf3 are tensors on cuda:0..cuda:3 respectively
// output is on cuda:0
// After this, output has the sum. Call broadcast separately.
torch::Tensor p2p_allreduce(
    torch::Tensor buf0,
    torch::Tensor buf1,
    torch::Tensor buf2,
    torch::Tensor buf3,
    torch::Tensor output
) {
    int n = buf0.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Get raw device pointers — these work across GPUs with P2P enabled
    auto* p0 = reinterpret_cast<const __nv_bfloat16*>(buf0.data_ptr());
    auto* p1 = reinterpret_cast<const __nv_bfloat16*>(buf1.data_ptr());
    auto* p2 = reinterpret_cast<const __nv_bfloat16*>(buf2.data_ptr());
    auto* p3 = reinterpret_cast<const __nv_bfloat16*>(buf3.data_ptr());
    auto* out = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());

    // Get the stream from the output tensor's device
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(output.device().index()).stream();

    p2p_allreduce_bf16_kernel<<<blocks, threads, 0, stream>>>(p0, p1, p2, p3, out, n);

    return output;
}

// Broadcast from output (on GPU 0) to peer buffers
void p2p_broadcast(
    torch::Tensor output,
    torch::Tensor dst1,
    torch::Tensor dst2,
    torch::Tensor dst3
) {
    int n = output.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    auto* src = reinterpret_cast<const __nv_bfloat16*>(output.data_ptr());
    auto* d1 = reinterpret_cast<__nv_bfloat16*>(dst1.data_ptr());
    auto* d2 = reinterpret_cast<__nv_bfloat16*>(dst2.data_ptr());
    auto* d3 = reinterpret_cast<__nv_bfloat16*>(dst3.data_ptr());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(output.device().index()).stream();

    // Launch 3 broadcast kernels on the same stream
    p2p_broadcast_bf16_kernel<<<blocks, threads, 0, stream>>>(src, d1, n);
    p2p_broadcast_bf16_kernel<<<blocks, threads, 0, stream>>>(src, d2, n);
    p2p_broadcast_bf16_kernel<<<blocks, threads, 0, stream>>>(src, d3, n);
}

// Combined: allreduce + broadcast in one call
torch::Tensor p2p_allreduce_and_broadcast(
    torch::Tensor buf0,
    torch::Tensor buf1,
    torch::Tensor buf2,
    torch::Tensor buf3,
    torch::Tensor output
) {
    int n = buf0.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    auto* p0 = reinterpret_cast<const __nv_bfloat16*>(buf0.data_ptr());
    auto* p1 = reinterpret_cast<const __nv_bfloat16*>(buf1.data_ptr());
    auto* p2 = reinterpret_cast<const __nv_bfloat16*>(buf2.data_ptr());
    auto* p3 = reinterpret_cast<const __nv_bfloat16*>(buf3.data_ptr());
    auto* out = reinterpret_cast<__nv_bfloat16*>(output.data_ptr());

    // Writable peer pointers for broadcast
    auto* w1 = reinterpret_cast<__nv_bfloat16*>(buf1.data_ptr());
    auto* w2 = reinterpret_cast<__nv_bfloat16*>(buf2.data_ptr());
    auto* w3 = reinterpret_cast<__nv_bfloat16*>(buf3.data_ptr());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(output.device().index()).stream();

    // Reduce
    p2p_allreduce_bf16_kernel<<<blocks, threads, 0, stream>>>(p0, p1, p2, p3, out, n);

    // Broadcast result to peer buffers (overwrite their input)
    p2p_broadcast_bf16_kernel<<<blocks, threads, 0, stream>>>(out, w1, n);
    p2p_broadcast_bf16_kernel<<<blocks, threads, 0, stream>>>(out, w2, n);
    p2p_broadcast_bf16_kernel<<<blocks, threads, 0, stream>>>(out, w3, n);

    return output;
}
"""

cpp_src = r"""
torch::Tensor p2p_allreduce(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void p2p_broadcast(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
torch::Tensor p2p_allreduce_and_broadcast(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void enable_peer_access();
"""


def get_module():
    """Build and load the P2P allreduce CUDA extension."""
    return load_inline(
        name="p2p_allreduce",
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=["p2p_allreduce", "p2p_broadcast",
                    "p2p_allreduce_and_broadcast", "enable_peer_access"],
        verbose=False,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
    )


def benchmark():
    print("Building CUDA extension...")
    mod = get_module()

    print("Enabling P2P access...")
    mod.enable_peer_access()

    n_gpus = torch.cuda.device_count()
    assert n_gpus == 4

    test_sizes = [
        (2048, "2K elems (4KB)"),
        (4096, "4K elems (8KB) — hidden_dim"),
        (8192, "8K elems (16KB)"),
        (16384, "16K elems (32KB)"),
    ]

    for n_elements, label in test_sizes:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        # Allocate buffers
        buffers = []
        for i in range(n_gpus):
            torch.cuda.set_device(i)
            buf = torch.full((n_elements,), float(i + 1),
                           dtype=torch.bfloat16, device=f'cuda:{i}')
            buffers.append(buf)

        torch.cuda.set_device(0)
        output = torch.zeros(n_elements, dtype=torch.bfloat16, device='cuda:0')

        # === Correctness: reduce only ===
        print("\n--- Reduce only ---")
        mod.p2p_allreduce(buffers[0], buffers[1], buffers[2], buffers[3], output)
        torch.cuda.synchronize()
        r = output[0].item()
        print(f"  Result: {r} {'✓' if abs(r - 10.0) < 0.1 else '✗ FAIL'}")

        # === Correctness: allreduce + broadcast ===
        print("\n--- AllReduce + Broadcast ---")
        for i in range(n_gpus):
            buffers[i].fill_(float(i + 1))
        output.zero_()
        torch.cuda.synchronize()

        mod.p2p_allreduce_and_broadcast(
            buffers[0], buffers[1], buffers[2], buffers[3], output)
        torch.cuda.synchronize()

        r = output[0].item()
        print(f"  GPU 0 output: {r} {'✓' if abs(r - 10.0) < 0.1 else '✗'}")
        for i in range(1, n_gpus):
            val = buffers[i][0].item()
            print(f"  GPU {i} buf:    {val} {'✓' if abs(val - 10.0) < 0.1 else '✗'}")

        # === Benchmark: reduce only ===
        print("\n--- Benchmark: Reduce only (no broadcast) ---")
        for _ in range(50):
            mod.p2p_allreduce(buffers[0], buffers[1], buffers[2], buffers[3], output)
        torch.cuda.synchronize()

        n_iters = 2000
        start = time.perf_counter()
        for _ in range(n_iters):
            mod.p2p_allreduce(buffers[0], buffers[1], buffers[2], buffers[3], output)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        avg_us = (elapsed / n_iters) * 1e6
        print(f"  Eager: {avg_us:.1f} μs (vs NCCL ~254 μs → {254/avg_us:.1f}x)")

        # === Benchmark: allreduce + broadcast ===
        print("\n--- Benchmark: AllReduce + Broadcast ---")
        for _ in range(50):
            mod.p2p_allreduce_and_broadcast(
                buffers[0], buffers[1], buffers[2], buffers[3], output)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_iters):
            mod.p2p_allreduce_and_broadcast(
                buffers[0], buffers[1], buffers[2], buffers[3], output)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        avg_full = (elapsed / n_iters) * 1e6
        print(f"  Eager: {avg_full:.1f} μs (vs NCCL ~254 μs → {254/avg_full:.1f}x)")

        # === CUDA Graph Capture ===
        print("\n--- CUDA Graph Capture ---")
        torch.cuda.set_device(0)

        # Reset
        for i in range(n_gpus):
            buffers[i].fill_(float(i + 1))
        output.zero_()
        torch.cuda.synchronize()

        stream = torch.cuda.Stream(device='cuda:0')

        # Warmup in stream
        with torch.cuda.stream(stream):
            mod.p2p_allreduce_and_broadcast(
                buffers[0], buffers[1], buffers[2], buffers[3], output)
        stream.synchronize()

        # Reset for capture
        for i in range(n_gpus):
            buffers[i].fill_(float(i + 1))
        output.zero_()
        torch.cuda.synchronize()

        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                mod.p2p_allreduce_and_broadcast(
                    buffers[0], buffers[1], buffers[2], buffers[3], output)

            # Test replay
            for i in range(n_gpus):
                buffers[i].fill_(float(i + 1))
            output.zero_()
            torch.cuda.synchronize()

            graph.replay()
            torch.cuda.synchronize()

            r = output[0].item()
            graph_ok = abs(r - 10.0) < 0.1
            print(f"  Graph replay: {r} {'✓ GRAPH WORKS!' if graph_ok else '✗ FAIL'}")

            if graph_ok:
                for i in range(1, n_gpus):
                    val = buffers[i][0].item()
                    ok = abs(val - 10.0) < 0.1
                    print(f"  GPU {i} broadcast: {val} {'✓' if ok else '✗'}")

                # Benchmark graph
                for _ in range(100):
                    graph.replay()
                torch.cuda.synchronize()

                start = time.perf_counter()
                for _ in range(n_iters):
                    graph.replay()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                avg_graph = (elapsed / n_iters) * 1e6
                print(f"\n  Graph latency: {avg_graph:.1f} μs")
                print(f"  vs NCCL: ~254 μs → {254/avg_graph:.1f}x speedup")
                print(f"  vs Eager: {avg_full:.1f} μs → {avg_full/avg_graph:.1f}x from graph")

        except Exception as e:
            print(f"  Graph capture FAILED: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    benchmark()
