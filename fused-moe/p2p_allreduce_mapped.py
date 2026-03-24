#!/usr/bin/env python3
"""
P2P All-Reduce for 4 PCIe GPUs.

Approach: Copy peer buffers to local staging → reduce locally → copy result back.
All buffers pre-allocated at init → CUDA graph capturable.
"""

import torch
import torch.cuda
import time


def benchmark():
    n_gpus = torch.cuda.device_count()
    assert n_gpus == 4

    test_sizes = [
        (2048, "2K elems (4KB)"),
        (4096, "4K elems (8KB) — hidden_dim"),
        (8192, "8K elems (16KB)"),
    ]

    for n_elements, label in test_sizes:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        # Each GPU's input buffer (the data to all-reduce)
        buffers = []
        for i in range(n_gpus):
            torch.cuda.set_device(i)
            buf = torch.zeros(n_elements, dtype=torch.bfloat16, device=f'cuda:{i}')
            buffers.append(buf)

        # GPU 0: staging buffers (local copies of peer data)
        torch.cuda.set_device(0)
        staging = [
            torch.zeros(n_elements, dtype=torch.bfloat16, device='cuda:0')
            for _ in range(n_gpus)
        ]
        output = torch.zeros(n_elements, dtype=torch.bfloat16, device='cuda:0')

        def p2p_allreduce_on_gpu0(bufs, stg, out, n):
            """
            Reduce all 4 GPU buffers on GPU 0, broadcast result back.
            Steps:
              1. Copy peer buffers to local staging (P2P reads)
              2. Reduce locally
              3. Broadcast result to peers (P2P writes)
            """
            # Step 1: P2P copy to local staging
            for i in range(n_gpus):
                stg[i][:n].copy_(bufs[i][:n], non_blocking=True)

            # Step 2: Reduce locally (all staging buffers are on GPU 0)
            out[:n].copy_(stg[0][:n])
            out[:n].add_(stg[1][:n])
            out[:n].add_(stg[2][:n])
            out[:n].add_(stg[3][:n])

            # Step 3: Broadcast back to peers
            for i in range(1, n_gpus):
                bufs[i][:n].copy_(out[:n], non_blocking=True)

        # Fill buffers
        for i in range(n_gpus):
            buffers[i].fill_(float(i + 1))
        torch.cuda.synchronize()

        # === Test correctness ===
        print("\n--- Correctness ---")
        p2p_allreduce_on_gpu0(buffers, staging, output, n_elements)
        torch.cuda.synchronize()

        result = output[0].item()
        print(f"  GPU 0 result: {result} {'✓' if abs(result - 10.0) < 0.1 else '✗'}")
        for i in range(1, n_gpus):
            val = buffers[i][0].item()
            print(f"  GPU {i} result: {val} {'✓' if abs(val - 10.0) < 0.1 else '✗'}")

        # === Benchmark: Full allreduce (copy + reduce + broadcast) ===
        print("\n--- Benchmark: Full allreduce ---")
        # Warmup
        for _ in range(50):
            for i in range(n_gpus):
                buffers[i].fill_(float(i + 1))
            p2p_allreduce_on_gpu0(buffers, staging, output, n_elements)
        torch.cuda.synchronize()

        n_iters = 1000
        # Time just the allreduce (buffers already filled)
        for i in range(n_gpus):
            buffers[i].fill_(float(i + 1))
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_iters):
            p2p_allreduce_on_gpu0(buffers, staging, output, n_elements)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        avg_us = (elapsed / n_iters) * 1e6
        print(f"  Eager latency: {avg_us:.1f} μs")
        print(f"  vs NCCL: ~254 μs → {254.0 / avg_us:.1f}x speedup")

        # === CUDA Graph Capture ===
        print("\n--- CUDA Graph Capture ---")
        torch.cuda.set_device(0)

        for i in range(n_gpus):
            buffers[i].fill_(float(i + 1))
        torch.cuda.synchronize()

        stream = torch.cuda.Stream(device='cuda:0')

        # Warmup in stream
        with torch.cuda.stream(stream):
            p2p_allreduce_on_gpu0(buffers, staging, output, n_elements)
        stream.synchronize()

        # Reset
        for i in range(n_gpus):
            buffers[i].fill_(float(i + 1))
        output.zero_()
        torch.cuda.synchronize()

        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                p2p_allreduce_on_gpu0(buffers, staging, output, n_elements)

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
                # Check broadcast
                for i in range(1, n_gpus):
                    val = buffers[i][0].item()
                    ok = abs(val - 10.0) < 0.1
                    print(f"  GPU {i} broadcast: {val} {'✓' if ok else '✗'}")

                # Benchmark graph
                for _ in range(50):
                    graph.replay()
                torch.cuda.synchronize()

                start = time.perf_counter()
                for _ in range(n_iters):
                    graph.replay()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                avg_g = (elapsed / n_iters) * 1e6
                print(f"  Graph latency: {avg_g:.1f} μs")
                print(f"  vs NCCL: ~254 μs → {254.0 / avg_g:.1f}x speedup")

        except Exception as e:
            print(f"  Graph capture FAILED: {e}")

        # === Approach 2: Reduce-scatter pattern ===
        # Each GPU reduces a chunk, then all-gather
        print("\n--- Approach 2: Tree reduce (GPU pairs) ---")

        # GPU 0 staging for tree reduce
        tree_staging = torch.zeros(n_elements, dtype=torch.bfloat16, device='cuda:0')

        def tree_allreduce(bufs, stg, out, n):
            """
            Tree all-reduce:
            Round 1: GPU 0 += GPU 1, GPU 2 += GPU 3 (parallel)
            Round 2: GPU 0 += GPU 2
            Round 3: Broadcast GPU 0 → GPU 1,2,3
            """
            # Round 1: GPU 0 accumulates GPU 1
            stg[:n].copy_(bufs[1][:n], non_blocking=True)
            out[:n].copy_(bufs[0][:n])
            out[:n].add_(stg[:n])

            # Round 2: GPU 0 accumulates GPU 2's partial (GPU 2 + GPU 3)
            # But GPU 2's accumulation happened on GPU 2... we can't easily
            # do parallel tree on different GPUs without multi-GPU streams.
            # Simpler: just sequential copy + add
            stg[:n].copy_(bufs[2][:n], non_blocking=True)
            out[:n].add_(stg[:n])
            stg[:n].copy_(bufs[3][:n], non_blocking=True)
            out[:n].add_(stg[:n])

            # Broadcast
            for i in range(1, 4):
                bufs[i][:n].copy_(out[:n], non_blocking=True)

        for i in range(n_gpus):
            buffers[i].fill_(float(i + 1))
        torch.cuda.synchronize()

        tree_allreduce(buffers, tree_staging, output, n_elements)
        torch.cuda.synchronize()
        r = output[0].item()
        print(f"  Correctness: {r} {'✓' if abs(r - 10.0) < 0.1 else '✗'}")

        # Benchmark
        for _ in range(50):
            for i in range(n_gpus):
                buffers[i].fill_(float(i + 1))
            tree_allreduce(buffers, tree_staging, output, n_elements)
        torch.cuda.synchronize()

        for i in range(n_gpus):
            buffers[i].fill_(float(i + 1))
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_iters):
            tree_allreduce(buffers, tree_staging, output, n_elements)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        avg_tree = (elapsed / n_iters) * 1e6
        print(f"  Tree reduce: {avg_tree:.1f} μs")
        print(f"  vs NCCL: ~254 μs → {254.0 / avg_tree:.1f}x speedup")

        # Graph capture tree reduce
        try:
            for i in range(n_gpus):
                buffers[i].fill_(float(i + 1))
            torch.cuda.synchronize()

            stream3 = torch.cuda.Stream(device='cuda:0')
            with torch.cuda.stream(stream3):
                tree_allreduce(buffers, tree_staging, output, n_elements)
            stream3.synchronize()

            graph3 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph3, stream=stream3):
                tree_allreduce(buffers, tree_staging, output, n_elements)

            for i in range(n_gpus):
                buffers[i].fill_(float(i + 1))
            output.zero_()
            torch.cuda.synchronize()
            graph3.replay()
            torch.cuda.synchronize()

            r3 = output[0].item()
            g3_ok = abs(r3 - 10.0) < 0.1
            print(f"  Graph capture: {r3} {'✓' if g3_ok else '✗'}")

            if g3_ok:
                for _ in range(50):
                    graph3.replay()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(n_iters):
                    graph3.replay()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                avg_g3 = (elapsed / n_iters) * 1e6
                print(f"  Graph latency: {avg_g3:.1f} μs")
                print(f"  vs NCCL: ~254 μs → {254.0 / avg_g3:.1f}x speedup")

        except Exception as e:
            print(f"  Graph capture FAILED: {e}")


if __name__ == "__main__":
    benchmark()
