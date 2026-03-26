#!/usr/bin/env python3
"""
Sprint 13: NCCL AllReduce p99 tail latency benchmark.
Measures p50/p95/p99/p999/max for small tensors (8KB-32KB) typical of MoE decode.
Uses torchrun for multi-GPU NCCL process group.
"""
import os
import sys
import time
import argparse
import torch
import torch.distributed as dist
import numpy as np

def benchmark_allreduce(tensor_size_bytes: int, num_iters: int, warmup: int = 200):
    """Run AllReduce benchmark and return latency array in microseconds."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Create tensor matching MoE decode size: [M, hidden_dim] in BF16
    num_elements = tensor_size_bytes // 2  # BF16 = 2 bytes
    tensor = torch.randn(num_elements, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Benchmark
    latencies_us = []
    for i in range(num_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        t1 = time.perf_counter_ns()
        latencies_us.append((t1 - t0) / 1000.0)

    return np.array(latencies_us)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--sizes", type=str, default="8192,16384,32768",
                        help="Comma-separated tensor sizes in bytes")
    parser.add_argument("--label", type=str, default="baseline",
                        help="Label for this run")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0,
                        help="Local rank (set by torch.distributed.launch)")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"NCCL AllReduce p99 Benchmark — {args.label}")
        print(f"{'='*70}")
        print(f"World size: {world_size}, Iters: {args.iters}, Warmup: {args.warmup}")
        # Print NCCL env vars
        nccl_vars = {k: v for k, v in os.environ.items() if k.startswith("NCCL")}
        if nccl_vars:
            print(f"NCCL env vars: {nccl_vars}")
        print()

    sizes = [int(s) for s in args.sizes.split(",")]

    for size in sizes:
        dist.barrier()
        latencies = benchmark_allreduce(size, args.iters, args.warmup)

        if rank == 0:
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
            p999 = np.percentile(latencies, 99.9)
            pmax = np.max(latencies)
            pmin = np.min(latencies)

            # Count spikes (>10x median)
            spike_threshold = p50 * 10
            num_spikes = np.sum(latencies > spike_threshold)
            spike_pct = 100.0 * num_spikes / len(latencies)

            # Find spike pattern (are they periodic?)
            spike_indices = np.where(latencies > spike_threshold)[0]
            if len(spike_indices) > 1:
                gaps = np.diff(spike_indices)
                gap_info = f"gaps: min={gaps.min()}, max={gaps.max()}, mean={gaps.mean():.0f}"
            else:
                gap_info = "N/A"

            print(f"Size: {size:>6d}B ({size/1024:.0f}KB)")
            print(f"  p50={p50:>8.1f}μs  p95={p95:>8.1f}μs  p99={p99:>8.1f}μs  "
                  f"p999={p999:>8.1f}μs  max={pmax:>8.1f}μs  min={pmin:>8.1f}μs")
            print(f"  Spikes (>{spike_threshold:.0f}μs): {num_spikes}/{len(latencies)} "
                  f"({spike_pct:.2f}%)  {gap_info}")
            print(f"  p99/p50 ratio: {p99/p50:.1f}x")
            print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
