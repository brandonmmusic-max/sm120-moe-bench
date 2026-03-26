#!/usr/bin/env python3
"""
Sprint 13: NCCL AllReduce p99 tail latency benchmark v2.
Uses CUDA events for accurate GPU-side timing (no host sync overhead).
"""
import os
import sys
import time
import argparse
import torch
import torch.distributed as dist
import numpy as np


def benchmark_allreduce_events(tensor_size_bytes: int, num_iters: int, warmup: int = 200):
    """Run AllReduce benchmark using CUDA events for GPU-side timing."""
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    num_elements = tensor_size_bytes // 2  # BF16
    tensor = torch.randn(num_elements, dtype=torch.bfloat16, device=device)

    # Pre-allocate CUDA events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    # Warmup
    for _ in range(warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    # Benchmark with CUDA events (no host sync between iterations)
    for i in range(num_iters):
        start_events[i].record()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        end_events[i].record()

    # Sync once at end, then read all timings
    torch.cuda.synchronize()
    latencies_us = []
    for i in range(num_iters):
        elapsed_ms = start_events[i].elapsed_time(end_events[i])
        latencies_us.append(elapsed_ms * 1000.0)  # ms -> us

    return np.array(latencies_us)


def print_results(label: str, size: int, latencies: np.ndarray):
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    p999 = np.percentile(latencies, 99.9)
    pmax = np.max(latencies)
    pmin = np.min(latencies)

    spike_threshold = p50 * 10
    num_spikes = np.sum(latencies > spike_threshold)
    spike_pct = 100.0 * num_spikes / len(latencies)

    spike_indices = np.where(latencies > spike_threshold)[0]
    if len(spike_indices) > 1:
        gaps = np.diff(spike_indices)
        gap_info = f"gaps: min={gaps.min()}, max={gaps.max()}, mean={gaps.mean():.0f}"
    else:
        gap_info = "no spikes" if len(spike_indices) == 0 else "single spike"

    # Also check moderate spikes (>5x median)
    mod_threshold = p50 * 5
    mod_spikes = np.sum(latencies > mod_threshold)

    print(f"  Size: {size:>6d}B ({size/1024:.0f}KB)")
    print(f"    p50={p50:>8.1f}μs  p95={p95:>8.1f}μs  p99={p99:>8.1f}μs  "
          f"p999={p999:>8.1f}μs  max={pmax:>8.1f}μs")
    print(f"    p99/p50={p99/p50:.1f}x  max/p50={pmax/p50:.1f}x")
    print(f"    Spikes >10x median: {num_spikes}/{len(latencies)} ({spike_pct:.2f}%)  {gap_info}")
    print(f"    Spikes >5x median:  {mod_spikes}/{len(latencies)} ({100*mod_spikes/len(latencies):.2f}%)")

    # Show histogram of top 20 worst latencies
    worst = np.sort(latencies)[-20:]
    print(f"    Top 20 worst (μs): {', '.join(f'{x:.0f}' for x in worst)}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--sizes", type=str, default="8192,16384,32768")
    parser.add_argument("--label", type=str, default="baseline")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"NCCL AllReduce p99 Benchmark (CUDA events) — {args.label}")
        print(f"{'='*70}")
        print(f"World size: {world_size}, Iters: {args.iters}, Warmup: {args.warmup}")
        nccl_vars = {k: v for k, v in os.environ.items() if k.startswith("NCCL")}
        if nccl_vars:
            print(f"NCCL env vars: {nccl_vars}")
        # Print GPU states
        for i in range(world_size):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
        print()

    sizes = [int(s) for s in args.sizes.split(",")]

    for size in sizes:
        dist.barrier()
        latencies = benchmark_allreduce_events(size, args.iters, args.warmup)
        if rank == 0:
            print_results(args.label, size, latencies)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
