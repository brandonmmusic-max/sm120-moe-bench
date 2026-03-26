"""
bench_nccl_allreduce.py — NCCL AllReduce latency for small tensors

Launch with: torchrun --nproc_per_node=4 bench_nccl_allreduce.py

Tests AllReduce latency for tensors matching MoE output sizes:
  M=1: [1, 4096] BF16 = 8 KB
  M=4: [4, 4096] BF16 = 32 KB
  Also tests float32 variants (16 KB, 64 KB)
"""

import os
import time
import torch
import torch.distributed as dist


def benchmark_allreduce(tensor_size, dtype, warmup=100, iters=500, label=""):
    """Benchmark NCCL AllReduce for a given tensor size."""
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    buf = torch.randn(tensor_size, dtype=dtype, device=device)
    nbytes = buf.numel() * buf.element_size()

    # Warmup
    for _ in range(warmup):
        dist.all_reduce(buf)
    torch.cuda.synchronize()

    # Timed
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        dist.all_reduce(buf)
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)]  # μs
    times.sort()
    # Trim outliers: middle 80%
    lo, hi = len(times) // 10, len(times) * 9 // 10
    avg = sum(times[lo:hi]) / (hi - lo)
    p50 = times[len(times) // 2]
    p99 = times[int(len(times) * 0.99)]

    if rank == 0:
        print(f"  {label:30s} | {nbytes:>8d} B | avg={avg:7.1f} μs | p50={p50:7.1f} μs | p99={p99:7.1f} μs")

    return avg


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    if rank == 0:
        print(f"NCCL AllReduce Benchmark — {world_size} GPUs")
        print(f"NCCL version: {torch.cuda.nccl.version()}")
        env_symm = os.environ.get("VLLM_USE_NCCL_SYMM_MEM", "not set")
        print(f"VLLM_USE_NCCL_SYMM_MEM={env_symm}")
        print()
        print(f"  {'Label':30s} | {'Size':>8s}   | {'avg':>7s}    | {'p50':>7s}    | {'p99':>7s}")
        print(f"  {'-'*30} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")

    # MoE output sizes (BF16)
    for M in [1, 4]:
        benchmark_allreduce((M, 4096), torch.bfloat16, label=f"MoE M={M} BF16 [{M},4096]")

    # Same in float32 (if kernel outputs f32 before cast)
    for M in [1, 4]:
        benchmark_allreduce((M, 4096), torch.float32, label=f"MoE M={M} F32 [{M},4096]")

    # Attention output sizes
    for M in [1, 4]:
        benchmark_allreduce((M, 4096), torch.bfloat16, label=f"Attn M={M} BF16 [{M},4096]")

    # Larger tensors for reference
    for size in [128*1024, 1024*1024]:
        benchmark_allreduce((size,), torch.bfloat16, label=f"Large {size*2//1024} KB BF16")

    if rank == 0:
        print()
        print("Note: MoE has 60 AllReduces/token, Attention has 60.")
        print("Total AllReduce per token = 120 calls.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
