#!/usr/bin/env python3
"""
Baseline MoE Layer Benchmark — Measure current FLASHINFER_CUTLASS performance
==============================================================================
Runs inside the vLLM container to profile individual MoE layer latency.
Measures GEMM1, activation, GEMM2 separately and end-to-end.

Usage (inside vLLM container):
    python3 /benchmarks/bench_baseline.py

Usage (from host):
    docker exec vllm-qwen35 python3 /benchmarks/bench_baseline.py
"""

import torch
import torch.nn as nn
import time
import json
import sys

def bench_moe_layer():
    """Benchmark a single MoE layer with Qwen3.5-397B dimensions at TP=4."""

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Qwen3.5-397B dimensions at TP=4
    hidden_size = 4096
    intermediate_size_per_partition = 256  # 1024 / TP=4
    num_experts = 512
    top_k = 10

    # GEMM shapes per expert per GPU:
    # GEMM1: [M, 4096] x [4096, 512] -> [M, 512]  (gate_up fused: 256 gate + 256 up)
    # GEMM2: [M, 256] x [256, 4096] -> [M, 4096]

    gate_up_size = intermediate_size_per_partition * 2  # 512

    print("=" * 70)
    print("SM120 Fused MoE Kernel — Baseline Benchmark")
    print("=" * 70)
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  SM count: {torch.cuda.get_device_properties(0).multi_processor_count}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  intermediate/GPU: {intermediate_size_per_partition}")
    print(f"  gate_up_size: {gate_up_size}")
    print(f"  num_experts: {num_experts}")
    print(f"  top_k: {top_k}")
    print()

    results = {}

    for M in [1, 2, 4, 8, 16, 32]:
        num_active_experts = min(top_k, M * top_k)  # at most top_k unique experts
        # For decode (M=1), ~10 active experts each with 1 token
        tokens_per_expert = max(1, M)

        print(f"--- M={M} (tokens_per_expert≈{tokens_per_expert}, active_experts={top_k}) ---")

        # Create dummy tensors matching the shapes
        # Input: already FP4-quantized in real pipeline, use BF16 for this benchmark
        input_tensor = torch.randn(M * top_k, hidden_size, dtype=torch.bfloat16, device=device)

        # Expert weights (simulate 10 active experts)
        w1 = torch.randn(top_k, gate_up_size, hidden_size, dtype=torch.bfloat16, device=device)
        w2 = torch.randn(top_k, hidden_size, intermediate_size_per_partition, dtype=torch.bfloat16, device=device)

        # Routing
        expert_ids = torch.randint(0, num_experts, (M, top_k), device=device)
        routing_weights = torch.softmax(torch.randn(M, top_k, device=device), dim=-1).to(torch.bfloat16)

        # Warmup
        for _ in range(10):
            # Simulate grouped GEMM1
            gate_up = torch.bmm(
                input_tensor.view(top_k, M, hidden_size),
                w1.transpose(1, 2)
            )  # [top_k, M, 512]
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = torch.nn.functional.silu(gate) * up  # [top_k, M, 256]
            # Simulate grouped GEMM2
            output = torch.bmm(
                intermediate,
                w2.transpose(1, 2)
            )  # [top_k, M, 4096]

        torch.cuda.synchronize()

        # Benchmark with CUDA events
        num_iters = 200

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        gemm1_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        act_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        gemm2_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

        for i in range(num_iters):
            start_events[i].record()

            # GEMM1 (gate_up projection)
            gate_up = torch.bmm(
                input_tensor.view(top_k, M, hidden_size),
                w1.transpose(1, 2)
            )
            gemm1_events[i].record()

            # Activation (SwiGLU)
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = torch.nn.functional.silu(gate) * up
            act_events[i].record()

            # GEMM2 (down projection)
            output = torch.bmm(
                intermediate,
                w2.transpose(1, 2)
            )
            gemm2_events[i].record()

            # Weighted reduction (simulate scatter-add)
            final = (output * routing_weights.view(1, M, top_k, 1).sum(dim=2)).sum(dim=0)
            end_events[i].record()

        torch.cuda.synchronize()

        gemm1_times = [start_events[i].elapsed_time(gemm1_events[i]) for i in range(num_iters)]
        act_times = [gemm1_events[i].elapsed_time(act_events[i]) for i in range(num_iters)]
        gemm2_times = [act_events[i].elapsed_time(gemm2_events[i]) for i in range(num_iters)]
        reduce_times = [gemm2_events[i].elapsed_time(end_events[i]) for i in range(num_iters)]
        total_times = [start_events[i].elapsed_time(end_events[i]) for i in range(num_iters)]

        # Remove first 20 iterations (warmup)
        gemm1_times = gemm1_times[20:]
        act_times = act_times[20:]
        gemm2_times = gemm2_times[20:]
        reduce_times = reduce_times[20:]
        total_times = total_times[20:]

        avg = lambda x: sum(x) / len(x)
        med = lambda x: sorted(x)[len(x)//2]

        r = {
            "M": M,
            "gemm1_ms": {"avg": avg(gemm1_times), "med": med(gemm1_times)},
            "activation_ms": {"avg": avg(act_times), "med": med(act_times)},
            "gemm2_ms": {"avg": avg(gemm2_times), "med": med(gemm2_times)},
            "reduce_ms": {"avg": avg(reduce_times), "med": med(reduce_times)},
            "total_ms": {"avg": avg(total_times), "med": med(total_times)},
        }
        results[f"M={M}"] = r

        print(f"  GEMM1:      {r['gemm1_ms']['avg']:.3f} ms (median {r['gemm1_ms']['med']:.3f})")
        print(f"  Activation: {r['activation_ms']['avg']:.3f} ms (median {r['activation_ms']['med']:.3f})")
        print(f"  GEMM2:      {r['gemm2_ms']['avg']:.3f} ms (median {r['gemm2_ms']['med']:.3f})")
        print(f"  Reduce:     {r['reduce_ms']['avg']:.3f} ms (median {r['reduce_ms']['med']:.3f})")
        print(f"  TOTAL:      {r['total_ms']['avg']:.3f} ms (median {r['total_ms']['med']:.3f})")
        print()

    # Also measure kernel launch overhead
    print("--- Kernel Launch Overhead ---")
    dummy = torch.randn(1, device=device)
    launch_times = []
    for _ in range(500):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        torch.add(dummy, dummy)  # minimal kernel
        e.record()
        torch.cuda.synchronize()
        launch_times.append(s.elapsed_time(e))

    launch_times = launch_times[50:]  # remove warmup
    results["kernel_launch_overhead_us"] = {
        "avg": avg(launch_times) * 1000,
        "med": med(launch_times) * 1000,
    }
    print(f"  Single kernel launch: {results['kernel_launch_overhead_us']['avg']:.1f} μs avg "
          f"(median {results['kernel_launch_overhead_us']['med']:.1f} μs)")
    print(f"  5 launches (current MoE path): ~{5 * results['kernel_launch_overhead_us']['avg']:.1f} μs")
    print(f"  × 60 layers: ~{60 * 5 * results['kernel_launch_overhead_us']['avg'] / 1000:.2f} ms per token")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY — Per-Layer MoE Latency (decode, M=1)")
    print("=" * 70)
    m1 = results["M=1"]
    print(f"  Current total:     {m1['total_ms']['avg']:.3f} ms")
    print(f"  Launch overhead:   ~{5 * results['kernel_launch_overhead_us']['avg'] / 1000:.3f} ms (5 launches)")
    print(f"  GMEM intermediate: ~{m1['activation_ms']['avg']:.3f} ms (write+read)")
    print(f"  Target (fused):    ~{m1['gemm1_ms']['avg'] + m1['gemm2_ms']['avg']:.3f} ms (GEMM1+GEMM2 only)")
    potential = (1 - (m1['gemm1_ms']['avg'] + m1['gemm2_ms']['avg']) / m1['total_ms']['avg']) * 100
    print(f"  Potential savings: ~{potential:.0f}%")

    # Save results
    with open("/tmp/moe_baseline_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /tmp/moe_baseline_benchmark.json")


if __name__ == "__main__":
    bench_moe_layer()
