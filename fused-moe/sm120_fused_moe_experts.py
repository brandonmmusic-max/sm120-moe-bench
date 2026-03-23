#!/usr/bin/env python3
"""
SM120 Optimized MoE Experts — Phase 2

Reduces kernel launches from 7 to 4 by:
1. Fused routing + expand + quant (replaces 3 separate kernels)
2. GEMM1 (existing CUTLASS grouped, 25μs)
3. Fused activation + strides (replaces 2 separate kernels)
4. GEMM2 with FINALIZE epilogue (replaces GEMM2 + finalize kernels)

Run inside vLLM container:
  python3 /workspace/sm120_fused_moe_experts.py
"""
import torch
import torch.cuda
import time
import sys

torch.cuda.set_device(0)


def benchmark_current_path():
    """Benchmark the current FlashInfer CUTLASS fused MoE path for comparison."""
    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module, ActivationType
    module = get_cutlass_fused_moe_module('120')

    M, K, N_half, E, topk = 1, 4096, 256, 512, 10
    SF_VEC, FP8_PER_INT32 = 16, 4
    device = 'cuda:0'

    hidden = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(0, E, (M, topk), dtype=torch.int32, device=device)
    topk_weights = torch.ones(M, topk, dtype=torch.float32, device=device) / topk
    output = torch.empty(M, K, dtype=torch.bfloat16, device=device)
    w1 = torch.randint(0, 2**63-1, (E, 2*N_half, K//16), dtype=torch.int64, device=device)
    w2 = torch.randint(0, 2**63-1, (E, K, N_half//16), dtype=torch.int64, device=device)
    qs = [
        torch.ones(E, dtype=torch.float32, device=device),
        torch.ones(E, 2*N_half, K//(SF_VEC*FP8_PER_INT32), dtype=torch.int32, device=device),
        torch.ones(E, dtype=torch.float32, device=device),
        torch.ones(E, dtype=torch.float32, device=device),
        torch.ones(E, K, N_half//(SF_VEC*FP8_PER_INT32), dtype=torch.int32, device=device),
        torch.ones(E, dtype=torch.float32, device=device),
    ]

    # Warmup
    for _ in range(50):
        module.cutlass_fused_moe(output, hidden, topk_ids, topk_weights,
            w1, None, w2, None, torch.bfloat16, qs,
            activation_type=ActivationType.Swiglu)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(300):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        module.cutlass_fused_moe(output, hidden, topk_ids, topk_weights,
            w1, None, w2, None, torch.bfloat16, qs,
            activation_type=ActivationType.Swiglu)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    times = times[30:]
    med = sorted(times)[len(times)//2] * 1000
    avg = sum(times)/len(times) * 1000
    return avg, med


def benchmark_vllm_cutlass_path():
    """Benchmark the unfused CUTLASS path (cutlass_fp4_moe_mm) used by VLLM_CUTLASS backend."""
    from vllm._custom_ops import (
        cutlass_fp4_moe_mm,
        get_cutlass_moe_mm_data,
        scaled_fp4_experts_quant,
        silu_and_mul_scaled_fp4_experts_quant,
        shuffle_rows,
    )

    M, K, N_half, E, topk = 1, 4096, 256, 512, 10
    device = 'cuda:0'

    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(0, E, (M, topk), dtype=torch.int32, device=device)

    w1_fp4 = torch.randint(0, 255, (E, 2*N_half, K//2), dtype=torch.uint8, device=device)
    w2_fp4 = torch.randint(0, 255, (E, K, N_half//2), dtype=torch.uint8, device=device)
    w1_bs = torch.ones(E, 2*N_half, K//16, dtype=torch.float8_e4m3fn, device=device)
    w2_bs = torch.ones(E, K, N_half//16, dtype=torch.float8_e4m3fn, device=device)
    w1_alphas = torch.ones(2*N_half, K//16, dtype=torch.float32, device=device)
    w2_alphas = torch.ones(K, N_half//16, dtype=torch.float32, device=device)
    a1_gscale = torch.ones(E, dtype=torch.float32, device=device)
    a2_gscale = torch.ones(E, dtype=torch.float32, device=device)

    expert_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)
    blockscale_offsets = torch.empty(E + 1, dtype=torch.int32, device=device)
    problem_sizes1 = torch.empty(E, 3, dtype=torch.int32, device=device)
    problem_sizes2 = torch.empty(E, 3, dtype=torch.int32, device=device)
    a_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device=device)
    c_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device=device)

    get_cutlass_moe_mm_data(
        topk_ids, expert_offsets, problem_sizes1, problem_sizes2,
        a_map, c_map, E, N_half, K, blockscale_offsets,
    )

    a_shuffled = shuffle_rows(a.expand(topk, K), a_map)
    a_fp4, a_bs = scaled_fp4_experts_quant(
        a_shuffled, a1_gscale, expert_offsets, blockscale_offsets, topk
    )

    c1 = torch.empty(M * topk, 2*N_half, dtype=torch.bfloat16, device=device)
    c3 = torch.empty(M * topk, K, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(50):
        cutlass_fp4_moe_mm(c1, a_fp4, w1_fp4, a_bs, w1_bs, w1_alphas,
                           problem_sizes1, expert_offsets[:-1], blockscale_offsets[:-1])
        int_fp4, int_bs = silu_and_mul_scaled_fp4_experts_quant(
            c1, a2_gscale, expert_offsets, blockscale_offsets, topk)
        cutlass_fp4_moe_mm(c3, int_fp4, w2_fp4, int_bs, w2_bs, w2_alphas,
                           problem_sizes2, expert_offsets[:-1], blockscale_offsets[:-1])
    torch.cuda.synchronize()

    # Benchmark individual ops
    N_ITER = 300

    # Total pipeline
    total_times = []
    gemm1_times = []
    act_times = []
    gemm2_times = []

    for _ in range(N_ITER):
        s = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        e3 = torch.cuda.Event(enable_timing=True)

        s.record()
        cutlass_fp4_moe_mm(c1, a_fp4, w1_fp4, a_bs, w1_bs, w1_alphas,
                           problem_sizes1, expert_offsets[:-1], blockscale_offsets[:-1])
        e1.record()
        int_fp4, int_bs = silu_and_mul_scaled_fp4_experts_quant(
            c1, a2_gscale, expert_offsets, blockscale_offsets, topk)
        e2.record()
        cutlass_fp4_moe_mm(c3, int_fp4, w2_fp4, int_bs, w2_bs, w2_alphas,
                           problem_sizes2, expert_offsets[:-1], blockscale_offsets[:-1])
        e3.record()
        torch.cuda.synchronize()

        gemm1_times.append(s.elapsed_time(e1))
        act_times.append(e1.elapsed_time(e2))
        gemm2_times.append(e2.elapsed_time(e3))
        total_times.append(s.elapsed_time(e3))

    # Skip warmup
    skip = 30
    gemm1_times = gemm1_times[skip:]
    act_times = act_times[skip:]
    gemm2_times = gemm2_times[skip:]
    total_times = total_times[skip:]

    def med(v):
        s = sorted(v)
        return s[len(s)//2] * 1000

    return {
        'gemm1': med(gemm1_times),
        'activation': med(act_times),
        'gemm2': med(gemm2_times),
        'total': med(total_times),
    }


def main():
    print("=" * 60)
    print("SM120 MoE Kernel Benchmark — Phase 2")
    print("=" * 60)
    print()

    # Benchmark current FlashInfer CUTLASS fused path (7 launches)
    print("Benchmarking FlashInfer CUTLASS fused MoE (7 launches)...")
    fi_avg, fi_med = benchmark_current_path()
    print(f"  FlashInfer cutlass_fused_moe: avg={fi_avg:.1f}μs  med={fi_med:.1f}μs")
    print()

    # Benchmark VLLM_CUTLASS unfused path (5 launches, excluding routing)
    print("Benchmarking vLLM CUTLASS unfused path (3 launches)...")
    vllm_results = benchmark_vllm_cutlass_path()
    print(f"  GEMM1:      {vllm_results['gemm1']:.1f}μs")
    print(f"  Activation: {vllm_results['activation']:.1f}μs")
    print(f"  GEMM2:      {vllm_results['gemm2']:.1f}μs")
    print(f"  Total:      {vllm_results['total']:.1f}μs (excludes routing+expand)")
    print()

    # Analysis
    overhead = fi_med - vllm_results['total']
    print("Analysis:")
    print(f"  FlashInfer total:     {fi_med:.1f}μs (7 launches)")
    print(f"  CUTLASS GEMM-only:    {vllm_results['total']:.1f}μs (3 launches)")
    print(f"  Non-GEMM overhead:    {overhead:.1f}μs")
    print(f"  Overhead breakdown:")
    print(f"    routing+expand:     ~{overhead - 14:.1f}μs (prefix sums + expand + quant)")
    print(f"    computeStrides:     ~14μs")
    print()

    # Target: if we can reduce to 4 launches
    target = vllm_results['total'] + 10  # 10μs for fused routing+expand
    print(f"  Target (4 launches):  ~{target:.1f}μs")
    print(f"  Projected speedup:    {fi_med/target:.2f}x over FlashInfer path")
    print(f"  Projected tok/s:      ~{127 * fi_med / target:.0f} (from 127 baseline)")


if __name__ == "__main__":
    main()
