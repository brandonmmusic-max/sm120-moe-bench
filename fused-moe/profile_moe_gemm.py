#!/usr/bin/env python3
"""
Profile the actual cutlass_fp4_moe_mm kernels with CUDA events + NCU-friendly isolation.
Run inside the vLLM container:
  python3 /workspace/profile_moe_gemm.py

For NCU profiling:
  ncu --target-processes all --set full -o /tmp/moe_profile python3 /workspace/profile_moe_gemm.py --ncu
"""
import torch
import sys

torch.cuda.set_device(0)

from vllm._custom_ops import (
    cutlass_fp4_moe_mm,
    get_cutlass_moe_mm_data,
    scaled_fp4_experts_quant,
    silu_and_mul_scaled_fp4_experts_quant,
    shuffle_rows,
)

# Qwen3.5 dims at TP=4
M_tokens = 1      # decode = 1 token
K = 4096           # hidden_size
N_half = 256       # moe_intermediate / TP
N = N_half * 2     # gate+up = 512
E = 512            # total experts
topk = 10
M = M_tokens * topk  # total rows after expansion
device = "cuda:0"

ncu_mode = "--ncu" in sys.argv

print(f"MoE GEMM Profile: M_tokens={M_tokens}, topk={topk}, K={K}, N_half={N_half}, E={E}")
print(f"  GEMM1: [{M},{K}] x [{K},{N}] grouped across {topk} experts")
print(f"  GEMM2: [{M},{N_half}] x [{N_half},{K}] grouped across {topk} experts")
print()

# Create tensors matching vLLM format
a = torch.randn(M_tokens, K, dtype=torch.bfloat16, device=device)
topk_ids = torch.randint(0, E, (M_tokens, topk), dtype=torch.int32, device=device)

# Weights
w1_fp4 = torch.randint(0, 255, (E, N, K // 2), dtype=torch.uint8, device=device)
w2_fp4 = torch.randint(0, 255, (E, K, N_half // 2), dtype=torch.uint8, device=device)
w1_bs = torch.ones(E, N, K // 16, dtype=torch.float8_e4m3fn, device=device)
w2_bs = torch.ones(E, K, N_half // 16, dtype=torch.float8_e4m3fn, device=device)
w1_alphas = torch.ones(N, K // 16, dtype=torch.float32, device=device)
w2_alphas = torch.ones(K, N_half // 16, dtype=torch.float32, device=device)
a1_gscale = torch.ones(E, dtype=torch.float32, device=device)
a2_gscale = torch.ones(E, dtype=torch.float32, device=device)

# Routing setup
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

c1 = torch.empty(M, N, dtype=torch.bfloat16, device=device)
c3 = torch.empty(M, K, dtype=torch.bfloat16, device=device)

# Warmup
print("Warming up...")
for _ in range(50):
    cutlass_fp4_moe_mm(c1, a_fp4, w1_fp4, a_bs, w1_bs, w1_alphas,
                       problem_sizes1, expert_offsets[:-1], blockscale_offsets[:-1])
    int_fp4, int_bs = silu_and_mul_scaled_fp4_experts_quant(
        c1, a2_gscale, expert_offsets, blockscale_offsets, topk)
    cutlass_fp4_moe_mm(c3, int_fp4, w2_fp4, int_bs, w2_bs, w2_alphas,
                       problem_sizes2, expert_offsets[:-1], blockscale_offsets[:-1])
torch.cuda.synchronize()

if ncu_mode:
    # NCU mode: run exactly 1 iteration for clean profiling
    print("NCU capture: 1 iteration...")
    torch.cuda.cudart().cudaProfilerStart()

    # GEMM1
    cutlass_fp4_moe_mm(c1, a_fp4, w1_fp4, a_bs, w1_bs, w1_alphas,
                       problem_sizes1, expert_offsets[:-1], blockscale_offsets[:-1])

    # SiLU + requant
    int_fp4, int_bs = silu_and_mul_scaled_fp4_experts_quant(
        c1, a2_gscale, expert_offsets, blockscale_offsets, topk)

    # GEMM2
    cutlass_fp4_moe_mm(c3, int_fp4, w2_fp4, int_bs, w2_bs, w2_alphas,
                       problem_sizes2, expert_offsets[:-1], blockscale_offsets[:-1])

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print("NCU capture complete. Use: ncu --import /tmp/moe_profile.ncu-rep")
else:
    # Event timing mode
    print("Profiling with CUDA events (500 iterations)...")
    N_ITER = 500

    gemm1_times = []
    act_times = []
    gemm2_times = []

    for _ in range(N_ITER):
        s1 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        s2 = torch.cuda.Event(enable_timing=True)
        e2 = torch.cuda.Event(enable_timing=True)
        s3 = torch.cuda.Event(enable_timing=True)
        e3 = torch.cuda.Event(enable_timing=True)

        s1.record()
        cutlass_fp4_moe_mm(c1, a_fp4, w1_fp4, a_bs, w1_bs, w1_alphas,
                           problem_sizes1, expert_offsets[:-1], blockscale_offsets[:-1])
        e1.record()

        s2.record()
        int_fp4, int_bs = silu_and_mul_scaled_fp4_experts_quant(
            c1, a2_gscale, expert_offsets, blockscale_offsets, topk)
        e2.record()

        s3.record()
        cutlass_fp4_moe_mm(c3, int_fp4, w2_fp4, int_bs, w2_bs, w2_alphas,
                           problem_sizes2, expert_offsets[:-1], blockscale_offsets[:-1])
        e3.record()

        torch.cuda.synchronize()
        gemm1_times.append(s1.elapsed_time(e1))
        act_times.append(s2.elapsed_time(e2))
        gemm2_times.append(s3.elapsed_time(e3))

    # Skip first 50 warmup
    gemm1_times = gemm1_times[50:]
    act_times = act_times[50:]
    gemm2_times = gemm2_times[50:]

    def stats(v):
        s = sorted(v)
        n = len(s)
        return {
            "avg": sum(s)/n * 1000,
            "med": s[n//2] * 1000,
            "p5": s[int(n*0.05)] * 1000,
            "p95": s[int(n*0.95)] * 1000,
            "min": s[0] * 1000,
            "max": s[-1] * 1000,
        }

    g1 = stats(gemm1_times)
    ac = stats(act_times)
    g2 = stats(gemm2_times)

    print(f"\n{'Kernel':<16} {'Avg':>8} {'Med':>8} {'P5':>8} {'P95':>8} {'Min':>8} {'Max':>8}  (μs)")
    print("-" * 80)
    print(f"{'GEMM1':<16} {g1['avg']:8.1f} {g1['med']:8.1f} {g1['p5']:8.1f} {g1['p95']:8.1f} {g1['min']:8.1f} {g1['max']:8.1f}")
    print(f"{'SiLU+requant':<16} {ac['avg']:8.1f} {ac['med']:8.1f} {ac['p5']:8.1f} {ac['p95']:8.1f} {ac['min']:8.1f} {ac['max']:8.1f}")
    print(f"{'GEMM2':<16} {g2['avg']:8.1f} {g2['med']:8.1f} {g2['p5']:8.1f} {g2['p95']:8.1f} {g2['min']:8.1f} {g2['max']:8.1f}")
    total_med = g1['med'] + ac['med'] + g2['med']
    print(f"{'TOTAL':<16} {'-':>8} {total_med:8.1f}")

    print(f"\nGEMM1 is {g1['med']/g2['med']:.1f}x slower than GEMM2")
    print(f"GEMM1 problem: [{M},{K}]x[{K},{N}] = {M*K*N/1e6:.1f}M MACs")
    print(f"GEMM2 problem: [{M},{N_half}]x[{N_half},{K}] = {M*N_half*K/1e6:.1f}M MACs")
    print(f"GEMM1/GEMM2 MAC ratio: {(M*K*N)/(M*N_half*K):.1f}x (explains latency ratio)")

    # Compute arithmetic intensity
    # GEMM1: 10 experts × [1,4096]×[4096,512]
    gemm1_flops = topk * 2 * M_tokens * K * N  # 2 for multiply-add
    gemm1_bytes = topk * (M_tokens * K / 2 + K * N / 2)  # FP4 packed
    gemm1_ai = gemm1_flops / gemm1_bytes
    gemm1_tflops = gemm1_flops / (g1['med'] * 1e-6) / 1e12

    gemm2_flops = topk * 2 * M_tokens * N_half * K
    gemm2_bytes = topk * (M_tokens * N_half / 2 + N_half * K / 2)
    gemm2_ai = gemm2_flops / gemm2_bytes
    gemm2_tflops = gemm2_flops / (g2['med'] * 1e-6) / 1e12

    print(f"\nGEMM1: {gemm1_tflops:.3f} TFLOPS, AI={gemm1_ai:.1f} flops/byte, weight={topk*K*N/2/1e6:.1f}MB")
    print(f"GEMM2: {gemm2_tflops:.3f} TFLOPS, AI={gemm2_ai:.1f} flops/byte, weight={topk*N_half*K/2/1e6:.1f}MB")
    print(f"Peak SM120 FP4: ~200 TFLOPS → GEMM1 util: {gemm1_tflops/200*100:.2f}%, GEMM2 util: {gemm2_tflops/200*100:.2f}%")

    # Memory bandwidth analysis
    bw_tbps = 1.5  # ~1.5 TB/s HBM for RTX PRO 6000
    gemm1_bw = gemm1_bytes / (g1['med'] * 1e-6) / 1e12
    gemm2_bw = gemm2_bytes / (g2['med'] * 1e-6) / 1e12
    print(f"\nGEMM1 BW: {gemm1_bw:.3f} TB/s ({gemm1_bw/bw_tbps*100:.1f}% of {bw_tbps} TB/s)")
    print(f"GEMM2 BW: {gemm2_bw:.3f} TB/s ({gemm2_bw/bw_tbps*100:.1f}% of {bw_tbps} TB/s)")
