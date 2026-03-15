"""
Benchmark SM120 Flash Attention vs PyTorch SDPA.

Measures TFLOPS and latency for various sequence lengths.
"""

import torch
import torch.nn.functional as F
import sm120_flash_attn
import time
import argparse


def benchmark_one(func, warmup=5, iters=20):
    """Run function and return median time in ms."""
    # Warmup
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        func()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times.sort()
    return times[len(times) // 2]  # median


def compute_tflops(batch, heads, seq_q, seq_kv, head_dim, time_ms):
    """Compute attention TFLOPS (forward only)."""
    # FLOPs = 2 * batch * heads * seq_q * seq_kv * head_dim (Q@K^T)
    #       + 2 * batch * heads * seq_q * seq_kv * head_dim (P@V)
    flops = 4.0 * batch * heads * seq_q * seq_kv * head_dim
    return flops / (time_ms * 1e-3) / 1e12


def main():
    parser = argparse.ArgumentParser(description="SM120 FA Benchmark")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16

    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    print(f"{'':>8s} | {'SM120 FA':>12s} {'TFLOPS':>8s} | {'torch SDPA':>12s} {'TFLOPS':>8s} | {'Speedup':>8s}")
    print("-" * 75)

    for seq_len in seq_lengths:
        B = args.batch
        Hq = args.heads
        Hkv = args.kv_heads
        D = args.head_dim

        Q = torch.randn(B, Hq, seq_len, D, device=device, dtype=dtype)
        K = torch.randn(B, Hkv, seq_len, D, device=device, dtype=dtype)
        V = torch.randn(B, Hkv, seq_len, D, device=device, dtype=dtype)

        # Expand for SDPA
        kv_repeat = Hq // Hkv
        Ke = K.repeat_interleave(kv_repeat, dim=1)
        Ve = V.repeat_interleave(kv_repeat, dim=1)

        # Benchmark SM120 FA
        def run_sm120():
            sm120_flash_attn.forward(Q, K, V, False)

        t_sm120 = benchmark_one(run_sm120, iters=args.iters)
        tflops_sm120 = compute_tflops(B, Hq, seq_len, seq_len, D, t_sm120)

        # Benchmark torch SDPA
        def run_sdpa():
            F.scaled_dot_product_attention(Q, Ke, Ve)

        t_sdpa = benchmark_one(run_sdpa, iters=args.iters)
        tflops_sdpa = compute_tflops(B, Hq, seq_len, seq_len, D, t_sdpa)

        speedup = t_sdpa / t_sm120

        print(f"{seq_len:>8d} | {t_sm120:>9.2f} ms {tflops_sm120:>7.1f} | {t_sdpa:>9.2f} ms {tflops_sdpa:>7.1f} | {speedup:>7.2f}x")

    # Decode-like benchmark (Q=1, varying KV length)
    print()
    print("Decode-like (Q=1 token, varying KV cache length):")
    print(f"{'KV len':>8s} | {'SM120 FA':>12s} {'TFLOPS':>8s} | {'torch SDPA':>12s} {'TFLOPS':>8s} | {'Speedup':>8s}")
    print("-" * 75)

    for kv_len in [1024, 4096, 8192, 16384, 32768]:
        Q = torch.randn(1, Hq, 1, D, device=device, dtype=dtype)
        K = torch.randn(1, Hkv, kv_len, D, device=device, dtype=dtype)
        V = torch.randn(1, Hkv, kv_len, D, device=device, dtype=dtype)

        Ke = K.repeat_interleave(kv_repeat, dim=1)
        Ve = V.repeat_interleave(kv_repeat, dim=1)

        def run_sm120():
            sm120_flash_attn.forward(Q, K, V, False)

        def run_sdpa():
            F.scaled_dot_product_attention(Q, Ke, Ve)

        t_sm120 = benchmark_one(run_sm120, iters=args.iters)
        t_sdpa = benchmark_one(run_sdpa, iters=args.iters)
        tflops_sm120 = compute_tflops(1, Hq, 1, kv_len, D, t_sm120)
        tflops_sdpa = compute_tflops(1, Hq, 1, kv_len, D, t_sdpa)
        speedup = t_sdpa / t_sm120

        print(f"{kv_len:>8d} | {t_sm120:>9.3f} ms {tflops_sm120:>7.2f} | {t_sdpa:>9.3f} ms {tflops_sdpa:>7.2f} | {speedup:>7.2f}x")


if __name__ == "__main__":
    main()
