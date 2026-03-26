#!/usr/bin/env python3
"""
Standalone benchmark: FP8 vs BF16 KV cache flash decode on SM120.

Measures μs/layer for both dtypes at various sequence lengths.
Expected: FP8 ~1.5-2x faster (half the KV bytes to load from HBM).
"""

import sys
import os
import torch
import time
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sm120_flash_decode_ext import sm120_flash_decode_paged, SM120FlashDecodeWorkspace

DEVICE = "cuda"
WARMUP = 50
ITERS = 200


def make_bench_data(batch, num_q_heads, num_kv_heads, head_dim, seq_len,
                    block_size, dtype_kv):
    """Create synthetic paged KV cache data for benchmarking."""
    Q = torch.randn(batch, num_q_heads, head_dim, dtype=torch.bfloat16, device=DEVICE)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=DEVICE)

    total_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = batch * total_blocks_per_seq
    max_blocks_per_seq = total_blocks_per_seq

    K_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim,
                          dtype=torch.bfloat16, device=DEVICE)
    V_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim,
                          dtype=torch.bfloat16, device=DEVICE)

    if dtype_kv == "fp8":
        K_cache = K_cache.to(torch.float8_e4m3fn)
        V_cache = V_cache.to(torch.float8_e4m3fn)

    # Simple sequential block table
    block_table = torch.zeros(batch, max_blocks_per_seq, dtype=torch.int32, device=DEVICE)
    for b in range(batch):
        for i in range(total_blocks_per_seq):
            block_table[b, i] = b * total_blocks_per_seq + i

    output = torch.empty(batch, num_q_heads, head_dim, dtype=torch.bfloat16, device=DEVICE)
    workspace = SM120FlashDecodeWorkspace(
        max_batch_size=batch, num_q_heads=num_q_heads, head_dim=head_dim,
        max_splits=32, device=DEVICE
    )

    return Q, K_cache, V_cache, block_table, seq_lens, output, workspace


def bench_one(batch, num_q_heads, num_kv_heads, head_dim, seq_len,
              block_size, dtype_kv, k_scale=1.0, v_scale=1.0):
    """Benchmark a single configuration, return μs/call."""
    Q, K_cache, V_cache, block_table, seq_lens, output, workspace = make_bench_data(
        batch, num_q_heads, num_kv_heads, head_dim, seq_len, block_size, dtype_kv
    )

    # Warmup
    for _ in range(WARMUP):
        sm120_flash_decode_paged(
            Q, K_cache, V_cache, block_table, seq_lens,
            output=output, workspace=workspace, max_seq_len=seq_len,
            k_scale=k_scale, v_scale=v_scale,
        )
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(ITERS):
        sm120_flash_decode_paged(
            Q, K_cache, V_cache, block_table, seq_lens,
            output=output, workspace=workspace, max_seq_len=seq_len,
            k_scale=k_scale, v_scale=v_scale,
        )
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    us_per_call = (total_ms / ITERS) * 1000
    return us_per_call


def main():
    print("=" * 90)
    print("SM120 Flash Decode v2 — FP8 vs BF16 KV Cache Benchmark")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Warmup: {WARMUP}, Iterations: {ITERS}")
    print("=" * 90)

    # Qwen3.5 config: 64 q_heads, 4 kv_heads, HD=128
    configs = [
        # (batch, q_heads, kv_heads, head_dim, seq_len, block_size, label)
        (1, 64, 4, 128, 256, 16, "256"),
        (1, 64, 4, 128, 1024, 16, "1K"),
        (1, 64, 4, 128, 4096, 16, "4K"),
        (1, 64, 4, 128, 16384, 16, "16K"),
        (1, 64, 4, 128, 32768, 16, "32K"),
        # HD=256
        (1, 64, 4, 256, 256, 16, "256 HD256"),
        (1, 64, 4, 256, 4096, 16, "4K HD256"),
        (1, 64, 4, 256, 16384, 16, "16K HD256"),
        # Multi-batch
        (8, 64, 4, 128, 4096, 16, "4K x8"),
    ]

    print(f"\n{'Config':<20} {'BF16 (μs)':<12} {'FP8 (μs)':<12} {'Speedup':<10} {'BW saved':<10}")
    print("-" * 70)

    for batch, qh, kvh, hd, sl, bs, label in configs:
        try:
            us_bf16 = bench_one(batch, qh, kvh, hd, sl, bs, "bf16")
            us_fp8 = bench_one(batch, qh, kvh, hd, sl, bs, "fp8", k_scale=1.0, v_scale=1.0)
            speedup = us_bf16 / us_fp8
            # KV bytes: BF16=2B, FP8=1B per element
            kv_bytes_bf16 = batch * sl * kvh * hd * 2 * 2  # K+V, 2B each
            kv_bytes_fp8 = batch * sl * kvh * hd * 2 * 1   # K+V, 1B each
            bw_saved = (1 - kv_bytes_fp8 / kv_bytes_bf16) * 100

            print(f"  {label:<18} {us_bf16:<12.1f} {us_fp8:<12.1f} {speedup:<10.2f}x {bw_saved:.0f}%")
        except Exception as e:
            print(f"  {label:<18} ERROR: {e}")

    # HBM bandwidth analysis
    print("\n--- HBM Bandwidth Utilization ---")
    for dtype_label, dtype_kv in [("BF16", "bf16"), ("FP8", "fp8")]:
        batch, qh, kvh, hd, sl, bs = 1, 64, 4, 128, 4096, 16
        us = bench_one(batch, qh, kvh, hd, sl, bs, dtype_kv)
        elem_bytes = 2 if dtype_kv == "bf16" else 1
        kv_bytes = batch * sl * kvh * hd * 2 * elem_bytes  # K+V
        q_bytes = batch * qh * hd * 2  # Q always BF16
        total_bytes = kv_bytes + q_bytes
        bw_gbps = total_bytes / (us * 1e-6) / 1e9
        # RTX PRO 6000 Blackwell: ~1152 GB/s HBM
        roofline_pct = bw_gbps / 1152 * 100
        print(f"  {dtype_label}: {us:.1f} μs, {bw_gbps:.0f} GB/s ({roofline_pct:.0f}% HBM roofline)")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
