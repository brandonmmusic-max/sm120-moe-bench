#!/usr/bin/env python3
"""
Head-to-head comparison: SM120 Flash Decode vs FlashInfer single_decode_with_kv_cache.

Same inputs (Q BF16, K/V FP8 E4M3, same scales) → compare BF16 outputs.
Target: max absolute error < 1e-5 (near bit-level match in BF16).
"""

import sys
import os
import torch
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sm120_flash_decode_ext import sm120_flash_decode_paged

import flashinfer

torch.manual_seed(42)
DEVICE = "cuda"


def run_comparison(num_q_heads, num_kv_heads, head_dim, seq_len,
                   k_scale, v_scale, block_size=16, label=""):
    """Run one head-to-head comparison."""
    batch = 1  # single_decode is single-request
    gqa_ratio = num_q_heads // num_kv_heads

    # Generate random Q (BF16), K/V (FP8 E4M3)
    Q = torch.randn(batch, num_q_heads, head_dim, dtype=torch.bfloat16, device=DEVICE) * 0.1

    # Create contiguous K/V for FlashInfer (NHD layout: [seq_len, num_kv_heads, head_dim])
    K_bf16 = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=DEVICE) * 0.1
    V_bf16 = torch.randn(seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=DEVICE) * 0.1
    K_fp8 = K_bf16.to(torch.float8_e4m3fn)
    V_fp8 = V_bf16.to(torch.float8_e4m3fn)

    # ---- FlashInfer reference ----
    # single_decode_with_kv_cache expects q: [num_qo_heads, head_dim], k/v: [kv_len, num_kv_heads, head_dim]
    fi_out = flashinfer.single_decode_with_kv_cache(
        Q[0],  # [num_q_heads, head_dim]
        K_fp8,  # [seq_len, num_kv_heads, head_dim]
        V_fp8,  # [seq_len, num_kv_heads, head_dim]
        kv_layout="NHD",
        k_scale=k_scale,
        v_scale=v_scale,
    )  # returns [num_q_heads, head_dim] BF16

    # ---- Our SM120 kernel ----
    # Need to create paged KV cache from contiguous K/V
    num_blocks = (seq_len + block_size - 1) // block_size
    total_blocks = num_blocks

    # Pack into paged format: [num_blocks, block_size, num_kv_heads, head_dim]
    K_paged = torch.zeros(total_blocks, block_size, num_kv_heads, head_dim,
                          dtype=torch.float8_e4m3fn, device=DEVICE)
    V_paged = torch.zeros(total_blocks, block_size, num_kv_heads, head_dim,
                          dtype=torch.float8_e4m3fn, device=DEVICE)

    for blk in range(num_blocks):
        start = blk * block_size
        end = min(start + block_size, seq_len)
        length = end - start
        K_paged[blk, :length] = K_fp8[start:end]
        V_paged[blk, :length] = V_fp8[start:end]

    block_table = torch.arange(total_blocks, dtype=torch.int32, device=DEVICE).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)

    our_out = sm120_flash_decode_paged(
        query=Q, key_cache=K_paged, value_cache=V_paged,
        block_table=block_table, seq_lens=seq_lens,
        max_seq_len=seq_len, k_scale=k_scale, v_scale=v_scale,
    )
    torch.cuda.synchronize()

    # ---- Compare ----
    fi_fp32 = fi_out.float()
    our_fp32 = our_out[0].float()  # remove batch dim

    abs_err = (our_fp32 - fi_fp32).abs()
    max_abs_err = abs_err.max().item()
    mean_abs_err = abs_err.mean().item()
    ref_norm = fi_fp32.abs().mean().item()
    rel_err_pct = max_abs_err / max(ref_norm, 1e-8) * 100

    cos_sim = torch.nn.functional.cosine_similarity(
        our_fp32.flatten().unsqueeze(0),
        fi_fp32.flatten().unsqueeze(0),
    ).item()

    # Count BF16-exact matches
    fi_bf16_bits = fi_out.view(torch.int16)
    our_bf16_bits = our_out[0].view(torch.int16)
    exact_match_pct = (fi_bf16_bits == our_bf16_bits).float().mean().item() * 100

    # Count within 1 ULP
    diff_ulps = (fi_bf16_bits.int() - our_bf16_bits.int()).abs()
    within_1ulp_pct = (diff_ulps <= 1).float().mean().item() * 100

    passed = max_abs_err < 1e-3 and cos_sim > 0.9999
    status = "PASS" if passed else "FAIL"

    print(f"  [{status}] {label}: qh={num_q_heads} kvh={num_kv_heads} hd={head_dim} "
          f"sl={seq_len} ks={k_scale:.4f} vs={v_scale:.4f}")
    print(f"    max_abs={max_abs_err:.2e}  mean_abs={mean_abs_err:.2e}  "
          f"rel={rel_err_pct:.3f}%  cos={cos_sim:.8f}")
    print(f"    BF16 exact match: {exact_match_pct:.1f}%  within 1 ULP: {within_1ulp_pct:.1f}%")

    return passed


def main():
    print("=" * 80)
    print("SM120 Flash Decode vs FlashInfer — Head-to-Head FP8 Comparison")
    print("=" * 80)

    results = []

    print("\n--- Standard configs (HD=128) ---")
    results.append(run_comparison(8, 2, 128, 64, k_scale=1.0, v_scale=1.0, label="noscale-short"))
    results.append(run_comparison(8, 2, 128, 256, k_scale=1.0, v_scale=1.0, label="noscale-256"))
    results.append(run_comparison(8, 2, 128, 256, k_scale=0.0078125, v_scale=0.0078125, label="fp8-scale-256"))
    results.append(run_comparison(32, 4, 128, 512, k_scale=0.0078125, v_scale=0.0078125, label="GQA32:4-fp8-512"))

    print("\n--- Longer sequences ---")
    results.append(run_comparison(8, 2, 128, 1024, k_scale=0.0078125, v_scale=0.0078125, label="fp8-1024"))
    results.append(run_comparison(8, 2, 128, 2048, k_scale=0.0078125, v_scale=0.0078125, label="fp8-2048"))

    print("\n--- HD=256 ---")
    results.append(run_comparison(4, 2, 256, 128, k_scale=1.0, v_scale=1.0, label="hd256-noscale"))
    results.append(run_comparison(4, 2, 256, 128, k_scale=0.0078125, v_scale=0.0078125, label="hd256-fp8"))

    print("\n" + "=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} PASS")
    print("ALL TESTS PASSED" if passed == total else "SOME TESTS FAILED")
    print("=" * 80)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
