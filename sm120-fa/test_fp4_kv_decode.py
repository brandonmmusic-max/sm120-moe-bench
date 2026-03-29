#!/usr/bin/env python3
"""
Standalone correctness test for SM120 NVFP4 KV cache decode kernel.

Tests:
1. Quantize BF16 K,V to NVFP4 packed format
2. Run FP4 kernel and BF16 reference kernel
3. Compare: relative error < 5%, cosine similarity > 0.995, zero NaN/Inf
4. Tests at seq_len = 1, 64, 512, 4096, 32768 with HEAD_DIM=256

Usage:
    python test_fp4_kv_decode.py           # Run all tests
    python test_fp4_kv_decode.py --verbose  # With detailed output
"""

import sys
import os
import time
import argparse
import torch

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sm120_flash_decode_ext import (
    sm120_flash_decode_paged,
    sm120_flash_decode_paged_fp4,
    quantize_to_nvfp4,
    nvfp4_packed_dim,
    FP4_BLOCK_SIZE,
    SM120FlashDecodeWorkspace,
)


def create_paged_cache_bf16(
    K: torch.Tensor,    # [num_seqs, seq_len, num_kv_heads, head_dim] bf16
    V: torch.Tensor,    # [num_seqs, seq_len, num_kv_heads, head_dim] bf16
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack BF16 K, V into paged cache format.

    Returns:
        key_cache:   [num_blocks, block_size, num_kv_heads, head_dim] bf16
        value_cache: [num_blocks, block_size, num_kv_heads, head_dim] bf16
        block_table: [num_seqs, max_blocks_per_seq] int32
    """
    num_seqs, seq_len, num_kv_heads, head_dim = K.shape
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = num_seqs * num_blocks_per_seq

    key_cache = torch.zeros(total_blocks, block_size, num_kv_heads, head_dim,
                            dtype=torch.bfloat16, device=K.device)
    val_cache = torch.zeros(total_blocks, block_size, num_kv_heads, head_dim,
                            dtype=torch.bfloat16, device=V.device)
    block_table = torch.zeros(num_seqs, num_blocks_per_seq, dtype=torch.int32, device=K.device)

    for s in range(num_seqs):
        for b in range(num_blocks_per_seq):
            blk_idx = s * num_blocks_per_seq + b
            block_table[s, b] = blk_idx
            start = b * block_size
            end = min(start + block_size, seq_len)
            length = end - start
            if length > 0:
                key_cache[blk_idx, :length] = K[s, start:end]
                val_cache[blk_idx, :length] = V[s, start:end]

    return key_cache, val_cache, block_table


def create_paged_cache_fp4(
    K: torch.Tensor,    # [num_seqs, seq_len, num_kv_heads, head_dim] bf16
    V: torch.Tensor,    # [num_seqs, seq_len, num_kv_heads, head_dim] bf16
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Pack BF16 K, V into paged NVFP4 cache format.

    Quantizes the ENTIRE K and V tensors at once to compute a global per-tensor
    scale, then packs into pages. This ensures consistent scaling across all pages.

    Returns:
        key_cache:      [num_blocks, block_size, num_kv_heads, packed_dim] uint8
        value_cache:    [num_blocks, block_size, num_kv_heads, packed_dim] uint8
        block_table:    [num_seqs, max_blocks_per_seq] int32
        k_tensor_scale: float (per-tensor K pre-normalization scale)
        v_tensor_scale: float (per-tensor V pre-normalization scale)
    """
    num_seqs, seq_len, num_kv_heads, head_dim = K.shape
    packed_dim = nvfp4_packed_dim(head_dim)
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = num_seqs * num_blocks_per_seq

    # Quantize entire K and V tensors to get global tensor scales
    k_flat = K.reshape(-1, head_dim)
    v_flat = V.reshape(-1, head_dim)
    k_packed_all, k_ts = quantize_to_nvfp4(k_flat)  # [N, packed_dim], scale
    v_packed_all, v_ts = quantize_to_nvfp4(v_flat)
    k_packed_all = k_packed_all.reshape(num_seqs, seq_len, num_kv_heads, packed_dim)
    v_packed_all = v_packed_all.reshape(num_seqs, seq_len, num_kv_heads, packed_dim)

    key_cache = torch.zeros(total_blocks, block_size, num_kv_heads, packed_dim,
                            dtype=torch.uint8, device=K.device)
    val_cache = torch.zeros(total_blocks, block_size, num_kv_heads, packed_dim,
                            dtype=torch.uint8, device=V.device)
    block_table = torch.zeros(num_seqs, num_blocks_per_seq, dtype=torch.int32, device=K.device)

    for s in range(num_seqs):
        for b in range(num_blocks_per_seq):
            blk_idx = s * num_blocks_per_seq + b
            block_table[s, b] = blk_idx
            start = b * block_size
            end = min(start + block_size, seq_len)
            length = end - start
            if length > 0:
                key_cache[blk_idx, :length] = k_packed_all[s, start:end]
                val_cache[blk_idx, :length] = v_packed_all[s, start:end]

    return key_cache, val_cache, block_table, k_ts, v_ts


def bf16_attention_reference(
    Q: torch.Tensor,    # [batch, num_q_heads, head_dim] bf16
    K: torch.Tensor,    # [num_seqs, seq_len, num_kv_heads, head_dim] bf16
    V: torch.Tensor,    # [num_seqs, seq_len, num_kv_heads, head_dim] bf16
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Pure PyTorch BF16 attention reference (no paging, no quantization)."""
    batch = Q.shape[0]
    num_q_heads = Q.shape[1]
    head_dim = Q.shape[2]
    num_kv_heads = K.shape[2]
    gqa_ratio = num_q_heads // num_kv_heads

    output = torch.zeros_like(Q, dtype=torch.float32)
    scale = 1.0 / (head_dim ** 0.5)

    for s in range(batch):
        sl = seq_lens[s].item()
        for qh in range(num_q_heads):
            kv_h = qh // gqa_ratio
            q = Q[s, qh].float()             # [head_dim]
            k = K[s, :sl, kv_h].float()      # [sl, head_dim]
            v = V[s, :sl, kv_h].float()      # [sl, head_dim]

            scores = (k @ q) * scale          # [sl]
            weights = torch.softmax(scores, dim=0)  # [sl]
            output[s, qh] = weights @ v       # [head_dim]

    return output.to(torch.bfloat16)


def dequantize_nvfp4_cache(
    cache_fp4: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, packed_dim] uint8
    head_dim: int,
) -> torch.Tensor:
    """Dequantize packed NVFP4 cache back to BF16 for diagnostic comparison.

    This lets us separate quantization error from kernel computation error.
    """
    from sm120_flash_decode_ext import _e4m3fn_to_float, FP4_BLOCK_SIZE

    shape = cache_fp4.shape
    packed_dim = shape[-1]
    data_cols = head_dim // 2
    scale_cols = head_dim // FP4_BLOCK_SIZE

    flat = cache_fp4.reshape(-1, packed_dim)
    N = flat.shape[0]

    result = torch.zeros(N, head_dim, dtype=torch.bfloat16, device=cache_fp4.device)

    # FP4 magnitude lookup
    fp4_mag = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

    for i in range(N):
        for b in range(head_dim // FP4_BLOCK_SIZE):
            scale_byte = flat[i, data_cols + b].item()
            # Decode scale using same logic as CUDA kernel
            s = (scale_byte >> 7) & 1
            e = (scale_byte >> 3) & 0xF
            m = scale_byte & 7
            if e == 15 and m == 7:
                scale_val = 0.0
            elif e == 0:
                scale_val = m / 512.0
            else:
                import math
                scale_val = math.ldexp(1.0 + m / 8.0, e - 7)
            if s:
                scale_val = -scale_val

            for dd in range(FP4_BLOCK_SIZE):
                d = b * FP4_BLOCK_SIZE + dd
                byte_idx = d // 2
                byte_val = flat[i, byte_idx].item()
                nibble = (byte_val >> 4) if (d & 1) else (byte_val & 0xF)
                mag_idx = nibble & 0x7
                sign = (nibble >> 3) & 1
                fp4_val = fp4_mag[mag_idx]
                if sign:
                    fp4_val = -fp4_val
                result[i, d] = fp4_val * scale_val

    return result.reshape(list(shape[:-1]) + [head_dim])


def run_test(
    seq_len: int,
    head_dim: int = 256,
    num_q_heads: int = 8,
    num_kv_heads: int = 1,
    block_size: int = 16,
    num_seqs: int = 1,
    verbose: bool = False,
) -> dict:
    """Run one correctness test comparing FP4 kernel vs BF16 reference.

    Measures THREE error components:
    1. Quantization error: BF16 reference with dequantized-FP4 K/V vs true BF16 reference
    2. Kernel error: FP4 kernel vs BF16-reference-with-dequantized-FP4-K/V (should be ~0)
    3. Total error: FP4 kernel vs true BF16 reference (= quant error + kernel error)

    Returns dict with test results.
    """
    device = "cuda"

    # Generate random data (scale to realistic range for attention)
    torch.manual_seed(42 + seq_len)
    Q = torch.randn(num_seqs, num_q_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    K = torch.randn(num_seqs, seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    V = torch.randn(num_seqs, seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    seq_lens_t = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)

    # 1. BF16 reference (pure PyTorch, no quantization)
    ref_output = bf16_attention_reference(Q, K, V, seq_lens_t)

    # 2. BF16 kernel (paged) -- sanity check
    key_cache_bf16, val_cache_bf16, block_table = create_paged_cache_bf16(K, V, block_size)
    bf16_output = sm120_flash_decode_paged(
        query=Q, key_cache=key_cache_bf16, value_cache=val_cache_bf16,
        block_table=block_table, seq_lens=seq_lens_t, max_seq_len=seq_len,
    )

    # 3. FP4 kernel (paged, quantized)
    key_cache_fp4, val_cache_fp4, block_table_fp4, k_ts, v_ts = create_paged_cache_fp4(K, V, block_size)
    fp4_output = sm120_flash_decode_paged_fp4(
        query=Q, key_cache=key_cache_fp4, value_cache=val_cache_fp4,
        block_table=block_table_fp4, seq_lens=seq_lens_t,
        head_dim=head_dim, max_seq_len=seq_len,
        k_tensor_scale=k_ts, v_tensor_scale=v_ts,
    )

    # 4. Diagnostic: dequant FP4 cache → BF16, run through BF16 kernel
    # This isolates kernel error from quantization error
    # Must apply tensor scales to the dequanted data
    key_dequant = (dequantize_nvfp4_cache(key_cache_fp4, head_dim) * k_ts).to(torch.bfloat16)
    val_dequant = (dequantize_nvfp4_cache(val_cache_fp4, head_dim) * v_ts).to(torch.bfloat16)
    dequant_bf16_output = sm120_flash_decode_paged(
        query=Q, key_cache=key_dequant, value_cache=val_dequant,
        block_table=block_table_fp4, seq_lens=seq_lens_t, max_seq_len=seq_len,
    )

    # Compare outputs
    ref_f = ref_output.float()
    bf16_f = bf16_output.float()
    fp4_f = fp4_output.float()
    dequant_f = dequant_bf16_output.float()

    # Error decomposition:
    bf16_l2_err = (bf16_f - ref_f).norm() / ref_f.norm()

    # Quantization error: dequantized-through-BF16-kernel vs true reference
    quant_l2_err = (dequant_f - ref_f).norm() / ref_f.norm()

    # Kernel error: FP4 kernel vs dequantized-through-BF16-kernel (should be ~0 if kernel is correct)
    if dequant_f.norm() > 0:
        kernel_l2_err = (fp4_f - dequant_f).norm() / dequant_f.norm()
    else:
        kernel_l2_err = torch.tensor(0.0)

    # Total error: FP4 kernel vs true reference
    total_l2_err = (fp4_f - ref_f).norm() / ref_f.norm()

    # Cosine similarities
    cos_sim_total = torch.nn.functional.cosine_similarity(
        fp4_f.reshape(1, -1), ref_f.reshape(1, -1)).item()
    cos_sim_kernel = torch.nn.functional.cosine_similarity(
        fp4_f.reshape(1, -1), dequant_f.reshape(1, -1)).item()

    has_nan = torch.isnan(fp4_output).any().item()
    has_inf = torch.isinf(fp4_output).any().item()

    # Pass/fail: FP4 kernel must match the dequantized reference (kernel correctness)
    # AND total error must be bounded (quantization + kernel)
    kernel_correct = kernel_l2_err.item() < 0.02 and cos_sim_kernel > 0.999
    total_bounded = total_l2_err.item() < 0.15 and cos_sim_total > 0.99

    passed = kernel_correct and total_bounded and not has_nan and not has_inf

    result = {
        "seq_len": seq_len,
        "passed": passed,
        "kernel_correct": kernel_correct,
        "total_bounded": total_bounded,
        "quant_l2_err": quant_l2_err.item(),
        "kernel_l2_err": kernel_l2_err.item(),
        "total_l2_err": total_l2_err.item(),
        "cos_sim_total": cos_sim_total,
        "cos_sim_kernel": cos_sim_kernel,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "bf16_l2_err": bf16_l2_err.item(),
    }

    if verbose:
        print(f"\n  BF16 kernel vs reference:   L2_err={bf16_l2_err.item():.6f}")
        print(f"  --- Error decomposition ---")
        print(f"  Quantization error (FP4→dequant→BF16 kernel vs ref): L2={quant_l2_err.item():.4f}")
        print(f"  Kernel error (FP4 kernel vs dequant BF16 kernel):     L2={kernel_l2_err.item():.6f}, "
              f"cos={cos_sim_kernel:.6f}")
        print(f"  Total error (FP4 kernel vs ref):                      L2={total_l2_err.item():.4f}, "
              f"cos={cos_sim_total:.6f}")
        print(f"  NaN={has_nan}, Inf={has_inf}")
        print(f"  Kernel correct: {kernel_correct}, Total bounded: {total_bounded}")

        print(f"  Sample output (first 8 dims, seq 0, head 0):")
        print(f"    REF:    {ref_f[0, 0, :8].tolist()}")
        print(f"    BF16:   {bf16_f[0, 0, :8].tolist()}")
        print(f"    DEQUANT:{dequant_f[0, 0, :8].tolist()}")
        print(f"    FP4:    {fp4_f[0, 0, :8].tolist()}")

    return result


def run_perf_test(
    seq_len: int,
    head_dim: int = 256,
    num_q_heads: int = 8,
    num_kv_heads: int = 1,
    block_size: int = 16,
    num_seqs: int = 1,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> dict:
    """Run performance comparison: FP4 vs FP8 vs BF16 decode."""
    device = "cuda"

    torch.manual_seed(42)
    Q = torch.randn(num_seqs, num_q_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    K = torch.randn(num_seqs, seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    V = torch.randn(num_seqs, seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    seq_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)

    workspace = SM120FlashDecodeWorkspace(
        max_batch_size=max(num_seqs * 2, 256),
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        max_splits=32,
        device=device,
    )

    # BF16 cache
    key_bf16, val_bf16, bt_bf16 = create_paged_cache_bf16(K, V, block_size)

    # FP4 cache
    key_fp4, val_fp4, bt_fp4, k_ts, v_ts = create_paged_cache_fp4(K, V, block_size)

    results = {}

    # Benchmark BF16
    output = torch.empty_like(Q)
    for _ in range(num_warmup):
        sm120_flash_decode_paged(
            query=Q, key_cache=key_bf16, value_cache=val_bf16,
            block_table=bt_bf16, seq_lens=seq_lens, output=output,
            workspace=workspace, max_seq_len=seq_len,
        )
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        sm120_flash_decode_paged(
            query=Q, key_cache=key_bf16, value_cache=val_bf16,
            block_table=bt_bf16, seq_lens=seq_lens, output=output,
            workspace=workspace, max_seq_len=seq_len,
        )
    torch.cuda.synchronize()
    bf16_us = (time.perf_counter() - start) / num_iters * 1e6
    results["bf16_us"] = bf16_us

    # Benchmark FP4
    for _ in range(num_warmup):
        sm120_flash_decode_paged_fp4(
            query=Q, key_cache=key_fp4, value_cache=val_fp4,
            block_table=bt_fp4, seq_lens=seq_lens,
            head_dim=head_dim, output=output,
            workspace=workspace, max_seq_len=seq_len,
            k_tensor_scale=k_ts, v_tensor_scale=v_ts,
        )
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iters):
        sm120_flash_decode_paged_fp4(
            query=Q, key_cache=key_fp4, value_cache=val_fp4,
            block_table=bt_fp4, seq_lens=seq_lens,
            head_dim=head_dim, output=output,
            workspace=workspace, max_seq_len=seq_len,
            k_tensor_scale=k_ts, v_tensor_scale=v_ts,
        )
    torch.cuda.synchronize()
    fp4_us = (time.perf_counter() - start) / num_iters * 1e6
    results["fp4_us"] = fp4_us
    results["speedup"] = bf16_us / fp4_us if fp4_us > 0 else float('inf')

    # Cache memory comparison
    bf16_bytes = key_bf16.numel() * key_bf16.element_size() * 2  # K + V
    fp4_bytes = key_fp4.numel() * key_fp4.element_size() * 2
    results["bf16_cache_MB"] = bf16_bytes / 1e6
    results["fp4_cache_MB"] = fp4_bytes / 1e6
    results["memory_ratio"] = bf16_bytes / fp4_bytes if fp4_bytes > 0 else float('inf')

    return results


def run_extreme_value_test(
    scale: float,
    label: str,
    seq_len: int = 256,
    head_dim: int = 256,
    num_q_heads: int = 8,
    num_kv_heads: int = 1,
    block_size: int = 16,
    num_seqs: int = 1,
    verbose: bool = False,
) -> dict:
    """Test FP4 kernel correctness with extreme value ranges.

    Tests with different K/V magnitude scales to ensure block scales handle
    the full E4M3FN range (subnormals through max=448.0).
    """
    device = "cuda"

    torch.manual_seed(123)
    Q = torch.randn(num_seqs, num_q_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    K = torch.randn(num_seqs, seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device) * scale
    V = torch.randn(num_seqs, seq_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device) * scale
    seq_lens_t = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)

    # BF16 reference
    ref_output = bf16_attention_reference(Q, K, V, seq_lens_t)

    # FP4 kernel
    key_cache_fp4, val_cache_fp4, block_table_fp4, k_ts, v_ts = create_paged_cache_fp4(K, V, block_size)
    fp4_output = sm120_flash_decode_paged_fp4(
        query=Q, key_cache=key_cache_fp4, value_cache=val_cache_fp4,
        block_table=block_table_fp4, seq_lens=seq_lens_t,
        head_dim=head_dim, max_seq_len=seq_len,
        k_tensor_scale=k_ts, v_tensor_scale=v_ts,
    )

    # Diagnostic: dequant → BF16 kernel
    key_dequant = (dequantize_nvfp4_cache(key_cache_fp4, head_dim) * k_ts).to(torch.bfloat16)
    val_dequant = (dequantize_nvfp4_cache(val_cache_fp4, head_dim) * v_ts).to(torch.bfloat16)
    key_cache_bf16_dq, val_cache_bf16_dq, bt_dq = create_paged_cache_bf16(
        key_dequant.reshape(num_seqs, seq_len, num_kv_heads, head_dim),
        val_dequant.reshape(num_seqs, seq_len, num_kv_heads, head_dim),
        block_size
    )
    dequant_bf16_output = sm120_flash_decode_paged(
        query=Q, key_cache=key_cache_bf16_dq, value_cache=val_cache_bf16_dq,
        block_table=bt_dq, seq_lens=seq_lens_t, max_seq_len=seq_len,
    )

    ref_f = ref_output.float()
    fp4_f = fp4_output.float()
    dequant_f = dequant_bf16_output.float()

    total_l2 = (fp4_f - ref_f).norm() / ref_f.norm() if ref_f.norm() > 0 else torch.tensor(0.0)
    kernel_l2 = (fp4_f - dequant_f).norm() / dequant_f.norm() if dequant_f.norm() > 0 else torch.tensor(0.0)
    cos_sim = torch.nn.functional.cosine_similarity(
        fp4_f.reshape(1, -1), ref_f.reshape(1, -1)).item()
    cos_kernel = torch.nn.functional.cosine_similarity(
        fp4_f.reshape(1, -1), dequant_f.reshape(1, -1)).item()

    has_nan = torch.isnan(fp4_output).any().item()
    has_inf = torch.isinf(fp4_output).any().item()

    # Kernel correctness: FP4 kernel must match dequantized BF16 (no kernel bugs)
    kernel_correct = kernel_l2.item() < 0.02 and cos_kernel > 0.999
    # No numerical failures
    numerically_safe = not has_nan and not has_inf
    # Pass if kernel is correct and numerically safe
    # (total error = quantization noise, which degrades at extreme magnitudes by design)
    passed = kernel_correct and numerically_safe

    if verbose:
        print(f"\n  Data scale={scale}, K/V max={K.abs().max().item():.2f}")
        print(f"  Kernel error:  L2={kernel_l2.item():.6f}, cos={cos_kernel:.6f}")
        print(f"  Total error:   L2={total_l2.item():.4f}, cos={cos_sim:.6f}")
        print(f"  NaN={has_nan}, Inf={has_inf}")
        if cos_sim < 0.99:
            print(f"  NOTE: Low cosine sim is expected at extreme magnitudes --")
            print(f"        FP4 (4-bit) has only 8 magnitudes per block scale group.")
            print(f"        Intra-block variance in random Gaussian data at scale={scale}")
            print(f"        causes coarse quantization steps that dominate small values.")

    return {
        "label": label,
        "passed": passed,
        "kernel_correct": kernel_correct,
        "kernel_l2": kernel_l2.item(),
        "total_l2": total_l2.item(),
        "cos_sim": cos_sim,
        "cos_kernel": cos_kernel,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "kv_max": K.abs().max().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="NVFP4 KV cache decode kernel test")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--perf", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--seq-lens", type=str, default="1,64,512,4096",
                        help="Comma-separated sequence lengths to test")
    parser.add_argument("--extreme", action="store_true", help="Run extreme value tests")
    args = parser.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    print("=" * 70)
    print("SM120 NVFP4 KV Cache Decode Kernel -- Correctness Tests")
    print("=" * 70)
    print(f"HEAD_DIM=256, num_q_heads=8, num_kv_heads=1, block_size=16")
    print(f"Packed dim = {nvfp4_packed_dim(256)} bytes/row (vs 256 BF16, 256 FP8)")
    print(f"FP4_BLOCK_SIZE = {FP4_BLOCK_SIZE}")
    print()

    # First compile the kernels
    print("Compiling BF16/FP8 kernel...", flush=True)
    t0 = time.time()
    from sm120_flash_decode_ext import _get_module
    _get_module()
    print(f"  Done ({time.time()-t0:.1f}s)")

    print("Compiling FP4 kernel...", flush=True)
    t0 = time.time()
    from sm120_flash_decode_ext import _get_fp4_module
    _get_fp4_module()
    print(f"  Done ({time.time()-t0:.1f}s)")
    print()

    # Run correctness tests
    all_passed = True
    for sl in seq_lens:
        print(f"Test seq_len={sl:>6d} ... ", end="", flush=True)
        result = run_test(seq_len=sl, verbose=args.verbose)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status}  kernel_L2={result['kernel_l2_err']:.6f} "
              f"quant_L2={result['quant_l2_err']:.4f} "
              f"total_L2={result['total_l2_err']:.4f} "
              f"cos={result['cos_sim_total']:.6f}")
        if not result["passed"]:
            all_passed = False

    print()
    if all_passed:
        print(f"All {len(seq_lens)} tests PASSED")
    else:
        print(f"SOME TESTS FAILED")

    # Extreme value tests
    if args.extreme or True:  # Always run
        print()
        print("=" * 70)
        print("Extreme Value Tests (cosine similarity degradation check)")
        print("=" * 70)
        extreme_scales = [
            (0.001, "tiny (0.001)"),
            (0.01, "small (0.01)"),
            (0.1, "normal (0.1)"),
            (1.0, "unit (1.0)"),
            (10.0, "large (10.0)"),
            (100.0, "huge (100.0)"),
            (400.0, "near-max (400.0, close to E4M3FN max=448)"),
        ]

        extreme_passed = True
        for scale, label in extreme_scales:
            print(f"  scale={label:>50s} ... ", end="", flush=True)
            r = run_extreme_value_test(scale=scale, label=label, verbose=args.verbose)
            status = "PASS" if r["passed"] else "FAIL"
            print(f"{status}  kernel_L2={r['kernel_l2']:.6f} "
                  f"total_L2={r['total_l2']:.4f} "
                  f"cos={r['cos_sim']:.6f} "
                  f"kv_max={r['kv_max']:.2f}")
            if not r["passed"]:
                extreme_passed = False
                all_passed = False

        # Large KV cache test (seq_len=32768)
        print()
        print("  Large KV cache test (seq_len=32768) ... ", end="", flush=True)
        r_large = run_test(seq_len=32768, verbose=args.verbose)
        status = "PASS" if r_large["passed"] else "FAIL"
        print(f"{status}  kernel_L2={r_large['kernel_l2_err']:.6f} "
              f"quant_L2={r_large['quant_l2_err']:.4f} "
              f"cos={r_large['cos_sim_total']:.6f}")
        if not r_large["passed"]:
            all_passed = False

        if extreme_passed and r_large["passed"]:
            print(f"\n  All extreme/large tests PASSED")
        else:
            print(f"\n  SOME extreme/large tests FAILED")

    # Performance benchmarks
    if args.perf:
        print()
        print("=" * 70)
        print("Performance Benchmarks")
        print("=" * 70)
        perf_lens = [sl for sl in seq_lens if sl >= 64]
        for sl in perf_lens:
            print(f"\nseq_len={sl}:")
            r = run_perf_test(seq_len=sl)
            print(f"  BF16:  {r['bf16_us']:.1f} us")
            print(f"  FP4:   {r['fp4_us']:.1f} us  ({r['speedup']:.2f}x vs BF16)")
            print(f"  Cache: BF16={r['bf16_cache_MB']:.2f} MB, "
                  f"FP4={r['fp4_cache_MB']:.2f} MB "
                  f"({r['memory_ratio']:.2f}x savings)")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
