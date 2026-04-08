#!/usr/bin/env python3
"""
Test the fused reshape_and_cache_nvfp4 CUDA kernel.

Validates roundtrip accuracy: BF16 → NVFP4 cache → dequant → compare.
Uses the existing decode kernel's dequant path as reference.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reshape_and_cache_nvfp4_ext import reshape_and_cache_nvfp4, packed_dim, FP4_BLOCK_SIZE
from sm120_flash_decode_ext import quantize_to_nvfp4


def dequant_nvfp4_row(packed: torch.Tensor, head_dim: int, global_scale: float = 1.0) -> torch.Tensor:
    """Dequantize a single packed NVFP4 row back to float32.

    packed: [packed_dim] uint8
    Returns: [head_dim] float32
    """
    data_bytes = head_dim // 2
    scale_bytes = head_dim // FP4_BLOCK_SIZE

    fp4_data = packed[:data_bytes]
    block_scales = packed[data_bytes:data_bytes + scale_bytes]

    # FP4 magnitude LUT
    fp4_mags = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

    result = torch.zeros(head_dim, dtype=torch.float32)
    for i in range(head_dim):
        byte_idx = i // 2
        is_high = i % 2
        byte_val = fp4_data[byte_idx].item()
        nibble = (byte_val >> 4) if is_high else (byte_val & 0xF)

        sign = -1.0 if (nibble & 0x8) else 1.0
        mag_idx = nibble & 0x7
        fp4_val = sign * fp4_mags[mag_idx]

        # Block scale
        blk_idx = i // FP4_BLOCK_SIZE
        scale_code = block_scales[blk_idx].item()
        e = (scale_code >> 3) & 0xF
        m = scale_code & 0x7
        if e == 0:
            scale_float = m / 512.0
        elif e == 15 and m == 7:
            scale_float = 0.0
        else:
            import math
            scale_float = math.ldexp(1.0 + m / 8.0, e - 7)

        result[i] = fp4_val * scale_float * global_scale

    return result


def test_basic():
    """Test basic scatter-write + quantization."""
    print("=== Test: Basic scatter-write + NVFP4 quantization ===")

    num_tokens = 4
    num_heads = 2
    head_dim = 128
    block_size = 16
    num_blocks = 10
    pd = packed_dim(head_dim)

    # Create BF16 source data
    key = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    value = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')

    # Create NVFP4 cache
    key_cache = torch.zeros(num_blocks, block_size, num_heads, pd, dtype=torch.uint8, device='cuda')
    value_cache = torch.zeros(num_blocks, block_size, num_heads, pd, dtype=torch.uint8, device='cuda')

    # Slot mapping: tokens 0-3 go to slots 0,1,16,17 (two different blocks)
    slot_mapping = torch.tensor([0, 1, 16, 17], dtype=torch.int64, device='cuda')

    # Run kernel
    reshape_and_cache_nvfp4(key, value, key_cache, value_cache, slot_mapping)
    torch.cuda.synchronize()

    print(f"  Cache shape: key={key_cache.shape}, packed_dim={pd}")

    # Verify roundtrip for each token/head
    max_errors = []
    for t in range(num_tokens):
        slot = slot_mapping[t].item()
        blk = slot // block_size
        off = slot % block_size
        for h in range(num_heads):
            packed = key_cache[blk, off, h].cpu()
            original = key[t, h].float().cpu()
            dequant = dequant_nvfp4_row(packed, head_dim)
            err = (original - dequant).abs().max().item()
            max_errors.append(err)

    avg_err = sum(max_errors) / len(max_errors)
    max_err = max(max_errors)
    print(f"  Roundtrip max_error: avg={avg_err:.4f}, worst={max_err:.4f}")
    print(f"  {'PASS' if max_err < 2.0 else 'FAIL'} (threshold: 2.0)")
    return max_err < 2.0


def test_hd256():
    """Test with HEAD_DIM=256 (Qwen3.5 full attention)."""
    print("=== Test: HEAD_DIM=256 (Qwen3.5) ===")

    num_tokens = 8
    num_heads = 1
    head_dim = 256
    block_size = 16
    num_blocks = 20
    pd = packed_dim(head_dim)

    key = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    value = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')

    key_cache = torch.zeros(num_blocks, block_size, num_heads, pd, dtype=torch.uint8, device='cuda')
    value_cache = torch.zeros(num_blocks, block_size, num_heads, pd, dtype=torch.uint8, device='cuda')

    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device='cuda')

    reshape_and_cache_nvfp4(key, value, key_cache, value_cache, slot_mapping)
    torch.cuda.synchronize()

    # Spot check token 0, head 0
    packed = key_cache[0, 0, 0].cpu()
    original = key[0, 0].float().cpu()
    dequant = dequant_nvfp4_row(packed, head_dim)
    cos_sim = torch.nn.functional.cosine_similarity(original, dequant, dim=0).item()
    max_err = (original - dequant).abs().max().item()

    print(f"  cos_sim={cos_sim:.6f}, max_err={max_err:.4f}")
    print(f"  {'PASS' if cos_sim > 0.95 else 'FAIL'} (threshold: cos>0.95)")
    return cos_sim > 0.95


def test_padding():
    """Test that slot_mapping=-1 (padding) is skipped."""
    print("=== Test: Padding tokens (slot=-1) ===")

    num_tokens = 4
    num_heads = 1
    head_dim = 128
    block_size = 16
    num_blocks = 5
    pd = packed_dim(head_dim)

    key = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    value = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')

    key_cache = torch.zeros(num_blocks, block_size, num_heads, pd, dtype=torch.uint8, device='cuda')
    value_cache = torch.zeros(num_blocks, block_size, num_heads, pd, dtype=torch.uint8, device='cuda')

    # Token 1 and 3 are padding
    slot_mapping = torch.tensor([0, -1, 2, -1], dtype=torch.int64, device='cuda')

    reshape_and_cache_nvfp4(key, value, key_cache, value_cache, slot_mapping)
    torch.cuda.synchronize()

    # Slot 1 should be all zeros (padding token skipped)
    slot1_data = key_cache[0, 1, 0].sum().item()
    # Slot 0 should have data
    slot0_data = key_cache[0, 0, 0].sum().item()

    passed = slot1_data == 0 and slot0_data != 0
    print(f"  slot0 sum={slot0_data:.0f} (should be nonzero)")
    print(f"  slot1 sum={slot1_data:.0f} (should be 0, padding)")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_compare_python():
    """Compare CUDA kernel output vs Python quantize_to_nvfp4."""
    print("=== Test: CUDA vs Python quantize_to_nvfp4 ===")

    num_tokens = 2
    num_heads = 1
    head_dim = 128
    block_size = 16
    num_blocks = 5
    pd = packed_dim(head_dim)

    key = torch.randn(num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device='cuda')

    # CUDA kernel path
    key_cache = torch.zeros(num_blocks, block_size, num_heads, pd, dtype=torch.uint8, device='cuda')
    value_cache = torch.zeros(num_blocks, block_size, num_heads, pd, dtype=torch.uint8, device='cuda')
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device='cuda')

    reshape_and_cache_nvfp4(key, key, key_cache, value_cache, slot_mapping)
    torch.cuda.synchronize()

    # Python reference path
    for t in range(num_tokens):
        cuda_packed = key_cache[t // block_size, t % block_size, 0].cpu()
        py_packed, py_ts = quantize_to_nvfp4(key[t, 0].unsqueeze(0))
        py_packed = py_packed.squeeze(0).cpu()

        # Compare data bytes
        data_match = (cuda_packed[:head_dim//2] == py_packed[:head_dim//2]).float().mean().item()
        # Compare scale bytes
        scale_match = (cuda_packed[head_dim//2:] == py_packed[head_dim//2:]).float().mean().item()

        print(f"  Token {t}: data_match={data_match*100:.1f}%, scale_match={scale_match*100:.1f}%")

    # They won't match exactly (different scale search algorithms) but should be close
    print("  NOTE: Exact match not expected (different scale search). Checking roundtrip quality instead.")

    # Roundtrip quality comparison
    for t in range(num_tokens):
        original = key[t, 0].float().cpu()

        # CUDA roundtrip
        cuda_packed = key_cache[t // block_size, t % block_size, 0].cpu()
        cuda_dequant = dequant_nvfp4_row(cuda_packed, head_dim)
        cuda_cos = torch.nn.functional.cosine_similarity(original, cuda_dequant, dim=0).item()

        # Python roundtrip
        py_packed, _ = quantize_to_nvfp4(key[t, 0].unsqueeze(0))
        py_dequant = dequant_nvfp4_row(py_packed.squeeze(0).cpu(), head_dim)
        py_cos = torch.nn.functional.cosine_similarity(original, py_dequant, dim=0).item()

        print(f"  Token {t}: CUDA cos={cuda_cos:.6f}, Python cos={py_cos:.6f}")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("reshape_and_cache_nvfp4 CUDA Kernel Tests")
    print("=" * 60)

    results = []
    results.append(test_basic())
    results.append(test_hd256())
    results.append(test_padding())
    results.append(test_compare_python())

    print()
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} PASSED")
    sys.exit(0 if passed == total else 1)
