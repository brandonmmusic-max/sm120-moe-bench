#!/usr/bin/env python3
"""
Standalone test for SM120 flash prefill kernel + paged KV gather.

Tests:
1. Contiguous BF16 prefill at HD=128 and HD=256
2. Paged KV gather (BF16 and FP8) → contiguous → prefill
3. GQA correctness (num_heads != num_kv_heads)
4. Causal masking correctness
5. Qwen3.5-like shapes: HD=256, GQA 8:1

Verifies against PyTorch SDPA reference.
"""

import os
import sys
import time
import torch
import torch.nn.functional as F

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def sdpa_reference(q, k, v, causal=True):
    """PyTorch SDPA reference for flash attention.

    Args:
        q: [batch, num_heads, Sq, HD] float32
        k: [batch, num_kv_heads, Skv, HD] float32
        v: [batch, num_kv_heads, Skv, HD] float32

    Returns: [batch, num_heads, Sq, HD] float32
    """
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    gqa_ratio = num_heads // num_kv_heads

    if gqa_ratio > 1:
        # Expand KV heads for GQA
        k = k.repeat_interleave(gqa_ratio, dim=1)
        v = v.repeat_interleave(gqa_ratio, dim=1)

    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def test_contiguous_prefill(head_dim=128, num_heads=8, num_kv_heads=8,
                             seq_len=256, causal=True, label=""):
    """Test contiguous BF16 prefill kernel against SDPA reference."""
    from sm120_flash_prefill_ext import sm120_flash_prefill_forward

    print(f"\n{'='*60}")
    print(f"Test: contiguous prefill {label}")
    print(f"  HD={head_dim}, Hq={num_heads}, Hkv={num_kv_heads}, "
          f"Sq={seq_len}, causal={causal}")
    print(f"{'='*60}")

    device = "cuda:0"
    batch = 1

    # Random inputs in BF16
    torch.manual_seed(42)
    q = torch.randn(batch * num_heads, seq_len, head_dim,
                     dtype=torch.bfloat16, device=device)
    k = torch.randn(batch * num_kv_heads, seq_len, head_dim,
                     dtype=torch.bfloat16, device=device)
    v = torch.randn(batch * num_kv_heads, seq_len, head_dim,
                     dtype=torch.bfloat16, device=device)
    output = torch.empty_like(q)

    # SM120 kernel
    sm120_flash_prefill_forward(
        query=q, key=k, value=v, output=output,
        batch=batch, Hq=num_heads, Hkv=num_kv_heads,
        Sq=seq_len, Skv=seq_len, causal=causal,
    )
    torch.cuda.synchronize()

    # Reference: reshape to [batch, heads, seq, dim]
    q_ref = q.view(batch, num_heads, seq_len, head_dim).float()
    k_ref = k.view(batch, num_kv_heads, seq_len, head_dim).float()
    v_ref = v.view(batch, num_kv_heads, seq_len, head_dim).float()
    ref = sdpa_reference(q_ref, k_ref, v_ref, causal=causal)
    ref_bf16 = ref.to(torch.bfloat16).view(batch * num_heads, seq_len, head_dim)

    # Compare
    out_flat = output.float()
    ref_flat = ref_bf16.float()
    max_err = (out_flat - ref_flat).abs().max().item()
    mean_err = (out_flat - ref_flat).abs().mean().item()

    # Cosine similarity per row
    cos_sim = F.cosine_similarity(
        out_flat.reshape(-1, head_dim),
        ref_flat.reshape(-1, head_dim),
        dim=1,
    ).mean().item()

    status = "PASS" if cos_sim > 0.99 and max_err < 0.1 else "FAIL"
    print(f"  max_err={max_err:.6f}, mean_err={mean_err:.6f}, cos_sim={cos_sim:.6f}")
    print(f"  [{status}]")

    return status == "PASS"


def test_paged_kv_gather(head_dim=256, num_kv_heads=8, block_size=16,
                          seq_len=128, fp8=False, label=""):
    """Test paged KV gather → contiguous conversion."""
    from sm120_flash_prefill_ext import gather_paged_kv

    print(f"\n{'='*60}")
    print(f"Test: paged KV gather {label}")
    print(f"  HD={head_dim}, Hkv={num_kv_heads}, BS={block_size}, "
          f"Skv={seq_len}, FP8={fp8}")
    print(f"{'='*60}")

    device = "cuda:0"
    torch.manual_seed(42)

    num_blocks_needed = (seq_len + block_size - 1) // block_size
    # Allocate more blocks than needed to simulate paged allocation
    total_blocks = num_blocks_needed + 10

    # HND layout: [num_blocks, num_kv_heads, block_size, head_dim]
    if fp8:
        key_cache = torch.randn(total_blocks, num_kv_heads, block_size, head_dim,
                                dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
        value_cache = torch.randn(total_blocks, num_kv_heads, block_size, head_dim,
                                  dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
        k_scale, v_scale = 0.5, 0.25
    else:
        key_cache = torch.randn(total_blocks, num_kv_heads, block_size, head_dim,
                                dtype=torch.bfloat16, device=device)
        value_cache = torch.randn(total_blocks, num_kv_heads, block_size, head_dim,
                                  dtype=torch.bfloat16, device=device)
        k_scale, v_scale = 1.0, 1.0

    # Non-sequential block table (simulate fragmented paged allocation)
    perm = torch.randperm(total_blocks, device=device)[:num_blocks_needed]
    block_table = perm.to(torch.int32)

    # Gather
    k_contig, v_contig = gather_paged_kv(
        key_cache, value_cache, block_table, seq_len,
        k_scale=k_scale, v_scale=v_scale, hnd_layout=True,
    )

    # Manual reference gather
    k_ref_blocks = key_cache[block_table.long()]  # [num_blocks_needed, Hkv, BS, HD]
    k_ref = k_ref_blocks.permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim)[:, :seq_len, :]
    if fp8:
        k_ref = k_ref.to(torch.bfloat16) * k_scale
    else:
        k_ref = k_ref.to(torch.bfloat16)

    max_err = (k_contig.float() - k_ref.float()).abs().max().item()

    status = "PASS" if max_err < 1e-3 else "FAIL"
    print(f"  K shape: {k_contig.shape}, V shape: {v_contig.shape}")
    print(f"  K max_err vs reference: {max_err:.6f}")
    print(f"  [{status}]")

    return status == "PASS"


def test_paged_prefill_e2e(head_dim=256, num_heads=64, num_kv_heads=8,
                            block_size=16, seq_len=128, fp8=False, label=""):
    """End-to-end test: paged KV gather → prefill kernel → compare to SDPA."""
    from sm120_flash_prefill_ext import gather_paged_kv, sm120_flash_prefill_forward

    print(f"\n{'='*60}")
    print(f"Test: paged prefill E2E {label}")
    print(f"  HD={head_dim}, Hq={num_heads}, Hkv={num_kv_heads}, "
          f"BS={block_size}, Sq={seq_len}, FP8={fp8}")
    print(f"{'='*60}")

    device = "cuda:0"
    batch = 1
    torch.manual_seed(42)

    num_blocks_needed = (seq_len + block_size - 1) // block_size
    total_blocks = num_blocks_needed + 10

    # Create reference contiguous KV in BF16 first
    k_contig_ref = torch.randn(num_kv_heads, seq_len, head_dim,
                                dtype=torch.bfloat16, device=device)
    v_contig_ref = torch.randn(num_kv_heads, seq_len, head_dim,
                                dtype=torch.bfloat16, device=device)

    # Pack into paged cache
    key_cache = torch.zeros(total_blocks, num_kv_heads, block_size, head_dim,
                            dtype=torch.bfloat16, device=device)
    value_cache = torch.zeros(total_blocks, num_kv_heads, block_size, head_dim,
                              dtype=torch.bfloat16, device=device)

    block_table = torch.randperm(total_blocks, device=device)[:num_blocks_needed].to(torch.int32)

    for b_idx in range(num_blocks_needed):
        start = b_idx * block_size
        end = min(start + block_size, seq_len)
        length = end - start
        phys_block = block_table[b_idx].item()
        key_cache[phys_block, :, :length, :] = k_contig_ref[:, start:end, :]
        value_cache[phys_block, :, :length, :] = v_contig_ref[:, start:end, :]

    k_scale, v_scale = 1.0, 1.0
    if fp8:
        key_cache = key_cache.to(torch.float8_e4m3fn)
        value_cache = value_cache.to(torch.float8_e4m3fn)
        k_scale, v_scale = 1.0, 1.0  # scales=1.0 since we cast directly

    # Gather paged KV
    k_gathered, v_gathered = gather_paged_kv(
        key_cache, value_cache, block_table, seq_len,
        k_scale=k_scale, v_scale=v_scale, hnd_layout=True,
    )

    # Query
    q = torch.randn(batch * num_heads, seq_len, head_dim,
                     dtype=torch.bfloat16, device=device)
    output = torch.empty_like(q)

    # SM120 prefill
    sm120_flash_prefill_forward(
        query=q, key=k_gathered, value=v_gathered, output=output,
        batch=batch, Hq=num_heads, Hkv=num_kv_heads,
        Sq=seq_len, Skv=seq_len, causal=True,
    )
    torch.cuda.synchronize()

    # SDPA reference
    q_ref = q.view(batch, num_heads, seq_len, head_dim).float()
    k_ref = k_gathered.unsqueeze(0).float()  # [1, Hkv, Skv, HD]
    v_ref = v_gathered.unsqueeze(0).float()
    ref = sdpa_reference(q_ref, k_ref, v_ref, causal=True)
    ref_bf16 = ref.to(torch.bfloat16).view(batch * num_heads, seq_len, head_dim)

    out_flat = output.float()
    ref_flat = ref_bf16.float()
    max_err = (out_flat - ref_flat).abs().max().item()
    mean_err = (out_flat - ref_flat).abs().mean().item()
    cos_sim = F.cosine_similarity(
        out_flat.reshape(-1, head_dim),
        ref_flat.reshape(-1, head_dim),
        dim=1,
    ).mean().item()

    status = "PASS" if cos_sim > 0.99 and max_err < 0.15 else "FAIL"
    print(f"  max_err={max_err:.6f}, mean_err={mean_err:.6f}, cos_sim={cos_sim:.6f}")
    print(f"  [{status}]")

    return status == "PASS"


def main():
    print("SM120 Flash Prefill — Standalone Tests")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    results = []

    # 1. Contiguous prefill HD=128
    results.append(("contiguous HD=128",
        test_contiguous_prefill(head_dim=128, num_heads=8, num_kv_heads=8,
                                 seq_len=256, label="HD=128 MHA")))

    # 2. Contiguous prefill HD=128 with GQA
    results.append(("contiguous HD=128 GQA",
        test_contiguous_prefill(head_dim=128, num_heads=32, num_kv_heads=8,
                                 seq_len=256, label="HD=128 GQA 4:1")))

    # 3. Contiguous prefill HD=256
    results.append(("contiguous HD=256",
        test_contiguous_prefill(head_dim=256, num_heads=8, num_kv_heads=8,
                                 seq_len=128, label="HD=256 MHA")))

    # 4. Contiguous prefill HD=256 GQA (Qwen3.5-like)
    results.append(("contiguous HD=256 GQA 8:1",
        test_contiguous_prefill(head_dim=256, num_heads=64, num_kv_heads=8,
                                 seq_len=128, label="HD=256 GQA 8:1 (Qwen3.5)")))

    # 5. Paged KV gather BF16
    results.append(("gather BF16",
        test_paged_kv_gather(head_dim=256, num_kv_heads=8, block_size=16,
                              seq_len=128, fp8=False, label="BF16")))

    # 6. Paged KV gather FP8
    results.append(("gather FP8",
        test_paged_kv_gather(head_dim=256, num_kv_heads=8, block_size=16,
                              seq_len=128, fp8=True, label="FP8")))

    # 7. E2E paged prefill HD=256 GQA (Qwen3.5-like)
    results.append(("E2E paged HD=256 GQA",
        test_paged_prefill_e2e(head_dim=256, num_heads=64, num_kv_heads=8,
                                block_size=16, seq_len=128, label="Qwen3.5-like")))

    # 8. E2E paged prefill HD=128
    results.append(("E2E paged HD=128",
        test_paged_prefill_e2e(head_dim=128, num_heads=32, num_kv_heads=8,
                                block_size=16, seq_len=256, label="HD=128 GQA")))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
