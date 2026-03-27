"""
Standalone correctness test for SM120 attention kernels.

Tests:
1. Decode: single-query attention against paged KV cache
2. Prefill: multi-query attention with causal masking
3. FP8 KV cache decode
4. GQA (grouped query attention) with Qwen3.5 ratios
5. Multi-query decode (MTP verification scenario)

Reference: torch.nn.functional.scaled_dot_product_attention

Run: python test_sm120_attention.py
"""

import sys
import os
import time
import torch
import torch.nn.functional as F

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def reference_attention(
    query: torch.Tensor,    # [batch, num_q_heads, seq_q, head_dim]
    key: torch.Tensor,      # [batch, num_kv_heads, seq_kv, head_dim]
    value: torch.Tensor,    # [batch, num_kv_heads, seq_kv, head_dim]
    is_causal: bool = False,
) -> torch.Tensor:
    """Reference attention using PyTorch SDPA (handles GQA via repeat)."""
    num_q_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    gqa_ratio = num_q_heads // num_kv_heads

    if gqa_ratio > 1:
        key = key.repeat_interleave(gqa_ratio, dim=1)
        value = value.repeat_interleave(gqa_ratio, dim=1)

    return F.scaled_dot_product_attention(
        query.float(), key.float(), value.float(), is_causal=is_causal,
    ).to(torch.bfloat16)


def create_paged_kv_cache(
    key: torch.Tensor,      # [batch, num_kv_heads, seq_len, head_dim]
    value: torch.Tensor,    # [batch, num_kv_heads, seq_len, head_dim]
    block_size: int = 16,
    dtype: torch.dtype = torch.bfloat16,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
):
    """Create paged KV cache from contiguous K, V tensors.

    Returns: key_cache, value_cache, block_table, seq_lens
    """
    batch, num_kv_heads, seq_len, head_dim = key.shape
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = batch * num_blocks_per_seq

    # Allocate cache
    key_cache = torch.zeros(total_blocks, block_size, num_kv_heads, head_dim,
                            dtype=dtype, device=key.device)
    value_cache = torch.zeros_like(key_cache)

    # Build block table
    block_table = torch.zeros(batch, num_blocks_per_seq, dtype=torch.int32, device=key.device)

    for b in range(batch):
        for blk in range(num_blocks_per_seq):
            block_idx = b * num_blocks_per_seq + blk
            block_table[b, blk] = block_idx

            start = blk * block_size
            end = min(start + block_size, seq_len)
            length = end - start

            # K, V are [batch, num_kv_heads, seq_len, head_dim]
            # Cache is [num_blocks, block_size, num_kv_heads, head_dim]
            k_block = key[b, :, start:end, :].permute(1, 0, 2)  # [length, num_kv_heads, head_dim]
            v_block = value[b, :, start:end, :].permute(1, 0, 2)

            if dtype == torch.float8_e4m3fn:
                # Quantize: divide by scale, clamp, cast
                key_cache[block_idx, :length] = (k_block.float() / k_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
                value_cache[block_idx, :length] = (v_block.float() / v_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            else:
                key_cache[block_idx, :length] = k_block
                value_cache[block_idx, :length] = v_block

    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=key.device)
    return key_cache, value_cache, block_table, seq_lens


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


# ============================================================================
# Test 1: Decode (Sq=1, BF16 KV)
# ============================================================================

def test_decode_bf16():
    print("=" * 60)
    print("Test 1: Decode (Sq=1, BF16 KV)")
    print("=" * 60)

    from sm120_flash_decode_ext import sm120_flash_decode_paged, SM120FlashDecodeWorkspace

    batch = 4
    num_q_heads = 16
    num_kv_heads = 2
    head_dim = 128
    seq_len = 512
    block_size = 16

    torch.manual_seed(42)
    Q = torch.randn(batch, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")

    # Reference
    q_ref = Q.unsqueeze(2)  # [batch, num_q_heads, 1, head_dim]
    ref_out = reference_attention(q_ref, K, V, is_causal=False).squeeze(2)  # [batch, num_q_heads, head_dim]

    # Paged KV
    key_cache, value_cache, block_table, seq_lens = create_paged_kv_cache(K, V, block_size)

    workspace = SM120FlashDecodeWorkspace(
        max_batch_size=batch, num_q_heads=num_q_heads, head_dim=head_dim, device="cuda"
    )

    our_out = sm120_flash_decode_paged(
        query=Q, key_cache=key_cache, value_cache=value_cache,
        block_table=block_table, seq_lens=seq_lens,
        workspace=workspace, max_seq_len=seq_len,
    )

    mae = max_abs_error(our_out, ref_out)
    cs = cos_sim(our_out, ref_out)
    print(f"  Max abs error: {mae:.6f}")
    print(f"  Cosine sim:    {cs:.8f}")
    passed = mae < 0.05 and cs > 0.999
    print(f"  {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ============================================================================
# Test 2: Decode with FP8 KV cache
# ============================================================================

def test_decode_fp8():
    print("=" * 60)
    print("Test 2: Decode (Sq=1, FP8 KV)")
    print("=" * 60)

    from sm120_flash_decode_ext import sm120_flash_decode_paged, SM120FlashDecodeWorkspace

    batch = 4
    num_q_heads = 16
    num_kv_heads = 2
    head_dim = 128
    seq_len = 1024
    block_size = 16
    k_scale = 0.8
    v_scale = 0.9

    torch.manual_seed(123)
    Q = torch.randn(batch, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda") * 0.5
    V = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda") * 0.5

    # Reference with BF16
    q_ref = Q.unsqueeze(2)
    ref_out = reference_attention(q_ref, K, V, is_causal=False).squeeze(2)

    # Paged FP8 KV
    key_cache, value_cache, block_table, seq_lens = create_paged_kv_cache(
        K, V, block_size, dtype=torch.float8_e4m3fn, k_scale=k_scale, v_scale=v_scale
    )

    workspace = SM120FlashDecodeWorkspace(
        max_batch_size=batch, num_q_heads=num_q_heads, head_dim=head_dim, device="cuda"
    )

    our_out = sm120_flash_decode_paged(
        query=Q, key_cache=key_cache, value_cache=value_cache,
        block_table=block_table, seq_lens=seq_lens,
        workspace=workspace, max_seq_len=seq_len,
        k_scale=k_scale, v_scale=v_scale,
    )

    mae = max_abs_error(our_out, ref_out)
    cs = cos_sim(our_out, ref_out)
    print(f"  Max abs error: {mae:.6f} (FP8 quantization adds noise)")
    print(f"  Cosine sim:    {cs:.8f}")
    # FP8 has higher tolerance due to quantization noise
    passed = mae < 0.15 and cs > 0.995
    print(f"  {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ============================================================================
# Test 3: Decode with GQA (Qwen3.5 ratios)
# ============================================================================

def test_decode_gqa():
    print("=" * 60)
    print("Test 3: Decode with GQA (Hq=64, Hkv=8, HD=128)")
    print("=" * 60)

    from sm120_flash_decode_ext import sm120_flash_decode_paged, SM120FlashDecodeWorkspace

    batch = 2
    num_q_heads = 64
    num_kv_heads = 8
    head_dim = 128
    seq_len = 256
    block_size = 16

    torch.manual_seed(7)
    Q = torch.randn(batch, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")

    # Reference
    q_ref = Q.unsqueeze(2)
    ref_out = reference_attention(q_ref, K, V, is_causal=False).squeeze(2)

    # Paged KV
    key_cache, value_cache, block_table, seq_lens = create_paged_kv_cache(K, V, block_size)

    workspace = SM120FlashDecodeWorkspace(
        max_batch_size=batch, num_q_heads=num_q_heads, head_dim=head_dim, device="cuda"
    )

    our_out = sm120_flash_decode_paged(
        query=Q, key_cache=key_cache, value_cache=value_cache,
        block_table=block_table, seq_lens=seq_lens,
        workspace=workspace, max_seq_len=seq_len,
    )

    mae = max_abs_error(our_out, ref_out)
    cs = cos_sim(our_out, ref_out)
    print(f"  Max abs error: {mae:.6f}")
    print(f"  Cosine sim:    {cs:.8f}")
    passed = mae < 0.05 and cs > 0.999
    print(f"  {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ============================================================================
# Test 4: Prefill (causal, BF16)
# ============================================================================

def test_prefill_causal():
    print("=" * 60)
    print("Test 4: Prefill (causal, BF16, Sq=Skv=128)")
    print("=" * 60)

    from sm120_flash_prefill_ext import sm120_flash_prefill_forward

    batch = 1
    num_q_heads = 16
    num_kv_heads = 2
    head_dim = 128
    seq_len = 128

    torch.manual_seed(99)
    # Kernel layout: [Hq, Sq, HD]
    Q = torch.randn(num_q_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")

    # Reference: SDPA expects [batch, heads, seq, dim]
    q_ref = Q.unsqueeze(0)  # [1, Hq, Sq, HD]
    k_ref = K.unsqueeze(0)  # [1, Hkv, Skv, HD]
    v_ref = V.unsqueeze(0)
    ref_out = reference_attention(q_ref, k_ref, v_ref, is_causal=True).squeeze(0)  # [Hq, Sq, HD]

    # Our kernel
    our_out = sm120_flash_prefill_forward(
        query=Q, key=K, value=V,
        batch=batch, Hq=num_q_heads, Hkv=num_kv_heads,
        Sq=seq_len, Skv=seq_len, causal=True,
    )

    mae = max_abs_error(our_out, ref_out)
    cs = cos_sim(our_out, ref_out)
    print(f"  Max abs error: {mae:.6f}")
    print(f"  Cosine sim:    {cs:.8f}")
    passed = mae < 0.05 and cs > 0.999
    print(f"  {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ============================================================================
# Test 5: Prefill with GQA
# ============================================================================

def test_prefill_gqa():
    print("=" * 60)
    print("Test 5: Prefill with GQA (Hq=16, Hkv=2, causal)")
    print("=" * 60)

    from sm120_flash_prefill_ext import sm120_flash_prefill_forward

    batch = 1
    num_q_heads = 16
    num_kv_heads = 2
    head_dim = 128
    seq_len = 256

    torch.manual_seed(314)
    Q = torch.randn(num_q_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")

    q_ref = Q.unsqueeze(0)
    k_ref = K.unsqueeze(0)
    v_ref = V.unsqueeze(0)
    ref_out = reference_attention(q_ref, k_ref, v_ref, is_causal=True).squeeze(0)

    our_out = sm120_flash_prefill_forward(
        query=Q, key=K, value=V,
        batch=batch, Hq=num_q_heads, Hkv=num_kv_heads,
        Sq=seq_len, Skv=seq_len, causal=True,
    )

    mae = max_abs_error(our_out, ref_out)
    cs = cos_sim(our_out, ref_out)
    print(f"  Max abs error: {mae:.6f}")
    print(f"  Cosine sim:    {cs:.8f}")
    passed = mae < 0.05 and cs > 0.999
    print(f"  {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ============================================================================
# Test 6: Prefill non-causal
# ============================================================================

def test_prefill_non_causal():
    print("=" * 60)
    print("Test 6: Prefill (non-causal, BF16)")
    print("=" * 60)

    from sm120_flash_prefill_ext import sm120_flash_prefill_forward

    batch = 1
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 128
    seq_q = 64
    seq_kv = 128

    torch.manual_seed(555)
    Q = torch.randn(num_q_heads, seq_q, head_dim, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(num_kv_heads, seq_kv, head_dim, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(num_kv_heads, seq_kv, head_dim, dtype=torch.bfloat16, device="cuda")

    q_ref = Q.unsqueeze(0)
    k_ref = K.unsqueeze(0)
    v_ref = V.unsqueeze(0)
    ref_out = reference_attention(q_ref, k_ref, v_ref, is_causal=False).squeeze(0)

    our_out = sm120_flash_prefill_forward(
        query=Q, key=K, value=V,
        batch=batch, Hq=num_q_heads, Hkv=num_kv_heads,
        Sq=seq_q, Skv=seq_kv, causal=False,
    )

    mae = max_abs_error(our_out, ref_out)
    cs = cos_sim(our_out, ref_out)
    print(f"  Max abs error: {mae:.6f}")
    print(f"  Cosine sim:    {cs:.8f}")
    passed = mae < 0.05 and cs > 0.999
    print(f"  {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ============================================================================
# Test 7: Multi-query decode (MTP verification)
# ============================================================================

def test_multi_query_decode():
    print("=" * 60)
    print("Test 7: Multi-query decode (MTP verify, q_per_req=4)")
    print("=" * 60)

    from sm120_flash_decode_ext import sm120_flash_decode_paged, SM120FlashDecodeWorkspace

    batch = 4
    num_q_heads = 16
    num_kv_heads = 2
    head_dim = 128
    base_seq_len = 256
    q_per_req = 4  # MTP with 3 speculative tokens
    block_size = 16

    torch.manual_seed(777)
    # Each request has q_per_req query tokens
    Q_all = torch.randn(batch, q_per_req, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(batch, num_kv_heads, base_seq_len + q_per_req, head_dim, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(batch, num_kv_heads, base_seq_len + q_per_req, head_dim, dtype=torch.bfloat16, device="cuda")

    # Paged KV with full seq_len (base + q_per_req tokens already in cache)
    full_seq_len = base_seq_len + q_per_req
    key_cache, value_cache, block_table, _ = create_paged_kv_cache(K, V, block_size)

    workspace = SM120FlashDecodeWorkspace(
        max_batch_size=batch, num_q_heads=num_q_heads, head_dim=head_dim, device="cuda"
    )

    all_passed = True
    for qi in range(q_per_req):
        # For position qi, the query sees base_seq_len + qi + 1 KV tokens (causal)
        visible_len = base_seq_len + qi + 1
        seq_lens = torch.full((batch,), visible_len, dtype=torch.int32, device="cuda")

        q_slice = Q_all[:, qi, :, :]  # [batch, num_q_heads, head_dim]

        our_out = sm120_flash_decode_paged(
            query=q_slice, key_cache=key_cache, value_cache=value_cache,
            block_table=block_table, seq_lens=seq_lens,
            workspace=workspace, max_seq_len=visible_len,
        )

        # Reference: Q[batch,1,Hq,HD] vs K[:,:visible_len]
        q_ref = q_slice.unsqueeze(2)  # [batch, Hq, 1, HD]
        k_ref = K[:, :, :visible_len, :]
        v_ref = V[:, :, :visible_len, :]
        ref_out = reference_attention(q_ref, k_ref, v_ref, is_causal=False).squeeze(2)

        mae = max_abs_error(our_out, ref_out)
        cs = cos_sim(our_out, ref_out)
        ok = mae < 0.05 and cs > 0.999
        print(f"  qi={qi}: mae={mae:.6f}, cos={cs:.8f} {'PASS' if ok else 'FAIL'}")
        all_passed = all_passed and ok

    print(f"  Overall: {'PASS' if all_passed else 'FAIL'}")
    print()
    return all_passed


# ============================================================================
# Test 8: Prefill with paged KV (via sm120_attention_ext gather)
# ============================================================================

def test_prefill_paged():
    print("=" * 60)
    print("Test 8: Prefill with paged KV (gather + prefill)")
    print("=" * 60)

    from sm120_attention_ext import sm120_prefill

    batch = 1
    num_q_heads = 16
    num_kv_heads = 2
    head_dim = 128
    seq_len = 128
    block_size = 16

    torch.manual_seed(42)
    # vLLM layout: Q[seq_len, num_q_heads, head_dim]
    Q = torch.randn(seq_len, num_q_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16, device="cuda")

    # Reference
    q_ref = Q.permute(1, 0, 2).unsqueeze(0)  # [1, Hq, Sq, HD]
    k_ref = K  # [1, Hkv, Skv, HD]
    v_ref = V
    ref_out = reference_attention(q_ref, k_ref, v_ref, is_causal=True).squeeze(0)  # [Hq, Sq, HD]
    ref_out = ref_out.permute(1, 0, 2)  # [Sq, Hq, HD]

    # Create paged KV
    key_cache, value_cache, block_table, seq_lens = create_paged_kv_cache(K, V, block_size)

    output = torch.empty_like(Q)
    sm120_prefill(
        query=Q,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table[0],  # single request
        seq_len=seq_len,
        query_len=seq_len,
        output=output,
        block_size=block_size,
        causal=True,
    )

    mae = max_abs_error(output, ref_out)
    cs = cos_sim(output, ref_out)
    print(f"  Max abs error: {mae:.6f}")
    print(f"  Cosine sim:    {cs:.8f}")
    passed = mae < 0.05 and cs > 0.999
    print(f"  {'PASS' if passed else 'FAIL'}")
    print()
    return passed


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("SM120 Attention Kernel Correctness Tests")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA:   {torch.version.cuda}")
    print()

    results = {}

    # Decode tests
    print(">>> Compiling decode kernel (first run)...")
    t0 = time.time()
    results["decode_bf16"] = test_decode_bf16()
    print(f"    Compile + run: {time.time()-t0:.1f}s\n")

    results["decode_fp8"] = test_decode_fp8()
    results["decode_gqa"] = test_decode_gqa()
    results["multi_query_decode"] = test_multi_query_decode()

    # Prefill tests
    print(">>> Compiling prefill kernel (first run)...")
    t0 = time.time()
    results["prefill_causal"] = test_prefill_causal()
    print(f"    Compile + run: {time.time()-t0:.1f}s\n")

    results["prefill_gqa"] = test_prefill_gqa()
    results["prefill_non_causal"] = test_prefill_non_causal()

    # Combined test (paged gather + prefill)
    results["prefill_paged"] = test_prefill_paged()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    for name, ok in results.items():
        print(f"  {name:30s} {'PASS' if ok else 'FAIL'}")
    print(f"\n  {passed}/{total} tests passed")

    sys.exit(0 if passed == total else 1)
