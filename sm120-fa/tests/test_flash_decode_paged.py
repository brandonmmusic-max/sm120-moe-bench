"""
Correctness test + benchmark for SM120 Flash Decode with paged KV cache.

Compares against torch SDPA reference implementation.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import math


def reference_paged_attention(
    query, key_cache, value_cache, block_table, seq_lens,
    num_q_heads, num_kv_heads, head_dim
):
    """
    Reference implementation: gather KV from paged cache, run torch SDPA.

    query: [batch, num_q_heads, head_dim]
    key_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: [num_blocks, block_size, num_kv_heads, head_dim]
    block_table: [batch, max_blocks_per_seq]
    seq_lens: [batch]

    Returns: [batch, num_q_heads, head_dim]
    """
    batch_size = query.shape[0]
    block_size = key_cache.shape[1]
    gqa_ratio = num_q_heads // num_kv_heads
    max_seq_len = seq_lens.max().item()

    output = torch.empty_like(query)

    for b in range(batch_size):
        sl = seq_lens[b].item()
        num_blocks_needed = (sl + block_size - 1) // block_size

        # Gather K and V from paged cache
        k_gathered = []
        v_gathered = []
        for blk_i in range(num_blocks_needed):
            blk_idx = block_table[b, blk_i].item()
            start = blk_i * block_size
            end = min(start + block_size, sl)
            tokens_in_block = end - start
            k_gathered.append(key_cache[blk_idx, :tokens_in_block])    # [tokens, kv_heads, hd]
            v_gathered.append(value_cache[blk_idx, :tokens_in_block])

        k_full = torch.cat(k_gathered, dim=0)  # [sl, kv_heads, hd]
        v_full = torch.cat(v_gathered, dim=0)  # [sl, kv_heads, hd]

        # GQA: expand KV heads to match Q heads
        # k_full: [sl, kv_heads, hd] -> [sl, q_heads, hd]
        k_exp = k_full.unsqueeze(2).expand(-1, num_kv_heads, gqa_ratio, head_dim)
        k_exp = k_exp.reshape(sl, num_q_heads, head_dim)
        v_exp = v_full.unsqueeze(2).expand(-1, num_kv_heads, gqa_ratio, head_dim)
        v_exp = v_exp.reshape(sl, num_q_heads, head_dim)

        # SDPA: Q=[1, heads, hd], K=V=[sl, heads, hd]
        # Reshape for F.scaled_dot_product_attention: [1, heads, 1, hd] @ [1, heads, sl, hd]
        q_b = query[b].unsqueeze(0).transpose(0, 1).unsqueeze(0)  # [1, heads, 1, hd]
        k_b = k_exp.transpose(0, 1).unsqueeze(0)                   # [1, heads, sl, hd]
        v_b = v_exp.transpose(0, 1).unsqueeze(0)                   # [1, heads, sl, hd]

        out_b = F.scaled_dot_product_attention(q_b, k_b, v_b, is_causal=False)
        output[b] = out_b.squeeze(0).squeeze(-2)  # [heads, hd]

    return output


def create_paged_kv_cache(batch_size, seq_lens, num_kv_heads, head_dim, block_size, device):
    """Create a paged KV cache with random data and a block table."""
    max_seq_len = max(seq_lens)
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    total_blocks = batch_size * max_blocks_per_seq + 10  # some extra

    key_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim,
                            dtype=torch.bfloat16, device=device) * 0.1
    value_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim,
                              dtype=torch.bfloat16, device=device) * 0.1

    # Create block table: simple sequential allocation
    block_table = torch.zeros(batch_size, max_blocks_per_seq, dtype=torch.int32, device=device)
    blk_counter = 0
    for b in range(batch_size):
        n_blks = (seq_lens[b] + block_size - 1) // block_size
        for i in range(n_blks):
            block_table[b, i] = blk_counter
            blk_counter += 1

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)

    return key_cache, value_cache, block_table, seq_lens_t


def test_correctness(head_dim=128, num_q_heads=32, num_kv_heads=2, block_size=16):
    """Test SM120 decode kernel against reference for various configs."""
    from sm120_flash_decode_ext import sm120_flash_decode_paged

    device = "cuda"
    configs = [
        # (batch_size, seq_lens)
        (1, [64]),
        (1, [256]),
        (1, [1024]),
        (1, [4096]),
        (4, [128, 256, 512, 1024]),
        (8, [100, 200, 300, 400, 500, 600, 700, 800]),
        (1, [16384]),
    ]

    print(f"\n{'='*70}")
    print(f"Correctness Test: head_dim={head_dim}, q_heads={num_q_heads}, kv_heads={num_kv_heads}")
    print(f"{'='*70}")

    all_pass = True
    for batch_size, seq_lens in configs:
        key_cache, value_cache, block_table, seq_lens_t = create_paged_kv_cache(
            batch_size, seq_lens, num_kv_heads, head_dim, block_size, device
        )

        query = torch.randn(batch_size, num_q_heads, head_dim,
                            dtype=torch.bfloat16, device=device) * 0.1

        # Reference
        ref_out = reference_paged_attention(
            query, key_cache, value_cache, block_table, seq_lens_t,
            num_q_heads, num_kv_heads, head_dim
        )

        # SM120 kernel
        sm120_out = sm120_flash_decode_paged(
            query, key_cache, value_cache, block_table, seq_lens_t
        )

        # Compare
        max_err = (ref_out.float() - sm120_out.float()).abs().max().item()
        mean_err = (ref_out.float() - sm120_out.float()).abs().mean().item()
        cos_sim = F.cosine_similarity(
            ref_out.float().reshape(-1).unsqueeze(0),
            sm120_out.float().reshape(-1).unsqueeze(0)
        ).item()

        ok = max_err < 0.05 and cos_sim > 0.999
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False

        print(f"  [{status}] batch={batch_size}, seq_lens={seq_lens}: "
              f"max_err={max_err:.6f}, mean_err={mean_err:.6f}, cos_sim={cos_sim:.6f}")

    return all_pass


def benchmark(head_dim=256, num_q_heads=32, num_kv_heads=2, block_size=16):
    """Benchmark SM120 decode kernel vs torch SDPA."""
    from sm120_flash_decode_ext import sm120_flash_decode_paged, SM120FlashDecodeWorkspace

    device = "cuda"
    configs = [
        # (batch_size, seq_len, label)
        (1, 512, "single-user short"),
        (1, 2048, "single-user medium"),
        (1, 8192, "single-user long"),
        (1, 32768, "single-user very long"),
        (8, 2048, "8-user medium"),
        (8, 8192, "8-user long"),
        (32, 2048, "32-user medium"),
    ]

    print(f"\n{'='*70}")
    print(f"Benchmark: head_dim={head_dim}, q_heads={num_q_heads}, kv_heads={num_kv_heads}")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'SM120 (us)':>12} {'SDPA (us)':>12} {'Speedup':>10}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    for batch_size, seq_len, label in configs:
        seq_lens = [seq_len] * batch_size
        key_cache, value_cache, block_table, seq_lens_t = create_paged_kv_cache(
            batch_size, seq_lens, num_kv_heads, head_dim, block_size, device
        )
        query = torch.randn(batch_size, num_q_heads, head_dim,
                            dtype=torch.bfloat16, device=device) * 0.1

        # Pre-allocate workspace
        workspace = SM120FlashDecodeWorkspace(
            max_batch_size=batch_size, num_q_heads=num_q_heads,
            head_dim=head_dim, max_splits=32, device=device
        )
        output = torch.empty_like(query)

        # Warmup SM120
        for _ in range(5):
            sm120_flash_decode_paged(
                query, key_cache, value_cache, block_table, seq_lens_t,
                output=output, workspace=workspace
            )
        torch.cuda.synchronize()

        # Benchmark SM120
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        n_iters = 50
        start.record()
        for _ in range(n_iters):
            sm120_flash_decode_paged(
                query, key_cache, value_cache, block_table, seq_lens_t,
                output=output, workspace=workspace
            )
        end.record()
        torch.cuda.synchronize()
        sm120_us = start.elapsed_time(end) * 1000 / n_iters  # ms -> us

        # Benchmark SDPA reference
        # Gather KV (simulating non-paged for SDPA)
        ref_out = reference_paged_attention(
            query, key_cache, value_cache, block_table, seq_lens_t,
            num_q_heads, num_kv_heads, head_dim
        )
        # For fair comparison, also time the gather+SDPA
        torch.cuda.synchronize()
        start.record()
        for _ in range(n_iters):
            reference_paged_attention(
                query, key_cache, value_cache, block_table, seq_lens_t,
                num_q_heads, num_kv_heads, head_dim
            )
        end.record()
        torch.cuda.synchronize()
        sdpa_us = start.elapsed_time(end) * 1000 / n_iters

        speedup = sdpa_us / sm120_us if sm120_us > 0 else float('inf')
        print(f"  {label:<25} {sm120_us:>10.1f} {sdpa_us:>10.1f} {speedup:>9.2f}x")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run correctness tests")
    parser.add_argument("--bench", action="store_true", help="Run benchmarks")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension (128 or 256)")
    parser.add_argument("--q-heads", type=int, default=32, help="Number of query heads")
    parser.add_argument("--kv-heads", type=int, default=2, help="Number of KV heads")
    parser.add_argument("--block-size", type=int, default=16, help="KV cache block size")
    args = parser.parse_args()

    if not args.test and not args.bench:
        args.test = True
        args.bench = True

    if args.test:
        ok = test_correctness(args.head_dim, args.q_heads, args.kv_heads, args.block_size)
        print(f"\nOverall: {'ALL PASS' if ok else 'SOME FAILURES'}")

    if args.bench:
        benchmark(args.head_dim, args.q_heads, args.kv_heads, args.block_size)
