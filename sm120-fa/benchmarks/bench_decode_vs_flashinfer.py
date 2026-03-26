"""
Benchmark: SM120 Flash Decode vs FlashInfer paged decode.

Tests at Qwen3.5-397B per-GPU dimensions:
  - num_q_heads = 8  (32 total / 4 TP)
  - num_kv_heads = 2  (replicated across TP)
  - head_dim = 256
  - block_size = 16
  - BF16 KV cache (FP8 comparison TODO)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time


def setup_flashinfer_paged(batch_size, seq_lens, num_q_heads, num_kv_heads, head_dim, block_size, device):
    """Set up FlashInfer's BatchDecodeWithPagedKVCacheWrapper."""
    import flashinfer

    max_seq_len = max(seq_lens)

    # Total pages needed
    total_pages = sum((sl + block_size - 1) // block_size for sl in seq_lens)
    total_pages += 10  # headroom

    # KV data: [max_num_pages, 2, num_kv_heads, page_size, head_dim]
    kv_data = torch.randn(
        total_pages, 2, num_kv_heads, block_size, head_dim,
        dtype=torch.bfloat16, device=device
    ) * 0.1

    # Build page table
    kv_page_indices = []
    kv_page_indptr = [0]
    page_counter = 0
    for sl in seq_lens:
        n_pages = (sl + block_size - 1) // block_size
        for i in range(n_pages):
            kv_page_indices.append(page_counter)
            page_counter += 1
        kv_page_indptr.append(len(kv_page_indices))

    kv_page_indices = torch.tensor(kv_page_indices, dtype=torch.int32, device=device)
    kv_page_indptr = torch.tensor(kv_page_indptr, dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor(
        [(sl - 1) % block_size + 1 for sl in seq_lens],
        dtype=torch.int32, device=device
    )

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        float_workspace_buffer=torch.empty(256 * 1024 * 1024, dtype=torch.float32, device=device),
        kv_layout="NHD",
    )
    wrapper.plan(
        indptr=kv_page_indptr,
        indices=kv_page_indices,
        last_page_len=kv_last_page_len,
        num_qo_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=block_size,
        q_data_type=torch.bfloat16,
        data_type=torch.bfloat16,
    )

    return wrapper, kv_data


def setup_sm120_paged(batch_size, seq_lens, num_q_heads, num_kv_heads, head_dim, block_size, device):
    """Set up SM120 decode with paged KV cache."""
    from sm120_flash_decode_ext import SM120FlashDecodeWorkspace

    max_seq_len = max(seq_lens)
    max_blocks = (max_seq_len + block_size - 1) // block_size
    total_blocks = sum((sl + block_size - 1) // block_size for sl in seq_lens) + 10

    # KV cache: [num_blocks, block_size, num_kv_heads, head_dim]
    key_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim,
                            dtype=torch.bfloat16, device=device) * 0.1
    value_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim,
                              dtype=torch.bfloat16, device=device) * 0.1

    # Block table
    block_table = torch.zeros(batch_size, max_blocks, dtype=torch.int32, device=device)
    blk = 0
    for b in range(batch_size):
        n = (seq_lens[b] + block_size - 1) // block_size
        for i in range(n):
            block_table[b, i] = blk
            blk += 1

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    workspace = SM120FlashDecodeWorkspace(batch_size, num_q_heads, head_dim, 32, device)

    return key_cache, value_cache, block_table, seq_lens_t, workspace


def benchmark_config(batch_size, seq_len, num_q_heads, num_kv_heads, head_dim, block_size, n_iters=100):
    device = "cuda"
    seq_lens = [seq_len] * batch_size

    # Setup FlashInfer
    try:
        fi_wrapper, fi_kv_data = setup_flashinfer_paged(
            batch_size, seq_lens, num_q_heads, num_kv_heads, head_dim, block_size, device
        )
        fi_available = True
    except Exception as e:
        print(f"  FlashInfer setup failed: {e}")
        fi_available = False

    # Setup SM120
    from sm120_flash_decode_ext import sm120_flash_decode_paged
    key_cache, value_cache, block_table, seq_lens_t, workspace = setup_sm120_paged(
        batch_size, seq_lens, num_q_heads, num_kv_heads, head_dim, block_size, device
    )

    query = torch.randn(batch_size, num_q_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    output = torch.empty_like(query)

    # Warmup SM120
    for _ in range(10):
        sm120_flash_decode_paged(query, key_cache, value_cache, block_table, seq_lens_t,
                                 output=output, workspace=workspace)
    torch.cuda.synchronize()

    # Benchmark SM120
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iters):
        sm120_flash_decode_paged(query, key_cache, value_cache, block_table, seq_lens_t,
                                 output=output, workspace=workspace)
    end.record()
    torch.cuda.synchronize()
    sm120_us = start.elapsed_time(end) * 1000 / n_iters

    fi_us = float('inf')
    if fi_available:
        # Warmup FlashInfer
        fi_query = query.unsqueeze(2)  # [batch, heads, 1, dim] -> FlashInfer expects [batch*heads, 1, dim] or [batch, heads, dim]
        # FlashInfer decode expects: q shape [batch_size, num_qo_heads, head_dim]
        for _ in range(10):
            fi_wrapper.run(query, fi_kv_data)
        torch.cuda.synchronize()

        start.record()
        for _ in range(n_iters):
            fi_wrapper.run(query, fi_kv_data)
        end.record()
        torch.cuda.synchronize()
        fi_us = start.elapsed_time(end) * 1000 / n_iters

    return sm120_us, fi_us


def main():
    # Qwen3.5-397B per-GPU dimensions (TP=4)
    num_q_heads = 8   # 32 / 4
    num_kv_heads = 2  # replicated
    head_dim = 256
    block_size = 16

    configs = [
        # (batch_size, seq_len, label)
        (1, 256, "1 user, 256 ctx"),
        (1, 1024, "1 user, 1K ctx"),
        (1, 4096, "1 user, 4K ctx"),
        (1, 16384, "1 user, 16K ctx"),
        (1, 32768, "1 user, 32K ctx"),
        (8, 1024, "8 users, 1K ctx"),
        (8, 4096, "8 users, 4K ctx"),
        (8, 16384, "8 users, 16K ctx"),
        (32, 2048, "32 users, 2K ctx"),
    ]

    print(f"\n{'='*75}")
    print(f"SM120 Flash Decode vs FlashInfer — Qwen3.5-397B per-GPU (TP=4)")
    print(f"  q_heads={num_q_heads}, kv_heads={num_kv_heads}, head_dim={head_dim}, block_size={block_size}")
    print(f"{'='*75}")
    print(f"{'Config':<25} {'SM120 (us)':>12} {'FlashInfer (us)':>16} {'Speedup':>10}")
    print(f"{'-'*25} {'-'*12} {'-'*16} {'-'*10}")

    for batch_size, seq_len, label in configs:
        try:
            sm120_us, fi_us = benchmark_config(
                batch_size, seq_len, num_q_heads, num_kv_heads, head_dim, block_size
            )
            if fi_us < float('inf'):
                speedup = fi_us / sm120_us
                print(f"  {label:<25} {sm120_us:>10.1f} {fi_us:>14.1f} {speedup:>9.2f}x")
            else:
                print(f"  {label:<25} {sm120_us:>10.1f} {'N/A':>14} {'N/A':>9}")
        except Exception as e:
            print(f"  {label:<25} ERROR: {e}")

    # Also test with full-model dimensions (all 32 q_heads, for comparison)
    print(f"\n{'='*75}")
    print(f"Full-model dimensions (no TP): q_heads=32, kv_heads=2, head_dim=256")
    print(f"{'='*75}")
    print(f"{'Config':<25} {'SM120 (us)':>12} {'FlashInfer (us)':>16} {'Speedup':>10}")
    print(f"{'-'*25} {'-'*12} {'-'*16} {'-'*10}")

    for batch_size, seq_len, label in [(1, 4096, "1 user, 4K"), (8, 4096, "8 users, 4K")]:
        try:
            sm120_us, fi_us = benchmark_config(
                batch_size, seq_len, 32, 2, head_dim, block_size
            )
            if fi_us < float('inf'):
                speedup = fi_us / sm120_us
                print(f"  {label:<25} {sm120_us:>10.1f} {fi_us:>14.1f} {speedup:>9.2f}x")
            else:
                print(f"  {label:<25} {sm120_us:>10.1f} {'N/A':>14}")
        except Exception as e:
            print(f"  {label:<25} ERROR: {e}")


if __name__ == "__main__":
    main()
