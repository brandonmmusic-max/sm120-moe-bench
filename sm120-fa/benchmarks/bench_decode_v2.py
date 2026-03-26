"""
Benchmark: SM120 Flash Decode v2 (tiled) vs v1 (scalar) vs FlashInfer.

Tests at Qwen3.5-397B per-GPU dimensions (TP=4):
  - num_q_heads = 8  (32 total / 4 TP)
  - num_kv_heads = 2  (replicated)
  - head_dim = 128 and 256
  - block_size = 16
  - BF16 KV cache
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F


def create_paged_kv_cache(batch_size, seq_lens, num_kv_heads, head_dim, block_size, device):
    """Create paged KV cache with random data and block table."""
    max_seq_len = max(seq_lens)
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    total_blocks = sum((sl + block_size - 1) // block_size for sl in seq_lens) + 10

    key_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim,
                            dtype=torch.bfloat16, device=device) * 0.1
    value_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim,
                              dtype=torch.bfloat16, device=device) * 0.1

    block_table = torch.zeros(batch_size, max_blocks_per_seq, dtype=torch.int32, device=device)
    blk_counter = 0
    for b in range(batch_size):
        n_blks = (seq_lens[b] + block_size - 1) // block_size
        for i in range(n_blks):
            block_table[b, i] = blk_counter
            blk_counter += 1

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    return key_cache, value_cache, block_table, seq_lens_t


def setup_flashinfer(batch_size, seq_lens, num_q_heads, num_kv_heads, head_dim, block_size, device):
    """Set up FlashInfer paged decode wrapper."""
    import flashinfer

    total_pages = sum((sl + block_size - 1) // block_size for sl in seq_lens) + 10
    kv_data = torch.randn(total_pages, 2, num_kv_heads, block_size, head_dim,
                          dtype=torch.bfloat16, device=device) * 0.1

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
        dtype=torch.int32, device=device)

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


def bench_kernel(fn, warmup=10, iters=100):
    """Benchmark a kernel, return latency in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / iters  # ms -> us


def benchmark_config(batch_size, seq_len, num_q_heads, num_kv_heads, head_dim, block_size):
    device = "cuda"
    seq_lens = [seq_len] * batch_size

    from sm120_flash_decode_ext import sm120_flash_decode_paged, SM120FlashDecodeWorkspace

    # SM120 v2 setup
    key_cache, value_cache, block_table, seq_lens_t = create_paged_kv_cache(
        batch_size, seq_lens, num_kv_heads, head_dim, block_size, device)
    query = torch.randn(batch_size, num_q_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1
    workspace = SM120FlashDecodeWorkspace(batch_size, num_q_heads, head_dim, 32, device)
    output = torch.empty_like(query)

    max_sl = max(seq_lens)
    sm120_us = bench_kernel(lambda: sm120_flash_decode_paged(
        query, key_cache, value_cache, block_table, seq_lens_t,
        output=output, workspace=workspace, max_seq_len=max_sl))

    # FlashInfer setup
    fi_us = float('inf')
    try:
        fi_wrapper, fi_kv_data = setup_flashinfer(
            batch_size, seq_lens, num_q_heads, num_kv_heads, head_dim, block_size, device)
        fi_us = bench_kernel(lambda: fi_wrapper.run(query, fi_kv_data))
    except Exception as e:
        pass

    return sm120_us, fi_us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    hd = args.head_dim
    qh = args.q_heads
    kvh = args.kv_heads
    bs = args.block_size

    configs = [
        (1, 256, "1 user, 256 ctx"),
        (1, 1024, "1 user, 1K ctx"),
        (1, 4096, "1 user, 4K ctx"),
        (1, 16384, "1 user, 16K ctx"),
        (1, 32768, "1 user, 32K ctx"),
        (1, 65536, "1 user, 64K ctx"),
        (8, 1024, "8 users, 1K ctx"),
        (8, 4096, "8 users, 4K ctx"),
        (8, 16384, "8 users, 16K ctx"),
        (32, 2048, "32 users, 2K ctx"),
    ]

    print(f"\n{'='*80}")
    print(f"SM120 Flash Decode v2 (tiled) vs FlashInfer")
    print(f"  q_heads={qh}, kv_heads={kvh}, head_dim={hd}, block_size={bs}")
    print(f"{'='*80}")
    print(f"{'Config':<25} {'SM120-v2 (us)':>14} {'FlashInfer (us)':>16} {'Ratio':>10}")
    print(f"{'-'*25} {'-'*14} {'-'*16} {'-'*10}")

    for batch_size, seq_len, label in configs:
        try:
            sm120_us, fi_us = benchmark_config(
                batch_size, seq_len, qh, kvh, hd, bs)
            if fi_us < float('inf'):
                ratio = sm120_us / fi_us
                marker = " <--" if ratio < 1.0 else ""
                print(f"  {label:<25} {sm120_us:>12.1f} {fi_us:>14.1f} {ratio:>9.2f}x{marker}")
            else:
                print(f"  {label:<25} {sm120_us:>12.1f} {'N/A':>14} {'N/A':>9}")
        except Exception as e:
            print(f"  {label:<25} ERROR: {e}")


if __name__ == "__main__":
    main()
