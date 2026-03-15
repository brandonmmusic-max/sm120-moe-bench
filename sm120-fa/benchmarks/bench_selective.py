"""
Benchmark selective attention vs full exact attention.
"""
import sys
sys.path.insert(0, "/tmp/sm120-fa")

import torch
import torch.nn.functional as F
import time
from sm120_selective_attn import forward_selective

try:
    import sm120_flash_attn
    HAS_SM120 = True
except ImportError:
    HAS_SM120 = False


def benchmark_one(func, warmup=3, iters=10):
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t = time.perf_counter()
        func()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    times.sort()
    return times[len(times) // 2]


def main():
    device = "cuda"
    dtype = torch.bfloat16
    Hq, Hkv, D = 32, 8, 128
    kv_repeat = Hq // Hkv

    print(f"{'':>8s} | {'Selective':>12s} {'sel/tot':>10s} {'err':>10s} | {'Full SM120':>12s} | {'SDPA':>12s} | {'Sel/Full':>10s} {'Sel/SDPA':>10s}")
    print("-" * 100)

    configs = [
        # (Sq, Skv, top_k, local_win, label)
        (1, 1024, 4, 2, "dec 1K"),
        (1, 4096, 8, 2, "dec 4K"),
        (1, 8192, 8, 4, "dec 8K"),
        (1, 16384, 12, 4, "dec 16K"),
        (1, 32768, 16, 4, "dec 32K"),
        (128, 4096, 8, 2, "pf 128/4K"),
        (128, 8192, 8, 4, "pf 128/8K"),
        (512, 8192, 12, 4, "pf 512/8K"),
    ]

    for Sq, Skv, top_k, local_win, label in configs:
        torch.manual_seed(42)
        Q = torch.randn(1, Hq, Sq, D, device=device, dtype=dtype)
        K = torch.randn(1, Hkv, Skv, D, device=device, dtype=dtype)
        V = torch.randn(1, Hkv, Skv, D, device=device, dtype=dtype)
        Ke = K.repeat_interleave(kv_repeat, dim=1)
        Ve = V.repeat_interleave(kv_repeat, dim=1)

        # Reference
        ref = F.scaled_dot_product_attention(Q, Ke, Ve)

        # Selective
        def run_sel():
            return forward_selective(Q, K, V, block_size=64, top_k_blocks=top_k,
                                      local_window_blocks=local_win,
                                      fallback_threshold=0.99)

        out_sel, debug = run_sel()
        sel_err = (out_sel.float() - ref.float()).abs().mean().item()
        sel_blocks = debug.get("selected_blocks", "full") if debug else "N/A"
        num_blocks = debug.get("num_blocks", "?") if debug else "?"

        t_sel = benchmark_one(lambda: forward_selective(Q, K, V, block_size=64,
                                                         top_k_blocks=top_k,
                                                         local_window_blocks=local_win,
                                                         fallback_threshold=0.99))

        # Full SM120
        if HAS_SM120:
            t_full = benchmark_one(lambda: sm120_flash_attn.forward(Q, K, V, False))
        else:
            t_full = benchmark_one(lambda: F.scaled_dot_product_attention(Q, Ke, Ve))

        # SDPA
        t_sdpa = benchmark_one(lambda: F.scaled_dot_product_attention(Q, Ke, Ve))

        speedup_full = t_full / t_sel if t_sel > 0 else 0
        speedup_sdpa = t_sdpa / t_sel if t_sel > 0 else 0

        print(f"{label:>8s} | {t_sel:>9.2f} ms {sel_blocks:>3s}/{num_blocks:>3s} {sel_err:>9.6f} | "
              f"{t_full:>9.2f} ms | {t_sdpa:>9.2f} ms | {speedup_full:>9.2f}x {speedup_sdpa:>9.2f}x")


if __name__ == "__main__":
    main()
