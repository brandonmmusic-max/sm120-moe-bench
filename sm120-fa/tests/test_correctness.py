"""
Correctness test: SM120 Flash Attention vs torch SDPA.

Compares outputs and reports max/mean absolute and relative errors.
"""

import torch
import torch.nn.functional as F
import sm120_flash_attn
import argparse


def test_correctness(
    batch=2, num_q_heads=32, num_kv_heads=8, seq_q=512, seq_kv=512, head_dim=128,
    atol=1e-2, rtol=5e-2
):
    """Compare SM120 FA output against PyTorch SDPA."""
    print(f"Config: B={batch}, Hq={num_q_heads}, Hkv={num_kv_heads}, "
          f"Sq={seq_q}, Skv={seq_kv}, D={head_dim}")

    device = "cuda"
    dtype = torch.bfloat16

    # Generate random inputs
    torch.manual_seed(42)
    Q = torch.randn(batch, num_q_heads, seq_q, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch, num_kv_heads, seq_kv, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch, num_kv_heads, seq_kv, head_dim, device=device, dtype=dtype)

    # Expand KV for GQA comparison with SDPA
    kv_repeat = num_q_heads // num_kv_heads
    K_expanded = K.repeat_interleave(kv_repeat, dim=1)
    V_expanded = V.repeat_interleave(kv_repeat, dim=1)

    # Reference: PyTorch SDPA
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(Q, K_expanded, V_expanded)

    # Our kernel
    with torch.no_grad():
        [out] = sm120_flash_attn.forward(Q, K, V, False)

    # Compare
    abs_err = (out.float() - ref.float()).abs()
    rel_err = abs_err / (ref.float().abs() + 1e-8)

    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    max_rel = rel_err.max().item()
    mean_rel = rel_err.mean().item()

    print(f"  Max  absolute error: {max_abs:.6f}")
    print(f"  Mean absolute error: {mean_abs:.6f}")
    print(f"  Max  relative error: {max_rel:.6f}")
    print(f"  Mean relative error: {mean_rel:.6f}")

    passed = max_abs < atol and mean_rel < rtol
    status = "PASS" if passed else "FAIL"
    print(f"  Status: {status} (atol={atol}, rtol={rtol})")
    print()

    return passed


def main():
    parser = argparse.ArgumentParser(description="SM120 FA Correctness Test")
    parser.add_argument("--quick", action="store_true", help="Run minimal tests")
    args = parser.parse_args()

    configs = [
        # (batch, num_q_heads, num_kv_heads, seq_q, seq_kv, head_dim)
        (1, 8, 8, 128, 128, 128),     # MHA, short
        (1, 8, 8, 512, 512, 128),     # MHA, medium
        (1, 32, 8, 512, 512, 128),    # GQA 4:1
        (2, 32, 8, 1024, 1024, 128),  # GQA, longer
    ]

    if not args.quick:
        configs += [
            (1, 32, 8, 2048, 2048, 128),   # GQA, 2K
            (1, 32, 8, 4096, 4096, 128),   # GQA, 4K
            (1, 32, 8, 128, 8192, 128),    # Decode-like (short Q, long KV)
            (1, 32, 8, 1, 4096, 128),      # Single-token decode
            (4, 32, 8, 256, 256, 128),     # Batched
        ]

    all_passed = True
    for config in configs:
        passed = test_correctness(*config)
        all_passed = all_passed and passed

    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        exit(1)


if __name__ == "__main__":
    main()
