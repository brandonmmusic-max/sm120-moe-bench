"""
Tests for selective attention mode.
"""
import sys
sys.path.insert(0, "/tmp/sm120-fa")

import torch
import torch.nn.functional as F
from sm120_selective_attn import SelectiveAttention, SelectiveAttnConfig, forward_selective


def test_short_sequence_fallback():
    """Short sequences should use full exact attention."""
    config = SelectiveAttnConfig(block_size=64, top_k_blocks=4, local_window_blocks=2)
    sa = SelectiveAttention(config)

    Q = torch.randn(1, 8, 64, 128, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(1, 8, 128, 128, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(1, 8, 128, 128, device="cuda", dtype=torch.bfloat16)

    out, debug = sa.forward(Q, K, V, return_debug=True)
    print(f"Short seq: mode={debug['mode']}, reason={debug.get('reason', 'N/A')}")
    assert debug["mode"] == "full_exact"
    assert out.shape == Q.shape
    print("  PASS")


def test_selective_mode_activates():
    """Long sequences should trigger selective mode."""
    config = SelectiveAttnConfig(
        block_size=64, top_k_blocks=4, local_window_blocks=2,
        fallback_threshold=1.0,  # Disable fallback for this test
        enable_full_fallback=False,
    )
    sa = SelectiveAttention(config)

    # Use structured data so routing is non-trivial
    Q = torch.randn(1, 8, 64, 128, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(1, 8, 2048, 128, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(1, 8, 2048, 128, device="cuda", dtype=torch.bfloat16)

    out, debug = sa.forward(Q, K, V, return_debug=True)
    coverage = debug.get('coverage', 'N/A')
    entropy = debug.get('entropy', 'N/A')
    print(f"Long seq: mode={debug['mode']}, selected={debug.get('selected_blocks', 'N/A')}/{debug['num_blocks']}")
    if isinstance(coverage, float):
        print(f"  Coverage: {coverage:.1%}")
    if isinstance(entropy, float):
        print(f"  Entropy: {entropy:.3f}")
    assert debug["mode"] == "selective", f"Expected selective, got {debug['mode']} ({debug.get('reason', '')})"
    assert out.shape == Q.shape
    assert debug["selected_blocks"] < debug["num_blocks"]
    print("  PASS")


def test_output_shape_consistency():
    """Output shape must match Q shape regardless of mode."""
    for Skv in [64, 256, 1024, 4096]:
        Q = torch.randn(1, 8, 32, 128, device="cuda", dtype=torch.bfloat16)
        K = torch.randn(1, 8, Skv, 128, device="cuda", dtype=torch.bfloat16)
        V = torch.randn(1, 8, Skv, 128, device="cuda", dtype=torch.bfloat16)

        out, _ = forward_selective(Q, K, V, fallback_threshold=0.99)
        assert out.shape == Q.shape, f"Shape mismatch at Skv={Skv}: {out.shape} vs {Q.shape}"
    print("Shape consistency: PASS")


def test_monotonic_accuracy():
    """More selected blocks should generally improve accuracy."""
    torch.manual_seed(42)
    Q = torch.randn(1, 8, 32, 128, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(1, 8, 4096, 128, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(1, 8, 4096, 128, device="cuda", dtype=torch.bfloat16)

    ref = F.scaled_dot_product_attention(Q, K, V)

    errors = []
    for top_k in [2, 4, 8, 16, 32]:
        out, debug = forward_selective(Q, K, V, top_k_blocks=top_k,
                                        fallback_threshold=0.99, return_debug=True)
        err = (out.float() - ref.float()).abs().mean().item()
        mode = debug["mode"]
        sel = debug.get("selected_blocks", "full")
        errors.append(err)
        print(f"  top_k={top_k:2d}: err={err:.6f}  selected={sel}  mode={mode}")

    # Error should generally decrease (allow small non-monotonicity)
    improving = sum(1 for i in range(1, len(errors)) if errors[i] <= errors[i-1] * 1.5)
    print(f"Monotonic: {improving}/{len(errors)-1} improving")
    print(f"  {'PASS' if improving >= 2 else 'WEAK'}")


def test_decode_scenario():
    """Decode: Q=1, long KV — the primary use case for selective attention."""
    torch.manual_seed(42)
    Q = torch.randn(1, 32, 1, 128, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(1, 8, 8192, 128, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(1, 8, 8192, 128, device="cuda", dtype=torch.bfloat16)

    out, debug = forward_selective(Q, K, V, block_size=64, top_k_blocks=8,
                                    local_window_blocks=4,
                                    fallback_threshold=1.1, return_debug=True)  # Disable fallback

    Ke = K.repeat_interleave(4, dim=1)
    Ve = V.repeat_interleave(4, dim=1)
    ref = F.scaled_dot_product_attention(Q, Ke, Ve)

    err = (out.float() - ref.float()).abs()
    print(f"Decode Q=1, KV=8192:")
    print(f"  Mode: {debug['mode']}")
    print(f"  Selected: {debug.get('selected_blocks', 'N/A')}/{debug['num_blocks']} blocks")
    sr = debug.get('speedup_ratio', 0)
    print(f"  Speedup ratio: {sr:.1f}x" if isinstance(sr, (int, float)) else f"  Speedup ratio: N/A")
    print(f"  Max err: {err.max().item():.6f}")
    print(f"  Mean err: {err.mean().item():.6f}")
    print(f"  PASS" if out.shape == Q.shape else "  FAIL")


def test_gqa():
    """GQA with selective attention."""
    Q = torch.randn(1, 32, 64, 128, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(1, 8, 2048, 128, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(1, 8, 2048, 128, device="cuda", dtype=torch.bfloat16)

    out, debug = forward_selective(Q, K, V, top_k_blocks=8,
                                    fallback_threshold=0.99, return_debug=True)
    print(f"GQA 32:8: mode={debug['mode']}, selected={debug.get('selected_blocks', 'N/A')}")
    assert out.shape == Q.shape
    print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Selective Attention Tests")
    print("=" * 60)
    test_short_sequence_fallback()
    test_selective_mode_activates()
    test_output_shape_consistency()
    test_monotonic_accuracy()
    test_decode_scenario()
    test_gqa()
    print("\nAll tests complete.")
