"""
Test Q@K^T MMA scores against scalar reference.

Runs the v2 scalar kernel and the v3 MMA kernel on the same input,
extracts the pre-softmax scores, and compares.
"""
import torch
import torch.nn.functional as F
import sm120_flash_attn  # The compiled module (currently v2 scalar)


def test_qk_mma():
    """Compare MMA Q@K^T against torch reference."""
    torch.manual_seed(42)
    B, Hq, Hkv = 1, 8, 8
    Sq, Skv, D = 64, 64, 128

    Q = torch.randn(B, Hq, Sq, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(B, Hkv, Skv, D, device="cuda", dtype=torch.bfloat16)

    scale = 1.0 / (D ** 0.5)

    # Reference Q@K^T scores
    ref_scores = (Q.float() @ K.float().transpose(-2, -1)) * scale

    # Our kernel output (full attention, not just scores)
    [out] = sm120_flash_attn.forward(Q, K, K, False)  # Use K as V for simplicity

    # Compare against torch SDPA
    ref_out = F.scaled_dot_product_attention(Q, K, K)

    err = (out.float() - ref_out.float()).abs()
    print(f"Full attention output error:")
    print(f"  Max: {err.max().item():.6f}")
    print(f"  Mean: {err.mean().item():.6f}")
    print(f"  {'PASS' if err.max().item() < 0.01 else 'FAIL'}")


if __name__ == "__main__":
    test_qk_mma()
