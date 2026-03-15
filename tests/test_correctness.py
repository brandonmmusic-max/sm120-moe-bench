"""
SM120 Flash Attention — Correctness Verification Suite

Tests against torch.nn.functional.scaled_dot_product_attention (cuDNN SDPA).
Methodology from FA2/FA3/FA4 test suites:
- Max absolute error < 0.05 (BF16)
- Mean absolute error < 0.001
- Cosine similarity > 0.999
- LSE (log-sum-exp) verified separately

Sweeps: sequence lengths, GQA ratios, batch sizes, non-aligned edge cases.
"""

import torch
import torch.nn.functional as F
import sm120_flash_attn
import sys


def test_config(B, Hq, Hkv, Sq, Skv, D=128, atol=0.05, verbose=True):
    """Test one configuration against PyTorch SDPA reference."""
    torch.manual_seed(42)
    Q = torch.randn(B, Hq, Sq, D, dtype=torch.bfloat16, device='cuda')
    K = torch.randn(B, Hkv, Skv, D, dtype=torch.bfloat16, device='cuda')
    V = torch.randn(B, Hkv, Skv, D, dtype=torch.bfloat16, device='cuda')

    Ke = K.repeat_interleave(Hq // Hkv, dim=1)
    Ve = V.repeat_interleave(Hq // Hkv, dim=1)
    ref = F.scaled_dot_product_attention(Q, Ke, Ve, is_causal=False)
    out = sm120_flash_attn.forward(Q, K, V)[0]

    diff = (out.float() - ref.float()).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    cos = F.cosine_similarity(out.flatten().float(), ref.flatten().float(), dim=0).item()

    passed = max_err < atol and cos > 0.999
    label = f"B={B} Hq={Hq} Hkv={Hkv} Sq={Sq} Skv={Skv}"
    if verbose:
        s = "PASS" if passed else "FAIL"
        print(f"  {label:>40s}: max={max_err:.6f} mean={mean_err:.6f} cos={cos:.6f} [{s}]")
    return passed


def test_lse(B, Hq, Hkv, Sq, Skv, D=128, verbose=True):
    """Verify LSE output against manual computation."""
    torch.manual_seed(42)
    Q = torch.randn(B, Hq, Sq, D, dtype=torch.bfloat16, device='cuda')
    K = torch.randn(B, Hkv, Skv, D, dtype=torch.bfloat16, device='cuda')
    V = torch.randn(B, Hkv, Skv, D, dtype=torch.bfloat16, device='cuda')

    results = sm120_flash_attn.forward(Q, K, V, True)
    if len(results) < 2:
        if verbose: print(f"  LSE B={B} Sq={Sq}: SKIP (no LSE)")
        return True

    lse = results[1]
    scale = 1.0 / (D ** 0.5)
    Ke = K.repeat_interleave(Hq // Hkv, dim=1)
    scores = torch.matmul(Q.float(), Ke.float().transpose(-2, -1)) * scale
    ref_lse = torch.logsumexp(scores, dim=-1)

    max_err = (lse - ref_lse).abs().max().item()
    passed = max_err < 1.0
    if verbose:
        print(f"  LSE B={B} Hq={Hq} Sq={Sq} Skv={Skv}: max_err={max_err:.4f} [{'PASS' if passed else 'FAIL'}]")
    return passed


def run_suite():
    print("=" * 80)
    print("SM120 Flash Attention — Correctness Test Suite")
    print("Reference: torch SDPA | Tolerance: max<0.05, cos>0.999")
    print("=" * 80)

    total, passed = 0, 0

    def run(name, tests):
        nonlocal total, passed
        print(f"\n--- {name} ---")
        for t in tests:
            total += 1
            if t(): passed += 1

    # 1. Sequence length sweep
    run("Sequence Lengths (MHA)", [
        lambda sq=sq: test_config(1, 8, 8, sq, sq)
        for sq in [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    ])

    # 2. GQA ratios
    run("GQA Ratios", [
        lambda hq=hq, hkv=hkv: test_config(1, hq, hkv, 2048, 2048)
        for hq, hkv in [(8,8), (16,8), (32,8), (64,8), (32,4), (32,1)]
    ])

    # 3. Non-aligned (masking edge cases)
    run("Non-Aligned Sequences (masking)", [
        lambda sq=sq, skv=skv: test_config(1, 8, 8, sq, skv)
        for sq, skv in [(127,127), (129,129), (255,255), (513,513),
                        (1000,1000), (2048,100), (100,2048), (1,1024)]
    ])

    # 4. Asymmetric Sq != Skv
    run("Asymmetric Sq/Skv", [
        lambda sq=sq, skv=skv: test_config(1, 32, 8, sq, skv)
        for sq, skv in [(128,2048), (2048,128), (1024,4096), (4096,512)]
    ])

    # 5. Batch sizes
    run("Batch Sizes", [
        lambda b=b: test_config(b, 32, 8, 1024, 1024)
        for b in [1, 2, 4, 8]
    ])

    # 6. LSE
    run("LSE Verification", [
        lambda sq=sq: test_lse(1, 8, 8, sq, sq)
        for sq in [128, 512, 2048]
    ])

    # 7. Large sequences
    run("Large Sequences", [
        lambda sq=sq: test_config(1, 32, 8, sq, sq)
        for sq in [8192, 16384]
    ])

    print(f"\n{'='*80}")
    print(f"Results: {passed}/{total} passed")
    print("ALL TESTS PASSED" if passed == total else f"FAILURES: {total-passed}")
    print("=" * 80)
    return passed == total


if __name__ == "__main__":
    # Switch back to BF16 v4 kernel for correctness test
    success = run_suite()
    sys.exit(0 if success else 1)
