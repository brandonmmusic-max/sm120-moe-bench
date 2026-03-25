#!/usr/bin/env python3
"""
Standalone test for Expert Union Routing (Sprint 8).

Generates M=4 tokens with INTENTIONALLY DIFFERENT expert routing:
  Token 0: experts [0,1,2,3,4,5,6,7,8,9]
  Token 1: experts [5,6,7,8,9,10,11,12,13,14]   (50% overlap with token 0)
  Token 2: experts [10,11,12,13,14,15,16,17,18,19] (0% overlap with token 0)
  Token 3: experts [0,1,2,3,4,15,16,17,18,19]    (50% overlap with token 0)

Expert union = 20 unique experts.

Runs union routing kernel for M=4, then runs each token individually at M=1
with its own routing. Compares per-token outputs.

Target: max per-token relative error < 15% (FP4 accumulation noise).
Key check: token 2 (zero overlap with token 0) produces DIFFERENT output
than it would with shared routing (which uses token 0's experts).
"""

import os
import sys
import torch
from pathlib import Path
from torch.utils.cpp_extension import load

# ============================================================================
# Configuration
# ============================================================================
HIDDEN = 4096
N_HALF = 1024      # EP=4 shape (k_groups=1 for union — stress tests Phase 2 tile loop)
NUM_EXPERTS = 64   # enough for 20 unique experts
TOPK = 10
M = 4
MAX_M = 4
SF_BLOCK = 16
BN = 64
BK = 64

CUDA_FLAGS = [
    "-gencode=arch=compute_120a,code=sm_120a",
    "-O2",
    "--expt-relaxed-constexpr",
    "-use_fast_math",
]


def load_extension():
    """JIT-compile the fused cooperative extension."""
    csrc_dir = Path(__file__).parent / "csrc"
    ext_src = csrc_dir / "verdict_fused_cooperative_ext.cu"
    assert ext_src.exists(), f"Source not found: {ext_src}"

    print("JIT-compiling fused cooperative extension (may take ~60s)...")
    ext = load(
        name="verdict_fused_cooperative_ext_test",
        sources=[str(ext_src)],
        extra_cuda_cflags=CUDA_FLAGS,
        verbose=False,
    )
    print("Compilation done.")
    return ext


def make_random_weights(num_experts, n_half, device):
    """Create random NVFP4 weights + E4M3FN scales."""
    K = HIDDEN
    K_packed = K // 2
    sf_cols = K // SF_BLOCK

    # W1: [E, 2*N_half, K/2] uint8 (packed FP4 nibbles)
    w1_fp4 = torch.randint(0, 256, (num_experts, 2 * n_half, K_packed),
                           dtype=torch.uint8, device=device)
    # W1 scales: [E, 2*N_half, K/16] uint8 (E4M3FN)
    # Use moderate scale values (exponent 5-10 → reasonable magnitudes)
    w1_sf = torch.randint(40, 80, (num_experts, 2 * n_half, sf_cols),
                          dtype=torch.uint8, device=device)

    # W2: [E, K, N_half/2] uint8
    w2_fp4 = torch.randint(0, 256, (num_experts, K, n_half // 2),
                           dtype=torch.uint8, device=device)
    w2_sf = torch.randint(40, 80, (num_experts, K, n_half // SF_BLOCK),
                          dtype=torch.uint8, device=device)

    return w1_fp4, w1_sf, w2_fp4, w2_sf


def make_alpha(num_experts, device):
    """Create random per-expert alpha values (weight × activation scale product)."""
    # Typical alpha range: 0.001 - 0.1
    w1_alpha_all = torch.rand(num_experts, dtype=torch.float32, device=device) * 0.05 + 0.01
    w2_alpha_all = torch.rand(num_experts, dtype=torch.float32, device=device) * 0.05 + 0.01
    return w1_alpha_all, w2_alpha_all


def build_union_routing(topk_ids, topk_weights, w1_alpha_all, w2_alpha_all,
                        m, topk, device):
    """Build expert union routing tables (mirrors verdict_moe.py logic)."""
    grid_experts = m * topk  # 40
    MAX_M_K = 4

    # Sort and dedup
    all_ids = topk_ids[:m].reshape(-1).int()  # [40]
    n_slots = m * topk

    sorted_ids, sort_perm = torch.sort(all_ids)

    dedup = torch.ones(n_slots, dtype=torch.int32, device=device)
    if n_slots > 1:
        dedup[1:] = (sorted_ids[1:] != sorted_ids[:n_slots - 1]).int()

    positions = torch.cumsum(dedup, 0) - 1

    # Union IDs
    union_ids = torch.zeros(grid_experts, dtype=torch.int32, device=device)
    union_ids.scatter_(0, positions.long(), sorted_ids)

    # Count unique
    num_union = (positions[-1] + 1).item()

    # Token mask and weights
    union_mask = torch.zeros(grid_experts * MAX_M_K, dtype=torch.uint8, device=device)
    union_weights = torch.zeros(grid_experts * MAX_M_K, dtype=torch.float32, device=device)

    uid_exp = union_ids.unsqueeze(0).unsqueeze(2)
    tid_exp = topk_ids[:m].int().unsqueeze(1)
    match = (uid_exp == tid_exp)

    mask_2d = match.any(dim=2)  # [M, grid_experts]

    # Zero out mask for padded positions (>= num_union) to avoid duplicate contributions
    pos_range = torch.arange(grid_experts, device=device)
    valid = (pos_range < num_union)  # [grid_experts]
    mask_2d = mask_2d & valid.unsqueeze(0)

    union_mask.view(grid_experts, MAX_M_K)[:, :m].copy_(mask_2d.T.to(torch.uint8))

    wts_exp = topk_weights[:m].unsqueeze(1).expand_as(match)
    wt_2d = (wts_exp * match.float()).sum(dim=2)
    wt_2d = wt_2d * valid.unsqueeze(0).float()  # zero weights for padded positions
    union_weights.view(grid_experts, MAX_M_K)[:, :m].copy_(wt_2d.T)

    # Alpha
    union_w1a = torch.zeros(grid_experts, dtype=torch.float32, device=device)
    union_w2a = torch.zeros(grid_experts, dtype=torch.float32, device=device)
    clamped = union_ids.long().clamp(min=0, max=w1_alpha_all.size(0) - 1)
    torch.index_select(w1_alpha_all, 0, clamped, out=union_w1a)
    torch.index_select(w2_alpha_all, 0, clamped, out=union_w2a)

    return (union_ids, union_mask, union_weights, union_w1a, union_w2a,
            grid_experts, num_union)


def build_m1_routing(tok_ids, tok_weights, w1_alpha_all, w2_alpha_all, topk, device):
    """Build M=1 routing tables for a single token."""
    MAX_M_K = 4
    mask = torch.zeros(topk * MAX_M_K, dtype=torch.uint8, device=device)
    mask.view(topk, MAX_M_K)[:, 0] = 1

    weights = torch.zeros(topk * MAX_M_K, dtype=torch.float32, device=device)
    weights.view(topk, MAX_M_K)[:, 0] = tok_weights

    w1a = w1_alpha_all[tok_ids.long()]
    w2a = w2_alpha_all[tok_ids.long()]

    return mask, weights, w1a, w2a


def run_kernel(ext, hidden, w1_fp4, w1_sf, w2_fp4, w2_sf,
               expert_ids, w1_alpha, w2_alpha, token_mask, token_weights,
               grid_experts, n_half, device):
    """Run the fused cooperative kernel and return BF16 output."""
    m_tokens = hidden.size(0)
    K = HIDDEN
    K_packed = K // 2
    sf_cols = K // SF_BLOCK
    n_half_packed = n_half // 2
    n_half_sf = n_half // SF_BLOCK

    output = torch.zeros(m_tokens, K, dtype=torch.bfloat16, device=device)
    output_f32 = torch.zeros(m_tokens * K, dtype=torch.float32, device=device)
    input_fp4 = torch.zeros(m_tokens * K_packed, dtype=torch.uint8, device=device)
    input_sf = torch.zeros(m_tokens * sf_cols, dtype=torch.uint8, device=device)

    # Size intermediate for grid_experts (not topk)
    inter_fp4 = torch.zeros(grid_experts * m_tokens * n_half_packed,
                            dtype=torch.uint8, device=device)
    inter_sf = torch.zeros(grid_experts * m_tokens * n_half_sf,
                           dtype=torch.uint8, device=device)

    # Partials: grid_experts * num_tiles * M * 128
    tiles_n = n_half // BN
    total_k_tiles = HIDDEN // BK
    k_groups = max(1, 640 // (grid_experts * tiles_n))
    while total_k_tiles % k_groups != 0 and k_groups > 1:
        k_groups -= 1
    num_tiles = tiles_n * k_groups
    partials = torch.zeros(grid_experts * num_tiles * m_tokens * 128,
                           dtype=torch.float32, device=device)

    barrier = torch.zeros(1, dtype=torch.int32, device=device)

    ext.forward(
        hidden.contiguous(),
        w1_fp4.contiguous(),
        w1_sf.contiguous(),
        w1_alpha.contiguous(),
        w2_fp4.contiguous(),
        w2_sf.contiguous(),
        w2_alpha.contiguous(),
        output,
        expert_ids.contiguous(),
        token_mask.contiguous(),
        token_weights.contiguous(),
        output_f32,
        input_fp4,
        input_sf,
        partials,
        inter_fp4,
        inter_sf,
        barrier,
        K, n_half, grid_experts,
    )

    return output


def main():
    if not torch.cuda.is_available():
        print("CUDA not available!")
        sys.exit(1)

    device = torch.device("cuda:0")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Load extension
    ext = load_extension()

    # Create random weights
    w1_fp4, w1_sf, w2_fp4, w2_sf = make_random_weights(NUM_EXPERTS, N_HALF, device)
    w1_alpha_all, w2_alpha_all = make_alpha(NUM_EXPERTS, device)

    # Create random BF16 input: [M=4, HIDDEN]
    hidden = torch.randn(M, HIDDEN, dtype=torch.bfloat16, device=device)

    # --- Define per-token routing (intentionally different) ---
    topk_ids = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],          # Token 0
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],       # Token 1: 50% overlap
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],   # Token 2: 0% overlap
        [0, 1, 2, 3, 4, 15, 16, 17, 18, 19],        # Token 3: 50% overlap
    ], dtype=torch.int32, device=device)

    topk_weights = torch.rand(M, TOPK, dtype=torch.float32, device=device) * 0.2 + 0.05
    # Normalize weights per token (softmax-like)
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)

    print(f"\n{'='*70}")
    print("Expert Union Routing Test (Sprint 8)")
    print(f"{'='*70}")
    print(f"M={M}, TOPK={TOPK}, HIDDEN={HIDDEN}, N_HALF={N_HALF}, E={NUM_EXPERTS}")
    print(f"\nRouting:")
    for t in range(M):
        ids = topk_ids[t].tolist()
        wts = topk_weights[t].tolist()
        print(f"  Token {t}: experts={ids}")
        print(f"           weights=[{', '.join(f'{w:.3f}' for w in wts)}]")

    # --- Build union routing ---
    (union_ids, union_mask, union_weights, union_w1a, union_w2a,
     grid_experts, num_union) = build_union_routing(
        topk_ids, topk_weights, w1_alpha_all, w2_alpha_all, M, TOPK, device)

    print(f"\nUnion: {num_union} unique experts out of {M*TOPK} slots")
    print(f"  union_ids[:num_union] = {union_ids[:num_union].tolist()}")
    print(f"  grid_experts = {grid_experts}")

    # Print mask summary
    mask_view = union_mask.view(grid_experts, MAX_M)[:num_union, :M]
    print(f"  Token mask (union_expert × token):")
    for i in range(min(num_union, 25)):
        row = mask_view[i].tolist()
        eid = union_ids[i].item()
        print(f"    expert {eid:2d}: [{', '.join(str(x) for x in row)}]")

    # --- Run M=4 union routing kernel ---
    print(f"\n--- Running M=4 union routing kernel ---")
    output_union = run_kernel(
        ext, hidden, w1_fp4, w1_sf, w2_fp4, w2_sf,
        union_ids, union_w1a, union_w2a,
        union_mask, union_weights,
        grid_experts, N_HALF, device,
    )
    torch.cuda.synchronize()
    print(f"  Output shape: {output_union.shape}")
    for t in range(M):
        row = output_union[t].float()
        print(f"  Token {t}: norm={row.norm():.4f}, "
              f"mean={row.mean():.6f}, std={row.std():.4f}")

    # --- Run M=1 per-token reference ---
    print(f"\n--- Running M=1 per-token reference ---")
    outputs_m1 = []
    for tok in range(M):
        tok_ids = topk_ids[tok]
        tok_wts = topk_weights[tok]

        mask, weights, w1a, w2a = build_m1_routing(
            tok_ids, tok_wts, w1_alpha_all, w2_alpha_all, TOPK, device)

        out = run_kernel(
            ext, hidden[tok:tok+1], w1_fp4, w1_sf, w2_fp4, w2_sf,
            tok_ids, w1a, w2a,
            mask, weights,
            TOPK, N_HALF, device,
        )
        torch.cuda.synchronize()
        outputs_m1.append(out[0])
        row = out[0].float()
        print(f"  Token {tok}: norm={row.norm():.4f}, "
              f"mean={row.mean():.6f}, std={row.std():.4f}")

    # --- Compare per-token outputs ---
    print(f"\n{'='*70}")
    print("Per-token comparison: union M=4 vs individual M=1")
    print(f"{'='*70}")

    all_pass = True
    for tok in range(M):
        union_out = output_union[tok].float()
        ref_out = outputs_m1[tok].float()

        # Absolute difference
        abs_diff = (union_out - ref_out).abs()
        max_abs_diff = abs_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()

        # Relative error (avoid div by zero)
        ref_norm = ref_out.norm().item()
        diff_norm = (union_out - ref_out).norm().item()
        rel_error = diff_norm / max(ref_norm, 1e-10) * 100.0

        status = "PASS" if rel_error < 15.0 else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"\n  Token {tok}:")
        print(f"    Ref norm:      {ref_norm:.4f}")
        print(f"    Union norm:    {union_out.norm().item():.4f}")
        print(f"    Max abs diff:  {max_abs_diff:.6f}")
        print(f"    Mean abs diff: {mean_abs_diff:.6f}")
        print(f"    Relative error: {rel_error:.2f}%  [{status}] (threshold: 15%)")

    # --- KEY CHECK: Token 2 with shared routing vs per-token ---
    print(f"\n{'='*70}")
    print("KEY CHECK: Token 2 (0% overlap with token 0)")
    print(f"{'='*70}")

    # Run token 2 with token 0's routing (shared routing, the OLD behavior)
    shared_mask, shared_weights, shared_w1a, shared_w2a = build_m1_routing(
        topk_ids[0], topk_weights[0], w1_alpha_all, w2_alpha_all, TOPK, device)

    # Apply token 0's routing to token 2's input
    out_shared = run_kernel(
        ext, hidden[2:3], w1_fp4, w1_sf, w2_fp4, w2_sf,
        topk_ids[0], shared_w1a, shared_w2a,
        shared_mask, shared_weights,
        TOPK, N_HALF, device,
    )
    torch.cuda.synchronize()

    shared_out = out_shared[0].float()
    correct_out = outputs_m1[2].float()
    union_out_2 = output_union[2].float()

    # How different is shared routing from correct routing for token 2?
    shared_diff = (shared_out - correct_out).norm().item()
    shared_rel = shared_diff / max(correct_out.norm().item(), 1e-10) * 100.0

    # How close is union routing to correct routing for token 2?
    union_diff = (union_out_2 - correct_out).norm().item()
    union_rel = union_diff / max(correct_out.norm().item(), 1e-10) * 100.0

    print(f"  Token 2 correct output norm: {correct_out.norm():.4f}")
    print(f"  Token 2 with shared routing (old): rel error = {shared_rel:.2f}%")
    print(f"  Token 2 with union routing (new):  rel error = {union_rel:.2f}%")
    print(f"  Shared routing produced DIFFERENT output: "
          f"{'YES' if shared_rel > 1.0 else 'NO'} ({shared_rel:.1f}%)")
    print(f"  Union routing matches correct:     "
          f"{'YES' if union_rel < 15.0 else 'NO'} ({union_rel:.1f}%)")

    key_pass = (shared_rel > 1.0) and (union_rel < 15.0)

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Per-token error (all tokens < 15%): {'PASS' if all_pass else 'FAIL'}")
    print(f"  Key check (token 2 correct routing): {'PASS' if key_pass else 'FAIL'}")
    overall = all_pass and key_pass
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    print()

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
