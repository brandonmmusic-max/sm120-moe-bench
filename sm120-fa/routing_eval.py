"""
Phase 1+3: Routing evaluation metrics + speed/accuracy frontier sweep.

Primary metric: softmax mass recall (not raw score recall).
Secondary: output contribution recall, cosine similarity, error.
"""

import torch
import torch.nn.functional as F
import time
from dataclasses import dataclass
from typing import Optional

try:
    import sm120_flash_attn
    HAS_SM120 = True
except ImportError:
    HAS_SM120 = False


@dataclass
class RoutingMetrics:
    """Comprehensive routing quality metrics."""
    raw_score_recall: float = 0.0
    softmax_mass_recall: float = 0.0
    output_contrib_recall: float = 0.0
    cosine_similarity: float = 0.0
    mean_abs_error: float = 0.0
    max_abs_error: float = 0.0
    selected_blocks: int = 0
    total_blocks: int = 0
    coverage: float = 0.0
    latency_ms: float = 0.0
    speedup_vs_full: float = 0.0
    speedup_vs_sdpa: float = 0.0


def compute_block_metrics(
    Q: torch.Tensor,   # [B, Hq, Sq, D]
    K: torch.Tensor,   # [B, Hkv, Skv, D]
    V: torch.Tensor,   # [B, Hkv, Skv, D]
    block_size: int = 64,
    top_k: int = 4,
) -> dict:
    """Compute true block-level importance by different metrics."""
    B, Hq, Sq, D = Q.shape
    _, Hkv, Skv, _ = K.shape
    kv_r = Hq // Hkv
    scale = 1.0 / (D ** 0.5)
    num_blocks = (Skv + block_size - 1) // block_size

    Ke = K.repeat_interleave(kv_r, dim=1)
    Ve = V.repeat_interleave(kv_r, dim=1)

    # Full attention scores and weights
    scores = (Q.float() @ Ke.float().transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)  # [B, Hq, Sq, Skv]

    # Per-block metrics
    block_raw = torch.zeros(B, Hq, Sq, num_blocks, device=Q.device)
    block_mass = torch.zeros(B, Hq, Sq, num_blocks, device=Q.device)
    block_contrib = torch.zeros(B, Hq, Sq, num_blocks, device=Q.device)

    for blk in range(num_blocks):
        s = blk * block_size
        e = min(s + block_size, Skv)
        block_raw[:,:,:,blk] = scores[:,:,:,s:e].sum(dim=-1)
        block_mass[:,:,:,blk] = attn_weights[:,:,:,s:e].sum(dim=-1)
        contrib = (attn_weights[:,:,:,s:e].unsqueeze(-1) * Ve[:,:,s:e,:].unsqueeze(2)).sum(dim=3)
        block_contrib[:,:,:,blk] = contrib.norm(dim=-1)

    return {
        "block_raw": block_raw,
        "block_mass": block_mass,
        "block_contrib": block_contrib,
        "num_blocks": num_blocks,
    }


def eval_routing(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    block_size: int = 64, top_k: int = 4, local_window: int = 4,
    routing_mode: str = "mean",  # "mean", "subblock_max", "hybrid"
    sub_block_size: int = 16,
    alpha: float = 0.5,  # for hybrid mode
) -> RoutingMetrics:
    """
    Evaluate routing quality using multiple metrics.

    Returns RoutingMetrics with all quality + speed measurements.
    """
    B, Hq, Sq, D = Q.shape
    _, Hkv, Skv, _ = K.shape
    kv_r = Hq // Hkv
    scale = 1.0 / (D ** 0.5)
    num_blocks = (Skv + block_size - 1) // block_size
    max_sel = top_k + local_window + 2

    # True block metrics
    true_metrics = compute_block_metrics(Q, K, V, block_size, top_k)
    _, true_raw_top = true_metrics["block_raw"].topk(top_k, dim=-1)
    _, true_mass_top = true_metrics["block_mass"].topk(top_k, dim=-1)
    _, true_contrib_top = true_metrics["block_contrib"].topk(top_k, dim=-1)

    # Build routing scores based on mode
    routing_scores = build_routing_scores(Q, K, block_size, routing_mode,
                                           sub_block_size, alpha, scale, kv_r)

    # Include local window in selection
    local_start = max(0, num_blocks - local_window)
    # Mask local blocks to -inf for top_k selection
    masked = routing_scores.clone()
    masked[:, :, :, local_start:] = -float('inf')

    _, global_top = masked.topk(min(top_k, num_blocks - local_window), dim=-1)

    # Combine local + global
    local_indices = torch.arange(local_start, num_blocks, device=Q.device)
    # For recall comparison, use just the global top_k (excluding local)
    pred_top = global_top

    # Recall computation
    def recall(pred, true):
        k = min(pred.shape[-1], true.shape[-1])
        overlap = 0
        total = 0
        for b in range(B):
            for h in range(Hq):
                for q in range(Sq):
                    p = set(pred[b, h, q, :k].tolist())
                    t = set(true[b, h, q, :k].tolist())
                    overlap += len(p & t)
                    total += k
        return overlap / max(total, 1)

    r_raw = recall(pred_top, true_raw_top)
    r_mass = recall(pred_top, true_mass_top)
    r_contrib = recall(pred_top, true_contrib_top)

    # Run selective attention and measure error
    Ke = K.repeat_interleave(kv_r, dim=1)
    Ve = V.repeat_interleave(kv_r, dim=1)
    ref = F.scaled_dot_product_attention(Q, Ke, Ve)

    if HAS_SM120:
        [sel_out] = sm120_flash_attn.forward_selective(Q, K, V, block_size, top_k,
                                                         local_window, max_sel)
    else:
        sel_out = ref  # Fallback

    abs_err = (sel_out.float() - ref.float()).abs()
    cos = F.cosine_similarity(sel_out.float().flatten().unsqueeze(0),
                               ref.float().flatten().unsqueeze(0)).item()

    # Timing
    def bench(fn, iters=10):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
        ts.sort()
        return ts[len(ts)//2]

    t_sel = bench(lambda: sm120_flash_attn.forward_selective(Q, K, V, block_size,
                                                              top_k, local_window, max_sel)) if HAS_SM120 else 0
    t_full = bench(lambda: sm120_flash_attn.forward(Q, K, V, False)) if HAS_SM120 else 0
    t_sdpa = bench(lambda: F.scaled_dot_product_attention(Q, Ke, Ve))

    return RoutingMetrics(
        raw_score_recall=r_raw,
        softmax_mass_recall=r_mass,
        output_contrib_recall=r_contrib,
        cosine_similarity=cos,
        mean_abs_error=abs_err.mean().item(),
        max_abs_error=abs_err.max().item(),
        selected_blocks=max_sel,
        total_blocks=num_blocks,
        coverage=max_sel / num_blocks,
        latency_ms=t_sel,
        speedup_vs_full=t_full / t_sel if t_sel > 0 else 0,
        speedup_vs_sdpa=t_sdpa / t_sel if t_sel > 0 else 0,
    )


def build_routing_scores(
    Q: torch.Tensor, K: torch.Tensor,
    block_size: int, mode: str, sub_block_size: int,
    alpha: float, scale: float, kv_repeat: int,
) -> torch.Tensor:
    """
    Build block-level routing scores using different summary strategies.

    Returns: [B, Hq, Sq, num_blocks]
    """
    B, Hkv, Skv, D = K.shape
    Hq = Q.shape[1]
    num_blocks = (Skv + block_size - 1) // block_size

    # Pad K
    pad = num_blocks * block_size - Skv
    if pad > 0:
        K_padded = F.pad(K, (0, 0, 0, pad))
    else:
        K_padded = K

    K_blocks = K_padded.float().reshape(B, Hkv, num_blocks, block_size, D)

    if mode == "mean":
        summaries = K_blocks.mean(dim=3)  # [B, Hkv, nb, D]
        sum_exp = summaries.repeat_interleave(kv_repeat, dim=1)
        scores = (Q.float() @ sum_exp.transpose(-2, -1)) * scale
        return scores

    elif mode == "subblock_max":
        # Split each block into sub-blocks, take mean of each, score, take max
        n_sub = block_size // sub_block_size
        sub_blocks = K_blocks.reshape(B, Hkv, num_blocks, n_sub, sub_block_size, D)
        sub_summaries = sub_blocks.mean(dim=4)  # [B, Hkv, nb, n_sub, D]

        # Score Q against each sub-summary
        sub_exp = sub_summaries.repeat_interleave(kv_repeat, dim=1)
        # [B, Hq, nb, n_sub, D]
        Q_exp = Q.float().unsqueeze(3)  # [B, Hq, Sq, 1, D]
        sub_scores = (Q_exp @ sub_exp.transpose(-2, -1).unsqueeze(2)).squeeze(-2) * scale
        # sub_scores: [B, Hq, Sq, nb, n_sub] — no, let me fix dimensions

        # Simpler: flatten sub-summaries and score
        sub_flat = sub_summaries.reshape(B, Hkv, num_blocks * n_sub, D)
        sub_flat_exp = sub_flat.repeat_interleave(kv_repeat, dim=1)
        flat_scores = (Q.float() @ sub_flat_exp.transpose(-2, -1)) * scale
        # [B, Hq, Sq, nb*n_sub]
        sub_scores = flat_scores.reshape(B, Hq, Q.shape[2], num_blocks, n_sub)
        # Take max across sub-blocks
        max_sub_scores = sub_scores.max(dim=-1).values  # [B, Hq, Sq, nb]
        return max_sub_scores

    elif mode == "hybrid":
        # Combination of mean block score and max sub-block score
        mean_scores = build_routing_scores(Q, K, block_size, "mean",
                                            sub_block_size, alpha, scale, kv_repeat)
        sub_max_scores = build_routing_scores(Q, K, block_size, "subblock_max",
                                               sub_block_size, alpha, scale, kv_repeat)
        return alpha * mean_scores + (1 - alpha) * sub_max_scores

    else:
        raise ValueError(f"Unknown routing mode: {mode}")


def run_frontier_sweep(
    Skv: int = 32768,
    Sq: int = 1,
    Hq: int = 32,
    Hkv: int = 8,
    D: int = 128,
    seed: int = 42,
):
    """Run full speed/accuracy frontier sweep and print results."""
    torch.manual_seed(seed)
    Q = torch.randn(1, Hq, Sq, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(1, Hkv, Skv, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(1, Hkv, Skv, D, device="cuda", dtype=torch.bfloat16)

    num_blocks = Skv // 64
    print(f"Frontier sweep: KV={Skv} ({num_blocks} blocks), Sq={Sq}")
    print(f"{'mode':>12s} {'top_k':>5s} {'local':>5s} {'sel':>4s}/{num_blocks:>4d} "
          f"{'ms':>7s} {'vs_full':>7s} {'vs_sdpa':>7s} | "
          f"{'raw_r':>6s} {'mass_r':>6s} {'cont_r':>6s} | "
          f"{'mae':>9s} {'cos':>8s}")
    print("-" * 110)

    for mode in ["mean", "subblock_max", "hybrid"]:
        for top_k in [2, 4, 8, 16]:
            for local_w in [4]:
                m = eval_routing(Q, K, V, block_size=64, top_k=top_k,
                                  local_window=local_w, routing_mode=mode)
                print(f"{mode:>12s} {top_k:>5d} {local_w:>5d} {m.selected_blocks:>4d}/{m.total_blocks:>4d} "
                      f"{m.latency_ms:>6.2f} {m.speedup_vs_full:>6.2f}x {m.speedup_vs_sdpa:>6.2f}x | "
                      f"{m.raw_score_recall:>5.1%} {m.softmax_mass_recall:>5.1%} {m.output_contrib_recall:>5.1%} | "
                      f"{m.mean_abs_error:>9.6f} {m.cosine_similarity:>8.4f}")
        print()


if __name__ == "__main__":
    for skv in [8192, 32768, 65536]:
        run_frontier_sweep(Skv=skv)
        print()
