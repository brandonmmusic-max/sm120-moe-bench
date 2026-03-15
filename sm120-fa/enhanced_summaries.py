"""
Enhanced KV block summaries for better selective attention routing.

Three summary types:
1. Mean + variance (richer statistics)
2. Exponential moving average (recency-weighted)
3. Learned projection (trainable compression)

All produce summaries that can be scored against Q for block selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MeanVarSummary(nn.Module):
    """
    Summary = [mean(K_block), std(K_block)] concatenated.
    Scoring uses a learned 2D→1D projection to combine mean and variance signals.
    """
    def __init__(self, head_dim: int = 128):
        super().__init__()
        self.head_dim = head_dim
        # Learnable scoring: project [mean, var] to a single D-dim summary
        self.score_proj = nn.Linear(head_dim * 2, head_dim, bias=False)
        # Initialize to just use mean (identity on first half, zero on second)
        with torch.no_grad():
            self.score_proj.weight.zero_()
            self.score_proj.weight[:, :head_dim] = torch.eye(head_dim)

    def build_summaries(self, K: torch.Tensor, block_size: int) -> torch.Tensor:
        """
        Args:
            K: [B, Hkv, Skv, D]
            block_size: int
        Returns:
            summaries: [B, Hkv, num_blocks, D] — scoring-ready
        """
        B, H, S, D = K.shape
        num_blocks = (S + block_size - 1) // block_size

        # Pad
        pad = num_blocks * block_size - S
        if pad > 0:
            K = F.pad(K, (0, 0, 0, pad))

        blocks = K.reshape(B, H, num_blocks, block_size, D).float()
        mean = blocks.mean(dim=3)   # [B, H, nb, D]
        var = blocks.var(dim=3)     # [B, H, nb, D]

        # Concatenate and project
        combined = torch.cat([mean, var], dim=-1)  # [B, H, nb, 2D]
        summaries = self.score_proj(combined)       # [B, H, nb, D]

        return summaries.to(K.dtype)


class EMAWeightedSummary(nn.Module):
    """
    Exponential moving average within each block.
    Weights later tokens more heavily, capturing recency within blocks.
    """
    def __init__(self, head_dim: int = 128, decay: float = 0.9):
        super().__init__()
        self.head_dim = head_dim
        self.decay = decay

    def build_summaries(self, K: torch.Tensor, block_size: int) -> torch.Tensor:
        """EMA-weighted mean per block."""
        B, H, S, D = K.shape
        num_blocks = (S + block_size - 1) // block_size

        pad = num_blocks * block_size - S
        if pad > 0:
            K = F.pad(K, (0, 0, 0, pad))

        blocks = K.reshape(B, H, num_blocks, block_size, D).float()

        # Build EMA weights: [1, decay, decay^2, ..., decay^(bs-1)]
        # Reversed so LATER tokens have HIGHER weight
        weights = torch.pow(
            torch.tensor(self.decay, device=K.device),
            torch.arange(block_size - 1, -1, -1, device=K.device, dtype=torch.float32)
        )
        weights = weights / weights.sum()  # Normalize

        # Apply weights: [B, H, nb, bs, D] × [bs] → [B, H, nb, D]
        summaries = (blocks * weights.unsqueeze(-1)).sum(dim=3)

        return summaries.to(K.dtype)


class LearnedProjectionSummary(nn.Module):
    """
    Trainable linear projection that compresses a block of K into a summary.

    The projection can be trained end-to-end or pre-trained on a dataset
    to maximize routing accuracy.
    """
    def __init__(self, head_dim: int = 128, block_size: int = 64):
        super().__init__()
        self.head_dim = head_dim
        self.block_size = block_size

        # Compress block_size × D → D via learned projection
        # Input: [bs, D] flattened to [bs * D], projected to [D]
        # Too large! Instead, use a pooling + projection approach:
        #   1. Linear per-token projection: D → D (shared across positions)
        #   2. Attention-weighted pooling across block positions
        self.token_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.pool_weights = nn.Linear(head_dim, 1, bias=False)

        # Initialize to approximate mean
        with torch.no_grad():
            self.token_proj.weight.copy_(torch.eye(head_dim))
            self.pool_weights.weight.fill_(1.0 / block_size)

    def build_summaries(self, K: torch.Tensor, block_size: int) -> torch.Tensor:
        """Learned summary per block."""
        B, H, S, D = K.shape
        num_blocks = (S + block_size - 1) // block_size

        pad = num_blocks * block_size - S
        if pad > 0:
            K = F.pad(K, (0, 0, 0, pad))

        blocks = K.reshape(B * H, num_blocks, block_size, D).float()

        # Per-token projection
        projected = self.token_proj(blocks)  # [B*H, nb, bs, D]

        # Attention-weighted pooling
        attn_logits = self.pool_weights(blocks).squeeze(-1)  # [B*H, nb, bs]
        attn_weights = F.softmax(attn_logits, dim=-1)        # [B*H, nb, bs]

        summaries = (projected * attn_weights.unsqueeze(-1)).sum(dim=2)  # [B*H, nb, D]
        summaries = summaries.reshape(B, H, num_blocks, D)

        return summaries.to(K.dtype)


class AdaptiveTopKRouter(nn.Module):
    """
    Adaptive top-k selection based on score distribution.

    Instead of fixed top_k, dynamically choose how many blocks to include
    based on the entropy/concentration of coarse scores.

    Low entropy (concentrated scores) → few blocks needed
    High entropy (uniform scores) → need more blocks or fallback
    """
    def __init__(self, min_k: int = 2, max_k: int = 32,
                 entropy_threshold: float = 0.7):
        super().__init__()
        self.min_k = min_k
        self.max_k = max_k
        self.entropy_threshold = entropy_threshold

    def select_k(self, scores: torch.Tensor) -> int:
        """
        Adaptively choose k based on score distribution.

        Args:
            scores: [num_blocks] coarse scores for one Q row
        Returns:
            k: number of blocks to select
        """
        probs = F.softmax(scores, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        max_entropy = torch.log(torch.tensor(float(len(scores))))
        norm_entropy = (entropy / max_entropy).item()

        if norm_entropy < 0.3:
            return self.min_k  # Very concentrated → few blocks
        elif norm_entropy < 0.6:
            return min(self.min_k * 2, self.max_k)
        elif norm_entropy < self.entropy_threshold:
            return min(self.min_k * 4, self.max_k)
        else:
            return self.max_k  # Very uniform → need more blocks


def build_enhanced_summaries(
    K: torch.Tensor,
    block_size: int = 64,
    summary_type: str = "mean_var",
    decay: float = 0.9,
) -> torch.Tensor:
    """
    Build enhanced block summaries (convenience function).

    Args:
        K: [B, Hkv, Skv, D]
        block_size: block size
        summary_type: "mean", "mean_var", "ema", "learned"

    Returns:
        summaries: [B, Hkv, num_blocks, D]
    """
    D = K.shape[-1]

    if summary_type == "mean":
        # Simple mean (same as CUDA kernel)
        B, H, S, _D = K.shape
        num_blocks = (S + block_size - 1) // block_size
        pad = num_blocks * block_size - S
        if pad > 0:
            K_padded = F.pad(K, (0, 0, 0, pad))
        else:
            K_padded = K
        blocks = K_padded.reshape(B, H, num_blocks, block_size, _D)
        return blocks.float().mean(dim=3).to(K.dtype)

    elif summary_type == "mean_var":
        model = MeanVarSummary(D).to(K.device)
        with torch.no_grad():
            return model.build_summaries(K, block_size)

    elif summary_type == "ema":
        model = EMAWeightedSummary(D, decay).to(K.device)
        with torch.no_grad():
            return model.build_summaries(K, block_size)

    elif summary_type == "learned":
        model = LearnedProjectionSummary(D, block_size).to(K.device)
        with torch.no_grad():
            return model.build_summaries(K, block_size)

    else:
        raise ValueError(f"Unknown summary type: {summary_type}")


def benchmark_summary_quality(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    block_size: int = 64, top_k: int = 4, local_window: int = 4,
):
    """
    Compare different summary types for routing quality.

    Measures: how well does the coarse routing match the true attention pattern?
    """
    B, Hq, Sq, D = Q.shape
    _, Hkv, Skv, _ = K.shape
    kv_r = Hq // Hkv
    scale = 1.0 / (D ** 0.5)

    # True attention scores
    Ke = K.repeat_interleave(kv_r, dim=1)
    true_scores = (Q.float() @ Ke.float().transpose(-2, -1)) * scale  # [B, Hq, Sq, Skv]

    # True block importance (sum of attention scores per block)
    num_blocks = (Skv + block_size - 1) // block_size
    true_block_scores = torch.zeros(B, Hq, Sq, num_blocks, device=Q.device)
    for blk in range(num_blocks):
        start = blk * block_size
        end = min(start + block_size, Skv)
        true_block_scores[:, :, :, blk] = true_scores[:, :, :, start:end].sum(dim=-1)

    # True top-k blocks
    _, true_top_k = true_block_scores.topk(top_k, dim=-1)

    results = {}
    for stype in ["mean", "mean_var", "ema"]:
        summaries = build_enhanced_summaries(K, block_size, stype)
        # Expand for GQA
        sum_exp = summaries.repeat_interleave(kv_r, dim=1)

        # Coarse scores
        coarse = (Q.float() @ sum_exp.float().transpose(-2, -1)) * scale
        _, coarse_top_k = coarse.topk(top_k, dim=-1)

        # Measure overlap with true top-k (recall)
        overlap = 0
        total = 0
        for b in range(B):
            for h in range(Hq):
                for q in range(Sq):
                    true_set = set(true_top_k[b, h, q].tolist())
                    pred_set = set(coarse_top_k[b, h, q].tolist())
                    overlap += len(true_set & pred_set)
                    total += top_k
        recall = overlap / total if total > 0 else 0

        results[stype] = recall

    return results
