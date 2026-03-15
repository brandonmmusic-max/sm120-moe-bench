"""
SM120 Selective Attention — Block-summary routing for long-context speedup.

Experimental approximate attention mode that:
1. Splits KV into fixed-size blocks
2. Builds cheap mean(K_block) summaries
3. Scores Q against summaries for coarse routing
4. Selects local_window + top_k blocks for exact attention
5. Runs exact SM120 MMA attention only on selected blocks
6. Falls back to full exact attention when confidence is low

Does NOT modify or replace the exact kernel.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

try:
    import sm120_flash_attn
    HAS_SM120_FA = True
except ImportError:
    HAS_SM120_FA = False


@dataclass
class SelectiveAttnConfig:
    """Configuration for selective attention."""
    block_size: int = 64           # KV block size for summaries
    top_k_blocks: int = 8         # Number of top-scoring blocks to include
    local_window_blocks: int = 2  # Always include N most recent blocks
    fallback_threshold: float = 0.5  # Score entropy threshold for fallback
    max_selected_blocks: int = 32    # Cap on total selected blocks
    enable_full_fallback: bool = True  # Fall back to exact when uncertain
    summary_type: str = "mean"       # "mean", "max_norm", "mean_norm"


class KVBlockSummary:
    """Manages block summaries for a KV cache."""

    def __init__(self, K: torch.Tensor, V: torch.Tensor, config: SelectiveAttnConfig):
        """
        Build block summaries from K and V tensors.

        Args:
            K: [B, Hkv, Skv, D] bfloat16
            V: [B, Hkv, Skv, D] bfloat16
            config: SelectiveAttnConfig
        """
        self.config = config
        self.B, self.Hkv, self.Skv, self.D = K.shape
        self.block_size = config.block_size
        self.num_blocks = (self.Skv + self.block_size - 1) // self.block_size

        # Build summaries
        self.k_summaries = self._build_summaries(K)  # [B, Hkv, num_blocks, D]
        self.k_norms = self._build_norms(K)           # [B, Hkv, num_blocks]

        # Keep references to full K, V for exact attention
        self.K = K
        self.V = V

    def _build_summaries(self, K: torch.Tensor) -> torch.Tensor:
        """Build per-block summaries. Currently: mean(K_block)."""
        B, H, S, D = K.shape
        # Pad to multiple of block_size
        pad = (self.block_size - S % self.block_size) % self.block_size
        if pad > 0:
            K_padded = F.pad(K, (0, 0, 0, pad))
        else:
            K_padded = K

        # Reshape to [B, H, num_blocks, block_size, D] and take mean
        K_blocks = K_padded.reshape(B, H, self.num_blocks, self.block_size, D)

        if self.config.summary_type == "mean":
            summaries = K_blocks.float().mean(dim=3).to(K.dtype)  # [B, H, num_blocks, D]
        elif self.config.summary_type == "max_norm":
            summaries = K_blocks.float().mean(dim=3).to(K.dtype)
        else:
            summaries = K_blocks.float().mean(dim=3).to(K.dtype)

        return summaries

    def _build_norms(self, K: torch.Tensor) -> torch.Tensor:
        """Build per-block L2 norms for confidence estimation."""
        B, H, S, D = K.shape
        pad = (self.block_size - S % self.block_size) % self.block_size
        if pad > 0:
            K_padded = F.pad(K, (0, 0, 0, pad))
        else:
            K_padded = K

        K_blocks = K_padded.reshape(B, H, self.num_blocks, self.block_size, D)
        norms = K_blocks.float().norm(dim=-1).mean(dim=-1)  # [B, H, num_blocks]
        return norms


class SelectiveAttention:
    """
    Selective attention: route Q to a subset of KV blocks for exact attention.

    Usage:
        config = SelectiveAttnConfig(block_size=64, top_k_blocks=8)
        sa = SelectiveAttention(config)
        out, debug_info = sa.forward(Q, K, V)
    """

    def __init__(self, config: SelectiveAttnConfig):
        self.config = config

    def forward(
        self,
        Q: torch.Tensor,  # [B, Hq, Sq, D]
        K: torch.Tensor,  # [B, Hkv, Skv, D]
        V: torch.Tensor,  # [B, Hkv, Skv, D]
        return_debug: bool = False,
    ) -> tuple:
        """
        Forward pass with selective attention.

        Returns:
            output: [B, Hq, Sq, D]
            debug_info: dict (if return_debug=True)
        """
        B, Hq, Sq, D = Q.shape
        _, Hkv, Skv, _ = K.shape
        cfg = self.config
        kv_repeat = Hq // Hkv

        # Build block summaries
        summaries = KVBlockSummary(K, V, cfg)
        num_blocks = summaries.num_blocks

        # If sequence is short enough, just use full exact attention
        if num_blocks <= cfg.top_k_blocks + cfg.local_window_blocks:
            if HAS_SM120_FA:
                [out] = sm120_flash_attn.forward(Q, K, V, False)
            else:
                Ke = K.repeat_interleave(kv_repeat, dim=1)
                Ve = V.repeat_interleave(kv_repeat, dim=1)
                out = F.scaled_dot_product_attention(Q, Ke, Ve)
            debug_info = {"mode": "full_exact", "reason": "short_sequence",
                          "num_blocks": num_blocks}
            return (out, debug_info) if return_debug else (out, None)

        # ================================================================
        # Coarse scoring: Q against block summaries
        # ================================================================
        # Expand summaries for GQA
        k_sum = summaries.k_summaries.repeat_interleave(kv_repeat, dim=1)
        # k_sum: [B, Hq, num_blocks, D]

        scale = 1.0 / (D ** 0.5)

        # Coarse scores: Q @ summary^T → [B, Hq, Sq, num_blocks]
        coarse_scores = torch.matmul(
            Q.float(), k_sum.float().transpose(-2, -1)
        ) * scale

        # Average across query positions for per-block importance
        # [B, Hq, num_blocks]
        block_importance = coarse_scores.mean(dim=2)

        # ================================================================
        # Block selection
        # ================================================================
        # Always include local window (most recent blocks)
        local_start = max(0, num_blocks - cfg.local_window_blocks)
        local_blocks = set(range(local_start, num_blocks))

        # Top-k by importance (excluding local blocks already selected)
        # Mask local blocks to -inf so they don't count against top_k
        masked_importance = block_importance.clone()
        for lb in local_blocks:
            masked_importance[:, :, lb] = -float('inf')

        remaining_k = min(cfg.top_k_blocks, num_blocks - len(local_blocks))
        if remaining_k > 0:
            _, top_indices = masked_importance.topk(remaining_k, dim=-1)
            # top_indices: [B, Hq, remaining_k]
        else:
            top_indices = torch.empty(B, Hq, 0, dtype=torch.long, device=Q.device)

        # ================================================================
        # Confidence check: should we fall back to full attention?
        # ================================================================
        # Compute score entropy — high entropy = uncertain routing
        score_probs = F.softmax(block_importance, dim=-1)
        entropy = -(score_probs * torch.log(score_probs + 1e-10)).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(float(num_blocks)))
        normalized_entropy = (entropy / max_entropy).mean().item()

        use_fallback = (cfg.enable_full_fallback and
                        normalized_entropy > cfg.fallback_threshold)

        if use_fallback:
            if HAS_SM120_FA:
                [out] = sm120_flash_attn.forward(Q, K, V, False)
            else:
                Ke = K.repeat_interleave(kv_repeat, dim=1)
                Ve = V.repeat_interleave(kv_repeat, dim=1)
                out = F.scaled_dot_product_attention(Q, Ke, Ve)
            debug_info = {
                "mode": "full_fallback",
                "reason": f"high_entropy={normalized_entropy:.3f}",
                "num_blocks": num_blocks,
                "entropy": normalized_entropy,
            }
            return (out, debug_info) if return_debug else (out, None)

        # ================================================================
        # Gather selected KV blocks
        # ================================================================
        # Collect unique block indices across batch and heads
        # For simplicity, use the same selection for all batch/head
        # (TODO: per-head selection for better accuracy)
        all_selected = set(local_blocks)
        if remaining_k > 0:
            # Use first batch/head's top-k as representative
            for idx in top_indices[0, 0].tolist():
                all_selected.add(idx)

        selected_list = sorted(all_selected)
        total_selected = len(selected_list)

        # Cap at max_selected_blocks
        if total_selected > cfg.max_selected_blocks:
            # Keep local blocks, trim global blocks
            global_blocks = [b for b in selected_list if b not in local_blocks]
            global_blocks = global_blocks[:cfg.max_selected_blocks - len(local_blocks)]
            selected_list = sorted(list(local_blocks) + global_blocks)
            total_selected = len(selected_list)

        # Gather selected K, V blocks into contiguous tensors
        selected_kv_len = total_selected * cfg.block_size
        K_selected = torch.zeros(B, Hkv, selected_kv_len, D,
                                  device=K.device, dtype=K.dtype)
        V_selected = torch.zeros(B, Hkv, selected_kv_len, D,
                                  device=V.device, dtype=V.dtype)

        for i, blk_idx in enumerate(selected_list):
            src_start = blk_idx * cfg.block_size
            src_end = min(src_start + cfg.block_size, Skv)
            dst_start = i * cfg.block_size
            actual_len = src_end - src_start
            K_selected[:, :, dst_start:dst_start+actual_len] = K[:, :, src_start:src_end]
            V_selected[:, :, dst_start:dst_start+actual_len] = V[:, :, src_start:src_end]

        # ================================================================
        # Run exact attention on selected blocks
        # ================================================================
        if HAS_SM120_FA:
            [out] = sm120_flash_attn.forward(Q, K_selected, V_selected, False)
        else:
            Ke = K_selected.repeat_interleave(kv_repeat, dim=1)
            Ve = V_selected.repeat_interleave(kv_repeat, dim=1)
            out = F.scaled_dot_product_attention(Q, Ke, Ve)

        debug_info = {
            "mode": "selective",
            "num_blocks": num_blocks,
            "selected_blocks": total_selected,
            "selected_block_ids": selected_list,
            "local_blocks": list(local_blocks),
            "coverage": total_selected / num_blocks,
            "entropy": normalized_entropy,
            "selected_kv_len": selected_kv_len,
            "original_kv_len": Skv,
            "speedup_ratio": Skv / selected_kv_len if selected_kv_len > 0 else 0,
        }

        return (out, debug_info) if return_debug else (out, None)


def forward_selective(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = 64,
    top_k_blocks: int = 8,
    local_window_blocks: int = 2,
    fallback_threshold: float = 0.5,
    return_debug: bool = False,
):
    """Convenience function for selective attention."""
    config = SelectiveAttnConfig(
        block_size=block_size,
        top_k_blocks=top_k_blocks,
        local_window_blocks=local_window_blocks,
        fallback_threshold=fallback_threshold,
    )
    sa = SelectiveAttention(config)
    return sa.forward(Q, K, V, return_debug=return_debug)
