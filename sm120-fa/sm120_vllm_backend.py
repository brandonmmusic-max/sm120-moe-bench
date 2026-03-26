"""
SM120 Flash Attention Backend for vLLM.

Subclasses FlashAttentionBackend to intercept decode-only batches (max_query_len=1)
and dispatch to the SM120-native split-KV decode kernel. Prefill and mixed batches
fall through to the standard FlashAttention path.

Supports both BF16 and FP8 E4M3 KV cache dtypes.

Usage:
    # In vLLM startup or plugin initialization:
    from vllm.v1.attention.backends.registry import register_backend, AttentionBackendEnum
    register_backend(AttentionBackendEnum.FLASH_ATTN,
                     "sm120_vllm_backend.SM120FlashAttentionBackend")

    # Or set: --attention-backend CUSTOM  and register CUSTOM instead.
"""

import torch
from typing import ClassVar

from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backend import AttentionType
from vllm.logger import init_logger

logger = init_logger(__name__)

# Lazy import of our kernel
_sm120_decode_module = None
_workspace = None


def _get_sm120_decode():
    """Lazy-load the SM120 flash decode kernel."""
    global _sm120_decode_module
    if _sm120_decode_module is None:
        import importlib.util, os
        ext_path = os.path.join(os.path.dirname(__file__), "sm120_flash_decode_ext.py")
        spec = importlib.util.spec_from_file_location("sm120_flash_decode_ext", ext_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _sm120_decode_module = mod
        logger.info("SM120 Flash Decode kernel loaded successfully")
    return _sm120_decode_module


def _get_workspace(batch_size, num_q_heads, head_dim, device):
    """Get or create workspace for SM120 decode kernel."""
    global _workspace
    ext = _get_sm120_decode()
    if (_workspace is None or
        _workspace.max_batch_size < batch_size or
        _workspace.num_q_heads < num_q_heads or
        _workspace.head_dim < head_dim):
        # Allocate with generous headroom
        max_bs = max(batch_size * 2, 256)
        _workspace = ext.SM120FlashDecodeWorkspace(
            max_batch_size=max_bs,
            num_q_heads=num_q_heads,
            head_dim=head_dim,
            max_splits=32,
            device=str(device),
        )
        logger.info(
            "SM120 decode workspace allocated: batch=%d, heads=%d, dim=%d",
            max_bs, num_q_heads, head_dim,
        )
    return _workspace


class SM120FlashAttentionBackend(FlashAttentionBackend):
    """SM120-optimized attention backend.

    Intercepts pure-decode batches to use our SM120-native split-KV kernel.
    Everything else (prefill, mixed, cascade) falls through to FlashAttention.
    """

    @staticmethod
    def get_name() -> str:
        return "SM120_FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SM120FlashAttentionImpl"]:
        return SM120FlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:
        # Reuse FA's metadata builder — we use the same metadata format
        return FlashAttentionMetadataBuilder


class SM120FlashAttentionImpl(FlashAttentionImpl):
    """Attention impl that dispatches decode to SM120 kernel."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sm120_decode_available = False
        self._sm120_init_attempted = False

    def _try_init_sm120(self, device):
        """Attempt to initialize SM120 decode kernel (once)."""
        if self._sm120_init_attempted:
            return
        self._sm120_init_attempted = True
        try:
            _get_sm120_decode()
            self._sm120_decode_available = True
            kv_dtype_str = self.kv_cache_dtype
            logger.info("SM120 decode kernel: enabled (head_dim=%d, GQA=%d:%d, kv_dtype=%s)",
                        self.head_size, self.num_heads, self.num_kv_heads, kv_dtype_str)
        except Exception as e:
            logger.warning("SM120 decode kernel: disabled (%s), falling back to FA", e)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            return output.fill_(0)

        # One-time init
        if not self._sm120_init_attempted:
            self._try_init_sm120(query.device)

        # Dispatch decode to SM120 kernel when conditions are met:
        # 1. Pure decode batch (max_query_len == 1)
        # 2. SM120 kernel available
        # 3. Decoder attention (not encoder)
        # 4. No cascade (cascade uses different metadata)
        # 5. No DCP (decode context parallelism)
        # 6. BF16 or FP8 E4M3 KV cache (both supported)
        can_use_sm120 = (
            self._sm120_decode_available
            and attn_metadata.max_query_len == 1
            and self.attn_type == AttentionType.DECODER
            and not attn_metadata.use_cascade
            and self.alibi_slopes is None
            and self.sliding_window == (-1, -1)
        )

        if can_use_sm120:
            return self._forward_sm120_decode(
                layer, query, kv_cache, attn_metadata, output
            )

        # Fall through to standard FlashAttention for everything else
        return super().forward(
            layer, query, key, value, kv_cache, attn_metadata,
            output, output_scale, output_block_scale,
        )

    def _forward_sm120_decode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,        # [num_tokens, num_q_heads, head_dim]
        kv_cache: torch.Tensor,      # [2, num_blocks, block_size, num_kv_heads, head_dim]
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,        # [num_tokens, num_q_heads, head_dim]
    ) -> torch.Tensor:
        ext = _get_sm120_decode()

        num_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.unbind(0)

        # query shape: [num_tokens, num_q_heads, head_dim]
        # For decode, num_tokens == batch_size (one token per request)
        q = query[:num_tokens]  # [batch, num_q_heads, head_dim]

        workspace = _get_workspace(
            num_tokens, self.num_heads, self.head_size, query.device
        )

        # Get KV dequant scales from layer (default 1.0 for BF16)
        k_scale = getattr(layer, '_k_scale_float', 1.0)
        v_scale = getattr(layer, '_v_scale_float', 1.0)

        # Call SM120 decode kernel
        max_sl = getattr(attn_metadata, 'max_seq_len', None)
        result = ext.sm120_flash_decode_paged(
            query=q,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=attn_metadata.block_table,
            seq_lens=attn_metadata.seq_lens,
            output=output[:num_tokens],
            workspace=workspace,
            max_seq_len=max_sl,
            k_scale=k_scale,
            v_scale=v_scale,
        )

        return output
