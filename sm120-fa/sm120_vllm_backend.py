"""
SM120 Flash Decode Backend for vLLM.

Subclasses FlashInferBackend to intercept decode-only batches (max_query_len=1)
and dispatch to the SM120-native split-KV decode kernel. Prefill and mixed batches
fall through to FlashInfer (which handles FP8 KV natively on SM120).

Supports both BF16 and FP8 E4M3 KV cache dtypes.

Registered as FLASH_ATTN in the backend registry to override the default.
"""

import torch

from vllm.v1.attention.backends.flashinfer import (
    FlashInferBackend,
    FlashInferImpl,
    FlashInferMetadata,
    FlashInferMetadataBuilder,
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


class SM120FlashAttentionBackend(FlashInferBackend):
    """SM120-optimized attention backend.

    Extends FlashInfer to intercept pure-decode batches and use SM120 kernel.
    Everything else (prefill, mixed, cascade) falls through to FlashInfer.
    """

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER"

    @staticmethod
    def get_impl_cls() -> type["SM120FlashInferImpl"]:
        return SM120FlashInferImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferMetadataBuilder"]:
        return FlashInferMetadataBuilder


class SM120FlashInferImpl(FlashInferImpl):
    """FlashInfer impl that dispatches decode to SM120 kernel."""

    def __init__(self, *args, **kwargs):
        # Capture attn_type before super().__init__ which doesn't store it
        attn_type = kwargs.get('attn_type', AttentionType.DECODER)
        if len(args) > 8:
            attn_type = args[8]
        super().__init__(*args, **kwargs)
        self.attn_type = attn_type
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
            logger.warning("SM120 decode kernel: disabled (%s), falling back to FlashInfer", e)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
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
        # Pure decode, no cascade, has TRTLLMDecode with raw block_tables
        can_use_sm120 = (
            self._sm120_decode_available
            and attn_metadata.num_prefills == 0
            and attn_metadata.num_decodes > 0
            and self.attn_type == AttentionType.DECODER
            and not attn_metadata.use_cascade
            and self.alibi_slopes is None
            and self.sliding_window == (-1, -1)
            and attn_metadata.decode is not None
            and hasattr(attn_metadata.decode, 'block_tables')
        )

        if can_use_sm120:
            return self._forward_sm120_decode(
                layer, query, kv_cache, attn_metadata, output
            )

        # Fall through to FlashInfer for everything else (prefill, mixed, etc.)
        return super().forward(
            layer, query, key, value, kv_cache, attn_metadata,
            output, output_scale, output_block_scale,
        )

    def _forward_sm120_decode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        ext = _get_sm120_decode()

        decode_meta = attn_metadata.decode
        num_tokens = attn_metadata.num_decode_tokens
        key_cache, value_cache = kv_cache.unbind(0)

        q = query[:num_tokens]

        workspace = _get_workspace(
            num_tokens, self.num_heads, self.head_size, query.device
        )

        # Get KV dequant scales from layer (default 1.0 for BF16)
        k_scale = getattr(layer, '_k_scale_float', 1.0)
        v_scale = getattr(layer, '_v_scale_float', 1.0)

        # Use block_tables and seq_lens from TRTLLMDecode metadata
        ext.sm120_flash_decode_paged(
            query=q,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=decode_meta.block_tables,
            seq_lens=decode_meta.seq_lens,
            output=output[:num_tokens],
            workspace=workspace,
            max_seq_len=decode_meta.max_seq_len,
            k_scale=k_scale,
            v_scale=v_scale,
        )

        return output
