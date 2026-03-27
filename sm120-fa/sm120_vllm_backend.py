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
    TRTLLMPrefill,
)
from vllm.v1.attention.backend import AttentionType, AttentionCGSupport
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
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
    def get_builder_cls() -> type["SM120MetadataBuilder"]:
        return SM120MetadataBuilder


class SM120MetadataBuilder(FlashInferMetadataBuilder):
    """Override CUDA graph support to UNIFORM_BATCH.

    The SM120 decode kernel handles any batch size (Sq=1..N),
    so we don't need TRTLLMDecode for UNIFORM_BATCH support.
    Stores seq_lens and block_tables on metadata for SM120 MTP verify.
    """

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_BATCH

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool = False) -> FlashInferMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build=fast_build)
        # Store seq_lens and block_tables for SM120 MTP verify path
        # These are needed when prefill is FIPrefill (no block_tables/seq_lens)
        metadata._sm120_seq_lens = common_attn_metadata.seq_lens
        metadata._sm120_block_tables = common_attn_metadata.block_table_tensor
        metadata._sm120_query_start_loc = common_attn_metadata.query_start_loc
        metadata._sm120_max_seq_len = common_attn_metadata.max_seq_len
        return metadata


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

        # DEBUG: log dispatch info
        if hasattr(attn_metadata, '_debug_logged') is False or not getattr(attn_metadata, '_debug_logged', False):
            logger.info("SM120 dispatch: num_decodes=%d, num_decode_tokens=%d, num_prefills=%d, num_prefill_tokens=%d",
                        attn_metadata.num_decodes, attn_metadata.num_decode_tokens,
                        attn_metadata.num_prefills, attn_metadata.num_prefill_tokens)

        # Dispatch decode to SM120 kernel when conditions are met
        can_use_sm120_decode = (
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

        if can_use_sm120_decode:
            return self._forward_sm120_decode(
                layer, query, kv_cache, attn_metadata, output
            )

        # MTP verify falls through to FlashInfer — repeat_interleave is not CUDA-graph-safe.
        # Sprint 14 showed 68% MTP acceptance with SM120 decode + FlashInfer verify.
        # Fall through to FlashInfer for everything else (prefill, MTP verify, mixed, etc.)
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
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_decodes = attn_metadata.num_decodes
        key_cache, value_cache = kv_cache[:, 0], kv_cache[:, 1]

        # Get KV dequant scales from layer (default 1.0 for BF16)
        k_scale = getattr(layer, '_k_scale_float', 1.0)
        v_scale = getattr(layer, '_v_scale_float', 1.0)

        # Determine query tokens per request
        # For MTP verify: num_decode_tokens = num_decodes * q_per_req
        q_per_req = num_decode_tokens // num_decodes

        if q_per_req == 1:
            # Fast path: standard Sq=1 decode
            q = query[:num_decode_tokens]

            workspace = _get_workspace(
                num_decode_tokens, self.num_heads, self.head_size, query.device
            )

            ext.sm120_flash_decode_paged(
                query=q,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=decode_meta.block_tables,
                seq_lens=decode_meta.seq_lens,
                output=output[:num_decode_tokens],
                workspace=workspace,
                max_seq_len=decode_meta.max_seq_len,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        else:
            # MTP verification: multiple query tokens per request
            # Token layout is contiguous per request:
            # [req0_q0, req0_q1, ..., req0_qN, req1_q0, ...]
            # Reshape to [num_decodes, q_per_req, num_heads, head_size]
            q_all = query[:num_decode_tokens].view(
                num_decodes, q_per_req, self.num_heads, self.head_size
            )
            o_all = output[:num_decode_tokens].view(
                num_decodes, q_per_req, self.num_heads, self.head_size
            )

            workspace = _get_workspace(
                num_decodes, self.num_heads, self.head_size, query.device
            )

            for qi in range(q_per_req):
                q_slice = q_all[:, qi, :, :].contiguous()  # [num_decodes, num_heads, head_size]
                o_slice = o_all[:, qi, :, :]

                # Adjust seq_lens for causal: position qi sees qi more KV entries
                if qi == 0:
                    adj_seq_lens = decode_meta.seq_lens
                    adj_max_seq_len = decode_meta.max_seq_len
                else:
                    adj_seq_lens = decode_meta.seq_lens + qi
                    adj_max_seq_len = decode_meta.max_seq_len + qi

                ext.sm120_flash_decode_paged(
                    query=q_slice,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_table=decode_meta.block_tables,
                    seq_lens=adj_seq_lens,
                    output=o_slice,
                    workspace=workspace,
                    max_seq_len=adj_max_seq_len,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )

        return output

    def _forward_sm120_mtp_verify(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Handle MTP verification through SM120 decode kernel.

        MTP verify is classified as prefill (num_prefills>0, small token count).
        We call the decode kernel once per query token to match draft numerics.

        CUDA graph safe: no .item() calls, fixed loop count, tensor-only ops.
        """
        ext = _get_sm120_decode()
        num_tokens = attn_metadata.num_prefill_tokens
        num_prefills = attn_metadata.num_prefills
        key_cache, value_cache = kv_cache[:, 0], kv_cache[:, 1]

        k_scale = getattr(layer, '_k_scale_float', 1.0)
        v_scale = getattr(layer, '_v_scale_float', 1.0)

        # Use stored metadata from SM120MetadataBuilder.build()
        # seq_lens: [num_reqs], block_tables: [num_reqs, max_blocks]
        # query_start_loc: [num_reqs + 1] cumulative query token offsets
        seq_lens_kv = attn_metadata._sm120_seq_lens  # [num_reqs]
        block_tables = attn_metadata._sm120_block_tables  # [num_reqs, max_blocks]
        query_start_loc = attn_metadata._sm120_query_start_loc  # [num_reqs + 1]
        max_seq_len = attn_metadata._sm120_max_seq_len

        # For MTP verify: num_prefills requests, each with some query tokens.
        # Token layout: [req0_q0, req0_q1, ..., req0_qN, req1_q0, ...]
        # We process ALL tokens as individual Sq=1 decode calls.
        # Each token at position i within its request sees seq_lens_kv + i KV entries.

        # Expand block_tables: [num_prefills, max_blocks] -> [num_tokens, max_blocks]
        q_lens = query_start_loc[1:] - query_start_loc[:-1]  # [num_reqs]
        bt_expanded = block_tables.repeat_interleave(q_lens, dim=0)

        # Build per-token seq_lens with causal offset
        global_idx = torch.arange(num_tokens, dtype=seq_lens_kv.dtype, device=seq_lens_kv.device)
        req_starts = query_start_loc[:-1].repeat_interleave(q_lens)
        offsets = global_idx - req_starts

        sl_expanded = seq_lens_kv.repeat_interleave(q_lens) + offsets

        max_sl = max_seq_len + num_tokens

        workspace = _get_workspace(
            num_tokens, self.num_heads, self.head_size, query.device
        )

        ext.sm120_flash_decode_paged(
            query=query[:num_tokens],
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=bt_expanded,
            seq_lens=sl_expanded,
            output=output[:num_tokens],
            workspace=workspace,
            max_seq_len=max_sl,
            k_scale=k_scale,
            v_scale=v_scale,
        )

        return output
