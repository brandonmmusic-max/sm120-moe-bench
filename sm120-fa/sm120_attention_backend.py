"""
SM120 Attention Backend for vLLM v1.

Complete standalone backend — no FlashInfer dependency.
Routes decode to SM120 split-KV paged kernel, prefill to SM120 MMA kernel.
Handles FP8 KV cache, GQA, CUDA graphs, and MTP verification.

Registration: Set VLLM_ATTENTION_BACKEND=SM120 or patch the backend registry
to point FLASHINFER → this module's SM120AttentionBackend.

NOTE: No -use_fast_math anywhere (per memory: causes MTP acceptance regression)
"""

import os
import enum
from dataclasses import dataclass
from typing import Optional, List, Type

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl as AttentionImplBase,
    AttentionMetadataBuilder as AttentionMetadataBuilderBase,
    AttentionType,
)

logger = init_logger(__name__)

# Lazy-load our kernel extension
_attn_ext = None


def _get_attn_ext():
    global _attn_ext
    if _attn_ext is None:
        import importlib.util
        ext_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "sm120_attention_ext.py")
        spec = importlib.util.spec_from_file_location("sm120_attention_ext", ext_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _attn_ext = mod
        logger.info("SM120 attention extension loaded")
    return _attn_ext


# ============================================================================
# Metadata
# ============================================================================

@dataclass
class SM120AttentionMetadata:
    """Attention metadata for SM120 backend."""
    # Token counts
    num_actual_tokens: int
    slot_mapping: torch.Tensor          # [num_actual_tokens]

    # Global limits
    max_query_len: int
    max_seq_len: int

    # Decode portion (tokens [0, num_decode_tokens))
    num_decodes: int
    num_decode_tokens: int
    decode_seq_lens: Optional[torch.Tensor] = None      # [num_decodes] int32
    decode_block_tables: Optional[torch.Tensor] = None   # [num_decodes, max_blocks] int32
    decode_max_seq_len: int = 0

    # Prefill portion (tokens [num_decode_tokens, num_actual_tokens))
    num_prefills: int = 0
    num_prefill_tokens: int = 0
    prefill_query_start_loc: Optional[torch.Tensor] = None  # [num_prefills + 1] int32
    prefill_seq_lens: Optional[torch.Tensor] = None          # [num_prefills] int32
    prefill_block_tables: Optional[torch.Tensor] = None      # [num_prefills, max_blocks] int32


# ============================================================================
# CUDA Graph Support
# ============================================================================

class AttentionCGSupport(enum.IntEnum):
    """CUDA graph support levels (must match vLLM's enum)."""
    NEVER = 0
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    UNIFORM_BATCH = 2


# ============================================================================
# Metadata Builder
# ============================================================================

class SM120MetadataBuilder(AttentionMetadataBuilderBase):
    """Builds SM120AttentionMetadata from CommonAttentionMetadata."""

    def __init__(self, kv_cache_spec, vllm_config, device):
        self.kv_cache_spec = kv_cache_spec
        self.vllm_config = vllm_config
        self.device = device
        self.block_size = kv_cache_spec.block_size

        # MTP decode threshold: requests with query_len <= threshold are decode
        num_spec = getattr(vllm_config.speculative_config,
                           'num_speculative_tokens', 0) if hasattr(vllm_config, 'speculative_config') and vllm_config.speculative_config else 0
        self.decode_threshold = 1 + 2 * num_spec if num_spec > 0 else 1

        # Pre-allocated tensors for CUDA graph replay
        self._cg_seq_lens = None
        self._cg_block_tables = None
        self._cg_slot_mapping = None

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
    ) -> SM120AttentionMetadata:
        """Build metadata from CommonAttentionMetadata."""
        cam = common_attn_metadata
        num_reqs = cam.num_reqs
        num_actual_tokens = cam.num_actual_tokens

        # Compute per-request query lengths
        qsl = cam.query_start_loc_cpu
        query_lens = qsl[1:] - qsl[:-1]  # [num_reqs]

        # Split into decode and prefill
        is_decode = query_lens <= self.decode_threshold
        num_decodes = int(is_decode.sum().item())
        num_prefills = num_reqs - num_decodes

        # Batch is pre-sorted: decodes first, prefills after
        num_decode_tokens = int(qsl[num_decodes].item()) if num_decodes > 0 else 0
        num_prefill_tokens = num_actual_tokens - num_decode_tokens

        # Decode metadata
        decode_seq_lens = None
        decode_block_tables = None
        decode_max_seq_len = 0
        if num_decodes > 0:
            decode_seq_lens = cam.seq_lens[:num_decodes]
            decode_block_tables = cam.block_table_tensor[:num_decodes]
            decode_max_seq_len = int(cam.seq_lens_cpu[:num_decodes].max().item()) if num_decodes > 0 else 0

        # Prefill metadata
        prefill_query_start_loc = None
        prefill_seq_lens = None
        prefill_block_tables = None
        if num_prefills > 0:
            # Prefill query_start_loc: rebase to start from 0
            pf_qsl = cam.query_start_loc[num_decodes:] - num_decode_tokens
            prefill_query_start_loc = pf_qsl
            prefill_seq_lens = cam.seq_lens[num_decodes:]
            prefill_block_tables = cam.block_table_tensor[num_decodes:]

        return SM120AttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=cam.slot_mapping,
            max_query_len=int(cam.max_query_len),
            max_seq_len=int(cam.max_seq_len),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            decode_seq_lens=decode_seq_lens,
            decode_block_tables=decode_block_tables,
            decode_max_seq_len=decode_max_seq_len,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_seq_lens=prefill_seq_lens,
            prefill_block_tables=prefill_block_tables,
        )

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata,
    ) -> SM120AttentionMetadata:
        """Build metadata for CUDA graph capture (decode-only, pre-allocated tensors)."""
        cam = common_attn_metadata
        num_reqs = cam.num_reqs
        num_tokens = cam.num_actual_tokens
        max_blocks = cam.block_table_tensor.shape[1] if cam.block_table_tensor is not None else 0

        # Pre-allocate or reuse tensors for in-place update during graph replay
        if (self._cg_seq_lens is None or
                self._cg_seq_lens.shape[0] < num_reqs):
            self._cg_seq_lens = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
            self._cg_block_tables = torch.zeros(num_reqs, max_blocks, dtype=torch.int32, device=self.device)
            self._cg_slot_mapping = torch.zeros(num_tokens, dtype=torch.int64, device=self.device)

        # Copy current values into pre-allocated tensors
        sl = self._cg_seq_lens[:num_reqs]
        sl.copy_(cam.seq_lens[:num_reqs])
        bt = self._cg_block_tables[:num_reqs, :max_blocks]
        bt.copy_(cam.block_table_tensor[:num_reqs, :max_blocks])
        sm = self._cg_slot_mapping[:num_tokens]
        sm.copy_(cam.slot_mapping[:num_tokens])

        max_seq_len = int(cam.seq_lens_cpu[:num_reqs].max().item()) if num_reqs > 0 else 0

        return SM120AttentionMetadata(
            num_actual_tokens=num_tokens,
            slot_mapping=sm,
            max_query_len=int(cam.max_query_len),
            max_seq_len=max_seq_len,
            num_decodes=num_reqs,
            num_decode_tokens=num_tokens,
            decode_seq_lens=sl,
            decode_block_tables=bt,
            decode_max_seq_len=max_seq_len,
            num_prefills=0,
            num_prefill_tokens=0,
        )

    def build_for_drafting(
        self,
        common_attn_metadata,
        draft_index: int,
    ) -> SM120AttentionMetadata:
        """Build metadata for MTP draft step (single-token decode per request)."""
        # Draft steps are single-token decodes — same as normal decode
        return self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            fast_build=True,
        )

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        """Support CUDA graphs for uniform decode batches (including MTP verify)."""
        return AttentionCGSupport.UNIFORM_BATCH


# ============================================================================
# Attention Implementation
# ============================================================================

class SM120AttentionImpl(AttentionImplBase):
    """SM120 attention implementation — decode + prefill via custom kernels."""

    # Class attributes for vLLM backend interface
    accept_output_buffer = True
    forward_includes_kv_cache_update = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[list] = None,
        sliding_window: Optional[int] = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type

        # Decode workspace (lazy-init)
        self._workspace = None
        self._decode_init_done = False

    def _ensure_workspace(self, batch_size: int, device: torch.device):
        """Ensure decode workspace is large enough."""
        ext = _get_attn_ext()
        WS = ext.get_decode_workspace_class()

        if (self._workspace is None or
                self._workspace.max_batch_size < batch_size):
            max_bs = max(batch_size * 2, 256)
            self._workspace = WS(
                max_batch_size=max_bs,
                num_q_heads=self.num_heads,
                head_dim=self.head_size,
                max_splits=32,
                device=str(device),
            )
            logger.info("SM120 decode workspace: batch=%d, heads=%d, dim=%d",
                        max_bs, self.num_heads, self.head_size)

    def forward(
        self,
        layer,
        query: torch.Tensor,           # [num_tokens, num_heads, head_size]
        key: torch.Tensor,             # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,           # [num_tokens, num_kv_heads, head_size]
        kv_cache: torch.Tensor,        # [num_blocks, 2, block_size, num_kv_heads, head_size]
        attn_metadata: SM120AttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert output is not None, "SM120 backend requires pre-allocated output"

        if attn_metadata is None:
            return output.fill_(0)

        num_actual = attn_metadata.num_actual_tokens

        # ---- Decode ----
        if attn_metadata.num_decode_tokens > 0:
            self._forward_decode(
                layer, query, kv_cache, attn_metadata, output,
            )

        # ---- Prefill ----
        if attn_metadata.num_prefill_tokens > 0:
            self._forward_prefill(
                layer, query, kv_cache, attn_metadata, output,
            )

        return output

    def do_kv_cache_update(
        self,
        layer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Update KV cache via vLLM's reshape_and_cache_flash (backend-independent)."""
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key, value,
            kv_cache[:, 0], kv_cache[:, 1],
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale, layer._v_scale,
        )

    # ---- Decode path ----

    def _forward_decode(
        self,
        layer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: SM120AttentionMetadata,
        output: torch.Tensor,
    ):
        """Dispatch decode tokens to SM120 split-KV paged kernel."""
        ext = _get_attn_ext()

        num_decode = metadata.num_decode_tokens
        num_decodes = metadata.num_decodes

        # Split KV cache: [num_blocks, 2, BS, Hkv, HD] → K, V each [num_blocks, BS, Hkv, HD]
        key_cache = kv_cache[:, 0]
        value_cache = kv_cache[:, 1]

        # Get FP8 scale factors
        k_scale = getattr(layer, '_k_scale_float', 1.0)
        v_scale = getattr(layer, '_v_scale_float', 1.0)

        # Determine query tokens per request
        # For MTP verify: num_decode_tokens = num_decodes * q_per_req
        q_per_req = num_decode // num_decodes if num_decodes > 0 else 1

        if q_per_req == 1:
            # Fast path: standard Sq=1 decode
            q_decode = query[:num_decode].view(num_decodes, self.num_heads, self.head_size)
            o_decode = output[:num_decode].view(num_decodes, self.num_heads, self.head_size)

            self._ensure_workspace(num_decodes, query.device)

            ext.sm120_decode(
                query=q_decode,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=metadata.decode_block_tables,
                seq_lens=metadata.decode_seq_lens,
                output=o_decode,
                workspace=self._workspace,
                max_seq_len=metadata.decode_max_seq_len,
                k_scale=k_scale,
                v_scale=v_scale,
            )
        else:
            # Multi-query decode (MTP verification): loop over query positions
            # Token layout: [req0_q0, req0_q1, ..., req0_qN, req1_q0, ...]
            # Reshape to [num_decodes, q_per_req, num_heads, head_size]
            q_all = query[:num_decode].view(num_decodes, q_per_req, self.num_heads, self.head_size)
            o_all = output[:num_decode].view(num_decodes, q_per_req, self.num_heads, self.head_size)

            self._ensure_workspace(num_decodes, query.device)

            for qi in range(q_per_req):
                q_slice = q_all[:, qi, :, :]  # [num_decodes, num_heads, head_size]
                o_slice = o_all[:, qi, :, :]

                # Adjust seq_lens for causal: position qi sees seq_lens_base + qi tokens
                if qi == 0:
                    adj_seq_lens = metadata.decode_seq_lens
                else:
                    adj_seq_lens = metadata.decode_seq_lens + qi
                    adj_max = metadata.decode_max_seq_len + qi

                ext.sm120_decode(
                    query=q_slice.contiguous(),
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_table=metadata.decode_block_tables,
                    seq_lens=adj_seq_lens,
                    output=o_slice,
                    workspace=self._workspace,
                    max_seq_len=metadata.decode_max_seq_len + qi if qi > 0 else metadata.decode_max_seq_len,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )

    # ---- Prefill path ----

    def _forward_prefill(
        self,
        layer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: SM120AttentionMetadata,
        output: torch.Tensor,
    ):
        """Dispatch prefill tokens to SM120 MMA prefill kernel."""
        ext = _get_attn_ext()

        num_decode_tokens = metadata.num_decode_tokens
        key_cache = kv_cache[:, 0]
        value_cache = kv_cache[:, 1]
        block_size = key_cache.shape[1]  # [num_blocks, block_size, ...]

        k_scale = getattr(layer, '_k_scale_float', 1.0)
        v_scale = getattr(layer, '_v_scale_float', 1.0)

        # Process each prefill request individually
        pf_qsl = metadata.prefill_query_start_loc
        for i in range(metadata.num_prefills):
            # Query range for this request
            q_start = int(pf_qsl[i].item())
            q_end = int(pf_qsl[i + 1].item())
            query_len = q_end - q_start
            seq_len = int(metadata.prefill_seq_lens[i].item())

            # Slice query (offset by num_decode_tokens into the full query tensor)
            q_req = query[num_decode_tokens + q_start:num_decode_tokens + q_end]
            o_req = output[num_decode_tokens + q_start:num_decode_tokens + q_end]

            # Block table for this request
            bt_req = metadata.prefill_block_tables[i]

            ext.sm120_prefill(
                query=q_req,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=bt_req,
                seq_len=seq_len,
                query_len=query_len,
                output=o_req,
                block_size=block_size,
                k_scale=k_scale,
                v_scale=v_scale,
                causal=True,
            )


# ============================================================================
# Backend Registration
# ============================================================================

class SM120AttentionBackend(AttentionBackend):
    """SM120-native attention backend — zero FlashInfer dependency."""

    @staticmethod
    def get_name() -> str:
        return "SM120_FLASH"

    @staticmethod
    def get_impl_cls() -> type:
        return SM120AttentionImpl

    @staticmethod
    def get_builder_cls() -> type:
        return SM120MetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple:
        # NHD layout: [num_blocks, 2(K/V), block_size, num_kv_heads, head_size]
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple:
        if include_num_layers_dimension:
            return (0, 1, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list:
        return [16]

    @classmethod
    def get_supported_head_sizes(cls) -> list:
        return [128]

    @classmethod
    def supports_compute_capability(cls, cap) -> bool:
        return cap.major == 12

    @classmethod
    def supports_kv_cache_dtype(cls, dtype: str) -> bool:
        return dtype in ("auto", "fp8", "fp8_e4m3")

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        return dtype == torch.bfloat16

    @classmethod
    def get_required_kv_cache_layout(cls):
        return None  # NHD (default)

    @classmethod
    def validate_configuration(
        cls,
        num_heads: int,
        head_size: int,
        num_kv_heads: int,
        dtype: torch.dtype,
        kv_cache_dtype: str,
        block_size: int,
        sliding_window: Optional[int] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
    ) -> list:
        """Return list of error strings if configuration is invalid."""
        errors = []
        if head_size not in cls.get_supported_head_sizes():
            errors.append(f"SM120 backend only supports head_size={cls.get_supported_head_sizes()}, got {head_size}")
        if dtype != torch.bfloat16:
            errors.append(f"SM120 backend only supports bfloat16, got {dtype}")
        if attn_type != AttentionType.DECODER:
            errors.append(f"SM120 backend only supports decoder attention, got {attn_type}")
        if sliding_window is not None:
            errors.append("SM120 backend does not support sliding window attention")
        if logits_soft_cap is not None:
            errors.append("SM120 backend does not support logits soft capping")
        return errors

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False
