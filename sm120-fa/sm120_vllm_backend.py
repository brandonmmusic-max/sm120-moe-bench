"""
SM120 Flash Attention Backend for vLLM.

Subclasses FlashInferBackend to intercept:
  - Decode (Sq=1): SM120 split-KV decode kernel
  - Prefill (Sq>8): SM120 MMA prefill kernel (gather paged KV → contiguous)
  - MTP verify (Sq<=8): SM120 decode kernel with per-token causal offsets

Zero FlashInfer in the attention forward path.

Supports both BF16 and FP8 E4M3 KV cache dtypes.
Registered as FLASH_ATTN in the backend registry to override the default.

NOTE: No -use_fast_math (per memory: causes MTP acceptance regression)
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

# Lazy import of our kernels
_sm120_decode_module = None
_sm120_prefill_module = None
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


def _get_sm120_prefill():
    """Lazy-load the SM120 flash prefill kernel."""
    global _sm120_prefill_module
    if _sm120_prefill_module is None:
        import importlib.util, os
        ext_path = os.path.join(os.path.dirname(__file__), "sm120_flash_prefill_ext.py")
        spec = importlib.util.spec_from_file_location("sm120_flash_prefill_ext", ext_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _sm120_prefill_module = mod
        logger.info("SM120 Flash Prefill kernel loaded successfully")
    return _sm120_prefill_module


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

    Intercepts decode AND prefill to use SM120 kernels.
    Zero FlashInfer in the forward path.
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

    @classmethod
    def get_required_kv_cache_layout(cls) -> str:
        # Force HND layout: CUDA kernel assumes HND for K/V address computation.
        # NHD is the default on SM120 but our kernel's address formula is:
        #   blk * stride + (kv_head * block_size + token) * head_dim  (HND)
        # HND stride_order permutes logical (N,2,BS,Hkv,HD) to physical (N,2,Hkv,BS,HD).
        return "HND"


class SM120MetadataBuilder(FlashInferMetadataBuilder):
    """Override CUDA graph support to UNIFORM_SINGLE_TOKEN_DECODE.

    The SM120 decode kernel handles any batch size (Sq=1..N).
    Stores seq_lens, block_tables, query_start_loc on metadata for SM120 paths.
    """

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec) -> AttentionCGSupport:
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata, fast_build: bool = False) -> FlashInferMetadata:
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build=fast_build)
        # Store metadata for SM120 decode/prefill/MTP paths
        metadata._sm120_seq_lens = common_attn_metadata.seq_lens
        metadata._sm120_block_tables = common_attn_metadata.block_table_tensor
        metadata._sm120_query_start_loc = common_attn_metadata.query_start_loc
        metadata._sm120_max_seq_len = common_attn_metadata.max_seq_len
        return metadata


class SM120FlashInferImpl(FlashInferImpl):
    """FlashInfer impl that dispatches decode AND prefill to SM120 kernels."""

    def __init__(self, *args, **kwargs):
        # Capture attn_type before super().__init__ which doesn't store it
        attn_type = kwargs.get('attn_type', AttentionType.DECODER)
        if len(args) > 8:
            attn_type = args[8]
        super().__init__(*args, **kwargs)
        self.attn_type = attn_type
        self._sm120_decode_available = False
        self._sm120_prefill_available = False
        self._sm120_init_attempted = False

    def _try_init_sm120(self, device):
        """Attempt to initialize SM120 kernels (once)."""
        if self._sm120_init_attempted:
            return
        self._sm120_init_attempted = True

        # Init decode kernel
        try:
            _get_sm120_decode()
            self._sm120_decode_available = True
            kv_dtype_str = self.kv_cache_dtype
            logger.info("SM120 decode kernel: enabled (head_dim=%d, GQA=%d:%d, kv_dtype=%s)",
                        self.head_size, self.num_heads, self.num_kv_heads, kv_dtype_str)
        except Exception as e:
            logger.warning("SM120 decode kernel: disabled (%s)", e)

        # Init prefill kernel
        try:
            _get_sm120_prefill()
            self._sm120_prefill_available = True
            logger.info("SM120 prefill kernel: enabled (head_dim=%d)", self.head_size)
        except Exception as e:
            logger.warning("SM120 prefill kernel: disabled (%s), falling back to FlashInfer", e)

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

        # Common guards for SM120 dispatch
        sm120_eligible = (
            self.attn_type == AttentionType.DECODER
            and not attn_metadata.use_cascade
            and self.alibi_slopes is None
            and self.sliding_window == (-1, -1)
            and hasattr(attn_metadata, '_sm120_block_tables')
            and attn_metadata._sm120_block_tables is not None
        )

        # ---- Pure decode (Sq=1 per request) ----
        if (self._sm120_decode_available
                and sm120_eligible
                and attn_metadata.num_prefills == 0
                and attn_metadata.num_decodes > 0):
            if not hasattr(self, '_logged_decode'):
                self._logged_decode = True
                logger.info("SM120 dispatch → decode (num_decodes=%d, tokens=%d)",
                            attn_metadata.num_decodes, attn_metadata.num_decode_tokens)
            return self._forward_sm120_decode(
                layer, query, kv_cache, attn_metadata, output
            )

        # ---- MTP verify (classified as prefill, ≤8 tokens per req) ----
        is_mtp_verify = (
            self._sm120_decode_available
            and sm120_eligible
            and attn_metadata.num_prefills > 0
            and attn_metadata.num_prefill_tokens <= attn_metadata.num_prefills * 8
            and attn_metadata.num_decodes == 0
        )

        # ---- True prefill (Sq > 8 per request on average) ----
        is_true_prefill = (
            self._sm120_prefill_available
            and sm120_eligible
            and attn_metadata.num_prefills > 0
            and attn_metadata.num_decodes == 0
            and not is_mtp_verify
        )

        if is_true_prefill:
            if not hasattr(self, '_logged_prefill'):
                self._logged_prefill = True
                logger.info("SM120 dispatch → prefill (num_prefills=%d, tokens=%d)",
                            attn_metadata.num_prefills, attn_metadata.num_prefill_tokens)
            return self._forward_sm120_prefill(
                layer, query, kv_cache, attn_metadata, output
            )

        if is_mtp_verify:
            if not hasattr(self, '_logged_mtp'):
                self._logged_mtp = True
                logger.info("SM120 dispatch → MTP verify (num_prefills=%d, tokens=%d)",
                            attn_metadata.num_prefills, attn_metadata.num_prefill_tokens)
            return self._forward_sm120_mtp_verify(
                layer, query, kv_cache, attn_metadata, output
            )

        # Fallback to FlashInfer (should rarely happen)
        if not hasattr(self, '_logged_fallback'):
            self._logged_fallback = True
            logger.warning(
                "SM120 fallback to FlashInfer: eligible=%s, num_prefills=%d, "
                "num_decodes=%d, num_prefill_tokens=%d, decode=%s, "
                "has_block_tables=%s",
                sm120_eligible, attn_metadata.num_prefills,
                attn_metadata.num_decodes, attn_metadata.num_prefill_tokens,
                attn_metadata.decode is not None,
                hasattr(attn_metadata, '_sm120_block_tables'),
            )
        return super().forward(
            layer, query, key, value, kv_cache, attn_metadata,
            output, output_scale, output_block_scale,
        )

    # ================================================================
    # Decode path (Sq=1)
    # ================================================================

    def _forward_sm120_decode(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        ext = _get_sm120_decode()

        num_decode_tokens = attn_metadata.num_decode_tokens
        num_decodes = attn_metadata.num_decodes
        key_cache, value_cache = kv_cache[:, 0], kv_cache[:, 1]

        # Use our stored metadata (always available via SM120MetadataBuilder)
        block_tables = attn_metadata._sm120_block_tables
        seq_lens = attn_metadata._sm120_seq_lens
        max_seq_len = attn_metadata._sm120_max_seq_len

        # Debug: log shapes/strides on first call
        if not hasattr(self, '_sm120_debug_logged'):
            self._sm120_debug_logged = True
            logger.info(
                "SM120 decode dispatch: kv_cache.shape=%s kv_cache.stride=%s "
                "key_cache.shape=%s key_cache.stride=%s key_cache.dtype=%s "
                "block_tables.shape=%s seq_lens=%s num_decode_tokens=%d",
                kv_cache.shape, kv_cache.stride(),
                key_cache.shape, key_cache.stride(), key_cache.dtype,
                block_tables.shape, seq_lens[:min(4, num_decodes)],
                num_decode_tokens,
            )

        # Get KV dequant scales from layer (default 1.0 for BF16)
        k_scale = getattr(layer, '_k_scale_float', 1.0)
        v_scale = getattr(layer, '_v_scale_float', 1.0)

        # Determine query tokens per request
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
                block_table=block_tables,
                seq_lens=seq_lens,
                output=output[:num_decode_tokens],
                workspace=workspace,
                max_seq_len=max_seq_len,
                k_scale=k_scale,
                v_scale=v_scale,
            )

            # DEBUG: check for kernel completion / CUDA errors
            if not hasattr(self, '_decode_sync_logged'):
                self._decode_sync_logged = True
                torch.cuda.synchronize()
                out_slice = output[:num_decode_tokens]
                has_nan = torch.isnan(out_slice).any().item()
                has_inf = torch.isinf(out_slice).any().item()
                logger.info("SM120 decode kernel completed: has_nan=%s, has_inf=%s, out_norm=%.4f",
                            has_nan, has_inf, out_slice.float().norm().item())

        else:
            # MTP verification: multiple query tokens per request
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
                q_slice = q_all[:, qi, :, :].contiguous()
                o_slice = o_all[:, qi, :, :]

                if qi == 0:
                    adj_seq_lens = seq_lens
                    adj_max_seq_len = max_seq_len
                else:
                    adj_seq_lens = seq_lens + qi
                    adj_max_seq_len = max_seq_len + qi

                ext.sm120_flash_decode_paged(
                    query=q_slice,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_table=block_tables,
                    seq_lens=adj_seq_lens,
                    output=o_slice,
                    workspace=workspace,
                    max_seq_len=adj_max_seq_len,
                    k_scale=k_scale,
                    v_scale=v_scale,
                )

        return output

    # ================================================================
    # Prefill path (Sq > 8) — gather paged KV → contiguous → MMA kernel
    # ================================================================

    def _forward_sm120_prefill(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch prefill through SM120 MMA prefill kernel.

        For each prefill request:
        1. Gather paged KV → contiguous BF16 [num_kv_heads, seq_len, HD]
        2. Reshape Q → [num_q_heads, query_len, HD]
        3. Call contiguous prefill kernel
        """
        prefill_ext = _get_sm120_prefill()

        num_prefills = attn_metadata.num_prefills
        query_start_loc = attn_metadata._sm120_query_start_loc  # [num_reqs + 1]
        seq_lens = attn_metadata._sm120_seq_lens                # [num_reqs]
        block_tables = attn_metadata._sm120_block_tables         # [num_reqs, max_blocks]

        key_cache, value_cache = kv_cache[:, 0], kv_cache[:, 1]
        # vLLM logical shape after [:, 0]: [num_blocks, block_size, num_kv_heads, head_dim]
        # (NHD logical order — get_kv_cache_shape returns (N, 2, BS, Hkv, HD))
        block_size = key_cache.shape[1]

        k_scale = getattr(layer, '_k_scale_float', 1.0)
        v_scale = getattr(layer, '_v_scale_float', 1.0)

        is_fp8 = key_cache.dtype in (torch.float8_e4m3fn, torch.uint8)

        # Log on first prefill call
        if not hasattr(self, '_sm120_prefill_logged'):
            self._sm120_prefill_logged = True
            logger.info(
                "SM120 prefill dispatch: num_prefills=%d, kv_cache.shape=%s, "
                "key_cache.dtype=%s, block_size=%d, head_dim=%d, GQA=%d:%d, "
                "k_scale=%.4f, v_scale=%.4f",
                num_prefills, kv_cache.shape, key_cache.dtype, block_size,
                self.head_size, self.num_heads, self.num_kv_heads,
                k_scale, v_scale,
            )

        # For pure prefill batches (num_decodes==0), all requests are prefill.
        # query_start_loc covers all requests.
        num_decodes = attn_metadata.num_decodes  # should be 0 for this path

        for i in range(num_prefills):
            req_idx = num_decodes + i  # global request index

            # Query token range
            q_start = int(query_start_loc[req_idx].item())
            q_end = int(query_start_loc[req_idx + 1].item())
            query_len = q_end - q_start

            # KV sequence length (includes current prefill tokens already in cache)
            kv_seq_len = int(seq_lens[req_idx].item())

            # Block table for this request
            bt = block_tables[req_idx]

            # Gather paged KV → contiguous BF16
            # key_cache logical shape: [num_blocks, block_size, num_kv_heads, head_dim] (NHD)
            k_contig, v_contig = prefill_ext.gather_paged_kv(
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=bt,
                seq_len=kv_seq_len,
                k_scale=k_scale,
                v_scale=v_scale,
                hnd_layout=False,
            )
            # k_contig, v_contig: [num_kv_heads, kv_seq_len, head_dim] bf16

            # Q slice: [query_len, num_heads, head_dim] → [num_heads, query_len, head_dim]
            q_req = query[q_start:q_end].permute(1, 0, 2).contiguous()
            # q_req: [num_heads, query_len, head_dim]

            # Output slice
            o_req = torch.empty(
                self.num_heads, query_len, self.head_size,
                dtype=torch.bfloat16, device=query.device,
            )

            # Call prefill kernel
            # Layout: Q[batch*Hq, Sq, HD], K[batch*Hkv, Skv, HD], V[batch*Hkv, Skv, HD]
            # batch=1 for per-request dispatch
            prefill_ext.sm120_flash_prefill_forward(
                query=q_req,
                key=k_contig,
                value=v_contig,
                output=o_req,
                batch=1,
                Hq=self.num_heads,
                Hkv=self.num_kv_heads,
                Sq=query_len,
                Skv=kv_seq_len,
                causal=True,
            )

            # Write back: [num_heads, query_len, head_dim] → [query_len, num_heads, head_dim]
            output[q_start:q_end] = o_req.permute(1, 0, 2)

        return output

    # ================================================================
    # MTP verify path (Sq <= 8, use decode kernel with causal offsets)
    # ================================================================

    def _forward_sm120_mtp_verify(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Handle MTP verification through SM120 decode kernel.

        MTP verify is classified as prefill with small token count (e.g., 4 tokens
        for MTP=3). We treat each query token as an independent Sq=1 decode call
        with incrementing seq_lens for causal masking.

        For single-user MTP=3: 1 request, 4 query tokens, same block_table.
        CUDA graph safe: expand() not copy, pre-built offset tensor.
        """
        ext = _get_sm120_decode()
        num_tokens = attn_metadata.num_prefill_tokens
        key_cache, value_cache = kv_cache[:, 0], kv_cache[:, 1]

        k_scale = getattr(layer, '_k_scale_float', 1.0)
        v_scale = getattr(layer, '_v_scale_float', 1.0)

        seq_lens_kv = attn_metadata._sm120_seq_lens       # [num_reqs]
        block_tables = attn_metadata._sm120_block_tables   # [num_reqs, max_blocks]
        max_seq_len = attn_metadata._sm120_max_seq_len

        query_start_loc = attn_metadata._sm120_query_start_loc  # [num_reqs + 1]
        q_lens = query_start_loc[1:] - query_start_loc[:-1]    # [num_reqs]

        # Expand block_tables: repeat each request's row by its query token count
        bt_expanded = block_tables.repeat_interleave(q_lens, dim=0)  # [num_tokens, max_blocks]

        # Build per-token seq_lens with causal offset
        if not hasattr(self, '_mtp_offsets') or self._mtp_offsets.size(0) < num_tokens:
            self._mtp_offsets = torch.arange(
                max(num_tokens, 16), dtype=seq_lens_kv.dtype, device=seq_lens_kv.device
            )
        global_idx = self._mtp_offsets[:num_tokens]
        req_starts = query_start_loc[:-1].repeat_interleave(q_lens)  # [num_tokens]
        offsets = global_idx - req_starts  # per-token offset within its request

        sl_expanded = seq_lens_kv.repeat_interleave(q_lens) + offsets  # [num_tokens]

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
