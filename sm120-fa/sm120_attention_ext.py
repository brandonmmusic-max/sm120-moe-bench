"""
SM120 Attention Extension — Combined API for decode + prefill kernels.

Provides:
- sm120_decode: Paged KV decode attention (existing kernel)
- sm120_prefill: Contiguous KV prefill attention (new kernel)
- gather_paged_kv: Utility to gather paged KV cache into contiguous tensors
- SM120FlashDecodeWorkspace: Pre-allocated workspace for CUDA graph compat
"""

import os
import torch
from typing import Optional, Tuple

# Lazy imports of kernel extensions
_decode_ext = None
_prefill_ext = None


def _get_decode_ext():
    """Lazy-load the SM120 decode kernel extension."""
    global _decode_ext
    if _decode_ext is None:
        import importlib.util
        ext_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "sm120_flash_decode_ext.py")
        spec = importlib.util.spec_from_file_location("sm120_flash_decode_ext", ext_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _decode_ext = mod
    return _decode_ext


def _get_prefill_ext():
    """Lazy-load the SM120 prefill kernel extension."""
    global _prefill_ext
    if _prefill_ext is None:
        import importlib.util
        ext_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "sm120_flash_prefill_ext.py")
        spec = importlib.util.spec_from_file_location("sm120_flash_prefill_ext", ext_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _prefill_ext = mod
    return _prefill_ext


# Re-export workspace class
def get_decode_workspace_class():
    return _get_decode_ext().SM120FlashDecodeWorkspace


def gather_paged_kv(
    kv_cache: torch.Tensor,     # [num_blocks, block_size, num_kv_heads, head_dim]
    block_table: torch.Tensor,  # [max_blocks_per_seq] int32
    seq_len: int,
    block_size: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    is_key: bool = True,
) -> torch.Tensor:
    """
    Gather paged KV cache into contiguous [num_kv_heads, seq_len, head_dim] BF16 tensor.

    If the KV cache is FP8, dequantizes using the provided scale factor.
    """
    num_blocks_needed = (seq_len + block_size - 1) // block_size
    blocks = block_table[:num_blocks_needed]

    # Gather: [num_blocks_needed, block_size, num_kv_heads, head_dim]
    gathered = kv_cache[blocks]

    # Flatten blocks: [total_positions, num_kv_heads, head_dim]
    gathered = gathered.reshape(-1, gathered.shape[-2], gathered.shape[-1])

    # Trim to actual seq_len
    gathered = gathered[:seq_len]

    # Transpose to [num_kv_heads, seq_len, head_dim]
    gathered = gathered.permute(1, 0, 2).contiguous()

    # Dequant FP8 → BF16 if needed
    if gathered.dtype == torch.float8_e4m3fn:
        scale = k_scale if is_key else v_scale
        gathered = gathered.to(torch.bfloat16) * scale

    return gathered


def sm120_decode(
    query: torch.Tensor,        # [batch, num_q_heads, head_dim] bf16
    key_cache: torch.Tensor,    # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    block_table: torch.Tensor,  # [num_seqs, max_blocks_per_seq] int32
    seq_lens: torch.Tensor,     # [num_seqs] int32
    output: torch.Tensor | None = None,
    workspace=None,
    max_seq_len: int | None = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> torch.Tensor:
    """
    SM120 decode attention with paged KV cache.
    Supports BF16 and FP8 E4M3 KV.
    """
    ext = _get_decode_ext()
    return ext.sm120_flash_decode_paged(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        output=output,
        workspace=workspace,
        max_seq_len=max_seq_len,
        k_scale=k_scale,
        v_scale=v_scale,
    )


def sm120_prefill(
    query: torch.Tensor,        # [Sq, num_q_heads, head_dim] bf16 (vLLM layout)
    key_cache: torch.Tensor,    # [num_blocks, block_size, num_kv_heads, head_dim]
    value_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim]
    block_table: torch.Tensor,  # [max_blocks_per_seq] int32 (single request)
    seq_len: int,               # KV sequence length for this request
    query_len: int,             # Query length for this request
    output: torch.Tensor | None = None,  # [query_len, num_q_heads, head_dim]
    block_size: int = 16,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    causal: bool = True,
) -> torch.Tensor:
    """
    SM120 prefill attention with paged KV cache.

    Gathers paged KV into contiguous buffers, then runs MMA prefill kernel.
    Input Q is [query_len, num_q_heads, head_dim] (vLLM token-first layout).
    Output is same shape as Q.
    """
    ext = _get_prefill_ext()

    num_q_heads = query.shape[1]
    num_kv_heads = key_cache.shape[2]
    head_dim = query.shape[2]

    # Gather paged KV into contiguous tensors: [num_kv_heads, seq_len, head_dim]
    k_contig = gather_paged_kv(
        key_cache, block_table, seq_len, block_size,
        k_scale=k_scale, v_scale=v_scale, is_key=True,
    )
    v_contig = gather_paged_kv(
        value_cache, block_table, seq_len, block_size,
        k_scale=k_scale, v_scale=v_scale, is_key=False,
    )

    # Transpose Q: [query_len, Hq, HD] → [Hq, query_len, HD]
    q_transposed = query[:query_len].permute(1, 0, 2).contiguous()

    # Allocate output in kernel layout: [Hq, query_len, HD]
    o_kernel = torch.empty(num_q_heads, query_len, head_dim,
                           dtype=torch.bfloat16, device=query.device)

    # Run prefill kernel
    ext.sm120_flash_prefill_forward(
        query=q_transposed,
        key=k_contig,
        value=v_contig,
        output=o_kernel,
        batch=1,
        Hq=num_q_heads,
        Hkv=num_kv_heads,
        Sq=query_len,
        Skv=seq_len,
        causal=causal,
    )

    # Transpose output back: [Hq, query_len, HD] → [query_len, Hq, HD]
    o_result = o_kernel.permute(1, 0, 2).contiguous()

    # Write to pre-allocated output if provided
    if output is not None:
        output[:query_len].copy_(o_result)
        return output
    return o_result
