"""
SM120 Flash Prefill — PyTorch extension for contiguous-KV prefill attention.

JIT-compiles the CUDA kernel and provides a Python interface.
Input layout: Q[batch*Hq, Sq, HD], K[batch*Hkv, Skv, HD], V[batch*Hkv, Skv, HD]
Output: O[batch*Hq, Sq, HD]

Supports HD=128 and HD=256 (templated in CUDA).
Also provides gather_paged_kv() for paged→contiguous KV conversion with FP8 dequant.

NOTE: No -use_fast_math (per memory: causes MTP acceptance regression)
"""

import os
import torch
from torch.utils.cpp_extension import load_inline

_CSRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")

with open(os.path.join(_CSRC_DIR, "sm120_flash_prefill_contiguous.cu"), "r") as f:
    _CUDA_SOURCE = f.read()

_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

extern "C" void sm120_flash_prefill_launch(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* LSE,
    int batch, int Hq, int Hkv, int Sq, int Skv, int hd,
    bool causal, cudaStream_t stream
);

torch::Tensor sm120_flash_prefill(
    torch::Tensor query,    // [batch*Hq, Sq, HD] bf16
    torch::Tensor key,      // [batch*Hkv, Skv, HD] bf16
    torch::Tensor value,    // [batch*Hkv, Skv, HD] bf16
    torch::Tensor output,   // [batch*Hq, Sq, HD] bf16 (pre-allocated)
    int batch,
    int Hq,
    int Hkv,
    int Sq,
    int Skv,
    bool causal
) {
    int hd = query.size(2);
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    sm120_flash_prefill_launch(
        reinterpret_cast<const __nv_bfloat16*>(query.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(value.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        nullptr,  // LSE not needed
        batch, Hq, Hkv, Sq, Skv, hd,
        causal, stream
    );

    return output;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="sm120_flash_prefill",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["sm120_flash_prefill"],
            extra_cuda_cflags=[
                "-O3",
                "-arch=sm_120",
                "--threads=4",
                "-lineinfo",
            ],
            verbose=False,
        )
    return _module


def gather_paged_kv(
    key_cache: torch.Tensor,     # [num_blocks, num_kv_heads, block_size, head_dim] (HND) or [num_blocks, block_size, num_kv_heads, head_dim] (NHD)
    value_cache: torch.Tensor,   # same layout as key_cache
    block_table: torch.Tensor,   # [max_blocks_per_seq] int32 — single request
    seq_len: int,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    hnd_layout: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather paged KV cache into contiguous BF16 tensors for one request.

    Returns: (K_contig, V_contig) each [num_kv_heads, seq_len, head_dim] bf16

    Handles FP8→BF16 dequant via k_scale/v_scale.
    """
    is_fp8 = key_cache.dtype in (torch.float8_e4m3fn, torch.uint8)

    if hnd_layout:
        # HND: [num_blocks, num_kv_heads, block_size, head_dim]
        num_kv_heads = key_cache.shape[1]
        block_size = key_cache.shape[2]
        head_dim = key_cache.shape[3]
    else:
        # NHD: [num_blocks, block_size, num_kv_heads, head_dim]
        block_size = key_cache.shape[1]
        num_kv_heads = key_cache.shape[2]
        head_dim = key_cache.shape[3]

    num_blocks_needed = (seq_len + block_size - 1) // block_size
    block_indices = block_table[:num_blocks_needed].long()

    if hnd_layout:
        # Gather: [num_blocks_needed, num_kv_heads, block_size, head_dim]
        k_gathered = key_cache[block_indices]
        v_gathered = value_cache[block_indices]

        # Reshape to [num_kv_heads, num_blocks_needed * block_size, head_dim]
        k_contig = k_gathered.permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim)
        v_contig = v_gathered.permute(1, 0, 2, 3).reshape(num_kv_heads, -1, head_dim)
    else:
        # NHD layout
        k_gathered = key_cache[block_indices]
        v_gathered = value_cache[block_indices]
        k_contig = k_gathered.permute(2, 0, 1, 3).reshape(num_kv_heads, -1, head_dim)
        v_contig = v_gathered.permute(2, 0, 1, 3).reshape(num_kv_heads, -1, head_dim)

    # Trim to actual seq_len
    k_contig = k_contig[:, :seq_len, :].contiguous()
    v_contig = v_contig[:, :seq_len, :].contiguous()

    # Dequant FP8 → BF16
    if is_fp8:
        # View-cast uint8 → float8_e4m3fn if needed (bit-identical)
        if k_contig.dtype == torch.uint8:
            k_contig = k_contig.view(torch.float8_e4m3fn)
            v_contig = v_contig.view(torch.float8_e4m3fn)
        k_contig = k_contig.to(torch.bfloat16) * k_scale
        v_contig = v_contig.to(torch.bfloat16) * v_scale
    else:
        k_contig = k_contig.to(torch.bfloat16)
        v_contig = v_contig.to(torch.bfloat16)

    return k_contig, v_contig


def sm120_flash_prefill_forward(
    query: torch.Tensor,    # [batch*Hq, Sq, HD] bf16
    key: torch.Tensor,      # [batch*Hkv, Skv, HD] bf16
    value: torch.Tensor,    # [batch*Hkv, Skv, HD] bf16
    output: torch.Tensor | None = None,
    batch: int = 1,
    Hq: int | None = None,
    Hkv: int | None = None,
    Sq: int | None = None,
    Skv: int | None = None,
    causal: bool = True,
) -> torch.Tensor:
    """
    SM120-native flash prefill attention with contiguous KV.

    Q layout: [batch*Hq, Sq, HD], K/V layout: [batch*Hkv, Skv, HD]
    Output: [batch*Hq, Sq, HD]

    Supports HD=128 and HD=256.
    """
    mod = _get_module()

    if Hq is None:
        Hq = query.shape[0] // batch
    if Hkv is None:
        Hkv = key.shape[0] // batch
    if Sq is None:
        Sq = query.shape[1]
    if Skv is None:
        Skv = key.shape[1]

    if output is None:
        output = torch.empty_like(query)

    mod.sm120_flash_prefill(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        output,
        batch, Hq, Hkv, Sq, Skv, causal,
    )

    return output
