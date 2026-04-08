"""
Python wrapper for the fused reshape_and_cache_nvfp4 CUDA kernel.

Provides a drop-in replacement for vLLM's reshape_and_cache_flash when
kv_cache_dtype="nvfp4". Handles both key and value in one call.

Usage:
    from reshape_and_cache_nvfp4_ext import reshape_and_cache_nvfp4

    reshape_and_cache_nvfp4(
        key,            # [num_tokens, num_heads, head_dim] bf16
        value,          # [num_tokens, num_heads, head_dim] bf16
        key_cache,      # [num_blocks, block_size, num_heads, packed_dim] uint8
        value_cache,    # [num_blocks, block_size, num_heads, packed_dim] uint8
        slot_mapping,   # [num_tokens] int64
        k_global_scale, # float (per-tensor scale for K, 1.0 default)
        v_global_scale, # float (per-tensor scale for V, 1.0 default)
    )

Cache layout (packed_dim = head_dim/2 + head_dim/16):
    First head_dim/2 bytes: packed FP4 data (2 nibbles per byte)
    Last head_dim/16 bytes: E4M3FN block scales (1 per 16 elements)
"""

import os
import torch
from torch.utils.cpp_extension import load_inline

_CSRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
_CUDA_PATH = os.path.join(_CSRC_DIR, "reshape_and_cache_nvfp4.cu")

with open(_CUDA_PATH, "r") as f:
    _CUDA_SOURCE = f.read()

_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

void reshape_and_cache_nvfp4_launch(
    const __nv_bfloat16* src, uint8_t* cache,
    const int64_t* slot_mapping, float global_scale,
    int num_tokens, int num_heads, int head_dim, int block_size,
    int64_t cache_stride_block, int64_t cache_stride_page,
    int64_t cache_stride_head, cudaStream_t stream);

void reshape_and_cache_nvfp4_torch(
    torch::Tensor src,           // [num_tokens, num_heads, head_dim] bf16
    torch::Tensor cache,         // [num_blocks, block_size, num_heads, packed_dim] uint8
    torch::Tensor slot_mapping,  // [num_tokens] int64
    double global_scale
) {
    int num_tokens = src.size(0);
    int num_heads = src.size(1);
    int head_dim = src.size(2);
    int block_size = cache.size(1);

    // Cache strides (in uint8 elements)
    int64_t cache_stride_block = cache.stride(0);
    int64_t cache_stride_page = cache.stride(1);
    int64_t cache_stride_head = cache.stride(2);

    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    reshape_and_cache_nvfp4_launch(
        reinterpret_cast<const __nv_bfloat16*>(src.data_ptr()),
        cache.data_ptr<uint8_t>(),
        slot_mapping.data_ptr<int64_t>(),
        (float)global_scale,
        num_tokens, num_heads, head_dim, block_size,
        cache_stride_block, cache_stride_page, cache_stride_head,
        stream
    );
}
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="reshape_and_cache_nvfp4",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["reshape_and_cache_nvfp4_torch"],
            extra_cuda_cflags=[
                "-O3",
                "-arch=sm_120",
                "--threads=4",
                "-lineinfo",
            ],
            verbose=False,
        )
    return _module


FP4_BLOCK_SIZE = 16


def packed_dim(head_dim: int) -> int:
    """Compute packed dimension for NVFP4: data + scales."""
    return head_dim // 2 + head_dim // FP4_BLOCK_SIZE


def reshape_and_cache_nvfp4(
    key: torch.Tensor,          # [num_tokens, num_heads, head_dim] bf16
    value: torch.Tensor,        # [num_tokens, num_heads, head_dim] bf16
    key_cache: torch.Tensor,    # [num_blocks, block_size, num_heads, packed_dim] uint8
    value_cache: torch.Tensor,  # [num_blocks, block_size, num_heads, packed_dim] uint8
    slot_mapping: torch.Tensor, # [num_tokens] int64
    k_global_scale: float = 1.0,
    v_global_scale: float = 1.0,
) -> None:
    """Fused scatter-write + NVFP4 quantization for paged KV cache.

    Replaces vLLM's reshape_and_cache_flash for nvfp4 dtype.
    """
    mod = _get_module()

    # Key
    mod.reshape_and_cache_nvfp4_torch(
        key.contiguous(),
        key_cache,
        slot_mapping.contiguous(),
        k_global_scale,
    )

    # Value
    mod.reshape_and_cache_nvfp4_torch(
        value.contiguous(),
        value_cache,
        slot_mapping.contiguous(),
        v_global_scale,
    )
