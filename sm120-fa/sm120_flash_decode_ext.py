"""
SM120 Flash Decode — PyTorch extension for paged KV cache decode attention.

Builds the CUDA kernel as a JIT torch extension and provides a Python interface
matching vLLM's attention conventions.
"""

import os
import torch
from torch.utils.cpp_extension import load_inline

_CSRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")

# Read the CUDA source
with open(os.path.join(_CSRC_DIR, "sm120_flash_decode_paged.cu"), "r") as f:
    _CUDA_SOURCE = f.read()

# C++ wrapper that calls the extern "C" launcher via torch tensors
_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

extern "C" void sm120_flash_decode_paged_launch(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* key_cache,
    const __nv_bfloat16* val_cache,
    const int* block_table,
    const int* seq_lens,
    __nv_bfloat16* O,
    float* partial_O,
    float* partial_lse,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int block_size,
    int max_blocks_per_seq,
    int max_splits,
    cudaStream_t stream
);

std::vector<torch::Tensor> sm120_flash_decode(
    torch::Tensor query,           // [batch, num_q_heads, head_dim] bf16
    torch::Tensor key_cache,       // [num_blocks, block_size, num_kv_heads, head_dim] bf16
    torch::Tensor value_cache,     // [num_blocks, block_size, num_kv_heads, head_dim] bf16
    torch::Tensor block_table,     // [num_seqs, max_blocks_per_seq] int32
    torch::Tensor seq_lens,        // [num_seqs] int32
    torch::Tensor output,          // [batch, num_q_heads, head_dim] bf16 (pre-allocated)
    torch::Tensor partial_O,       // [max_splits, batch*num_q_heads, head_dim] float32
    torch::Tensor partial_lse,     // [max_splits, batch*num_q_heads] float32
    int max_seq_len
) {
    int batch_size = query.size(0);
    int num_q_heads = query.size(1);
    int head_dim = query.size(2);
    int num_kv_heads = key_cache.size(2);
    int block_size = key_cache.size(1);
    int max_blocks_per_seq = block_table.size(1);
    int max_splits = partial_O.size(0);

    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    sm120_flash_decode_paged_launch(
        reinterpret_cast<const __nv_bfloat16*>(query.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(key_cache.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(value_cache.data_ptr()),
        block_table.data_ptr<int>(),
        seq_lens.data_ptr<int>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        partial_O.data_ptr<float>(),
        partial_lse.data_ptr<float>(),
        batch_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        max_seq_len,
        block_size,
        max_blocks_per_seq,
        max_splits,
        stream
    );

    return {output};
}
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        # NOTE: No -use_fast_math (per memory: causes MTP acceptance regression)
        _module = load_inline(
            name="sm120_flash_decode_paged",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["sm120_flash_decode"],
            extra_cuda_cflags=[
                "-O3",
                "-arch=sm_120",
                "--threads=4",
                "-lineinfo",
            ],
            verbose=False,
        )
    return _module


# Pre-allocated workspace for CUDA graph compatibility
class SM120FlashDecodeWorkspace:
    """Pre-allocates partial_O and partial_lse buffers for CUDA graph capture."""

    def __init__(self, max_batch_size: int, num_q_heads: int, head_dim: int,
                 max_splits: int = 32, device: str = "cuda"):
        total_heads = max_batch_size * num_q_heads
        self.partial_O = torch.zeros(
            max_splits, total_heads, head_dim,
            dtype=torch.float32, device=device
        )
        self.partial_lse = torch.full(
            (max_splits, total_heads),
            -float('inf'),
            dtype=torch.float32, device=device
        )
        self.max_splits = max_splits
        self.max_batch_size = max_batch_size
        self.num_q_heads = num_q_heads
        self.head_dim = head_dim


def sm120_flash_decode_paged(
    query: torch.Tensor,        # [batch, num_q_heads, head_dim] bf16
    key_cache: torch.Tensor,    # [num_blocks, block_size, num_kv_heads, head_dim] bf16
    value_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim] bf16
    block_table: torch.Tensor,  # [num_seqs, max_blocks_per_seq] int32
    seq_lens: torch.Tensor,     # [num_seqs] int32
    output: torch.Tensor | None = None,
    workspace: SM120FlashDecodeWorkspace | None = None,
) -> torch.Tensor:
    """
    SM120-native flash decode attention with paged KV cache.

    For decode: query has shape [batch, num_q_heads, head_dim] (one token per seq).
    Output shape: [batch, num_q_heads, head_dim].
    """
    mod = _get_module()

    batch_size = query.shape[0]
    num_q_heads = query.shape[1]
    head_dim = query.shape[2]
    max_seq_len = int(seq_lens.max().item())

    if output is None:
        output = torch.empty_like(query)

    max_splits = 32
    total_heads = batch_size * num_q_heads

    if workspace is not None:
        partial_O = workspace.partial_O[:max_splits, :total_heads, :head_dim]
        partial_lse = workspace.partial_lse[:max_splits, :total_heads]
    else:
        partial_O = torch.empty(max_splits, total_heads, head_dim,
                                dtype=torch.float32, device=query.device)
        partial_lse = torch.empty(max_splits, total_heads,
                                  dtype=torch.float32, device=query.device)

    mod.sm120_flash_decode(
        query.contiguous(),
        key_cache.contiguous(),
        value_cache.contiguous(),
        block_table.contiguous(),
        seq_lens.contiguous(),
        output,
        partial_O,
        partial_lse,
        max_seq_len,
    )

    return output
