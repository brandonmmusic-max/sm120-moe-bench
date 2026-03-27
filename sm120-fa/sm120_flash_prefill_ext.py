"""
SM120 Flash Prefill — PyTorch extension for contiguous-KV prefill attention.

JIT-compiles the CUDA kernel and provides a Python interface.
Input layout: Q[batch*Hq, Sq, HD], K[batch*Hkv, Skv, HD], V[batch*Hkv, Skv, HD]
Output: O[batch*Hq, Sq, HD]

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
