"""
SM120 Flash Decode — PyTorch extension for paged KV cache decode attention.

Builds the CUDA kernel as a JIT torch extension and provides a Python interface
matching vLLM's attention conventions.

Supports BF16, FP8 E4M3, and NVFP4 (E2M1 + E4M3FN block scales) KV cache dtypes.
"""

import os
import torch
from torch.utils.cpp_extension import load_inline

_CSRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")

# Read the CUDA source (v2: tiled vectorized with cp.async bulk loads + FP8 support)
_V2_PATH = os.path.join(_CSRC_DIR, "sm120_flash_decode_v2_paged.cu")
_V1_PATH = os.path.join(_CSRC_DIR, "sm120_flash_decode_paged.cu")
_KERNEL_PATH = _V2_PATH if os.path.exists(_V2_PATH) else _V1_PATH
with open(_KERNEL_PATH, "r") as f:
    _CUDA_SOURCE = f.read()

# C++ wrapper that calls the extern "C" launchers via torch tensors
_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// BF16 KV cache launcher
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
    int kv_block_stride,
    cudaStream_t stream
);

// FP8 E4M3 KV cache launcher
extern "C" void sm120_flash_decode_paged_fp8_launch(
    const __nv_bfloat16* Q,
    const __nv_fp8_e4m3* key_cache,
    const __nv_fp8_e4m3* val_cache,
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
    int kv_block_stride,
    float k_scale,
    float v_scale,
    cudaStream_t stream
);

std::vector<torch::Tensor> sm120_flash_decode(
    torch::Tensor query,           // [batch, num_q_heads, head_dim] bf16
    torch::Tensor key_cache,       // [num_blocks, block_size, num_kv_heads, head_dim] bf16 or fp8 (may be non-contiguous view)
    torch::Tensor value_cache,     // [num_blocks, block_size, num_kv_heads, head_dim] bf16 or fp8
    torch::Tensor block_table,     // [num_seqs, max_blocks_per_seq] int32
    torch::Tensor seq_lens,        // [num_seqs] int32
    torch::Tensor output,          // [batch, num_q_heads, head_dim] bf16 (pre-allocated)
    torch::Tensor partial_O,       // [max_splits, batch*num_q_heads, head_dim] float32
    torch::Tensor partial_lse,     // [max_splits, batch*num_q_heads] float32
    int max_seq_len,
    double k_scale,                // KV dequant scale for K (1.0 for BF16)
    double v_scale                 // KV dequant scale for V (1.0 for BF16)
) {
    int batch_size = query.size(0);
    int num_q_heads = query.size(1);
    int head_dim = query.size(2);
    // kv_cache[:, 0] may be NHD or HND layout.
    // block_size is always >= num_kv_heads for practical models.
    // (Qwen3.5-397B TP=4: block_size=16, num_kv_heads=1)
    int dim1 = key_cache.size(1);
    int dim2 = key_cache.size(2);
    int block_size = (dim1 >= dim2) ? dim1 : dim2;
    int num_kv_heads = (dim1 >= dim2) ? dim2 : dim1;
    int max_blocks_per_seq = block_table.size(1);
    int max_splits = partial_O.size(0);

    // Use stride(0) for block stride — supports both contiguous and interleaved K/V views
    int kv_block_stride = (int)key_cache.stride(0);

    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    bool is_fp8 = (key_cache.scalar_type() == torch::kFloat8_e4m3fn ||
                    key_cache.scalar_type() == torch::kByte);

    if (is_fp8) {
        sm120_flash_decode_paged_fp8_launch(
            reinterpret_cast<const __nv_bfloat16*>(query.data_ptr()),
            reinterpret_cast<const __nv_fp8_e4m3*>(key_cache.data_ptr()),
            reinterpret_cast<const __nv_fp8_e4m3*>(value_cache.data_ptr()),
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
            kv_block_stride,
            (float)k_scale,
            (float)v_scale,
            stream
        );
    } else {
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
            kv_block_stride,
            stream
        );
    }

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
    key_cache: torch.Tensor,    # [num_blocks, block_size, num_kv_heads, head_dim] bf16 or fp8
    value_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_dim] bf16 or fp8
    block_table: torch.Tensor,  # [num_seqs, max_blocks_per_seq] int32
    seq_lens: torch.Tensor,     # [num_seqs] int32
    output: torch.Tensor | None = None,
    workspace: SM120FlashDecodeWorkspace | None = None,
    max_seq_len: int | None = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> torch.Tensor:
    """
    SM120-native flash decode attention with paged KV cache.

    Supports both BF16 and FP8 E4M3 KV cache dtypes.
    For FP8: pass k_scale and v_scale for dequantization.
    For decode: query has shape [batch, num_q_heads, head_dim] (one token per seq).
    Pass max_seq_len to avoid GPU->CPU sync from seq_lens.max().item().
    """
    mod = _get_module()

    batch_size = query.shape[0]
    num_q_heads = query.shape[1]
    head_dim = query.shape[2]
    if max_seq_len is None:
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
        key_cache,  # non-contiguous OK: kernel uses stride(0) for block access
        value_cache,
        block_table.contiguous(),
        seq_lens.contiguous(),
        output,
        partial_O,
        partial_lse,
        max_seq_len,
        k_scale,
        v_scale,
    )

    # Apply v_scale post-kernel on BF16 output (matches FlashInfer decode.py:1434-1440)
    is_fp8 = key_cache.dtype == torch.float8_e4m3fn
    if is_fp8 and v_scale != 1.0:
        output.mul_(v_scale)

    return output


# ============================================================================
# NVFP4 KV cache support
# ============================================================================

# NVFP4 block scale group size (must match CUDA kernel's FP4_BLOCK_SIZE)
FP4_BLOCK_SIZE = 16

# Read the FP4 CUDA source
_FP4_PATH = os.path.join(_CSRC_DIR, "sm120_flash_decode_v2_paged_fp4.cu")
with open(_FP4_PATH, "r") as f:
    _FP4_CUDA_SOURCE = f.read()

# C++ wrapper for FP4 launcher
_FP4_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

// NVFP4 KV cache launcher
extern "C" void sm120_flash_decode_paged_fp4_launch(
    const __nv_bfloat16* Q,
    const uint8_t* key_cache,
    const uint8_t* val_cache,
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
    int kv_block_stride,
    float k_tensor_scale,
    float v_tensor_scale,
    cudaStream_t stream
);

std::vector<torch::Tensor> sm120_flash_decode_fp4(
    torch::Tensor query,           // [batch, num_q_heads, head_dim] bf16
    torch::Tensor key_cache,       // [num_blocks, block_size, num_kv_heads, packed_dim] uint8
    torch::Tensor value_cache,     // [num_blocks, block_size, num_kv_heads, packed_dim] uint8
    torch::Tensor block_table,     // [num_seqs, max_blocks_per_seq] int32
    torch::Tensor seq_lens,        // [num_seqs] int32
    torch::Tensor output,          // [batch, num_q_heads, head_dim] bf16
    torch::Tensor partial_O,       // [max_splits, batch*num_q_heads, head_dim] float32
    torch::Tensor partial_lse,     // [max_splits, batch*num_q_heads] float32
    int max_seq_len,
    int head_dim,                  // true head_dim (256), not packed_dim (144)
    double k_tensor_scale,         // per-tensor K pre-normalization scale (1.0 default)
    double v_tensor_scale          // per-tensor V pre-normalization scale (1.0 default)
) {
    int batch_size = query.size(0);
    int num_q_heads = query.size(1);
    // Infer block_size and num_kv_heads from cache shape
    // Cache: [num_blocks, block_size, num_kv_heads, packed_dim]
    int dim1 = key_cache.size(1);
    int dim2 = key_cache.size(2);
    int block_size = (dim1 >= dim2) ? dim1 : dim2;
    int num_kv_heads = (dim1 >= dim2) ? dim2 : dim1;
    int max_blocks_per_seq = block_table.size(1);
    int max_splits = partial_O.size(0);

    int kv_block_stride = (int)key_cache.stride(0);

    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    sm120_flash_decode_paged_fp4_launch(
        reinterpret_cast<const __nv_bfloat16*>(query.data_ptr()),
        reinterpret_cast<const uint8_t*>(key_cache.data_ptr()),
        reinterpret_cast<const uint8_t*>(value_cache.data_ptr()),
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
        kv_block_stride,
        (float)k_tensor_scale,
        (float)v_tensor_scale,
        stream
    );

    return {output};
}
"""

_fp4_module = None

def _get_fp4_module():
    global _fp4_module
    if _fp4_module is None:
        _fp4_module = load_inline(
            name="sm120_flash_decode_fp4",
            cpp_sources=_FP4_CPP_SOURCE,
            cuda_sources=_FP4_CUDA_SOURCE,
            functions=["sm120_flash_decode_fp4"],
            extra_cuda_cflags=[
                "-O3",
                "-arch=sm_120",
                "--threads=4",
                "-lineinfo",
            ],
            verbose=False,
        )
    return _fp4_module


def nvfp4_packed_dim(head_dim: int) -> int:
    """Compute packed dimension for NVFP4 cache: head_dim/2 (data) + head_dim/16 (scales)."""
    return head_dim // 2 + head_dim // FP4_BLOCK_SIZE


def quantize_to_nvfp4(tensor: torch.Tensor, tensor_scale: float | None = None) -> tuple[torch.Tensor, float]:
    """Quantize a BF16/FP32 tensor to packed NVFP4 format with optimal scale search.

    Input:  [..., head_dim] in BF16 or FP32
    Output: (packed_data, tensor_scale) where:
      - packed_data: [..., packed_dim] in uint8 (packed_dim = head_dim/2 + head_dim/16)
      - tensor_scale: float — per-tensor pre-normalization scale

    Packed layout per row:
      [fp4_data: head_dim/2 bytes] [scales: head_dim/16 bytes]

    Two-level quantization:
      1. Per-tensor scale: normalizes global max_abs into E4M3FN block scale range.
         If max_abs / 6.0 > 448 (E4M3FN max), a per-tensor scale is computed so that
         block scales stay representable. For normal-range data, tensor_scale = 1.0.
      2. Per-block scale (E4M3FN): handles local variation within each 16-element block.
      3. Optimal scale search: tries K nearest E4M3FN candidates, picks lowest MSE.

    Dequantization formula:
      real_value = decode_fp4(nibble) * decode_e4m3fn(block_scale) * tensor_scale
    """
    orig_shape = tensor.shape
    head_dim = orig_shape[-1]
    assert head_dim % FP4_BLOCK_SIZE == 0, f"head_dim {head_dim} must be divisible by {FP4_BLOCK_SIZE}"

    flat = tensor.float().reshape(-1, head_dim)
    num_rows = flat.shape[0]
    data_cols = head_dim // 2
    scale_cols = head_dim // FP4_BLOCK_SIZE
    packed_dim = data_cols + scale_cols

    FP4_MAX_MAG = 6.0
    E4M3FN_MAX = 448.0  # Maximum E4M3FN representable value

    # Per-tensor pre-normalization:
    # If the data's max_abs / FP4_MAX_MAG > E4M3FN_MAX, block scales would overflow.
    # Apply a per-tensor scale to bring data into representable range.
    # Use 80% of E4M3FN max as target to leave headroom for block-level variation.
    global_max_abs = flat.abs().max().item()
    max_representable = E4M3FN_MAX * FP4_MAX_MAG  # 448 * 6 = 2688
    if tensor_scale is not None:
        ts = tensor_scale
    elif global_max_abs > max_representable * 0.8:
        # Need per-tensor scale: target max_block_scale = E4M3FN_MAX * 0.8
        ts = global_max_abs / (max_representable * 0.8)
    else:
        ts = 1.0

    if ts != 1.0:
        flat = flat / ts

    # Pre-build E4M3FN LUT and FP4 magnitude table
    e4m3_codes, e4m3_decoded = _build_e4m3fn_lut(tensor.device)
    fp4_mags = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                            dtype=torch.float32, device=tensor.device)

    result = torch.zeros(num_rows, packed_dim, dtype=torch.uint8, device=tensor.device)

    NUM_SCALE_CANDIDATES = 7  # Try 7 nearest E4M3FN values

    num_blocks = head_dim // FP4_BLOCK_SIZE
    for b in range(num_blocks):
        start = b * FP4_BLOCK_SIZE
        end = start + FP4_BLOCK_SIZE
        block = flat[:, start:end]  # [num_rows, 16]

        # Initial scale estimate
        max_abs = block.abs().max(dim=1).values.clamp(min=1e-12)  # [num_rows]
        target_scale = max_abs / FP4_MAX_MAG  # [num_rows]

        # Find best E4M3FN scale per row via optimal search
        best_scale_codes, best_scale_floats = _optimal_e4m3fn_scale(
            block, target_scale, e4m3_codes, e4m3_decoded, fp4_mags,
            NUM_SCALE_CANDIDATES
        )
        result[:, data_cols + b] = best_scale_codes

        # Quantize with the optimal scale
        scale_decoded = best_scale_floats.unsqueeze(1).clamp(min=1e-12)  # [num_rows, 1]
        normalized = block / scale_decoded  # [num_rows, 16], range approximately [-6, 6]

        # Vectorized FP4 quantization: find nearest signed FP4 value
        for i in range(0, FP4_BLOCK_SIZE, 2):
            nibble0 = _quantize_fp4_element(normalized[:, i], fp4_mags)
            nibble1 = _quantize_fp4_element(normalized[:, i+1], fp4_mags)
            byte_idx = (start + i) // 2
            result[:, byte_idx] = nibble0 | (nibble1 << 4)

    out_shape = list(orig_shape[:-1]) + [packed_dim]
    return result.reshape(out_shape), ts


def _optimal_e4m3fn_scale(
    block: torch.Tensor,       # [N, 16] float values
    target: torch.Tensor,      # [N] target scale
    codes: torch.Tensor,       # [K] E4M3FN codes
    decoded: torch.Tensor,     # [K] decoded floats
    fp4_mags: torch.Tensor,    # [8] FP4 magnitudes
    num_candidates: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the E4M3FN scale that minimizes quantization MSE per block.

    For each row, tries num_candidates nearest E4M3FN values to the target
    and picks the one with lowest roundtrip MSE (quantize→dequantize→compare).
    """
    N = block.shape[0]
    device = block.device

    # Find K nearest E4M3FN values to each target [N, K]
    diffs = (target.unsqueeze(1) - decoded.unsqueeze(0)).abs()  # [N, len(decoded)]
    _, topk_idx = diffs.topk(num_candidates, dim=1, largest=False)  # [N, K]

    candidate_codes = codes[topk_idx]    # [N, K] uint8
    candidate_floats = decoded[topk_idx]  # [N, K] float

    # For each candidate scale, compute roundtrip MSE
    # block: [N, 16], candidate_floats: [N, K]
    best_codes = torch.zeros(N, dtype=torch.uint8, device=device)
    best_floats = torch.zeros(N, dtype=torch.float32, device=device)
    best_mse = torch.full((N,), float('inf'), device=device)

    for k in range(num_candidates):
        scale_k = candidate_floats[:, k].clamp(min=1e-12).unsqueeze(1)  # [N, 1]
        normalized = block / scale_k  # [N, 16]

        # Quantize each element: find nearest FP4 magnitude
        abs_norm = normalized.abs()  # [N, 16]
        signs = (normalized < 0).float()  # [N, 16]

        # Vectorized nearest-magnitude search
        diff_to_mags = (abs_norm.unsqueeze(2) - fp4_mags.unsqueeze(0).unsqueeze(0)).abs()  # [N, 16, 8]
        best_mag_idx = diff_to_mags.argmin(dim=2)  # [N, 16]
        quantized_abs = fp4_mags[best_mag_idx]  # [N, 16]
        quantized = quantized_abs * (1 - 2 * signs)  # restore sign

        # Dequantize: quantized * scale
        dequantized = quantized * scale_k  # [N, 16]

        # MSE
        mse_k = ((block - dequantized) ** 2).mean(dim=1)  # [N]

        # Update best
        improved = mse_k < best_mse
        best_mse[improved] = mse_k[improved]
        best_codes[improved] = candidate_codes[improved, k]
        best_floats[improved] = candidate_floats[improved, k]

    return best_codes, best_floats


def _quantize_fp4_element(values: torch.Tensor, fp4_mags: torch.Tensor) -> torch.Tensor:
    """Quantize float values to FP4 E2M1 nibbles (uint8, 0-15).

    E2M1 magnitudes: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    Nibble encoding: [sign:1][exp:2][mantissa:1]
    """
    fp4_codes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8, device=values.device)

    signs = (values < 0).to(torch.uint8) * 8  # sign bit = bit 3
    abs_vals = values.abs()

    # Find nearest magnitude by minimum distance
    diffs = (abs_vals.unsqueeze(1) - fp4_mags.unsqueeze(0)).abs()  # [N, 8]
    best_idx = diffs.argmin(dim=1)  # [N]
    nibbles = fp4_codes[best_idx] | signs

    return nibbles.to(torch.uint8)


def _build_e4m3fn_lut(device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build lookup table of all positive E4M3FN values.

    Returns:
        codes:   [K] uint8 — E4M3FN byte codes (positive only, excluding NaN)
        decoded: [K] float — corresponding decoded values
    """
    import math
    codes = []
    decoded = []
    for code in range(128):
        e = (code >> 3) & 0xF
        m = code & 7
        if e == 15 and m == 7:
            continue  # NaN
        if code == 0:
            dec = 0.0
        elif e == 0:
            dec = m / 512.0
        else:
            dec = math.ldexp(1.0 + m / 8.0, e - 7)
        codes.append(code)
        decoded.append(dec)
    return (torch.tensor(codes, dtype=torch.uint8, device=device),
            torch.tensor(decoded, dtype=torch.float32, device=device))


def _float_to_e4m3fn(values: torch.Tensor) -> torch.Tensor:
    """Encode positive float values as E4M3FN bytes (vectorized).

    E4M3FN: [sign:1][exp:4][mantissa:3], bias=7, max=448.0
    Only encodes positive values (block scales are always positive).
    """
    codes, decoded = _build_e4m3fn_lut(values.device)
    v = values.float().clamp(min=0, max=448.0)
    diffs = (v.unsqueeze(1) - decoded.unsqueeze(0)).abs()
    best_idx = diffs.argmin(dim=1)
    return codes[best_idx]


def _e4m3fn_to_float(codes: torch.Tensor) -> torch.Tensor:
    """Decode E4M3FN bytes to float (vectorized). Matches CUDA decode_e4m3fn exactly."""
    import math
    # Build full 256-entry decode table
    lut = torch.zeros(256, dtype=torch.float32, device=codes.device)
    for x in range(256):
        s = (x >> 7) & 1
        e = (x >> 3) & 0xF
        m = x & 7
        if e == 15 and m == 7:
            val = 0.0
        elif e == 0:
            val = m / 512.0
        else:
            val = math.ldexp(1.0 + m / 8.0, e - 7)
        lut[x] = -val if s else val
    return lut[codes.long()]


def sm120_flash_decode_paged_fp4(
    query: torch.Tensor,        # [batch, num_q_heads, head_dim] bf16
    key_cache: torch.Tensor,    # [num_blocks, block_size, num_kv_heads, packed_dim] uint8
    value_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, packed_dim] uint8
    block_table: torch.Tensor,  # [num_seqs, max_blocks_per_seq] int32
    seq_lens: torch.Tensor,     # [num_seqs] int32
    head_dim: int,              # true head_dim (e.g., 256), NOT packed_dim
    output: torch.Tensor | None = None,
    workspace: SM120FlashDecodeWorkspace | None = None,
    max_seq_len: int | None = None,
    k_tensor_scale: float = 1.0,
    v_tensor_scale: float = 1.0,
) -> torch.Tensor:
    """
    SM120-native flash decode attention with paged NVFP4 KV cache.

    NVFP4 format: 2 FP4 values per byte + E4M3FN block scale per 16 values.
    No per-tensor k_scale/v_scale needed -- block scales capture full dynamic range.

    Packed cache layout per row: [fp4_data: head_dim/2] [scales: head_dim/16]
    packed_dim = head_dim/2 + head_dim/16 = 9*head_dim/16
    """
    mod = _get_fp4_module()

    batch_size = query.shape[0]
    num_q_heads = query.shape[1]
    if max_seq_len is None:
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

    mod.sm120_flash_decode_fp4(
        query.contiguous(),
        key_cache,
        value_cache,
        block_table.contiguous(),
        seq_lens.contiguous(),
        output,
        partial_O,
        partial_lse,
        max_seq_len,
        head_dim,
        k_tensor_scale,
        v_tensor_scale,
    )

    return output
