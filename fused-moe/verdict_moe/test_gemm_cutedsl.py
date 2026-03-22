#!/usr/bin/env python3
"""
VerdictMoE Sprint 1: Standalone FC1 GEMM test using CuteDSL.

Tests a single FP4 block-scaled GEMM on SM120:
  C[M,N] = A[M,K] @ B[N,K]^T  with NVF4 scale factors

Uses CUTLASS CuteDSL to handle all fragment loading, SMEM layout,
and scale factor routing correctly. This validates the GEMM before
we add SwiGLU and FC2.

Usage:
  python3 test_gemm_cutedsl.py
"""

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu.warp.mma import Field as WarpField


def nvfp4_quantize(x_f32: torch.Tensor, sf_vec_size: int = 16):
    """Quantize FP32 tensor to NVFP4 (E2M1 + E4M3FN block scales).

    Returns (packed_fp4: uint8, scales: float8_e4m3fn).
    """
    assert x_f32.dim() == 2
    M, K = x_f32.shape
    assert K % sf_vec_size == 0

    device = x_f32.device
    flat = x_f32.float().reshape(-1)
    num_blocks = flat.numel() // sf_vec_size

    # Reshape into blocks
    blocks = flat.reshape(num_blocks, sf_vec_size)
    block_max = blocks.abs().amax(dim=1).clamp(min=1e-12)

    # E2M1 max representable = 6.0
    scales_f32 = block_max / 6.0

    # Convert to E4M3FN (not UE8M0!)
    scales_e4m3 = scales_f32.to(torch.float8_e4m3fn)
    scales_actual = scales_e4m3.float()

    # Quantize each element to nearest E2M1
    e2m1_table = torch.tensor([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=device)
    scaled = blocks / scales_actual.unsqueeze(1)
    signs = (scaled < 0).to(torch.uint8)
    abs_scaled = scaled.abs()

    # Find nearest E2M1 value
    diffs = (abs_scaled.unsqueeze(-1) - e2m1_table).abs()
    indices = diffs.argmin(dim=-1).to(torch.uint8)  # 0-7

    # Pack: 4-bit = sign(1) + value(3)
    fp4_nibbles = (signs << 3) | indices  # [num_blocks, sf_vec_size] uint8

    # Pack 2 nibbles per byte
    fp4_flat = fp4_nibbles.reshape(-1)
    packed = (fp4_flat[0::2] & 0xF) | ((fp4_flat[1::2] & 0xF) << 4)

    return packed.reshape(M, K // 2), scales_e4m3.reshape(M, K // sf_vec_size)


def nvfp4_dequantize(packed: torch.Tensor, scales: torch.Tensor, sf_vec_size: int = 16):
    """Dequantize NVFP4 back to FP32 for reference comparison."""
    M = packed.shape[0]
    K_half = packed.shape[1]
    K = K_half * 2

    e2m1_table = torch.tensor([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                               device=packed.device, dtype=torch.float32)

    # Unpack nibbles
    lo = (packed & 0xF).to(torch.int32)
    hi = ((packed >> 4) & 0xF).to(torch.int32)

    # Interleave back
    unpacked = torch.zeros(M, K, dtype=torch.int32, device=packed.device)
    unpacked[:, 0::2] = lo
    unpacked[:, 1::2] = hi

    # Decode
    signs = ((unpacked >> 3) & 1).float() * -2 + 1  # 1 or -1
    indices = (unpacked & 7).long()
    values = e2m1_table[indices] * signs

    # Apply block scales
    scales_f32 = scales.float()
    num_blocks = K // sf_vec_size
    values = values.reshape(M, num_blocks, sf_vec_size)
    values = values * scales_f32.unsqueeze(-1)
    return values.reshape(M, K)


def reference_gemm(a_f32, b_f32):
    """FP32 reference: C = A @ B^T"""
    return a_f32 @ b_f32.T


def test_quantize_roundtrip():
    """Test quantize → dequantize roundtrip."""
    print("=== Quantize/Dequantize Roundtrip Test ===")
    torch.manual_seed(42)
    M, K = 4, 64
    x = torch.randn(M, K, device="cuda") * 0.5

    packed, scales = nvfp4_quantize(x, sf_vec_size=16)
    x_recon = nvfp4_dequantize(packed, scales, sf_vec_size=16)

    err = (x - x_recon).abs()
    rel_err = err.mean() / x.abs().mean()
    print(f"  Shape: {x.shape}")
    print(f"  Packed: {packed.shape}, Scales: {scales.shape}")
    print(f"  Mean abs error: {err.mean():.4f}")
    print(f"  Relative error: {rel_err:.1%}")
    print(f"  Max abs error: {err.max():.4f}")
    assert rel_err < 0.3, f"Quantization error too high: {rel_err:.1%}"
    print("  PASS\n")


def test_quantized_gemm_reference():
    """Test: quantize A and B, dequantize, do FP32 GEMM. This is the target."""
    print("=== Quantized GEMM Reference Test ===")
    torch.manual_seed(42)
    M, N, K = 1, 256, 4096  # Qwen3.5 FC1 gate shape at TP=4

    a_f32 = torch.randn(M, K, device="cuda") * 0.5
    b_f32 = torch.randn(N, K, device="cuda") * 0.5

    # FP32 reference
    c_ref = reference_gemm(a_f32, b_f32)

    # Quantized reference
    a_packed, a_sf = nvfp4_quantize(a_f32, sf_vec_size=16)
    b_packed, b_sf = nvfp4_quantize(b_f32, sf_vec_size=16)
    a_deq = nvfp4_dequantize(a_packed, a_sf, sf_vec_size=16)
    b_deq = nvfp4_dequantize(b_packed, b_sf, sf_vec_size=16)
    c_quant_ref = reference_gemm(a_deq, b_deq)

    # Compare
    fp32_err = (c_ref - c_quant_ref).abs().mean() / c_ref.abs().mean()
    print(f"  Shapes: A={a_f32.shape}, B={b_f32.shape}, C={c_ref.shape}")
    print(f"  FP32 ref[0:4]: {c_ref[0, :4].tolist()}")
    print(f"  Quant ref[0:4]: {c_quant_ref[0, :4].tolist()}")
    print(f"  FP32→Quant relative error: {fp32_err:.1%}")
    print(f"  (This is the FP4 quantization noise floor — kernel should match this)")
    print()

    return c_ref, c_quant_ref, a_packed, a_sf, b_packed, b_sf


if __name__ == "__main__":
    print("VerdictMoE Sprint 1: CuteDSL GEMM Validation\n")
    test_quantize_roundtrip()
    c_ref, c_quant_ref, a_packed, a_sf, b_packed, b_sf = test_quantized_gemm_reference()

    print("=== Next: CuteDSL GEMM kernel will be validated against c_quant_ref ===")
    print(f"Target: kernel output matches c_quant_ref within ~1% additional error")
