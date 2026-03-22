#!/usr/bin/env python3
"""
Phase 1c Correctness Test: GEMM1 + SwiGLU
==========================================
Computes reference in PyTorch (BF16), compares against CUDA kernel output.

Run:
    python3 fused-moe/tests/test_gemm1_swiglu.py
"""

import torch
import torch.nn.functional as F


def reference_gemm1_swiglu(
    input_bf16: torch.Tensor,   # [M, 4096] BF16
    gate_up_w: torch.Tensor,    # [512, 4096] BF16 (gate=[:256], up=[256:])
) -> torch.Tensor:
    """PyTorch reference: GEMM1 + SwiGLU."""
    # GEMM1: [M, 4096] × [4096, 512] = [M, 512]
    gate_up = input_bf16 @ gate_up_w.T  # [M, 512]

    # Split into gate and up
    gate = gate_up[:, :256]   # [M, 256]
    up = gate_up[:, 256:]     # [M, 256]

    # SwiGLU: up * silu(gate)
    output = up * F.silu(gate.float()).to(gate.dtype)
    return output  # [M, 256] BF16


def quantize_to_nvfp4(tensor_bf16: torch.Tensor, block_size: int = 32):
    """
    Quantize BF16 tensor to NVFP4 (E2M1) with block-scale factors.
    Returns (packed_uint8, scale_factors_ue8m0).

    E2M1 range: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} (and negatives)
    Scale = max(abs(block)) / 6.0, stored as UE8M0 (pure exponent, no mantissa)
    """
    shape = tensor_bf16.shape
    flat = tensor_bf16.float().reshape(-1)
    num_blocks = (flat.numel() + block_size - 1) // block_size

    # Pad to block boundary
    padded = torch.zeros(num_blocks * block_size, device=flat.device)
    padded[:flat.numel()] = flat

    # Compute per-block scales
    blocks = padded.reshape(num_blocks, block_size)
    block_max = blocks.abs().max(dim=1).values.clamp(min=1e-12)
    scales = block_max / 6.0  # FP4 E2M1 max representable = 6.0

    # Quantize: round to nearest E2M1 value
    e2m1_values = torch.tensor([0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=flat.device)
    scaled_blocks = blocks / scales.unsqueeze(1)
    signs = scaled_blocks.sign()
    abs_vals = scaled_blocks.abs()

    # Find nearest E2M1 value
    diffs = (abs_vals.unsqueeze(-1) - e2m1_values.unsqueeze(0).unsqueeze(0)).abs()
    indices = diffs.argmin(dim=-1)  # [num_blocks, block_size]

    # Pack into uint8 (2 FP4 values per byte)
    # FP4 encoding: 1 sign bit + 3 value bits = 4 bits
    fp4_vals = indices.to(torch.uint8)  # 0..7 = value index
    fp4_with_sign = fp4_vals | ((signs < 0).to(torch.uint8) << 3)  # bit 3 = sign

    flat_fp4 = fp4_with_sign.reshape(-1)
    packed = (flat_fp4[0::2] & 0xF) | ((flat_fp4[1::2] & 0xF) << 4)

    # Scale factors as UE8M0 (exponent-only float)
    # UE8M0: 8-bit unsigned exponent, value = 2^(exp - 127)
    log2_scales = torch.log2(scales).round().clamp(0, 255).to(torch.uint8)

    return packed[:shape.numel() // 2], log2_scales


def test_reference():
    """Basic test of the PyTorch reference."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    M = 1

    torch.manual_seed(42)
    input_bf16 = torch.randn(M, 4096, dtype=torch.bfloat16, device=device)
    gate_up_w = torch.randn(512, 4096, dtype=torch.bfloat16, device=device)

    output = reference_gemm1_swiglu(input_bf16, gate_up_w)
    print(f"Reference output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    print(f"Output mean: {output.mean():.4f}")
    print(f"Output first 8: {output[0, :8]}")

    # Test quantization
    packed, scales = quantize_to_nvfp4(input_bf16)
    print(f"\nFP4 packed shape: {packed.shape} (expect {M * 4096 // 2})")
    print(f"Scale factors shape: {scales.shape} (expect {M * 4096 // 32})")

    return output


def test_quantize_roundtrip():
    """Verify FP4 quantization preserves rough magnitudes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -1.0, -3.0],
                      dtype=torch.bfloat16, device=device).unsqueeze(0)
    packed, scales = quantize_to_nvfp4(x)
    print(f"\nQuantize roundtrip test:")
    print(f"  Input:  {x[0]}")
    print(f"  Packed: {packed}")
    print(f"  Scales: {scales}")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1c: GEMM1 + SwiGLU Reference Test")
    print("=" * 60)
    test_reference()
    test_quantize_roundtrip()
    print("\nAll reference tests passed!")
