#!/usr/bin/env python3
"""
Standalone correctness + benchmark test for VerdictMoE CUDA extension.

Tests the 3-kernel pipeline (GEMM1 distributed + SwiGLU reduce + GEMM2 scatter)
with NVFP4 weights against a PyTorch FP32 reference.

Run inside Docker container with nvcc + SM120 GPU:
    python3 /workspace/sm120-moe-bench/fused-moe/test_verdict_moe_ext.py
"""

import torch
import time
import sys
from pathlib import Path


def build_extension():
    """JIT-compile the CUDA extension."""
    from torch.utils.cpp_extension import load

    csrc_dir = Path(__file__).parent / "csrc"
    ext_src = csrc_dir / "verdict_moe_ext.cu"

    print(f"Compiling {ext_src}...")
    t0 = time.time()
    ext = load(
        name="verdict_moe_ext",
        sources=[str(ext_src)],
        extra_cuda_cflags=[
            "-gencode=arch=compute_120a,code=sm_120a",
            "-O2",
            "--expt-relaxed-constexpr",
            "-use_fast_math",
        ],
        verbose=False,
    )
    print(f"Compiled in {time.time()-t0:.1f}s")
    return ext


# ============================================================================
# FP4 E2M1 reference
# ============================================================================
FP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def encode_fp4_e2m1(val: float) -> int:
    """Encode float to 4-bit E2M1 (sign + 3-bit unsigned magnitude)."""
    sign = 1 if val < 0 else 0
    av = abs(val)
    best_nib = 0
    best_err = float("inf")
    for nib in range(8):
        err = abs(av - FP4_LUT[nib].item())
        if err < best_err:
            best_err = err
            best_nib = nib
    return (sign << 3) | best_nib


def quantize_nvfp4(t: torch.Tensor, block_size: int = 16):
    """
    Proper NVFP4 quantization:
    1. Compute per-block E4M3FN scale = max_abs / 6.0
    2. Divide values by scale, round to nearest FP4 E2M1
    3. Pack 2 FP4 per byte

    Returns: (packed_fp4 uint8, block_scales uint8/E4M3FN)
    """
    *batch, K = t.shape
    assert K % block_size == 0
    flat_batch = t.reshape(-1, K)
    B, _ = flat_batch.shape
    nblocks = K // block_size

    # Block scales: max_abs / 6.0 as E4M3FN
    blocks = flat_batch.reshape(B, nblocks, block_size)
    max_abs = blocks.abs().max(dim=-1).values  # [B, nblocks]
    scales_float = (max_abs / 6.0).clamp(min=1e-12)
    scales_e4m3 = scales_float.to(torch.float8_e4m3fn)
    scales_float_rt = scales_e4m3.float()  # round-tripped through E4M3FN

    # Normalize values by block scale, then quantize to FP4
    scales_expanded = scales_float_rt.unsqueeze(-1).expand(B, nblocks, block_size)
    normalized = blocks / scales_expanded  # now in ~[-6, 6] range

    # Quantize each value to nearest FP4 E2M1
    flat_norm = normalized.reshape(-1)
    n = flat_norm.numel()
    assert n % 2 == 0
    packed = torch.zeros(n // 2, dtype=torch.uint8)
    for i in range(0, n, 2):
        lo = encode_fp4_e2m1(flat_norm[i].item())
        hi = encode_fp4_e2m1(flat_norm[i + 1].item())
        packed[i // 2] = (hi << 4) | lo

    packed = packed.reshape(*batch, K // 2)
    scales_u8 = scales_e4m3.view(torch.uint8).reshape(*batch, nblocks)
    return packed, scales_u8


def dequant_nvfp4_ref(packed, scales_u8, alpha, out_shape, block_size=16):
    """Reference dequant: unpack FP4, multiply by block scale and alpha."""
    flat = packed.reshape(-1)
    n = flat.numel() * 2
    fp4_float = torch.zeros(n)
    for i in range(flat.numel()):
        byte = flat[i].item()
        lo_nib = byte & 0xF
        hi_nib = (byte >> 4) & 0xF
        lo_sign = -1.0 if (lo_nib & 8) else 1.0
        hi_sign = -1.0 if (hi_nib & 8) else 1.0
        fp4_float[2 * i] = lo_sign * FP4_LUT[lo_nib & 7].item()
        fp4_float[2 * i + 1] = hi_sign * FP4_LUT[hi_nib & 7].item()

    fp4_float = fp4_float.reshape(out_shape)
    *batch, K = out_shape
    nblocks = K // block_size
    scales_float = scales_u8.reshape(*batch, nblocks).view(torch.float8_e4m3fn).float()
    scales_expanded = scales_float.unsqueeze(-1).expand(*batch, nblocks, block_size)
    scales_expanded = scales_expanded.reshape(*batch, K)
    return fp4_float * scales_expanded * alpha


def test_correctness(ext):
    """Test VerdictMoE kernel output against FP32 reference."""
    print("\n" + "=" * 60)
    print("CORRECTNESS TEST: VerdictMoE 3-kernel pipeline")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda:0"

    # Use SMALLER shapes for faster Python-side quantization
    M, K, N_half = 1, 256, 32
    N2 = 2 * N_half
    num_experts = 3
    tiles_per_expert = 4  # K / COLS_PER_TILE = 256 / 64 = 4

    # Xavier-scaled weights: magnitude ~1/sqrt(K) so intermediate stays in E4M3FN range
    w1_float = torch.randn(num_experts, N2, K) * (2.0 / (K ** 0.5))
    w2_float = torch.randn(num_experts, K, N_half) * (2.0 / (N_half ** 0.5))
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    input_f32 = input_bf16.float().cpu()

    # --- Quantize weights to NVFP4 ---
    print(f"\nQuantizing weights to NVFP4 (K={K}, N_half={N_half}, {num_experts} experts)...")
    alpha = 1.0

    w1_fp4, w1_sf = quantize_nvfp4(w1_float)
    w2_fp4, w2_sf = quantize_nvfp4(w2_float)
    print(f"  w1_fp4: {w1_fp4.shape}, w1_sf: {w1_sf.shape}")
    print(f"  w2_fp4: {w2_fp4.shape}, w2_sf: {w2_sf.shape}")

    # Dequant reference
    w1_dequant = dequant_nvfp4_ref(w1_fp4, w1_sf, alpha, (num_experts, N2, K))
    w2_dequant = dequant_nvfp4_ref(w2_fp4, w2_sf, alpha, (num_experts, K, N_half))

    # Verify dequant produces non-zero values
    print(f"  w1_dequant range: [{w1_dequant.min():.4f}, {w1_dequant.max():.4f}]")
    print(f"  w2_dequant range: [{w2_dequant.min():.4f}, {w2_dequant.max():.4f}]")

    # Move to device
    w1_fp4_d = w1_fp4.to(device)
    w1_sf_d = w1_sf.to(device)
    w2_fp4_d = w2_fp4.to(device)
    w2_sf_d = w2_sf.to(device)

    # Routing
    expert_ids = torch.arange(num_experts, dtype=torch.int32, device=device)
    expert_wts = torch.ones(num_experts, dtype=torch.float32, device=device) / num_experts
    token_ids = torch.zeros(num_experts, dtype=torch.int32, device=device)
    w1_alpha_t = torch.full((num_experts,), alpha, dtype=torch.float32, device=device)
    w2_alpha_t = torch.full((num_experts,), alpha, dtype=torch.float32, device=device)

    # Buffers
    partials_size = num_experts * tiles_per_expert * 2 * N_half
    partials = torch.empty(partials_size, dtype=torch.float32, device=device)
    gmem_inter = torch.empty(num_experts * N_half, dtype=torch.float32, device=device)
    output_bf16 = torch.zeros(M, K, dtype=torch.bfloat16, device=device)
    output_f32 = torch.zeros(M, K, dtype=torch.float32, device=device)

    # Run kernel
    print("\nRunning VerdictMoE kernel...")
    ext.forward(
        input_bf16, w1_fp4_d, w1_sf_d, w1_alpha_t,
        w2_fp4_d, w2_sf_d, w2_alpha_t,
        output_bf16, expert_ids, expert_wts, token_ids,
        partials, gmem_inter, output_f32,
        K, N_half, num_experts, tiles_per_expert,
    )
    torch.cuda.synchronize()
    kernel_out = output_bf16.float().cpu()

    # FP32 reference with dequanted weights
    print("Computing FP32 reference (dequanted NVFP4 weights)...")
    ref_out = torch.zeros(M, K)
    for e in range(num_experts):
        w1_e = w1_dequant[e]  # [N2, K]
        w2_e = w2_dequant[e]  # [K, N_half]
        wt = expert_wts[e].item()

        # GEMM1: [1, K] @ [N2, K]^T = [1, N2]
        gemm1 = input_f32 @ w1_e.T
        gate = gemm1[:, :N_half]
        up = gemm1[:, N_half:]
        # SiLU(gate) * up
        swiglu = gate * torch.sigmoid(gate) * up

        # E4M3 requant (clamp to avoid NaN/overflow)
        swiglu_clamped = swiglu.clamp(-448.0, 448.0)
        inter_e4m3 = swiglu_clamped.to(torch.float8_e4m3fn).float()

        # GEMM2: [1, N_half] @ [K, N_half]^T = [1, K]
        gemm2 = inter_e4m3 @ w2_e.T

        ref_out += wt * gemm2

    # Compare
    print(f"\n  kernel[0:8] = {kernel_out[0, :8].tolist()}")
    print(f"  ref[0:8]    = {ref_out[0, :8].tolist()}")

    # Check for all-zeros
    kernel_nonzero = (kernel_out.abs() > 1e-10).sum().item()
    ref_nonzero = (ref_out.abs() > 1e-10).sum().item()
    print(f"  kernel nonzero: {kernel_nonzero}/{kernel_out.numel()}")
    print(f"  ref nonzero: {ref_nonzero}/{ref_out.numel()}")

    if ref_nonzero == 0:
        print("  WARNING: Reference is all zeros!")
        return False

    diff = (kernel_out - ref_out).abs()
    ref_norm = ref_out.abs().mean()
    norm_err = (diff.mean() / ref_norm).item() if ref_norm > 0 else 0
    max_err = diff.max().item()
    close = (diff < 0.1 * ref_out.abs().clamp(min=0.001)).sum().item()
    total = kernel_out.numel()

    print(f"  Normalized error: {norm_err*100:.4f}%")
    print(f"  Max abs error: {max_err:.6f}")
    print(f"  Within 10%: {close}/{total} ({100*close/total:.1f}%)")
    print(f"  NaN: {kernel_out.isnan().sum().item()}")

    # For FP4 quantized weights, expect ~5-20% normalized error (quantization noise)
    passed = norm_err < 0.5 and kernel_out.isnan().sum() == 0 and kernel_nonzero > 0
    print(f"\n  VERDICT: {'PASSED' if passed else 'FAILED'}")
    return passed


def test_benchmark(ext):
    """Benchmark the 3-kernel pipeline with full-size shapes."""
    print("\n" + "=" * 60)
    print("BENCHMARK: VerdictMoE 3-kernel pipeline")
    print("=" * 60)

    torch.manual_seed(42)
    device = "cuda:0"

    M, K, N_half = 1, 4096, 256
    N2 = 2 * N_half
    num_experts = 10
    tiles_per_expert = 64  # K / 64

    # Random NVFP4 weights (random bytes — correctness doesn't matter for timing)
    w1_fp4 = torch.randint(0, 256, (num_experts, N2, K // 2), dtype=torch.uint8, device=device)
    w1_sf = torch.randint(1, 128, (num_experts, N2, K // 16), dtype=torch.uint8, device=device)
    w2_fp4 = torch.randint(0, 256, (num_experts, K, N_half // 2), dtype=torch.uint8, device=device)
    w2_sf = torch.randint(1, 128, (num_experts, K, N_half // 16), dtype=torch.uint8, device=device)
    w1_alpha = torch.ones(num_experts, dtype=torch.float32, device=device)
    w2_alpha = torch.ones(num_experts, dtype=torch.float32, device=device)

    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    expert_ids = torch.arange(num_experts, dtype=torch.int32, device=device)
    expert_wts = torch.ones(num_experts, dtype=torch.float32, device=device) / num_experts
    token_ids = torch.zeros(num_experts, dtype=torch.int32, device=device)

    partials_size = num_experts * tiles_per_expert * 2 * N_half
    partials = torch.empty(partials_size, dtype=torch.float32, device=device)
    gmem_inter = torch.empty(num_experts * N_half, dtype=torch.float32, device=device)
    output_bf16 = torch.zeros(M, K, dtype=torch.bfloat16, device=device)
    output_f32 = torch.zeros(M, K, dtype=torch.float32, device=device)

    # Benchmark
    print(f"\nBenchmarking ({num_experts} experts, M={M}, K={K}, N_half={N_half})...")
    print("  (50 warmup, 200 iterations)")
    times = ext.benchmark(
        input_bf16, w1_fp4, w1_sf, w1_alpha,
        w2_fp4, w2_sf, w2_alpha,
        output_bf16, expert_ids, expert_wts, token_ids,
        partials, gmem_inter, output_f32,
        K, N_half, num_experts, tiles_per_expert,
        50, 200,
    )

    times = sorted(times)
    n = len(times)
    med = times[n // 2]
    avg = sum(times) / n
    p5 = times[int(n * 0.05)]
    p95 = times[int(n * 0.95)]

    print(f"\n  Results:")
    print(f"    Median:  {med:.1f} μs")
    print(f"    Average: {avg:.1f} μs")
    print(f"    P5-P95:  {p5:.1f} - {p95:.1f} μs")

    # Weight data loaded (theoretical minimum)
    w1_bytes = num_experts * N2 * K // 2  # FP4 packed
    w2_bytes = num_experts * K * N_half // 2
    total_weight_bytes = w1_bytes + w2_bytes
    print(f"\n  Weight data: {total_weight_bytes/1e6:.1f} MB ({w1_bytes/1e6:.1f} W1 + {w2_bytes/1e6:.1f} W2)")
    print(f"  Theoretical min @ 2TB/s: {total_weight_bytes/2e12*1e6:.1f} μs")

    print(f"\n  Comparison (per MoE layer, 10 experts, M=1):")
    print(f"    VLLM_CUTLASS (5 launches, NVFP4 MMA):   ~98 μs")
    print(f"    FlashInfer CUTLASS (7 launches):         ~130 μs")
    print(f"    VerdictMoE FP32 (1 coop launch, Task4):  38.9 μs")
    print(f"    VerdictMoE NVFP4 (4 launches, scalar):   {med:.1f} μs")
    if med > 0:
        print(f"    Speedup vs VLLM_CUTLASS: {98/med:.2f}x")

    return med


def test_per_kernel(ext):
    """Per-kernel timing breakdown."""
    print("\n" + "=" * 60)
    print("PER-KERNEL TIMING BREAKDOWN")
    print("=" * 60)

    device = "cuda:0"
    M, K, N_half = 1, 4096, 256
    N2 = 2 * N_half
    num_experts = 10
    tiles_per_expert = 64

    w1_fp4 = torch.randint(0, 256, (num_experts, N2, K // 2), dtype=torch.uint8, device=device)
    w1_sf = torch.randint(1, 128, (num_experts, N2, K // 16), dtype=torch.uint8, device=device)
    w2_fp4 = torch.randint(0, 256, (num_experts, K, N_half // 2), dtype=torch.uint8, device=device)
    w2_sf = torch.randint(1, 128, (num_experts, K, N_half // 16), dtype=torch.uint8, device=device)
    w1_alpha = torch.ones(num_experts, dtype=torch.float32, device=device)
    w2_alpha = torch.ones(num_experts, dtype=torch.float32, device=device)

    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    expert_ids = torch.arange(num_experts, dtype=torch.int32, device=device)
    expert_wts = torch.ones(num_experts, dtype=torch.float32, device=device) / num_experts
    token_ids = torch.zeros(num_experts, dtype=torch.int32, device=device)

    partials_size = num_experts * tiles_per_expert * 2 * N_half
    partials = torch.empty(partials_size, dtype=torch.float32, device=device)
    gmem_inter = torch.empty(num_experts * N_half, dtype=torch.float32, device=device)
    output_bf16 = torch.zeros(M, K, dtype=torch.bfloat16, device=device)
    output_f32 = torch.zeros(M, K, dtype=torch.float32, device=device)

    times = ext.per_kernel_timing(
        input_bf16, w1_fp4, w1_sf, w1_alpha,
        w2_fp4, w2_sf, w2_alpha,
        output_bf16, expert_ids, expert_wts, token_ids,
        partials, gmem_inter, output_f32,
        K, N_half, num_experts, tiles_per_expert,
    )

    names = ["memset", "gemm1_distributed", "swiglu_reduce", "gemm2_scatter", "f32_to_bf16"]
    total = sum(times)
    print(f"\n  {'Kernel':<22} {'Median μs':>10} {'%':>6}")
    print(f"  {'-'*40}")
    for name, t in zip(names, times):
        print(f"  {name:<22} {t:>10.1f} {100*t/total:>5.1f}%")
    print(f"  {'-'*40}")
    print(f"  {'TOTAL':<22} {total:>10.1f}")

    return times


if __name__ == "__main__":
    ext = build_extension()
    passed = test_correctness(ext)
    test_per_kernel(ext)
    med_us = test_benchmark(ext)
    print(f"\n{'='*60}")
    print(f"SUMMARY: Correctness={'PASS' if passed else 'FAIL'}, Median={med_us:.1f}μs")
    print(f"{'='*60}")
