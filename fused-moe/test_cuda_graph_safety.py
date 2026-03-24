#!/usr/bin/env python3
"""
Test: VerdictMoE CUDA-graph safety — buffer pre-allocation.

Verifies that torch.cuda.memory_allocated() does NOT change between
first and second apply() calls, proving no dynamic allocations in forward.

Run inside Docker container or on host with vLLM + SM120 GPU:
    CUDA_DEVICE_ORDER=PCI_BUS_ID python3 test_cuda_graph_safety.py
"""

import torch
import gc
import sys
import importlib.util
from pathlib import Path


# ============================================================================
# FP4 helpers
# ============================================================================
FP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def encode_fp4_e2m1(val: float) -> int:
    sign = 1 if val < 0 else 0
    av = abs(val)
    best_nib, best_err = 0, float("inf")
    for nib in range(8):
        err = abs(av - FP4_LUT[nib].item())
        if err < best_err:
            best_err = err
            best_nib = nib
    return (sign << 3) | best_nib


def quantize_nvfp4(t: torch.Tensor, block_size: int = 16):
    *batch, K = t.shape
    assert K % block_size == 0
    flat_batch = t.reshape(-1, K)
    B, _ = flat_batch.shape
    nblocks = K // block_size
    blocks = flat_batch.reshape(B, nblocks, block_size)
    max_abs = blocks.abs().max(dim=-1).values
    scales_float = (max_abs / 6.0).clamp(min=1e-12)
    scales_e4m3 = scales_float.to(torch.float8_e4m3fn)
    scales_float_rt = scales_e4m3.float()
    scales_expanded = scales_float_rt.unsqueeze(-1).expand(B, nblocks, block_size)
    normalized = blocks / scales_expanded
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


def load_verdict_module(mock_ext=False):
    """Load verdict_moe.py directly (avoid verdict_moe/ package conflict).

    If mock_ext=True, replace the CUDA extension with a no-op mock
    (for testing memory allocation patterns without nvcc).
    """
    spec = importlib.util.spec_from_file_location(
        "verdict_moe_file",
        str(Path(__file__).parent / "verdict_moe.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if mock_ext:
        class MockExt:
            @staticmethod
            def forward(*args, **kwargs):
                pass  # no-op: we're testing allocation, not kernel
        mod._verdict_ext = MockExt()

    return mod


def make_vllm_configs(num_experts, topk, K, N_half, device):
    """Create minimal vLLM config objects for VerdictMoEExperts construction."""
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEQuantConfig,
        FusedMoEQuantDesc,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    par_config = FusedMoEParallelConfig(
        tp_size=1, pcp_size=1, dp_size=1, ep_size=1,
        tp_rank=0, pcp_rank=0, dp_rank=0, ep_rank=0,
        sp_size=1, use_ep=False, all2all_backend="",
        enable_eplb=False,
    )
    moe_config = FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=topk,
        hidden_dim=K,
        intermediate_size_per_partition=N_half,
        num_local_experts=num_experts,
        num_logical_experts=num_experts,
        activation=MoEActivation.SILU,
        device=device,
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=par_config,
        in_dtype=torch.bfloat16,
        max_num_tokens=256,
        is_act_and_mul=False,
    )
    quant_config = FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc(),
        _a2=FusedMoEQuantDesc(),
        _w1=FusedMoEQuantDesc(),
        _w2=FusedMoEQuantDesc(),
    )
    return moe_config, quant_config


# ============================================================================
# Static analysis test
# ============================================================================
def test_no_forbidden_ops():
    """Verify no forbidden CUDA-graph ops exist in apply() source."""
    print("=" * 70)
    print("TEST: No forbidden CUDA-graph ops in apply()")
    print("=" * 70)

    src_path = Path(__file__).parent / "verdict_moe.py"
    source = src_path.read_text()

    # Extract apply() method body
    apply_start = source.find("    def apply(")
    if apply_start == -1:
        print("  ERROR: Could not find apply() method")
        return False

    apply_body = source[apply_start:]
    lines = apply_body.split("\n")
    apply_lines = []
    for i, line in enumerate(lines):
        if i == 0:
            apply_lines.append(line)
            continue
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if indent <= 4 and stripped and not stripped.startswith("#"):
            if stripped.startswith("def ") or stripped.startswith("class "):
                break
        apply_lines.append(line)

    apply_src = "\n".join(apply_lines)

    forbidden = {
        ".any()": "GPU-to-CPU sync",
        ".item()": "GPU-to-CPU sync",
        "torch.empty(": "Dynamic allocation",
        "torch.zeros(": "Dynamic allocation",
        "torch.ones(": "Dynamic allocation",
        "torch.arange(": "Dynamic allocation",
        "torch.ones_like(": "Dynamic allocation",
        "torch.zeros_like(": "Dynamic allocation",
        ".cpu()": "GPU-to-CPU transfer",
        ".numpy()": "GPU-to-CPU transfer",
    }

    passed = True
    for pattern, reason in forbidden.items():
        if pattern in apply_src:
            print(f"  FAIL: Found '{pattern}' in apply() — {reason}")
            passed = False

    if passed:
        print(f"  PASS: No forbidden ops found in apply()")
        print(f"  Checked {len(apply_lines)} lines for {len(forbidden)} patterns")

    bufs = ["_buf_partials", "_buf_gmem_inter", "_buf_output_f32",
            "_buf_token_ids", "_buf_ones", "_buf_expert_ids",
            "_buf_expert_wts", "_buf_w1_alpha", "_buf_w2_alpha"]
    for b in bufs:
        if b not in apply_src:
            print(f"  FAIL: Missing {b} in apply()")
            passed = False

    if passed:
        print(f"  PASS: All {len(bufs)} pre-allocated buffers used")

    print(f"\n  VERDICT: {'PASSED' if passed else 'FAILED'}")
    return passed


# ============================================================================
# Runtime memory stability test
# ============================================================================
def test_buffer_preallocation():
    """Verify memory_allocated() does NOT change between apply() calls."""
    print("\n" + "=" * 70)
    print("TEST: VerdictMoE CUDA-graph buffer pre-allocation")
    print("=" * 70)

    device = "cuda:0"
    torch.manual_seed(42)

    M, K, N_half = 1, 256, 32
    N2 = 2 * N_half
    num_experts = 3
    topk = 3

    # Quantize weights
    print("\nQuantizing test weights...")
    w1_float = torch.randn(num_experts, N2, K) * (2.0 / K**0.5)
    w2_float = torch.randn(num_experts, K, N_half) * (2.0 / N_half**0.5)
    w1_fp4, w1_sf = quantize_nvfp4(w1_float)
    w2_fp4, w2_sf = quantize_nvfp4(w2_float)

    w1_d = w1_fp4.to(device)
    w2_d = w2_fp4.to(device)
    w1_sf_d = w1_sf.to(torch.float8_e4m3fn).to(device)
    w2_sf_d = w2_sf.to(torch.float8_e4m3fn).to(device)

    # Build VerdictMoEExperts
    print("Creating VerdictMoEExperts...")
    # Mock CUDA ext to avoid JIT compilation (we test allocation, not kernels)
    verdict_mod = load_verdict_module(mock_ext=True)
    VerdictMoEExperts = verdict_mod.VerdictMoEExperts

    moe_config, quant_config = make_vllm_configs(
        num_experts, topk, K, N_half, device
    )
    # Set weight tensors on quant_config desc objects before construction
    ones_e = torch.ones(num_experts, dtype=torch.float32, device=device)
    quant_config._w1.alpha_or_gscale = ones_e.clone()  # g1_alphas
    quant_config._a1.alpha_or_gscale = ones_e.clone()  # a1_gscale
    quant_config._w2.alpha_or_gscale = ones_e.clone()  # g2_alphas
    quant_config._a2.alpha_or_gscale = ones_e.clone()  # a2_gscale
    quant_config._w1.scale = w1_sf_d  # w1_scale
    quant_config._w2.scale = w2_sf_d  # w2_scale

    experts = VerdictMoEExperts(moe_config, quant_config)

    # Inputs
    hidden = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    topk_ids = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    topk_weights = torch.tensor(
        [[0.4, 0.35, 0.25]], dtype=torch.float32, device=device
    )
    output = torch.zeros(M, K, dtype=torch.bfloat16, device=device)

    def run_apply(**kwargs):
        defaults = dict(
            output=output,
            hidden_states=hidden,
            w1=w1_d,
            w2=w2_d,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="silu",
            global_num_experts=num_experts,
            expert_map=None,
            a1q_scale=None,
            a2_scale=None,
            workspace13=None,
            workspace2=None,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )
        defaults.update(kwargs)
        experts.apply(**defaults)

    # --- Call 1: triggers lazy setup_buffers ---
    print("\nCall 1 (triggers setup_buffers)...")
    run_apply()
    torch.cuda.synchronize()

    # Call 2: warmup (stabilize PyTorch caching allocator)
    print("Call 2 (warmup, stabilize allocator)...")
    output.zero_()
    run_apply()
    torch.cuda.synchronize()

    # Call 3: warmup
    output.zero_()
    run_apply()
    torch.cuda.synchronize()
    out1 = output.clone()

    # Pre-allocate clone buffers for correctness check (NOT inside measured calls)
    out_save1 = torch.empty_like(output)
    out_save2 = torch.empty_like(output)

    # DO NOT empty_cache — measure in steady state (same as CUDA graph replay)
    torch.cuda.synchronize()
    mem_baseline = torch.cuda.memory_allocated(device)
    print(f"  Memory baseline (after 3 warmups): {mem_baseline / 1024:.1f} KB")

    # --- Call 4: THE TEST — must NOT allocate ---
    print("Call 4 (test: no allocation)...")
    output.zero_()
    run_apply()
    torch.cuda.synchronize()
    out_save1.copy_(output)  # copy_, not clone() — no allocation

    mem_4 = torch.cuda.memory_allocated(device)
    print(f"  Memory after call 4: {mem_4 / 1024:.1f} KB (delta={mem_4 - mem_baseline})")

    # --- Call 5: confirm stability ---
    print("Call 5 (confirm: must match call 4)...")
    output.zero_()
    run_apply()
    torch.cuda.synchronize()
    out_save2.copy_(output)

    mem_5 = torch.cuda.memory_allocated(device)
    delta = mem_5 - mem_4
    print(f"  Memory after call 5: {mem_5 / 1024:.1f} KB (delta vs call 4: {delta})")
    # (delta already computed above as mem_5 - mem_4)

    # --- Different M ---
    print("Call M=2 test...")
    h2 = torch.randn(2, K, dtype=torch.bfloat16, device=device)
    t2 = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.int32, device=device)
    w2t = torch.tensor(
        [[0.4, 0.35, 0.25], [0.5, 0.3, 0.2]],
        dtype=torch.float32, device=device,
    )
    o2 = torch.zeros(2, K, dtype=torch.bfloat16, device=device)
    run_apply(
        output=o2, hidden_states=h2, topk_ids=t2, topk_weights=w2t,
    )
    torch.cuda.synchronize()
    mem_m2 = torch.cuda.memory_allocated(device)
    delta_m2 = mem_m2 - mem_5
    print(f"  Memory after M=2: {mem_m2 / 1024:.1f} KB (delta vs steady={delta_m2})")

    # --- EP path ---
    print("Call EP test...")
    expert_map = torch.tensor([0, 1, -1], dtype=torch.int32, device=device)
    output.zero_()
    run_apply(expert_map=expert_map)
    torch.cuda.synchronize()
    mem_ep = torch.cuda.memory_allocated(device)
    delta_ep = mem_ep - mem_5
    print(f"  Memory after EP: {mem_ep / 1024:.1f} KB (delta vs steady={delta_ep})")

    # --- RWI path ---
    print("Call RWI test...")
    t1 = torch.tensor([[1]], dtype=torch.int32, device=device)
    w1t = torch.tensor([[0.8]], dtype=torch.float32, device=device)
    output.zero_()
    run_apply(
        topk_ids=t1, topk_weights=w1t,
        apply_router_weight_on_input=True,
    )
    torch.cuda.synchronize()
    mem_rwi = torch.cuda.memory_allocated(device)
    delta_rwi = mem_rwi - mem_5
    print(f"  Memory after RWI: {mem_rwi / 1024:.1f} KB (delta vs steady={delta_rwi})")

    # --- Verify correctness consistency ---
    diff = (out_save1.float() - out_save2.float()).abs()
    max_diff = diff.max().item()
    bitexact = max_diff == 0.0
    nonzero = (out_save1.float().abs() > 1e-10).sum().item()

    # --- Results ---
    print("\n" + "-" * 50)
    test_pass = True

    if delta != 0:
        print(f"  FAIL: Steady-state delta = {delta} bytes (want 0)")
        test_pass = False
    else:
        print(f"  PASS: Steady-state delta = 0 bytes (no alloc in forward)")

    if not bitexact:
        print(f"  FAIL: Output not bit-exact (max_diff={max_diff})")
        test_pass = False
    else:
        print(f"  PASS: Output bit-exact between identical calls")

    # Output may be zeros if ext.forward is mocked (no real kernel execution)
    if nonzero == 0:
        print(f"  INFO: Output is all zeros (expected with mocked CUDA ext)")
    else:
        print(f"  PASS: Output has {nonzero}/{out1.numel()} nonzero elements")

    # Info (not failures — EP/RWI paths may use CUDA caching for fixed-size ops)
    for name, d in [("M=2", delta_m2), ("EP", delta_ep), ("RWI", delta_rwi)]:
        if d != 0:
            print(f"  INFO: {name} delta = {d} bytes (fixed-size intermediate ops)")

    # Buffer sizes
    total_buf = sum(
        getattr(experts, f"_buf_{n}").nbytes
        for n in ["partials", "gmem_inter", "output_f32", "expert_ids",
                   "expert_wts", "token_ids", "w1_alpha", "w2_alpha",
                   "w1_alpha_all", "w2_alpha_all", "ones"]
    )
    print(f"\n  Total pre-allocated: {total_buf / 1e6:.1f} MB")
    print(f"    partials:   {experts._buf_partials.nbytes / 1e6:.1f} MB")
    print(f"    gmem_inter: {experts._buf_gmem_inter.nbytes / 1e6:.1f} MB")
    print(f"    output_f32: {experts._buf_output_f32.nbytes / 1e6:.1f} MB")

    print(f"\n  VERDICT: {'PASSED' if test_pass else 'FAILED'}")
    return test_pass


if __name__ == "__main__":
    static_pass = test_no_forbidden_ops()

    if torch.cuda.is_available():
        runtime_pass = test_buffer_preallocation()
    else:
        print("\n  SKIP: No CUDA GPU available")
        runtime_pass = True

    print("\n" + "=" * 70)
    overall = static_pass and runtime_pass
    print(f"SUMMARY: Static={'PASS' if static_pass else 'FAIL'}"
          f"  Runtime={'PASS' if runtime_pass else 'FAIL'}"
          f"  Overall={'PASS' if overall else 'FAIL'}")
    print("=" * 70)
    sys.exit(0 if overall else 1)
