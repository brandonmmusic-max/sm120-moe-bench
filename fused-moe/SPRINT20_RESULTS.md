# Sprint 20: vLLM 0.18.x + FlashInfer 0.6.7 + VerdictMoE Integration

**Date:** 2026-03-29
**Hardware:** 4x NVIDIA RTX PRO 6000 Blackwell (96GB each, SM 12.0, PCIe)
**Model:** Qwen3.5-397B-A17B-NVFP4 (512 experts, top-10 routing, 60 layers: 45 GDN linear + 15 full attention)

---

## Executive Summary

Sprint 20 ports VerdictMoE to vLLM 0.18.x from main + 15 Repne PRs + DFlash PR 36767, with FlashInfer 0.6.7 and torch 2.12.0.dev+cu130. VerdictMoE produces **correct, coherent output** with the scale pre-multiply fix, but MTP acceptance regressed from Sprint 9's 65.9% to ~38-43% due to a draft/target numerical mismatch in the attention path.

### Key Results

| Config | Decode tok/s | MTP Accept | Pos 0 | Pos 1 | Pos 2 |
|--------|-------------|-----------|-------|-------|-------|
| Sprint 9 (0.17.x, torch 2.12, FI 0.6.6) | 165.1 | 65.9% | 84.9% | 64.9% | 48.1% |
| Sprint 20 torch 2.10 + FI 0.6.6 (strict) | 112.3 | 53.7% | 73.2% | 51.7% | 36.1% |
| Sprint 20 torch 2.12 + FI 0.6.7 | 107-115 | 38-43% | 88-98% | 18-47% | 0% |

### Root Cause: MTP Acceptance Drop

**Position 0 is actually HIGHER (88-98%) but Position 1-2 collapsed.** Two compounding issues:

1. **build_for_drafting override (PR 36060)**: Forces MTP draft tokens through FlashInfer PREFILL wrapper, while target model uses DECODE wrapper for the 15 full_attention layers. Different kernels → numerical mismatch → compounds across draft positions.

2. **GDN layers use vLLM FLA Triton kernels, NOT FlashInfer's optimized GDN kernels**: FlashInfer 0.6.7 has `gated_delta_rule_decode` and `run_mtp_decode` (MTP-specific GDN with proper state carry-forward), but vLLM routes all 45 GDN layers through its own FLA Triton ops. The MTP-specific state management is not being used.

---

## VerdictMoE Integration (9 Patches)

### The Critical Fix: g1_alphas Pre-Multiply

**Root cause of garbage output on 0.18.x:** `make_nvfp4_moe_quant_config()` changed between 0.17.x and 0.18.x:

```python
# Sprint 9 (0.17.x):
g1_alphas = a13_scale * w13_scale_2   # Pre-multiplied
g2_alphas = a2_scale * w2_scale_2

# 0.18.x (broken for VerdictMoE):
g1_alphas = w13_scale_2               # Raw, NOT multiplied
g2_alphas = w2_scale_2
```

VerdictMoE's kernel computes `w1_alpha_all = g1_alphas * a1_gscale = g1_alphas * (1/a13_scale)`.
- With pre-multiply: `(a13_scale * w13_scale_2) * (1/a13_scale) = w13_scale_2` ✓
- Without: `w13_scale_2 * (1/a13_scale) = w13_scale_2/a13_scale` ✗ (wrong by factor of a13_scale)

**Fix:** Add VERDICT_MOE branch in `make_nvfp4_moe_quant_config()` that pre-multiplies.

### All 9 Integration Points in oracle/nvfp4.py

| Line | What | Purpose |
|------|------|---------|
| 44 | Enum `VERDICT_MOE` | Backend enum |
| 105 | `backend_to_kernel_cls` | Maps to VerdictMoEExperts class |
| 129 | `map_nvfp4_backend` | String mapping |
| 250-251 | `select_nvfp4_moe_backend` | Env var `VLLM_USE_VERDICT_MOE=1` early check |
| 309 | `convert_to_nvfp4_moe_kernel_format` | Routes through weight prep |
| 387 | `make_nvfp4_moe_quant_config` | Pre-multiply g1/g2_alphas for VERDICT_MOE |
| 399 | `is_nvfp4_scale_swizzled=False` | VerdictMoE uses linear scale indexing |

### flashinfer_fp4_moe.py (2 patches)

| Line | What |
|------|------|
| 225 | Assertion: add VERDICT_MOE to allowed backends |
| 270 | Swizzle guard: `elif backend == VERDICT_MOE: pass` (skip swizzle_blockscale) |

### Other Patches

| File | Change |
|------|--------|
| `qwen3_5_mtp.py` | Unquantized MTP head (ReplicatedLinear, quant_config=None) — Sprint 9 behavior |
| `pynccl_allocator.py` | NCCL SymmMem linker fix |
| `verdict_moe.py` | `_supports_parallel_config` compat: getattr for use_all2all_kernels |

---

## Build Configuration

### Docker Image: `vllm-qwen35-k64:repne-main`

| Component | Version |
|-----------|---------|
| vLLM | 0.18.1rc1.dev346 (main + 15 Repne PRs + DFlash 36767) |
| torch | 2.12.0.dev20260313+cu130 (CUDA 13.0 runtime, cuDNN 9.20) |
| FlashInfer | 0.6.7 (flashinfer-python + flashinfer-cubin) |
| Triton | 3.6.0 |
| NCCL | 2.29.7 (pip) / 2.29.3 (torch bundled) |
| Base image | voipmonitor/llm-pytorch-blackwell:nightly-cuda132 |

### PRs Merged (15 Repne + 1 DFlash)

Correctness: 35687, 36138, 37831, 37152, 37795, 35936, 38039, 37170, 37236
Performance: 37865, 36060, 37700, 37110, 38020, 36317
DFlash: 36767 (base, conflicts resolved)

### Key PR Effects

| PR | Files | Impact |
|----|-------|--------|
| 36060 | flashinfer.py | `build_for_drafting` — forces prefill for MTP draft on non-TRTLLM (HURTS MTP acceptance) |
| 37700 | fla/ops/chunk_o.py, utils.py | SM120 FLA fixes: BKV_LIST=[32,64,128], Hopper detection excludes SM12x |
| 37170 | spec_decode/eagle.py | prompt_embeds support for spec decode |
| 37110 | triton_attn.py, fp8_utils.py | Attention quant fusion |

---

## Debugging Findings

### torch 2.10 vs 2.12

| Aspect | torch 2.10.0+cu128 | torch 2.12.0.dev+cu130 |
|--------|--------------------|-----------------------|
| CUDA runtime | 12.8 | 13.0 |
| cuDNN | older (pre-9.7 fixes) | 9.20 (FP8 fusion fix, BF16 SDPA fix) |
| Triton | (from pip) | 3.6.0 |
| MTP acceptance | 50-54% | 38-43% (with FI 0.6.7) |
| VerdictMoE | Works | Works |

torch 2.12 is required to match Sprint 9's base image. vLLM 0.18.x specifies torch==2.10.0 in requirements but compiles and runs fine against 2.12 (all APIs present).

### OOM Issues

GPU 1 has Cosmic desktop processes (workspace manager, terminal, xdg-portal) consuming ~3.8 GiB that other GPUs don't have. At `gpu-memory-utilization=0.92`, CUDA graph capture OOMs on GPU 1. Fix: use 0.90.

### SymmMem AllReduce

`VLLM_USE_NCCL_SYMM_MEM=1` causes NCCL errors with FlashInfer 0.6.7 (NCCL version interaction). Disabled with `VLLM_ALLREDUCE_USE_SYMM_MEM=0`. Need to debug separately.

### FlashInfer 0.6.7 GDN Kernels Not Used

vLLM routes all 45 GDN linear_attention layers through its own FLA Triton kernels:
- Non-spec decode: `fused_recurrent_gated_delta_rule_packed_decode` (vLLM FLA)
- Spec/MTP decode: `fused_sigmoid_gating_delta_rule_update` (vLLM FLA)
- Prefill: `chunk_gated_delta_rule` (FlashInfer when `gdn_prefill_backend=flashinfer`)

FlashInfer 0.6.7 has `gated_delta_rule_decode` and `run_mtp_decode` (PR #2618, #2521) but vLLM doesn't call them. **Wiring these in is the next step to fix MTP acceptance.**

---

## Layer Architecture

```
Qwen3.5-397B: 60 layers, pattern repeats every 4:
  Layer 0: linear_attention (GDN) → vLLM FLA Triton kernel
  Layer 1: linear_attention (GDN) → vLLM FLA Triton kernel
  Layer 2: linear_attention (GDN) → vLLM FLA Triton kernel
  Layer 3: full_attention       → FlashInfer (decode or prefill wrapper)
  ... (repeats 15 times)

MTP head: 1 decoder layer (full_attention + MoE)
  Attention: FlashInfer via build_for_drafting (PREFILL on non-TRTLLM)
  MoE: VerdictMoE fused kernel

Target verify uses DECODE wrapper for full_attention.
MTP draft uses PREFILL wrapper (build_for_drafting).
This mismatch causes Position 1-2 acceptance collapse.
```

---

## Next Steps

1. **Wire FlashInfer GDN decode into vLLM** — Replace vLLM's FLA Triton kernels with FlashInfer's `gated_delta_rule_decode` and `run_mtp_decode` for GDN layers. This ensures draft and target use the same kernel with proper MTP state management.

2. **Fix build_for_drafting** — Either remove it (earlier OOM was from GPU 1 desktop, not kernel issue) or make both draft and target use the same attention path.

3. **Fix SymmMem AllReduce** — Debug NCCL error with FlashInfer 0.6.7 + NCCL 2.29.7.

4. **NVFP4 KV Cache** — FlashInfer 0.6.7 has `nvfp4_kv_quantize`/`nvfp4_kv_dequantize`. Wire into vLLM as `--kv-cache-dtype nvfp4` for 2x context length. VerdictMoE's `decode_fp4`/`decode_e4m3fn`/`read_nvfp4` can be reused in SM120 attention kernel.

5. **DFlash speculative decoding** — PR 36767 merged. z-lab/Qwen3.5-9B-DFlash model downloaded, config remapped for 60-layer target. Test with `--speculative-config '{"method":"dflash","model":"/draft-model","num_speculative_tokens":16}'`.

---

## Docker Image State

Current `vllm-qwen35-k64:repne-main` contains:
- vLLM 0.18.1rc1.dev346 compiled against torch 2.12.0.dev+cu130
- FlashInfer 0.6.7 (flashinfer-python + flashinfer-cubin)
- VerdictMoE with all 9+2 oracle/fp4 patches
- Unquantized MTP head (ReplicatedLinear)
- build_for_drafting RESTORED (prefill path for MTP draft)
- pynccl fix
- VLLM_ALLREDUCE_USE_SYMM_MEM=0

Launch script: `~/klc-linux/run_vllm_sprint20.sh`
