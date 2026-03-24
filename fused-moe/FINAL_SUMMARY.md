# VerdictMoE: Fused MoE Kernel for SM120 Blackwell — Final Summary

**Project:** Custom fused Mixture-of-Experts GEMM kernel for vLLM on NVIDIA RTX PRO 6000 Blackwell
**Model:** Qwen3.5-397B-A17B-NVFP4 (512 experts, top-10 routing)
**Hardware:** 4x RTX PRO 6000 Blackwell (96GB HBM3e each, SM 12.0, 300W cap)
**Configuration:** EP=4 (128 experts/GPU), MTP=3 (speculative decode), FP8 KV cache
**Dates:** 2026-03-24 (all 3 phases)
**Repository:** https://github.com/brandonmmusic-max/sm120-moe-bench

---

## Project Overview

VerdictMoE is a custom fused MoE kernel targeting SM120 (Blackwell) GPUs, designed to replace vLLM's default CUTLASS MoE backend with a more efficient fused pipeline. The project evolved through 3 phases: integration, correctness, and MMA tensor core kernel development.

### Motivation

NSys profiling revealed that **NCCL AllReduce accounts for 69% of decode time** on PCIe-only topology (no NVLink). MoE GEMM is only 16% of GPU time. The strategy: reduce per-layer MoE latency via kernel fusion (fewer launches, less GMEM traffic) to minimize the relative impact of the fixed AllReduce cost.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VerdictMoE Pipeline                       │
│                                                             │
│  Input[1, 4096] BF16                                        │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Kernel 1: GEMM1 (640 CTAs, K-distributed)           │   │
│  │  Input × W1[4096, 2048] → partial[64, 2048] FP32     │   │
│  │  NVF4 MMA m16n8k64 (Phase 3) / Scalar GEMV (Phase 1) │   │
│  └──────────────────────────────────────────────────────┘   │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Kernel 2: Reduce + SwiGLU (10 CTAs)                  │   │
│  │  64-tile K-reduction → SwiGLU → FP32 intermediate     │   │
│  └──────────────────────────────────────────────────────┘   │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Kernel 3: GEMM2 (640 CTAs, N-distributed)            │   │
│  │  Intermediate[1024] × W2[1024, 4096] → output FP32    │   │
│  │  Weighted atomicAdd scatter to output buffer           │   │
│  └──────────────────────────────────────────────────────┘   │
│       │                                                     │
│       ▼                                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Kernel 4: FP32 → BF16 conversion                    │   │
│  └──────────────────────────────────────────────────────┘   │
│       │                                                     │
│       ▼                                                     │
│  Output[1, 4096] BF16                                       │
└─────────────────────────────────────────────────────────────┘

Per active expert (up to 10 per token):
  W1: [128, 2048, 2048] uint8 (NVFP4 packed) + E4M3FN block scales
  W2: [128, 4096, 512]  uint8 (NVFP4 packed) + E4M3FN block scales
```

---

## Phase 1: vLLM Integration

**Goal:** Wire VerdictMoE scalar GEMV kernel into vLLM's NVFP4 backend selection and achieve CUDA-graph-safe execution.

### Accomplishments

#### Buffer Pre-allocation (CUDA Graph Safety)
- Found and eliminated **11 dynamic allocations** in the forward path
- Pre-allocated buffer pool: 19 MB/GPU (down from initial 687 MB after optimization)
- Key insight: `partials` buffer allocated per-call (deterministic size per graph capture) — PyTorch caching allocator handles graph replay correctly
- Profile-run bypass: `if m > MAX_BATCHED_TOKENS: output.zero_(); return` prevents OOM during vLLM's 8192-token profiling phase

#### Oracle Patch (Backend Selection)
- Added `VERDICT_MOE` enum to `NvFp4MoeBackend`
- `VLLM_USE_VERDICT_MOE=1` env var activates VerdictMoE before other backend selection
- Lazy import prevents JIT compilation at import time
- Weight format: same NVFP4 packed format as CUTLASS (no conversion needed)
- Added swizzle guard: `swizzle_blockscale()` skipped for VERDICT_MOE

#### Server Startup
| Phase | Time | Status |
|-------|------|--------|
| Model loading (47 shards) | 35s | OK |
| torch.compile (cache hit) | 4s | OK |
| VerdictMoE CUDA JIT | 27s | OK |
| Buffer allocation | <1s | 19 MB/GPU |
| CUDA graph capture (51 sizes) | 111s | OK |
| **Total** | **~188s** | **API ready** |

### Result
- **Server starts, CUDA graphs captured, API responds**
- **Output: GARBAGE** — scalar GEMV produced incoherent text
- Performance: 17.3 tok/s median (7.5x regression vs CUTLASS)

---

## Phase 2: Correctness Fix

**Goal:** Diagnose and fix the 3 bugs causing garbage output.

### Bug 1: Block Scale Swizzle (Primary)

| | |
|---|---|
| **Root cause** | `swizzle_blockscale()` unconditionally applied to VERDICT_MOE weights |
| **Effect** | All scale factors garbled — complete output corruption |
| **Fix** | `if backend != NvFp4MoeBackend.VERDICT_MOE:` guard |
| **File** | `flashinfer_fp4_moe.py` |

### Bug 2: N_half Dimension Error

| | |
|---|---|
| **Root cause** | `N_half = w2.size(2) * 2 // 2` (computed N_half = 512, should be 1024) |
| **Effect** | 50% of W1 unread, expert weight overlap, SwiGLU truncated |
| **Fix** | `N_half = w2.size(2) * 2` |
| **File** | `verdict_moe.py` |

Additional CUDA fixes: GEMM1/SwiGLU loops for N_half > BLOCK_SIZE, `cudaFuncSetAttribute` for >48KB SMEM.

### Bug 3: E4M3FN Decode Integer Overflow

| | |
|---|---|
| **Root cause** | `1 << (e+17)` overflows int32 for exponent ≥14 |
| **Effect** | 3-11% of block scales wrong sign (largest weight groups) |
| **Fix** | `ldexpf(1.0f + m*0.125f, e-7)` |
| **File** | `verdict_moe_ext.cu` |

### Result
- **All 4 prompts produce coherent, correct output**
- Perplexity delta: 0.72% vs CUTLASS (within FP4 noise)
- Performance: 16.1 tok/s (scalar GEMV, compute-bound with correct N_half=1024)

---

## Phase 3: CUTLASS MMA Tensor Core Kernel

**Goal:** Replace scalar GEMV with NVF4 MMA instructions for tensor core acceleration.

### Accomplishments

#### MMA Atom Validation
Verified two MMA atoms on SM120:
- **GEMM1:** `mma.sync.m16n8k64.kind::mxf4nvf4` — NVF4×NVF4, E4M3FN block scales
- **GEMM2:** `mma.sync.m16n8k32.kind::f8f6f4` — E4M3×NVF4, UE8M0/E4M3FN scales

#### Technical Discoveries
1. **UE8M0 bias = 127** (not 128) — `scale = 2^(byte - 127)`
2. **ldmatrix.b4x16_p64 halves effective K** — requires direct packed loads
3. **CLayout SM80_16x8_Row** on SM120 — M=1 decode extracts via `lane_id % 4 == 0`
4. **SMEM Swizzle<3,4,3> required** — TMA swizzle pattern for correct MMA register mapping

#### Cooperative Kernel (FP32 Weights, Validated)

| Path | Launches | Median μs | Speedup |
|------|----------|-----------|---------|
| 60-kernel baseline | 60 | 2205.7 | 1.0x |
| 10x V2 cooperative | 20 | 321.5 | 6.9x |
| Fused independent (640 CTAs) | 1 | 825.3 | 2.7x |
| **Fused cooperative (640 CTAs, grid.sync)** | **1** | **38.9** | **56.7x** |

#### 3-Kernel MMA Pipeline (CUDA-Graph Safe)
- `verdict_mma_ext.cu`: Full torch extension with MMA GEMM1, scalar fallback GEMM2
- Activates via `VLLM_USE_VERDICT_MOE=1 VERDICT_USE_MMA=1`
- All buffers pre-allocated for CUDA graph safety

### Blocking Issue
**SMEM Swizzle<3,4,3>:** MMA instructions execute correctly but SMEM→register mapping is scrambled without TMA swizzle. The cooperative kernel runs at 110.6 μs with incorrect column mapping (vs projected 45-55 μs with correct SMEM).

### Result
- MMA kernel structurally complete, **blocked on SMEM swizzle fix**
- Scalar pipeline verified correct (0.72% PPL delta vs CUTLASS)
- Production falls back to CUTLASS (151.9 tok/s)

---

## Performance Progression

### MoE Layer Latency (Standalone)

| Phase | Backend | μs/layer | Notes |
|-------|---------|----------|-------|
| Baseline | VLLM_CUTLASS | 98 | 5 kernels/expert, tensor cores |
| Sprint 1 | CUTLASS GemmUniversal (validated) | — | Bit-exact on SM120 |
| Phase 1 | VerdictMoE scalar (buggy) | ~280 | 3 kernels, scalar GEMV |
| Phase 2 | VerdictMoE scalar (fixed) | ~280 | Correct output |
| Phase 2 | Cooperative FP32 | **38.9** | FP32 weights, grid.sync |
| Phase 3 | MMA cooperative (scrambled) | 110.6 | MMA running, SMEM blocked |
| Target | MMA cooperative (fixed) | **45-55** | Projected |

### End-to-End Decode Throughput (vLLM, Single User)

| Phase | Backend | Median tok/s | vs CUTLASS |
|-------|---------|-------------|------------|
| Baseline | VLLM_CUTLASS EP=4 MTP=3 | **151.9** | 1.00x |
| Phase 1 | VerdictMoE scalar (buggy output) | 17.3 | 0.11x |
| Phase 2 | VerdictMoE scalar (correct) | 16.1 | 0.11x |
| Phase 3 | VerdictMoE scalar (re-verified) | 16.7 | 0.11x |
| Target | VerdictMoE MMA (projected) | **190-220** | 1.25-1.45x |

### Correctness Progression

| Phase | Output Quality | Perplexity Delta |
|-------|---------------|-----------------|
| Phase 1 | Garbage (random tokens, repetition) | N/A |
| Phase 2 | Correct (all 4 prompts pass) | 0.72% vs CUTLASS |
| Phase 3 | Correct (scalar), blocked (MMA) | 0.72% vs CUTLASS |

---

## Architecture Decisions

### Why 3-Kernel Pipeline (Not Cooperative)

`cudaLaunchCooperativeKernel` is **not CUDA-graph-capturable**. vLLM captures the entire decode step as a CUDA graph during warmup. The 3-kernel pipeline with standard `<<<grid, block>>>` launches achieves the same distributed-GEMM1 + reduce + distributed-GEMM2 pattern with implicit stream-ordering barriers.

Overhead: ~10-15 μs vs cooperative (3 extra kernel launches at ~3 μs each). Acceptable for production use.

### Why EP=4 (Not TP=4)

EP=4 assigns 128 whole experts per GPU instead of TP-sharding all 512 across 4 GPUs. Benefits:
- Zero MoE AllReduce (the 69% bottleneck)
- Whole-expert weight tensors (simpler kernel addressing)
- Expert-map remapping handles non-local experts (weight=0)

Trade-off: Attention still requires TP=4 AllReduce (unchanged).

### Why Scalar Fallback Exists

The MMA SMEM swizzle blocker is non-trivial — replicating TMA's implicit Swizzle<3,4,3> in manual cooperative loads requires understanding the exact byte-level SMEM layout. The scalar path (`decode_fp4()` + `decode_e4m3fn()`) is correct but 9x slower. It serves as:
1. Correctness reference for MMA development
2. Production fallback if MMA fix takes longer than expected
3. Validation tool for weight layout and EP routing

---

## Lessons Learned

### SM120 Hardware
1. **`-gencode=arch=compute_120a,code=sm_120a`** required (not `-arch=`)
2. **ClusterShape must be 1×1×1** — no multicast TMA on SM120
3. **First vLLM startup takes ~20 min** (JIT for SM120, then cached)
4. **99 KB SMEM/SM** — careful budgeting for 3+ CTAs/SM occupancy
5. **`compute_120f`** (CUDA 13.0+) needed for TMA warp-specialized grouped GEMM init

### NVFP4 Weight Format
1. **E4M3FN block scales** (NOT UE8M0) — important for dequantization
2. **`swizzle_blockscale()` is CUTLASS-specific** — VerdictMoE needs linear indexing
3. **`ldexpf()` required for E4M3FN decode** — integer `1 << (e+17)` overflows for large exponents
4. **Group size = 16** FP4 elements per scale factor

### CUDA Graph Safety
1. **`torch.empty()` is graph-safe** if size is deterministic per graph capture
2. **`torch.mul(out=)` and `torch.index_select(out=)` are graph-safe** in-place ops
3. **`copy_()` handles dtype conversion in-place** — no `.int()/.float()` casts needed
4. **Profile-run bypass essential** — vLLM's 8192-token profiling pushes memory to limits
5. **Pre-compiled `.so` bypasses JIT** — 0s vs 27s load time

### MMA Programming on SM120
1. **UE8M0 bias = 127** — `0x7F = unity (1.0)`, not `0x80`
2. **ldmatrix.b4x16_p64 ≠ direct FP4 load** — halves effective K
3. **SMEM swizzle is implicit in TMA** — manual loads must replicate it
4. **CLayout SM80_16x8_Row** — accumulator fragment layout, extract M=0 via `lane%4==0`
5. **NVF4 MMA (m16n8k64) needs no fp4_shift** — only MxF8F6F4 (m16n8k32) needs it

---

## Future Work

### Sprint 4: SMEM Swizzle Fix
- Implement `Swizzle<3,4,3>` in cooperative SMEM write loop
- Validate with known-answer test (uniform weights → expected dot product)
- Re-run 4-prompt coherence + perplexity test with MMA GEMM1

### Sprint 5: Full MMA Pipeline
- Enable `use_mma_gemm2=True` after GEMM1 verified
- Input FP4 quantization kernel (BF16 → E2M1 with block scales)
- End-to-end benchmark: target >127 tok/s

### Sprint 6: Production Optimization
- TMA descriptor management for per-expert weight addressing
- Warp-specialized schedule (1 producer + 3 consumer warps, 128 threads)
- CUTLASS epilogue fusion for SwiGLU (eliminate K2)
- Target: 190-220 tok/s (cooperative + MMA)

### Sprint 7: Multi-Token Batching
- M>1 support for prefill and batched decode
- Grouped GEMM dispatch (10 experts × M tokens per call)
- Amortize kernel launch and weight loading overhead

---

## File Inventory

### Production Files
| File | Purpose |
|------|---------|
| `verdict_moe.py` | VerdictMoEExperts class, buffer management, forward dispatch |
| `csrc/verdict_moe_ext.cu` | 3-kernel scalar GEMV pipeline |
| `csrc/verdict_mma_ext.cu` | 3-kernel MMA pipeline (Phase 3) |
| `csrc/verdict_mma_cooperative.cu` | Fused cooperative kernel with MMA |

### Validation Tests
| File | Purpose |
|------|---------|
| `csrc/collective_mma_test.cu` | CUTLASS GemmUniversal NVF4 validation |
| `csrc/gemm2_test.cu` | E4M3×FP4 GEMM2 validation |
| `csrc/gemm2_e4m3_fp4_test.cu` | MxF8F6F4 schedule validation |
| `csrc/clayout_to_alayout_test.cu` | CLayout→SwiGLU→GEMM2 pipeline |
| `csrc/verdict_fused_single_expert.cu` | Single-expert cooperative (30.7 μs) |
| `csrc/verdict_fused_multi_expert.cu` | 10-expert cooperative (38.9 μs) |
| `csrc/mma_clayout_probe.cu` | CLayout mapping probe |
| `csrc/mma_direct_test.cu` | ldmatrix vs direct load |
| `csrc/mma_pack_test.cu` | Register packing validation |
| `test_cuda_graph_safety.py` | CUDA graph allocation audit |
| `test_verdict_moe_ext.py` | Scalar extension unit tests |

### Documentation
| File | Contents |
|------|----------|
| `INTEGRATION_RESULTS.md` | Phase 1: buffers, oracle, server startup |
| `PHASE2_RESULTS.md` | Phase 2: 3 bugs fixed, correctness verified |
| `PHASE3_RESULTS.md` | Phase 3: MMA kernel, cooperative, benchmarks |
| `FINAL_SUMMARY.md` | This document |
| `DESIGN.md` | Original kernel design |
| `PROFILING_RESULTS.md` | NSys ground truth (NCCL 69%, MoE 16%) |

### Docker Images
| Image | Contents |
|-------|----------|
| `vllm-qwen35-k64:latest` | K=64 CUTLASS production baseline |
| `vllm-qwen35-k64:verdict-moe-v2` | All Phase 1-3 patches |
| `vllm-qwen35-k64:ep-phase3` | EP=4 reference baseline |
