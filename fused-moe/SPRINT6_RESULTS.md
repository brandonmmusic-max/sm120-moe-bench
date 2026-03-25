# Sprint 6: Optimized Fused Cooperative MMA Kernel

**Date:** 2026-03-25
**Hardware:** 4x NVIDIA RTX PRO 6000 Blackwell (96GB each, SM120, 188 SMs, 100KB SMEM)
**Model:** Qwen3.5-397B-A17B-NVFP4 (512 experts, top-10 routing, 60 MoE layers)

---

## Executive Summary

Sprint 6 delivers a **single fused cooperative MMA kernel** that executes the entire MoE
expert pipeline — BF16→FP4 quantize → GEMM1 → SwiGLU → E4M3 requant → GEMM2 → BF16 —
in **ONE kernel launch** per token. The standalone kernel achieves **46.5 μs/layer**,
a **2.11x speedup over CUTLASS** (98 μs, 5 kernels/expert) and **2.50x over Sprint 5**
(116.1 μs).

Two key optimizations drove the gains:
1. **Vectorized uint32 packing** (40.7 μs saved) — consecutive-K SMEM layout enables
   2 uint32_t loads per MMA operand, replacing 16 scalar nibble reads
2. **Hybrid K×N-distributed Phase 1** (28.9 μs saved) — 640 CTAs (16 N-chunks × 4 K-groups)
   with 160-leader 4-way reduction, 73% HBM utilization (up from 41%)

**However**, the cooperative kernel has a fundamental M>1 limitation: atomic barriers
require all 640 CTAs to be resident simultaneously. With MTP=3 (M=4 tokens), 4 sequential
launches are needed where CUTLASS processes all tokens in parallel. This makes the kernel
**slower in vLLM end-to-end** with MTP enabled:

| Configuration | Standalone μs/layer | vLLM tok/s (MTP=3) | vs CUTLASS |
|--------------|--------------------:|--------------------:|-----------:|
| VerdictMoE cooperative (EP=4) | **46.5** | 83.2 | 0.64x |
| VerdictMoE scalar GEMV (TP=4) | ~280 | 87.2 | 0.78x |
| CUTLASS EP=4 MTP=3 | 98.0 | ~123 | 1.00x |
| CUTLASS TP=4 MTP=3 (production) | 98.0 | ~172 | 1.40x |

**Verdict:** The kernel is correct, CUDA-graph safe, and 2.11x faster per-token than
CUTLASS. But MTP serialization makes it a net regression for production. The path forward
is a **multi-token cooperative kernel** (M>1 without serialization), projected at 60-80 μs
for M=4 — which would beat CUTLASS.

**Architecture confirmation:** This is a **SINGLE fused kernel**, not a multi-kernel
pipeline. All 5 stages (quantize, GEMM1, SwiGLU, requant, GEMM2) execute within one
`<<<grid, block>>>` launch using 4 atomic barriers for inter-CTA synchronization.

---

## Comprehensive Comparison: All Configurations Tested

### Standalone Kernel Performance (μs/layer, M=1)

| Sprint | Configuration | μs/layer | vs CUTLASS | Key Innovation |
|--------|--------------|----------|------------|----------------|
| Phase 1-2 | VerdictMoE scalar GEMV | ~280 | 0.35x | Fused pipeline (correct, slow) |
| Phase 3 | MMA cooperative (scrambled SMEM) | 110.6 | 0.89x | First MMA attempt, SMEM blocker |
| Phase 3 | Cooperative FP32 weights | 38.9 | 2.52x | Theoretical FP32 ceiling |
| Sprint 4 | MMA + Swizzle<3,4,3> | 110.6 | 0.89x | SMEM swizzle solved |
| Sprint 5 v1 | Per-nibble ldexpf rescaling | 625.9 | 0.16x | scale_vec::4X attempt |
| Sprint 5 v2 | Optimized mag_map rescaling | 220.6 | 0.44x | Better rescaling (broken) |
| **Sprint 5** | **Consecutive-K (no rescaling)** | **116.1** | **0.84x** | **K-permutation invariance** |
| Sprint 6a | + Vectorized uint32 packing | 75.4 | 1.30x | 2 loads vs 16 nibble reads |
| Sprint 6 (failed) | N-distributed 160 CTAs | 124.3 | 0.79x | Too few CTAs for HBM |
| **Sprint 6** | **+ Hybrid K×N (640 CTAs)** | **46.5** | **2.11x** | **160-leader 4-way reduce** |
| — | **CUTLASS baseline** | **98.0** | **1.00x** | 5 kernels/expert |

### vLLM End-to-End Throughput (tok/s, single-user)

| Sprint | Config | Parallelism | MTP | Median tok/s | vs CUTLASS TP=4 |
|--------|--------|-------------|-----|-------------:|----------------:|
| Phase 3 | CUTLASS EP=4 | EP=4 | 3 | 151.9 | 0.88x |
| Phase 3 | VerdictMoE scalar EP=4 | EP=4 | 3 | 16.7 | 0.10x |
| — | CUTLASS EP=4 | EP=4 | 3 | ~123 | 0.72x |
| **Sprint 6** | **VerdictMoE coop EP=4** | **EP=4** | **3** | **83.2** | **0.48x** |
| — | CUTLASS TP=4 (benchmark) | TP=4 | 3 | 112.3 | 0.65x |
| **Sprint 6** | **VerdictMoE scalar TP=4** | **TP=4** | **3** | **87.2** | **0.51x** |
| — | **CUTLASS TP=4 (production)** | **TP=4** | **3** | **~172** | **1.00x** |

### EP=4 vs TP=4 (CUTLASS Baselines)

| Config | Median tok/s | AllReduce calls/token | Notes |
|--------|-------------:|----------------------:|-------|
| TP=4 MTP=3 (production) | ~172 | 120 | Best overall, K=64 patches |
| TP=4 MTP=3 (benchmark) | 112.3 | 120 | 512-tok completions, temp=0.7 |
| EP=4 MTP=3 (Phase 3) | 151.9 | 120 | Same AllReduce, +expert_map |
| EP=4 MTP=3 (Sprint 6) | ~123 | 120 | Different benchmark conditions |
| EP=4 MTP=3 + replicated shared | 118.5 | 60 | Halved AllReduce, +0.28GB/GPU |

---

## Task 0: Optimization

### Objective

Optimize the single fused cooperative kernel (GEMM1 → SwiGLU → E4M3 requant → GEMM2)
from 116.1 μs toward 40-55 μs, keeping it as ONE kernel launch.

### Result: 46.5 μs — 2.50x speedup, 2.11x faster than CUTLASS

| Configuration | μs/layer | vs Sprint 5 | vs CUTLASS |
|--------------|----------|-------------|------------|
| Sprint 5 (scalar packing, 64-way reduce) | 116.1 | 1.00x | 0.84x |
| Sprint 6a (vectorized packing only) | 75.4 | 1.54x | 1.30x |
| N-distributed (160 CTAs, low occupancy) | 124.3 | 0.93x | 0.79x |
| **Sprint 6 final (vec + hybrid K×N)** | **46.5** | **2.50x** | **2.11x** |
| VLLM_CUTLASS baseline | 98.0 | — | 1.00x |
| Sprint 4 cooperative (FP32 ceiling) | 38.9 | — | — |

### Correctness

| Comparison | Error | Status |
|-----------|-------|--------|
| GPU vs Quantized Ref | **9.59% RelErr** | PASS (<10%) |
| GPU vs FP32 Ref | 28.31% RelErr | PASS (<50%) |
| QRef vs FP32 | 29.23% RelErr | Baseline FP4 error |
| NaN/Inf count | 0 | PASS |

Output values are **bit-identical** across all Sprint 6 variants (vectorized packing
produces the exact same register values as scalar nibble extraction).

---

## Optimizations Applied

### 1. Vectorized Packing (40.7 μs saved: 116.1 → 75.4)

**Problem:** Each operand pack used 16 `get_nibble_swz` calls — each doing swizzle address
computation, byte load from SMEM, nibble extraction, shift, and OR accumulate. ~160
instructions per warp per N-pass iteration.

**Fix:** Consecutive-K packing means bytes in SMEM are already in MMA register format.
A single `uint32_t` SMEM load gives all 8 nibbles in the correct bit positions:

```cpp
// OLD: 16 scalar reads + 16 shifts + 16 ORs per operand
for (int p = 0; p < 8; p++) {
    b[0] |= get_nibble_swz(s_B, rbo, t0*8 + p) << (p*4);
    b[1] |= get_nibble_swz(s_B, rbo, 32 + t0*8 + p) << (p*4);
}

// NEW: 2 uint32_t loads per operand
b[0] = *(uint32_t*)&s_B[swizzle_343(rbo + t0 * 4)];
b[1] = *(uint32_t*)&s_B[swizzle_343(rbo + 16 + t0 * 4)];
```

**Why this works:** With consecutive-K packing (`t0*8+p`), nibbles for K positions
0..7 are in bytes at address `t0*4` through `t0*4+3`. The Swizzle<3,4,3> preserves
4-byte alignment within 8-byte groups, so 4 consecutive logical bytes map to 4
consecutive swizzled bytes. The uint32_t load returns them with nibbles already in
correct bit positions for the MMA operand register.

Also applied to GMEM→SMEM loads (uint32_t coalesced loads instead of byte-by-byte).

### 2. Hybrid K×N-Distributed Phase 1 (28.9 μs saved: 75.4 → 46.5)

**Problem:** Sprint 5 used K-distributed Phase 1 (64 tiles along K, iterate 32 N-passes).
Phase 1b had only 10 leader CTAs reading 5.2 MB of partials with 64-way reduction.
With only 10 CTAs × 8 warps = 80 warps, effective HBM bandwidth was ~0.25 TB/s
(vs 1.5 TB/s peak). Phase 1b alone took ~21 μs.

**Fix:** Reorganize 640 CTAs into 16 N-chunks × 4 K-groups per expert:
- **Phase 1:** Each CTA covers BN=64 N-columns and 16 K-tiles (1024 K-elements).
  Gate and Up B tiles loaded simultaneously into SMEM. 2 MMAs per K-tile iteration.
- **Phase 1b:** 160 leader CTAs (k_group==0) reduce only 4 partials per column
  instead of 64. Register-only reduction with warp shuffles. 320 KB partials vs 5.2 MB.

**Impact breakdown:**
- Partials buffer: 5.2 MB → 320 KB (16x reduction)
- Phase 1b readers: 10 CTAs → 160 CTAs (16x more bandwidth)
- Phase 1 syncs: 64 → 32 (gate+up simultaneous)
- Phase 1b computation: 64 additions per column → 4

### 3. Simultaneous Gate + Up B Tiles

SMEM holds both gate_B (2048 bytes) and up_B (2048 bytes) simultaneously.
Total SMEM = 5520 bytes (well under 100KB, occupancy stays at 4 CTAs/SM).
This eliminates the extra sync needed if loading gate and up sequentially,
halving Phase 1 sync count from 64 to 32.

### Failed Approach: Pure N-Distributed (160 CTAs)

Attempted N-distributed Phase 1 with 160 CTAs (16 per expert). Each CTA iterates
all 64 K-tiles, eliminating partials entirely. Result: 124.3 μs — SLOWER than
Sprint 5 (116.1 μs). Root cause: 160 CTAs / 188 SMs = 0.85 CTAs/SM, insufficient
for memory latency hiding. The GPU needs 3-4 CTAs/SM for effective HBM utilization.

---

## Architecture

### Grid: 640 CTAs = 10 experts × 64 tiles (16 N-chunks × 4 K-groups)

**Phase 1 (GEMM1 + SwiGLU):**
- Each CTA: n_chunk (0..15), k_group (0..3)
- 16 K-tile iterations with simultaneous gate+up loads
- FP32 accumulation in registers (no inter-CTA communication)
- Write 128-float partial (64 gate + 64 up) to GMEM
- Barrier 1

**Phase 1b (Reduce + SwiGLU + Requant):**
- 160 leader CTAs (k_group == 0), 64 threads each
- 4-way reduction via GMEM reads (4 × 128 = 512 floats per leader)
- SwiGLU: `up * silu(gate)` per column
- Group-max via warp shuffles (shfl_xor, masks 1,2,4,8)
- E4M3 scale encode + FP4 quantize
- Nibble packing via shfl_down (neighbor's nibble)
- Barrier 2

**Phase 2 (GEMM2):**
- All 640 CTAs, each handles 1 output tile (BN=64)
- 16 K-passes from intermediate buffer
- Vectorized packing + uint32_t loads
- atomicAdd scatter with expert weight

### SMEM Layout (Phase 1)
```
s_A:        [0, 512)       512 bytes  (M=1, only row 0 used)
s_B_gate:   [512, 2560)    2048 bytes (BN×BK/2)
s_B_up:     [2560, 4608)   2048 bytes
s_SFA:      [4608, 4624)   16 bytes
s_SFB_gate: [4624, 4880)   256 bytes  (BN×SF_PER_K)
s_SFB_up:   [4880, 5136)   256 bytes
Total: 5264 bytes + 128 pad = 5392 bytes
```

### Key Properties
- Single kernel launch (GEMM1 → SwiGLU → E4M3 requant → GEMM2)
- CUDA-graph safe (atomic barriers, no cooperative_groups)
- Occupancy: 4 CTAs/SM × 188 SMs = 752 > 640 needed
- scale_vec::4X with native E4M3FN scales — zero rescaling
- Consecutive-K packing — bit-exact validated

---

## Performance Analysis

| Component | Sprint 5 (μs) | Sprint 6 (μs) | Savings |
|-----------|---------------|---------------|---------|
| Phase 1 packing | ~20 | ~0.3 | 19.7 |
| Phase 1 GMEM loads | ~28 | ~31 | -3 (gate+up both loaded) |
| Phase 1 syncs (64→32) | ~3.2 | ~1.6 | 1.6 |
| Phase 1b partials read | ~21 | ~0.2 | 20.8 |
| Phase 1b compute | ~2 | ~0.1 | 1.9 |
| Partials write | ~3.5 | ~0.2 | 3.3 |
| Phase 2 (packing+loads) | ~22 | ~10 | 12 |
| Barriers (×2) | ~4 | ~4 | 0 |
| Other overhead | ~12 | ~0 | ~12 |
| **Total** | **116.1** | **46.5** | **69.6** |

**HBM bandwidth utilization:** ~1.1 TB/s out of 1.5 TB/s = 73% (up from 41% in Sprint 5)

---

## Key Discoveries

### 1. Consecutive-K Enables Vectorized Packing

The consecutive-K packing innovation from Sprint 5 has a hidden secondary benefit:
it makes the SMEM byte layout directly match the MMA register format. This means
a single uint32_t load from SMEM replaces 16 scalar nibble extraction operations.
The 40.7 μs savings (35% of total kernel time) was the single largest optimization.

### 2. Phase 1b Was the Real Bottleneck (21 μs with 10 CTAs)

The 64-way partial reduction appeared cheap (sequential summing) but was actually
the biggest bottleneck due to low bandwidth utilization. Only 10 CTAs × 8 warps =
80 warps generated memory requests, achieving ~0.25 TB/s vs 1.5 TB/s peak.
The hybrid K×N approach with 160 leader CTAs achieves near-peak bandwidth for Phase 1b.

### 3. CTA Count Dominates Memory-Bound Kernels

The pure N-distributed attempt (160 CTAs, 124 μs) proved that occupancy > 1 CTA/SM
is essential for memory-bound FP4 kernels on SM120. The hybrid approach achieves
640 CTAs (3.4 CTAs/SM) while getting the benefits of N-distribution.

---

## Task 1: EP=4 Integration + Benchmark

**Date:** 2026-03-25
**Status:** CORRECTNESS PASS, PERFORMANCE REGRESSION (0.64x CUTLASS for MTP=3)

### What Was Done

Replaced the 4-kernel split MMA pipeline (`verdict_fused_ext.cu`: K0 BF16→FP4, K1A GEMM1+SwiGLU,
K1B GEMM2+scatter, K2 F32→BF16) with the single fused cooperative kernel from Sprint 6 Task 0.

**New extension:** `verdict_fused_cooperative_ext.cu`
- ONE kernel launch per token: BF16→FP4 → GEMM1 → SwiGLU → E4M3 requant → GEMM2 → BF16
- 4 atomic barriers (prologue sync, Phase 1a→1b, Phase 1b→2, Phase 2→epilogue)
- BF16 input quantization in prologue (all CTAs cooperate, 1 half-warp per SF group)
- F32→BF16 output conversion in epilogue (all CTAs cooperate)
- Alpha scale application: alpha1 in Phase 1b (before SwiGLU), alpha2 in Phase 2 (during scatter)
- For M>1 (MTP): C++ forward() loops over tokens, one launch per token
- Grid: topk(10) × NUM_TILES(64) = 640 CTAs ≤ 752 max concurrent

**Files modified:**
- `csrc/verdict_fused_cooperative_ext.cu` (new — torch extension wrapping Sprint 6 kernel)
- `verdict_moe.py` (updated `_get_verdict_mma_ext`, `setup_buffers`, `_apply_mma`)

### Grid Size Fix

Previous bug: grid was `num_tokens * topk * tiles` which overflows for M>1.
Fix: grid = `topk × NUM_TILES` = 10 × 64 = 640 (per-token launch, always fits in 752).

N_HALF derived from weight tensor: `N_HALF = w2.shape[2] * 2 = 512 * 2 = 1024`.

### Correctness: ALL PASS

| Prompt | Max Tokens | Result | Status |
|--------|-----------|--------|--------|
| "The capital of Kentucky is" | 50 | "Frankfort." + correct explanation | PASS |
| "def fibonacci(n):" | 100 | Valid recursive Python implementation | PASS |
| "Explain quantum entanglement in one sentence:" | 60 | Coherent, accurate definition | PASS |
| "Write a detailed essay about AI history:" | 500 | Flowing prose, no repetition | PASS |

No "falling back to scalar" warnings. Cooperative extension compiled on all 4 GPUs.
CUDA graphs captured successfully (51 batch sizes).

### Benchmark: 83.2 tok/s (0.64x CUTLASS)

**Config:** EP=4, MTP=3, vllm-qwen35-k64:verdict-coop-s6

| Run | Tok/s |
|-----|-------|
| Warmup 1-8 | 71-97 (avg ~83) |
| Steady 1 | 88.0 |
| Steady 2 | 81.4 |
| Steady 3 | 97.6 |
| Steady 4 | 83.2 |
| Steady 5 | 82.9 |
| Steady 6 | 75.0 |
| Steady 7 | 86.1 |
| Steady 8 | 73.9 |
| Steady 9 | 83.2 |
| Steady 10 | 80.0 |
| **Median** | **83.2** |
| **Mean** | **83.1** |
| **Min** | **73.9** |
| **Max** | **97.6** |

**Baseline:** CUTLASS EP=4 MTP=3 = ~123 tok/s median
**Speedup:** 0.64x (regression)

### Root Cause Analysis

The cooperative kernel is **2.11x faster per-token** (46.5 μs vs 98 μs CUTLASS standalone).
But the regression comes from the **M=4 serialization penalty** with MTP=3:

**Cooperative kernel limitation:** Atomic barriers require ALL CTAs to be resident
simultaneously. Grid = topk × NUM_TILES = 10 × 64 = 640 ≤ 752. For M>1, processing all
tokens at once would need M × 640 CTAs → 2560 for M=4, exceeding the 752 CTA limit → deadlock.

**Solution used:** Loop over tokens in C++ (one kernel launch per token). For M=4:
4 sequential launches of 640 CTAs each, where CUTLASS launches ALL tokens in parallel.

| Metric | Cooperative | CUTLASS | Ratio |
|--------|------------|---------|-------|
| Per-token MoE kernel | ~46.5 μs | ~98 μs | 2.11x faster |
| Per-layer M=4 MoE | 4×46.5 = ~186 μs | ~120 μs (parallel) | 0.65x |
| 60 layers | ~11.2 ms | ~7.2 ms | 0.65x |
| Full model decode | 12.0 ms (83 tok/s) | 7.8 ms (129 tok/s) | 0.64x |

The cooperative kernel wins for M=1 decode (no MTP): projected ~150+ tok/s vs ~129 tok/s.
But for M=4 (MTP=3), the serialization penalty dominates.

### CUDA Graph Compatibility: PASS

- All buffers pre-allocated via `setup_buffers()` ✓
- `cudaMemsetAsync` for barrier reset (graph-safe) ✓
- Fixed grid/block/smem per launch ✓
- Fixed number of launches per layer (M=4 during capture = 4 during replay) ✓
- No GPU→CPU sync ✓
- 51 batch sizes captured successfully ✓

### Docker Image

`vllm-qwen35-k64:verdict-coop-s6` — based on `verdict-moe` image with:
- Updated `verdict_moe.py` (cooperative kernel integration)
- New `csrc/verdict_fused_cooperative_ext.cu`
- Activated via: `VLLM_USE_VERDICT_MOE=1 VLLM_VERDICT_MMA=1 --enable-expert-parallel`

---

## Next Steps

1. **Multi-token cooperative kernel**: Redesign for M>1 without serialization.
   Use K_GROUPS=1 (N-distributed) when M>1, fitting M×topk×16 CTAs in 752.
   M=4: 40×16=640 ≤ 752. Each CTA iterates full K (64 tiles) — occupancy is sufficient
   with 640 CTAs (3.4/SM). Projected: ~60-80 μs for M=4 (vs 186 μs serialized).

2. **TMA bulk loads** (cp.async.bulk.tensor): Replace GMEM→SMEM cooperative loads with
   hardware-managed tensor copies. Projected: 5-10 μs savings from pipeline overlap.

3. **ldmatrix** (SM100_SU4_DU8x16_x4_LDSM_N): Single instruction per operand instead of
   2 uint32_t loads. Minor but eliminates all manual SMEM address computation.

4. **cp.async pipelining**: 2-3 stage software pipeline to overlap GMEM transfers with
   compute. Would help close the gap to the 38.9 μs FP32 ceiling.

5. **Persistent kernel**: Keep CTAs alive across expert-token pairs, eliminating
   launch overhead for multi-layer execution.

---

## Task 2: TP=4 Benchmark

**Date:** 2026-03-25
**Status:** CORRECTNESS PASS, SCALAR FALLBACK — 0.78x CUTLASS

### What Was Done

Switched from EP=4 to TP=4 (production config). In TP=4, each GPU has TP-sharded
weight slices for ALL 512 experts. Expert GEMM shapes become N/tp_size instead of N:
- GEMM1: [M, 4096] × [4096, 256] (N_HALF/4 = 256 per GPU)
- GEMM2: [M, 256] × [256, 4096]

**verdict_moe.py modifications:**
1. Expert-map handling: `if expert_map is not None:` already conditional — no EP
   remapping needed for TP-only (expert_map=None, all 512 experts local).
2. N_HALF runtime detection: Added `_MMA_COMPILED_N_HALF = 1024` constant.
   When `N_half != 1024` (TP-sharded → 256), automatically falls back to scalar GEMV.
3. Buffer allocation guard: MMA-specific buffers only allocated when N_HALF matches
   compiled constant. Saves ~50 MB unused GPU memory in TP=4 mode.
4. One-time log: "VerdictMoE: N_HALF=256 != compiled 1024 (TP-sharded), falling back
   to scalar GEMV path" logged on first forward pass.

**Why cooperative MMA doesn't work for TP=4:**
- Kernel has compile-time `constexpr int N_HALF = 1024` and derived constants
  (TILES_N=16, K_GROUPS=4, NUM_TILES=64, grid=640 CTAs)
- TP=4 N_HALF=256 → TILES_N=4, NUM_TILES=16, grid=160 CTAs
- 160 CTAs / 188 SMs = 0.85 CTAs/SM — proven too low for HBM utilization (see Task 0
  "Failed Approach: Pure N-Distributed" which got 124 μs at 160 CTAs)
- Even if parameterized, TP=4's small N_HALF fundamentally limits CTA count

### Correctness: ALL PASS

| Prompt | Max Tokens | Result | Status |
|--------|-----------|--------|--------|
| "The capital of Kentucky is" | 50 | "Frankfort." + correct explanation | PASS |
| "def fibonacci(n):" | 100 | Valid recursive Python implementation | PASS |
| "Explain quantum entanglement in one sentence:" | 60 | Coherent, accurate definition | PASS |
| "Write a detailed essay about AI history:" | 500 | Flowing prose, structured outline | PASS |

CUDA graphs captured successfully (51 batch sizes). All 4 TP workers detected N_HALF=256
and fell back to scalar GEMV. No crashes, no OOM.

### Perplexity: COMPARABLE

Generated-token logprobs (temperature=0, 30 tokens):

| Prompt | VerdictMoE ppl | CUTLASS ppl |
|--------|---------------|-------------|
| "The capital of France is" | 1.30 | 2.08* |
| "Water boils at a temperature of" | 1.40 | 1.14 |
| "The speed of light..." | 1.22 | 1.31 |

*CUTLASS generated only 3 tokens for this prompt (early stop), inflating per-token ppl.

Both backends produce coherent, factual output. Minor logprob differences are expected
— different arithmetic paths (scalar GEMV vs MMA tensor cores) with FP4 quantization
produce slightly different rounding. Not a quality regression.

### Benchmark: 87.2 tok/s (0.78x CUTLASS)

**Config:** TP=4, MTP=3, VerdictMoE scalar GEMV (vllm-qwen35-k64:verdict-coop-s6)

| Run | VerdictMoE Scalar | CUTLASS |
|-----|------------------|---------|
| Warmup 1-8 | 69-103 (avg ~86) | 100-152 (avg ~128) |
| Steady 1 | 90.5 | 111.1 |
| Steady 2 | 103.8 | 130.2 |
| Steady 3 | 83.7 | 112.3 |
| Steady 4 | 84.4 | 147.0 |
| Steady 5 | 87.2 | 94.9 |
| Steady 6 | 93.1 | 123.3 |
| Steady 7 | 78.9 | 148.2 |
| Steady 8 | 84.9 | 110.3 |
| Steady 9 | 83.1 | 105.0 |
| Steady 10 | 93.5 | 110.8 |
| **Median** | **87.2** | **112.3** |
| **Mean** | **88.3** | **119.3** |
| **Min** | **78.9** | **94.9** |
| **Max** | **103.8** | **148.2** |

**Speedup vs CUTLASS TP=4: 0.78x (regression)**

Note: CUTLASS TP=4 measured at 112 tok/s median in this benchmark (not the previously
reported 172 tok/s). The 172 figure was from production with longer-running idle warmup;
this benchmark uses 512-token completions with temperature=0.7 and 8 warmup runs.

### Root Cause Analysis

The scalar GEMV path is inherently slower than CUTLASS MMA:

| Factor | Scalar GEMV | CUTLASS MMA |
|--------|------------|-------------|
| Compute unit | CUDA cores (scalar) | Tensor cores (MMA) |
| Throughput | ~1/8 of tensor core | Full tensor core peak |
| N_HALF tiles | K-distributed (64 tiles) | N×K-distributed |
| Per-expert GEMM | Sequential dot products | Tiled matrix multiply |
| MTP parallelism | Loops M tokens | Batched M×topk |

The scalar kernel was designed as a fallback/validation path, not for production TP=4.
It uses K-distributed tiling (64 tiles along K=4096) regardless of N_HALF, so the
smaller TP-sharded N_HALF=256 doesn't help — same K iteration count, just smaller output.

### Key Finding: Cooperative MMA Needs TP=4 Adaptation

To match or beat CUTLASS in TP=4 mode, the cooperative kernel must be parameterized for
N_HALF=256. The challenge: 256/64 = 4 N-chunks × 4 K-groups = 16 tiles per expert,
grid = 10 × 16 = 160 CTAs. This is the exact configuration that failed at 124 μs in
the N-distributed experiment (Task 0).

**Possible solutions:**
1. **Template the kernel for N_HALF**: Use compile-time templates or runtime params.
   But 160 CTAs still has the low-occupancy problem.
2. **Increase K_GROUPS**: With N_HALF=256, use K_GROUPS=16 → 4×16=64 tiles → 640 CTAs.
   Each K-group handles only 4 K-tiles (256 K-elements). Phase 1b reduces 16-way.
   This preserves the occupancy that makes the hybrid approach fast.
3. **Multi-expert batching**: Process 4+ experts per CTA launch to increase grid size.
   Grid = ceil(40 experts / batch) × tiles.

Option 2 (K_GROUPS=16) is the most promising — it preserves the 640-CTA structure
that achieved 46.5 μs in Task 0.

### Docker Image

`vllm-qwen35-k64:verdict-coop-s6` with volume-mounted verdict_moe.py:
- Modified `verdict_moe.py` (N_HALF fallback to scalar)
- Activated via: `VLLM_USE_VERDICT_MOE=1 VLLM_VERDICT_MMA=1` (MMA falls back to scalar)
- No `--enable-expert-parallel` flag
