# Sprint 12: TMA Bulk Loads + PDL for VerdictMoE

## Summary

Ported TMA (cp.async.bulk.tensor.3d) weight loads and PDL (Programmatic Dependent Launch) into the production VerdictMoE cooperative kernel. **+13% throughput at C=4-8**, confirming the Sprint 11 standalone improvement now translates to E2E on driver 595.

## Background

- Sprint 11 found TMA was 11% faster standalone (16.0 μs vs 17.8 μs at M=1) on driver 595, but had a 3.1% E2E regression on driver 580/590 due to mbarrier overhead
- Driver 595.45.04 reduced mbarrier overhead significantly
- E2E retest was the #1 remaining optimization opportunity

## Changes

### New Files
- `csrc/verdict_fused_cooperative_tma_ext.cu` — TMA+PDL variant of the production kernel
  - Weight tile loads (GEMM1 gate, GEMM1 up, GEMM2 down) via `cp.async.bulk.tensor.3d.shared::cta`
  - mbarrier synchronization for TMA completion
  - PDL (`griddepcontrol.launch_dependents`) at kernel exit
  - SMEM rearranged: B tiles first (128-byte aligned for TMA)
  - No swizzle_343 on B operand reads (TMA loads linearly)
  - TMA descriptor creation via `cuTensorMapEncodeTiled` (host-side)
  - Both TMA and scalar-load paths in one extension module
- `csrc/verdict_dense_fp4.cu` — VerdictDense fused BF16→FP4 quantize + dense GEMM
  - Numerically identical to VerdictMoE (same quantize, MMA, Kahan)
  - Single kernel launch with atomic grid barrier
  - For attention projections (q/k/v/o_proj), bypasses scaled_fp4_quant

### Modified Files
- `verdict_moe.py` — TMA path integration
  - Lazy TMA descriptor creation (`_ensure_tma_descriptors`)
  - `VERDICT_USE_TMA=1` env var to enable (default on)
  - Falls back to scalar-load kernel if TMA source not found
  - Routes to `forward_tma` with TMA descriptor args

### Patch Files (for vLLM integration)
- `~/klc-linux/patches/nvfp4_utils_verdict_dense.py` — NvFp4LinearBackend.VERDICT_DENSE enum + bypass scaled_fp4_quant

## Benchmark Results

### Hardware
- 4x NVIDIA RTX PRO 6000 Blackwell (96GB GDDR7 each, SM 12.0, 99KB SMEM)
- TP=4, PCIe Gen5, driver 595.45.04, CUDA 13.2
- GPU clocks locked at 2100-3090 MHz, GPU 1 at 600W

### Model
- Qwen3.5-397B-A17B-NVFP4, FP8 KV cache, MTP=3
- vLLM 0.19.0, CUDA graphs (PIECEWISE)

### Decode Throughput (tok/s) — ctx=0

| Config | C=1 | C=2 | C=4 | C=8 |
|--------|-----|-----|-----|-----|
| **Baseline (scalar loads)** | **146.0** | **194.5** | **266.3** | **323.1** |
| b12x dense K=128 | 143.3 | 203.6 | 261.6 | 323.9 |
| b12x dense K=64 | 143.2 | 196.9 | 259.4 | 322.9 |
| VerdictDense v1 (fused quantize) | 144.2 | 197.7 | 271.6 | 314.9 |
| VerdictDense v2 (optimized) | 146.9 | 189.9 | 255.1 | 323.8 |
| **TMA+PDL** | **144.2** | **215.7** | **301.1** | **364.8** |

### TMA+PDL vs Baseline Delta

| Concurrency | Delta tok/s | Delta % |
|-------------|-----------|---------|
| C=1 | -1.8 | -1.2% |
| C=2 | **+21.2** | **+10.9%** |
| C=4 | **+34.8** | **+13.1%** |
| C=8 | **+41.7** | **+12.9%** |

### MTP Acceptance (TMA+PDL)
- Per-position: Pos0 ~81%, Pos1 ~59%, Pos2 ~43%
- Avg draft acceptance: ~62% (unchanged from baseline)
- Mean acceptance length: ~2.85

### TMA Standalone Kernel Timing (driver 595)

| Config | Baseline | TMA | Improvement |
|--------|----------|-----|-------------|
| M=1 TP=4 | 17.8 μs | 16.0 μs | **10.4%** |
| M=4 TP=4 | 44.4 μs | 40.3 μs | **9.2%** |

## b12x Dense GEMM Experiment (Negative Result)

Tested b12x (v0.7.2) as a dense FP4 GEMM replacement for attention projections:

- **b12x dense GEMM IS faster in isolation** (1.75x vs CUTLASS per the PR)
- **BUT it degrades MTP acceptance** from ~62% to ~60% due to different FP32 accumulation order
- The acceptance loss cancels out the speed gain → flat or slightly worse E2E
- K=64 tile variant didn't help acceptance either
- **Conclusion**: Dense GEMM backend must be numerically consistent with the MoE kernel for MTP. b12x uses CuTe DSL warp-MMA with different tiling/accumulation than FlashInfer CUTLASS.

## VerdictDense Experiment (Neutral Result)

Wrote a custom dense FP4 GEMM kernel with identical numerical path to VerdictMoE:
- Fused BF16→FP4 quantize + GEMM in single kernel launch
- Same MMA (mxf4nvf4 m16n8k64), swizzle_343, BK=64, Kahan reduction
- Bypasses vLLM's scaled_fp4_quant entirely

Results: At parity with FlashInfer CUTLASS (146.9 tok/s C=1, 323.8 C=8).
Both FlashInfer CUTLASS and VerdictDense are already consistent enough with VerdictMoE.
Dense GEMM is ~5% of decode time — not the bottleneck.

## SM120 Architecture Research

### What SM120 Lacks (vs SM100 datacenter Blackwell)
- No TMEM (256KB tensor memory)
- No tcgen05 (128x128x64 tiles per instruction)
- No DSMEM / distributed shared memory (cluster size = 1x1x1)
- No TMA multicast
- No WGMMA (warpgroup MMA)
- These are **physical hardware limitations** — the silicon doesn't exist on GB202

### What SM120 Has
- TMA unicast (cp.async.bulk.tensor) ✓
- PDL (griddepcontrol) ✓
- Warp-level mma.sync with FP4/FP8 data types ✓
- 128 MB L2 cache (2x datacenter — expert weights fit in L2)
- 188 SMs, 256KB register file per SM
- Programming model: "Ampere mma.sync + Blackwell data types + TMA unicast + PDL"

### Dead Ends (All Validated)
- Persistent kernels: 5.8x SLOWER (barriers cheap, work uniform)
- cp.async pipelining: 0% at M=1 (BW-bound, not latency-bound)
- L2 persistence: neutral (working set fits in 128MB L2)
- Grouped routing: regression (fewer CTAs = lower SM utilization)
- Cross-SM SMEM sharing: impossible (no DSMEM hardware)
- Cross-GPU PCIe GEMM: latency-prohibitive (200ns/cacheline)
- Register/occupancy tuning: BW-bound, more regs = worse

## Files

| File | Description |
|------|-------------|
| csrc/verdict_fused_cooperative_tma_ext.cu | TMA+PDL production kernel |
| csrc/verdict_dense_fp4.cu | VerdictDense fused quantize+GEMM |
| verdict_moe.py | Updated with TMA path + lazy descriptors |
| SPRINT12_TMA_PDL_RESULTS.md | This document |

## Docker

- Base image: `vllm-019-verdict:latest`
- TMA kernel mounted via `-v` (JIT compiled on first inference)
- Env: `VERDICT_USE_TMA=1` (default)
- Fallback: `VERDICT_USE_TMA=0` uses scalar-load kernel
