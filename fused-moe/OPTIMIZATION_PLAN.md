# SM120 Fused MoE Kernel — Optimization Plan

## What We Build Upon (Not Copy)

### Our Empirical SM120 Discoveries (Not in b12x or Any Public Source)

1. **Dual-Quadrant CLayout**: The `mxf4nvf4.block_scale.m16n8k64` MMA produces two 8×4
   output quadrants, not full 16×8. d[1]=d[3]=0 always. The two quadrants share registers
   via destructive sum. This is undocumented and affects every SM120 FP4 kernel.

2. **UE8M0 Bias = 128**: The exponent bias for UE8M0 (MXF4 format) scale factors is 128,
   not 127. 0x80 = 1.0, 0x7F = 0.5. Gets the wrong answer otherwise.

3. **NVF4 vs MXF4 Format Distinction**: NVFP4 models use sf_vec_size=16 with E4M3FN scale
   factors (NVF4), not sf_vec_size=32 with UE8M0 (MXF4). Different MMA instruction variants.

4. **ldmatrix.b4x16_p64**: Handles E2M1 bit-shift AND lane redistribution atomically.
   SMEM data layout must match what this instruction expects (CUTLASS SmemLayoutAtom).

5. **B-row→Output-col**: Identity mapping. B-rows 0-3 → active quadrant cols 0-3,
   B-rows 4-7 → inactive quadrant (zero for M≤4).

6. **Kernel Launch Overhead**: 6.3μs per launch × 5 launches = 32μs per MoE layer = 61% of
   total 52μs. Across 60 layers = 1.9ms/token (33% of decode latency at 142 tok/s).

### b12x Architecture (What It Does Well)

- Three-tier dispatch: Static (decode), Micro (tiny), Dynamic (prefill)
- Fused route/pack + compute in single persistent kernel
- FC1→SiLU→requant-to-FP4→FC2→scatter-add pipeline
- FC1 intermediate cached in SMEM, reused across all FC2 output tiles
- 4 MMA warps + 1 TMA warp = 160 threads, occupancy 1
- sf_vec_size=16, E4M3FN scales, tile_k=128

### b12x Limitations We Can Exploit

| Limitation | Impact | Our Solution |
|-----------|--------|-------------|
| Global barrier between route/pack and compute | Tail-latency stall | Per-expert ready flags |
| No SplitK for M=1 decode | 1.6% SM utilization (3/188 SMs) | SplitK=8-16 → 24-48 CTAs |
| Fixed 128×128 tiles for both FC1 and FC2 | Suboptimal for different shapes | Asymmetric tiles |
| No L2 cache persistence hints | Cold weight loads every layer | cudaAccessPolicyWindow |
| Sequential FC1→FC2 (no weight prefetch) | FC2 stalls on weight load | DMA warp prefetch overlap |
| bf16x2 atomic scatter with no coalescing | High contention on output | Warp-level reduction tree |
| No overlap between consecutive MoE layers | Inter-layer idle time | PDL between layers |

---

## Our Kernel Architecture: "VerdictMoE"

### Design Philosophy
- **Decode-first**: Optimize for M=1-8 where SM utilization is the bottleneck
- **SM120-native**: Exploit our empirical CLayout/fragment discoveries
- **Build on CuteDSL**: CUTLASS Python DSL for correctness, custom optimizations on top
- **Measure everything**: CUDA events per optimization, NCU profiling

### Architecture: Single Persistent Kernel with Per-Expert Pipelining

```
Phase 0: Cooperative init (zero counters, barrier setup)
Phase 1: Route/pack with per-expert ready flags (not global barrier)
Phase 2: Compute with three novel optimizations:
  2a. SplitK for M≤16 decode (partition K across CTAs)
  2b. FC1→FC2 weight prefetch overlap via DMA warp
  2c. Warp-level scatter reduction (reduce before atomic)
```

### Optimization Priority (Build Order)

#### Sprint 1: Foundation (CuteDSL GEMM + SwiGLU, correct on SM120)
- Write CuteDSL kernel for our Qwen3.5-397B shapes (not copy b12x)
- Use sf_vec_size=16, E4M3FN scales (NVF4 format we empirically verified)
- Validate GEMM1+SwiGLU+requant+GEMM2 against PyTorch reference with random data
- **Success metric**: Correct output matching reference within FP4 tolerance

#### Sprint 2: SplitK for Decode (the differentiator)
- For M=1: split K across 8 CTAs per expert
- 10 experts × 8 splits = 80 CTAs (vs 10 without SplitK = 1.6% SM util)
- Atomic reduction of partial sums, or two-pass reduce
- **Success metric**: Measurable decode speedup in isolated kernel benchmark

#### Sprint 3: vLLM Integration + End-to-End Benchmark + NCU Profile
- Wire into vLLM as SM120FusedMoEExperts
- Benchmark real end-to-end tok/s (target: 180-200 tok/s, 25-40% over 142.6)
- **NCU profile to identify actual bottleneck** before ANY further optimization
- **Success metric**: >180 tok/s end-to-end, NCU data in hand

#### Sprint 4+: Data-Driven (based on NCU profiling from Sprint 3)
- DO NOT pre-commit to specific optimizations
- NCU will show the real bottleneck: memory bandwidth? compute? launch overhead? scatter contention?
- Build the optimization that addresses the measured bottleneck
- Candidate optimizations (ranked by b12x gap analysis, to be validated by NCU):
  - Per-expert ready flags (if barrier stall is measured)
  - FC1→FC2 weight prefetch (if weight load latency is measured)
  - L2 cache persistence (if L2 miss rate is high)
  - Warp-level scatter reduction (if atomic contention is measured)
  - Asymmetric tiles (if compute utilization differs between FC1/FC2)

### Realistic Targets
- **Sprint 1-2**: Correct kernel + SplitK decode = ~180 tok/s
- **Sprint 3**: vLLM integration + profiling = 180-200 tok/s
- **Sprint 4+**: Data-driven optimization = potential 200-250 tok/s (uncertain)
- **Publishable result**: 25-40% improvement over FLASHINFER_CUTLASS baseline

---

## How Our Discoveries Directly Enable These Optimizations

| Our Discovery | Enables |
|--------------|---------|
| Dual-quadrant CLayout | Custom tile shapes that avoid wasting the inactive quadrant |
| NVF4 format (sf_vec=16, E4M3FN) | Correct scale factor encoding from day 1 |
| ldmatrix.b4x16_p64 behavior | Understanding of when CuteDSL handles layout vs manual code |
| UE8M0 bias=128 | Correct scale factor math if we ever need MXF4 fallback |
| B-row→col identity mapping | Simplified output indexing for scatter-add |
| 6.3μs launch overhead | Quantified motivation: fusing 5 launches saves 32μs/layer |
| 52μs/layer baseline | Target: ≤20μs/layer (60% reduction) |
| 142.6 tok/s decode baseline | Target: >250 tok/s (75% improvement) |

---

## File Structure

```
~/sm120-moe-bench/fused-moe/
├── verdict_moe/
│   ├── __init__.py
│   ├── gemm.py              # CuteDSL GEMM with NVF4 format (Sprint 1)
│   ├── swiglu.py             # Fused SwiGLU activation + requant (Sprint 1)
│   ├── splitk.py             # SplitK dispatch for M≤16 decode (Sprint 2)
│   ├── expert_pipeline.py    # Per-expert ready flags (Sprint 3)
│   ├── weight_prefetch.py    # DMA warp FC1→FC2 overlap (Sprint 4)
│   ├── l2_cache.py           # L2 persistence hints (Sprint 5)
│   ├── scatter.py            # Warp-level reduction scatter (Sprint 6)
│   └── vllm_backend.py       # vLLM SM120FusedMoEExperts integration
├── tests/
│   ├── test_gemm_correctness.py
│   ├── test_splitk.py
│   └── test_e2e.py
└── benchmarks/
    ├── bench_per_sprint.py
    └── bench_vs_baseline.py
```
