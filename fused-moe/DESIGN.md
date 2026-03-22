# SM120 Fused MoE GEMM Kernel

## Problem

Current vLLM MoE path (FLASHINFER_CUTLASS) launches 5+ kernels per layer.
Kernel launch overhead is **61% of total layer time** (32μs of 52μs at M=1).
Across 60 MoE layers, this wastes **1.9ms per token** — 33% of decode latency.

## Solution

Fuse GEMM1 (gate_up) → SiLU → GEMM2 (down) into a single kernel.
Intermediate stays in shared memory, never touches global memory.

## Baseline (2026-03-22)

| Metric | Value |
|--------|-------|
| Device | RTX PRO 6000 Blackwell Max-Q, 188 SMs |
| Model | Qwen3.5-397B-A17B, TP=4 |
| Per-layer MoE (M=1) | 52μs total |
| GEMM1 | 25μs |
| Activation | 8μs |
| GEMM2 | 9μs |
| Reduce | 9μs |
| Kernel launch overhead | 6.3μs × 5 = 32μs (61%) |
| Current decode throughput | ~172 tok/s |
| Target per-layer | ≤34μs (1 launch) |
| Expected speedup | ~26% decode throughput |

## Architecture

### Per-Expert GEMM Shapes (TP=4)

```
GEMM1: [M, 4096] × [4096, 512] → [M, 512]   (FP4 × FP4, K=64 tiles)
SwiGLU: [M, 512] → [M, 256]                   (in registers, FP32)
Requant: [M, 256] → E4M3 + block scales        (FP32 → FP8)
GEMM2: [M, 256] × [256, 4096] → [M, 4096]    (FP8 × FP4, K=32 tiles)
```

### Grid/Block

```
Grid:  (num_active_experts=10, N_gemm2_tiles=64, 1) = 640 CTAs
Block: 256 threads (8 warps), cooperative
```

### SMEM Layout (~53KB, fits 99KB)

```
GEMM1 weights (2-stage pipeline): 36KB
Input tile (2-stage): 1KB
Gate buffer (persistent): 8KB
Intermediate E4M3 + scales: 5KB
GEMM2 weights (2-stage, reuses GEMM1 SMEM): 5KB
```

### Key SM120 Constraints

- NO BF16 MMA — only FP4/FP8 via `f8f6f4.m16n8k32`
- 99KB SMEM (vs 228KB SM100)
- No tcgen05/TMEM — warp specialization hurts
- K=64 tiles for FP4 block-scaled MMA fit SMEM
- 188 SMs, 1×1×1 cluster only

## Phases

1. **Baseline** ✅ — 52μs/layer measured
2. **GEMM1+SiLU** — Single expert, validate FP4 MMA + SwiGLU
3. **Fused GEMM1+SiLU+GEMM2** — SMEM handoff, E4M3 intermediate
4. **Multi-expert grid** — 640 CTAs, token gather/scatter
5. **vLLM integration** — SM120FusedMoEExperts, oracle dispatch
6. **Optimize** — NCU profiling, tile tuning
