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

## Critical Implementation Notes

### K=64 vs K=32 Asymmetry (GEMM1 vs GEMM2)

GEMM1 uses `mxf4nvf4.block_scale.m16n8k64` (K=64 per MMA) while GEMM2 uses
`f8f6f4.m16n8k32` (K=32 per MMA). For the same K-tile loaded into SMEM:
- GEMM1: 1 MMA instruction per K=64 tile
- GEMM2: 2 MMA instructions per K=64 tile (or 1 per K=32 tile)

The inner loop structure must account for this. With BK=64 SMEM tiles:
- GEMM1 K-loop: 4096/64 = 64 iterations, 1 MMA each
- GEMM2 K-loop: 256/64 = 4 iterations, 2 MMAs each (or 256/32 = 8 iterations, 1 MMA each)

### Block-Scale Register Overhead (GEMM1 only)

The `mxf4nvf4.block_scale` MMA has 6 extra register operands:
`sfa0` (uint32), `bidA` (uint16), `tidA` (uint16),
`sfb0` (uint32), `bidB` (uint16), `tidB` (uint16)

These must be loaded from SMEM scale factor buffers and staged in registers
alongside weight/input data for every MMA instruction. GEMM2's `f8f6f4` path
does NOT have block-scale registers — simpler.

Register budget per thread (worst case):
- 32 FP32 accumulators (per N-tile)
- 4+2 A/B data regs = 6
- 6 scale factor regs = 6
- ~20 misc (addresses, loop counters, temp)
- **Total: ~64 regs/thread** — well within 255 limit

### FP32 → E4M3 Conversion for GEMM2 Input

Use `cvt.rn.satfinite.e4m3x2.f32` to pack two FP32 values into a uint16
containing two E4M3 values. The packing order must match GEMM2's B operand
lane layout:

```ptx
cvt.rn.satfinite.e4m3x2.f32 %rd0, %f1, %f0;  // pack f0,f1 → uint16
```

The B[2×uint32] operand for `f8f6f4.m16n8k32` expects E4M3 values in
column-major order within each uint32. The `cvt.rn.satfinite` output
must be arranged to match this or the MMA will produce garbage.

This conversion happens after SwiGLU, before writing the intermediate to SMEM.
Per-block scale factors for the E4M3 intermediate are computed as:
`scale = max(abs(block)) / 448.0` (E4M3 max = 448).

## Phases

1. **Baseline** ✅ — 52μs/layer measured
2. **GEMM1+SiLU** 🔨 — Single expert, validate FP4 MMA + SwiGLU against PyTorch ref
3. **Fused GEMM1+SiLU+GEMM2** — SMEM handoff, E4M3 intermediate with cvt.rn.satfinite
4. **Multi-expert grid** — 640 CTAs, token gather/scatter
5. **vLLM integration** — SM120FusedMoEExperts, oracle dispatch
6. **Optimize** — NCU profiling, tile tuning
