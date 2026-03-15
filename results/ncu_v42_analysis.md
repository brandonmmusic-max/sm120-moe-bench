# NCU Profiling — v4.2 (FP32 fused, 223 regs, 0 stack)

**Kernel:** `sm120_fa_v4` | **GPU:** RTX PRO 6000 (SM 12.0, 300W) | **Config:** B=1, Hq=32, Hkv=8, Sq=8192, HD=128

## Performance Summary

| Metric | v4.1 | v4.2 | Change |
|--------|------|------|--------|
| **Compute Throughput** | 83.6% | **84.0%** | +0.5% |
| **Duration** | 4.09ms | **4.08ms** | -0.2% |
| IPC | 1.19 | 1.20 | +0.8% |
| Eligible Warps | 0.42 | 0.41 | -2% |
| L1/TEX Hit Rate | 58.3% | 58.3% | — |
| Registers | 223 | 223 | — |
| Stack | 0B | 0B | — |
| Occupancy | 16.67% | 16.67% | — |

## Stall Analysis

| Source | Est. Speedup | Details |
|--------|-------------|---------|
| SMEM access patterns | **25.2%** | 42M excessive wavefronts (7% of 623M total) |
| Occupancy (math pipe) | 16.0% | 41.5% of cycles stalled on tensor pipe |
| FP32 fusion | 5.4% | Remaining unfused mul+add pairs |
| **SMEM bank conflicts** | **6.7%** | Uncoalesced shared accesses |

## Bank Conflict Analysis

All ldmatrix calls exhibit **2-way bank conflicts** (lanes 0-7 and 8-15 hit same banks):

### K ldmatrix.x2 (per ni iteration)
```
Lanes 0-7:   banks [0, 4, 8, 12, 16, 20, 24, 28]
Lanes 8-15:  banks [4, 0, 12, 8, 20, 16, 28, 24]  ← 2-way conflicts on all 8 banks
```
8 extra wavefronts per ldmatrix × 8 ni × 8 ki = 512 extra per KV iteration

### V ldmatrix.x2.trans (per di iteration)
```
Lanes 0-7:   banks [0, 4, 8, 12, 16, 20, 24, 28]
Lanes 8-15:  banks [0, 4, 8, 12, 16, 20, 24, 28]  ← 2-way conflicts
```
8 extra wavefronts per ldmatrix × 16 di × 4 ki = 512 extra per KV iteration

### Q ldmatrix.x4 (per ki iteration)
```
Lanes 0-7:   banks [0, 8, 16, 24]  (repeated 2x)
Lanes 8-15:  banks [0, 8, 16, 24]  (repeated 2x)
Lanes 16-23: banks [4, 12, 20, 28] (repeated 2x)
Lanes 24-31: banks [4, 12, 20, 28] (repeated 2x)
```
**4-way conflicts** — only 8 unique banks used by 32 threads

### Root Cause
The TMA SWIZZLE_128B pattern XORs bits[6:4] of the column byte offset with row[2:0].
For 128-byte rows (HALF_HD=64 bf16), the swizzle permutes 16-byte chunks within each row.
But ldmatrix loads 16 bytes per thread — the second group of 8 threads accesses the SAME
swizzled 16-byte chunks as the first group (just at different rows with the same bank mapping).

The fix would require a swizzle that separates the two thread groups into different banks.
Options: wider XOR (bits[7:5] for K/V), different row grouping, or padding.

## Multi-GPU Results (v4.2, 4x RTX PRO 6000)

| Sq | Single GPU | 4-GPU P2P | Scaling | % Linear 4x |
|----|-----------|-----------|---------|-------------|
| 16K | 211 TF | 237 TF | 1.13x | 28% |
| 32K | 204 TF | 397 TF | 1.95x | 49% |
| 65K | 221 TF | 565 TF | 2.56x | 64% |
| 131K | 208 TF | 671 TF | 3.23x | 81% |

Note: Single-GPU TFLOPS varies between runs due to GPU clock fluctuation.
Multi-GPU scaling improves with sequence length as fixed P2P overhead amortizes.
