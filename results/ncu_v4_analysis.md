# NCU Profiling Analysis — SM120 Flash Attention v4

**Kernel:** `sm120_fa_v4` (BM128 BN64 double-buffered, ldmatrix, register-P)
**GPU:** RTX PRO 6000 Blackwell (SM 12.0, 96GB, 300W)
**Config:** B=1, Hq=32, Hkv=8, Sq=8192, Skv=8192, HD=128, BF16

## Summary

| Metric | Value | Notes |
|--------|-------|-------|
| **Duration** | 4.18 ms | |
| **Compute Throughput** | **81.5%** | Tensor pipe over-utilized (bottleneck) |
| **Tensor (FP) Pipeline** | **81.5%** | BF16 MMA — the performance limiter |
| **IPC (Active)** | **1.21** | Up from 0.95 in v3 SS |
| **Issue Rate** | 1 per 3.3 cycles | Up from 1 per 4.2 in v3 |
| **Issue Slots Busy** | 30.0% | |
| **L1/TEX Cache Throughput** | 50.3% | |
| **L1/TEX Hit Rate** | **94.5%** | Excellent cache locality |
| **L2 Hit Rate** | **98.2%** | Almost no DRAM pressure |
| **DRAM Throughput** | 1.8% | Fully compute-bound |
| **Memory Throughput** | 29.4 GB/s | Negligible vs compute |

## Occupancy

| Metric | Value |
|--------|-------|
| Block Size | 256 threads (8 warps) |
| Registers Per Thread | 229 |
| Dynamic SMEM Per Block | 98.4 KB |
| **Block Limit (Registers)** | **1** |
| **Block Limit (Shared Mem)** | **1** |
| Block Limit (Warps) | 6 |
| Block Limit (Barriers) | 24 |
| **Theoretical Occupancy** | **16.67%** (1 block/SM) |
| **Achieved Occupancy** | **16.62%** |
| Active Warps Per Scheduler | 2.00 (max 12) |
| **Eligible Warps Per Scheduler** | **0.40** |
| Waves Per SM | 10.89 |

## Warp Stall Analysis

| Stall Reason | Impact |
|-------------|--------|
| **Math Pipe Throttle** | **36.9%** of 6.6 cycle avg stall — warps waiting for tensor pipe |
| Not Selected | Scheduler has no eligible warp to issue |
| Other | Remaining stalls (barrier, memory, etc.) |

**Key insight:** 36.9% of stall time is warps waiting for the tensor pipe (MMA) to become available. This is the expected bottleneck at 81.5% tensor utilization with only 16.67% occupancy (2 active warps per scheduler, 0.40 eligible).

## Local Memory

| Metric | Value |
|--------|-------|
| Local Memory Access | **93.2%** of L1TEX sectors |
| Local Load Hit Rate | 98.4% (well cached) |
| Local Store Hit Rate | 88.1% |
| Spilling Requests | 0 |

The 80-byte stack frame (from double-buffer pointer arrays) causes local memory access but it's 98.4% cached in L1. This accounts for most of the L1/TEX traffic but doesn't significantly hurt performance since it hits L1 nearly every time.

## Performance Assessment

### What's working well:
- **81.5% tensor utilization** at only 16.67% occupancy is excellent
- **94.5% L1 hit rate** — ldmatrix + TMA swizzle eliminates bank conflicts
- **98.2% L2 hit rate** — KV data fits in L2 across iterations
- **1.21 IPC** — good instruction throughput for the occupancy level
- **Zero SMEM spills** — all register pressure goes to well-cached local memory

### What limits further improvement:
- **Occupancy ceiling:** Both registers (229) AND SMEM (98KB) cap at 1 block/SM
- **Tensor pipe saturation:** 81.5% → diminishing returns from instruction scheduling
- **SM120 architecture:** No wgmma (async MMA), no TMEM, no tcgen05 — mma.sync holds registers during execution

### Estimated speedup opportunities (from NCU):
- Reduce local memory access: **~21%** (if stack spills eliminated)
- Better math pipe scheduling: **~18.5%** (if more warps available)
- FP32 fusion opportunities: **~5.7%** (fma pairs in softmax)

## Multi-GPU Scaling

Single-GPU is compute-bound at 81.5% tensor utilization. Multi-GPU sequence-parallel attention distributes KV across GPUs, achieving near-linear scaling at long sequences:

| Sq | Single GPU | 4-GPU P2P | Scaling |
|----|-----------|-----------|---------|
| 16K | 229 TF | 261 TF | 1.14x |
| 32K | 229 TF | 409 TF | 1.80x |
| 65K | 225 TF | 579 TF | 2.57x |
| 131K | 224 TF | 670 TF | 2.99x |

The fixed ~3ms P2P overhead amortizes at Sq≥16K. At 131K, we achieve 2.99x scaling (theoretical max 4x), limited by PCIe Gen5 bandwidth and the online softmax combine step.
