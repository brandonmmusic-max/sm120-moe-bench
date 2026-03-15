# SM120 Flash Attention — Baseline Benchmark

## Correctness Status

**v0.1.0-correctness** — tagged 2026-03-15

| Test | Status |
|------|--------|
| MHA 128×128 | PASS (0.004 max err) |
| MHA 512×512 | PASS (0.002 max err) |
| GQA 32:8 512×512 | PASS (0.004 max err) |
| GQA 32:8 1024×1024 batched | PASS (0.004 max err) |
| GQA 32:8 2048×2048 | PASS (0.002 max err) |
| GQA 32:8 4096×4096 | PASS (0.001 max err) |
| Decode Q=128, KV=8192 | PASS (0.0005 max err) |
| Decode Q=1, KV=4096 | PASS (0.0005 max err) |
| Batched B=4, 256×256 | PASS (0.004 max err) |
| compute-sanitizer memcheck | CLEAN (container permission errors only) |
| compute-sanitizer racecheck | CLEAN |

### Known Issues (v0.1.0)
- Non-BLOCK_M-aligned sequences (e.g., 63, 65, 100) have boundary errors
- Very small sequences (Sq=1, Sq=3) produce NaN — softmax division by zero for empty warp tiles
- These are boundary condition bugs, not algorithmic issues

## Performance

**Hardware:** NVIDIA RTX PRO 6000 Blackwell (SM120, 99KB SMEM)

### Prefill (Q=KV, B=1, Hq=32, Hkv=8, D=128)

| Seq Len | SM120 FA (ms) | SM120 FA TFLOPS | torch SDPA (ms) | SDPA TFLOPS | Ratio |
|---------|--------------|-----------------|-----------------|-------------|-------|
| 128 | 0.03 | 9.9 | 0.02 | 14.2 | 0.70x |
| 256 | 0.04 | 25.2 | 0.03 | 36.6 | 0.69x |
| 512 | 0.13 | 32.0 | 0.05 | 79.9 | 0.40x |
| 1024 | 0.37 | 46.3 | 0.10 | 167.1 | 0.28x |
| 2048 | 1.40 | 49.2 | 0.28 | 244.4 | 0.20x |
| 4096 | 5.07 | 54.2 | 1.11 | 247.5 | 0.22x |
| 8192 | 20.47 | 53.7 | 4.16 | 264.5 | 0.20x |
| 16384 | 80.77 | 54.5 | 18.17 | 242.1 | 0.22x |

### Decode (Q=1, varying KV, B=1, Hq=32, Hkv=8, D=128)

| KV Len | SM120 FA (ms) | SDPA (ms) | Ratio |
|--------|--------------|-----------|-------|
| 1024 | 0.122 | 0.024 | 0.20x |
| 4096 | 0.458 | 0.036 | 0.08x |
| 8192 | 0.912 | 0.060 | 0.07x |
| 16384 | 1.817 | 0.187 | 0.10x |
| 32768 | 3.626 | 0.359 | 0.10x |

### Comparison across all implementations

| Implementation | Peak TFLOPS | Status |
|----------------|------------|--------|
| SM120 FA scalar (v2) | 0.3 | Correct, reference |
| **SM120 FA MMA (v3)** | **54.5** | **Correct, this kernel** |
| torch SDPA (SM80 path) | 264.5 | Production baseline |
| Theoretical SM120 BF16 | ~330 | Hardware peak |

### Gap Analysis

Current kernel is at **54.5 / 264.5 = 20.6%** of torch SDPA.

Identified optimization targets (in priority order):

1. **P staging overhead** — BF16 downcast to SMEM + re-read for P@V adds ~30% latency. Direct register-to-register P@V would eliminate this.

2. **SMEM bank conflicts** — No swizzle applied yet. CUTLASS XOR swizzle would reduce bank conflicts on ldmatrix and pack reads.

3. **Register pressure** — 255 registers (max). Reducing tile size or refactoring accumulator layout could free registers for higher occupancy.

4. **Occupancy** — At 255 registers, occupancy is 1 block per SM. Reducing to ~128 registers would allow 2 blocks.

5. **Split-KV for decode** — Single-token decode (Q=1) is severely underutilized. Split-KV across multiple CTAs would parallelize the KV iteration.

6. **TMA path** — Replace cp.async with TMA bulk copies for higher bandwidth.

7. **Warp specialization** — Dedicated producer/consumer warps to overlap memory and compute.

### Theoretical speedup from optimizations

| Optimization | Estimated Impact | Cumulative |
|-------------|-----------------|-----------|
| Swizzle (bank conflicts) | +20-30% | 65-70 TFLOPS |
| Eliminate P staging | +30-40% | 85-100 TFLOPS |
| Better occupancy | +30-50% | 110-150 TFLOPS |
| Split-KV for decode | +200-400% decode | (decode specific) |
| TMA + warp spec | +20-30% | 130-195 TFLOPS |

Target: **150-200 TFLOPS** (60-80% of SDPA) would make this kernel competitive enough to improve end-to-end inference.
