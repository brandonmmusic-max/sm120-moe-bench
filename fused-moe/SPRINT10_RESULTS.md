# Sprint 10: Group-by-Expert Fused Cooperative Kernel

**Date:** 2026-03-25
**Hardware:** 4x NVIDIA RTX PRO 6000 Blackwell (96GB each, SM120, 188 SMs, 100KB SMEM)
**Model:** Qwen3.5-397B-A17B-NVFP4 (512 experts, top-10 routing, 60 MoE layers)

---

## Task 0: Group-by-Expert with Per-Token Correctness

### Problem

Sprint 9 achieves 165 tok/s with correct per-token routing (66% MTP acceptance matching
CUTLASS). But the kernel is **44.4 μs/layer at M=4** because each CTA loads one expert's
weights for one token — when multiple tokens route to the same expert, weights are
loaded redundantly from GMEM (L2 cache helps but doesn't eliminate the cost).

Sprint 7 was **26 μs/layer** because weights were loaded ONCE per expert and reused across
all M tokens — but the routing was WRONG (all tokens forced to use token 0's experts).

### Approach: Group-by-Expert Sort + Cooperative MMA

Sort token-expert pairs by expert_id (counting sort), then each CTA handles ALL tokens
routed to one expert for a given (N-tile, K-group). Weights loaded ONCE per expert,
reused across 1-4 tokens. Per-token correctness preserved.

**Data structures (kernel args):**
- `expert_list[num_groups]`: which experts are active (~25 unique for M=4)
- `expert_token_count[num_groups]`: how many tokens per expert (1-4)
- `expert_token_offset[num_groups+1]`: prefix sum for indexing into gather_idx
- `gather_idx[total_pairs]`: for each slot, which original token to read input from
- `scatter_weights[total_pairs]`: routing weight for each (token, expert) pair

**Grid:** `num_groups × tiles_n × k_groups` CTAs

**Test routing (50% overlap):**
- Token 0 → experts {0,1,2,3,4,5,6,7,8,9}
- Token 1 → experts {5,6,7,8,9,10,11,12,13,14} (50% overlap with token 0)
- Token 2 → experts {20,21,22,23,24,25,26,27,28,29} (0% overlap)
- Token 3 → experts {0,1,2,3,4,25,26,27,28,29} (50% overlap with token 0)

**Result:** 25 unique experts, 15 with m_count=2, 10 with m_count=1

### Correctness — ALL PASS

| Config | Token 0 | Token 1 | Token 2 | Token 3 | Aggregate | Status |
|--------|---------|---------|---------|---------|-----------|--------|
| TP=4 M=4 (K_GROUPS=4) | 10.36% | 9.08% | 10.20% | 9.32% | **9.78%** | **PASS** |
| EP=4 M=4 (K_GROUPS=1) | 9.79% | 9.41% | 9.14% | 9.34% | **9.41%** | **PASS** |
| TP=4 M=1 (K_GROUPS=16) | 10.40% | — | — | — | **10.40%** | **PASS** |
| EP=4 M=1 (K_GROUPS=4) | 9.90% | — | — | — | **9.90%** | **PASS** |

All configs: 0 NaN, < 11% aggregate RelErr vs quantized reference. Bit-exact match with Sprint 9's
per-token reference (same computation, different CTA assignment).

### Standalone Benchmark Results

| Config | M | N_HALF | K_GROUPS | Grid (CTAs) | μs/layer | vs Sprint 9 | vs Sprint 7 | vs CUTLASS |
|--------|---|--------|----------|-------------|----------|-------------|-------------|------------|
| **TP=4** | 1 | 256 | 16 | 640 | **17.8** | **1.00x** | 1.11x | **5.50x** |
| **TP=4** | 4 | 256 | 4 | 400 | **56.7** | 0.78x | 0.46x | **2.12x** |
| EP=4 | 1 | 1024 | 4 | 640 | 48.5 | — | — | 2.02x |
| EP=4 | 4 | 1024 | 1 | 400 | 306.6 | — | — | 0.39x |

### Critical Finding: CTA Utilization Bottleneck

**The group-by-expert approach regresses M=4 performance vs Sprint 9 (56.7 μs vs 44.4 μs).**

Root cause: **CTA count too low to saturate the GPU.**

| Metric | Sprint 9 (per-pair) | Grouped (by expert) |
|--------|--------------------|--------------------|
| Grid (M=4 TP=4) | 40 pairs × 4 × 4 = **640 CTAs** | 25 groups × 4 × 4 = **400 CTAs** |
| CTAs/SM (avg) | 640/188 = **3.4** | 400/188 = **2.1** |
| SM utilization | **85%** | **53%** |
| Weight loads (total B tiles) | 640 × 16 = 10240 | 400 × 16 = 6400 |
| Bandwidth savings | — | **37.5% fewer B loads** |

The grouped kernel loads 37.5% fewer B tiles, but uses only 53% of the SM's concurrent execution
slots (vs 85% for Sprint 9). The bandwidth savings is dwarfed by the latency-hiding penalty from
lower CTA utilization.

### Register Pressure Journey

The kernel went through 3 register optimization stages:

| Stage | Registers | Stack | Spills | μs (M=4 TP=4) | Approach |
|-------|-----------|-------|--------|----------------|----------|
| 1. Naive `gate_acc[4][4]` | 64 | 208B | 4B | 91.6 | Runtime-indexed arrays → stack |
| 2. Template MAX_M_T | 80 | 192B | 0B | 91.6 | Still runtime `for(t<m_count)` → stack |
| 3. `#pragma unroll` + `--maxrregcount=64` | 56 | **0B** | **0B** | **56.7** | Compile-time indices → registers |

**Key discovery:** `float acc[N][4]` with runtime index `acc[t][i]` forces the compiler to place
the array on the stack (local memory), because GPU registers aren't addressable. The fix:
`#pragma unroll` over `for (int t = 0; t < MAX_M_T; t++) if (t < m_count)` makes `t` a
compile-time constant after unrolling, enabling scalar replacement into registers.

Combined with `--maxrregcount=64`, NVCC fits the full kernel (including Phase 2 accumulators)
in 56 registers with zero stack frame, achieving 4 CTAs/SM occupancy.

### Why Grouping Doesn't Help at TP=4

The fundamental constraint: `num_groups × tiles_n × k_groups ≤ max_concurrent_CTAs`.

At TP=4: `tiles_n = N_HALF / BN = 256 / 64 = 4`. With 25 groups:
- k_groups=4: 25 × 4 × 4 = 400 CTAs (only 53% of 752 capacity)
- k_groups=8: 25 × 4 × 8 = 800 > 752 → **DEADLOCK**

The low `tiles_n=4` (from TP=4 weight sharding) means grouping can't generate enough CTAs
to compensate for the reduced group count. Sprint 9's 640 CTAs (40 pairs × 16 tiles) has
60% more CTAs despite doing 37% more weight loads.

**Trade-off:** Grouping trades bandwidth efficiency for CTA count. On 188 SMs with atomic
barriers requiring all CTAs to be concurrently resident, CTA count wins.

### Where Grouping WOULD Help

1. **Larger N dimension (EP=4 or future models):** tiles_n=16 → more k_groups headroom
2. **Persistent kernels:** Decouple grid size from SM count — CTAs loop over expert groups
3. **Higher overlap (MTP with similar routing):** More tokens per group → more bandwidth savings
4. **Fewer SMs:** Smaller GPUs need fewer CTAs to saturate

### Files Created

| File | Status | Description |
|------|--------|-------------|
| `csrc/verdict_fused_grouped.cu` | **NEW** | Group-by-expert kernel + counting sort + test harness |
| `SPRINT10_RESULTS.md` | **NEW** | This file |

### Build & Test

```bash
# Compile (must use --maxrregcount=64 for 4 CTAs/SM occupancy):
/usr/local/cuda-13.2/bin/nvcc -std=c++17 -O2 \
  -gencode=arch=compute_120a,code=sm_120a \
  --expt-relaxed-constexpr --compiler-options '-fPIC' \
  --maxrregcount=64 \
  -o verdict_fused_grouped csrc/verdict_fused_grouped.cu
./verdict_fused_grouped
```

### Conclusion

Group-by-expert achieves **correct per-token routing** (matching Sprint 9 exactness) while
loading weights once per unique expert. However, at TP=4 with N_HALF=256, the reduced CTA count
(400 vs 640) causes a **28% regression** vs Sprint 9 at M=4.

**Sprint 9 remains the production kernel.** The grouped approach is architecturally sound and
would benefit from persistent kernel or TMA-based implementations that decouple grid size
from the number of unique expert groups.

The key technical discovery — that runtime-indexed accumulator arrays force stack allocation on
GPUs, and `#pragma unroll` with compile-time bounds enables register scalarization — is
applicable to all future multi-token kernel work.
