# SM120 Flash Attention — Optimization Roadmap

## Current Performance

| Mode | TFLOPS | vs SDPA | Status |
|------|--------|---------|--------|
| Exact (v5 swizzle) | 99 | 38-95% | Production |
| Selective decode (v2) | — | 1.64x at 131K | Beats SDPA! |

## Why 99 → 300 TFLOPS is Hard

The gap to SDPA (264 TFLOPS) comes from:

1. **Single block/SM occupancy** (255 registers, can't reduce below 254)
2. **No compute/memory overlap** (no warp specialization)
3. **P staging overhead** (~30% of kernel time)
4. **No TMA** (using cp.async, slower than TMA bulk copies)

## Optimization Tiers

### Tier 1: Already Done (0 → 99 TFLOPS)
- [x] MMA m16n8k16 fragment layout discovery
- [x] XOR swizzle (+83%)
- [x] Score masking for non-aligned sequences
- [x] Cross-thread softmax reduction
- [x] ldmatrix for A fragment loading
- [x] Selective attention for decode speedup

### Tier 2: Diminishing Returns (tried, didn't help enough)
- [x] Register reduction via 2-pass HEAD_DIM (v6: slower, compiler fills all 255)
- [x] Register limit via __launch_bounds__ or --maxrregcount (no effect)
- [x] P@V via warp shuffles (v4: 3.6x slower than SMEM staging)
- [x] BLOCK_N=32 (no register savings)
- [x] Sub-block routing (worse softmax mass recall than mean)
- [x] EMA-weighted summaries (worse than mean)
- [x] Mean+variance summaries (no improvement over mean)

### Tier 3: Worth Trying Next
- [ ] Remove unnecessary __syncthreads__ between P write and P@V (each warp reads own rows)
- [ ] BLOCK_M=64, BLOCK_N=128, STAGES=1 (96KB SMEM, fewer N iterations)
- [ ] Swizzle on P staging buffer (currently unswizzled)
- [ ] Software pipelining: overlap K loading for next iter with current P@V
- [ ] Use __ldg() for global memory reads (read-only cache hint)

### Tier 4: Architectural Changes (significant engineering)
- [ ] Warp specialization: 2 producer warps + 2 consumer warps
- [ ] TMA bulk copies (requires different SMEM management)
- [ ] Persistent kernel with tile scheduling
- [ ] Split-KV CUDA kernel for decode
- [ ] Block-sparse exact attention (skip masked blocks)

### Tier 5: Integration
- [ ] vLLM attention backend plugin
- [ ] Paged KV cache support
- [ ] Real-model activation testing
- [ ] Causal mask support (upper triangle skip)

## Selective Attention Findings

- Mean(K) summary: 99.2% raw score recall, 27-44% softmax mass recall
- Sub-block max: WORSE than mean for softmax mass
- Softmax mass recall scales slowly with top_k (27% → 56% for 4x→64x blocks)
- On random data, Q-independent summaries fundamentally can't predict softmax mass
- Real model data is more structured — local window likely captures most mass
- Current selective path at top_k=4, local=4 already beats SDPA at 32K+

## Recommended Focus

1. **Real-model evaluation** — measure actual accuracy on Qwen3.5 attention patterns
2. **Adaptive top_k in CUDA** — already implemented in v3, needs tuning
3. **Summary amortization** — separate build from use for decode loops
4. **vLLM integration** — wire as custom attention backend
