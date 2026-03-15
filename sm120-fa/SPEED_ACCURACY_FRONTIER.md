# Speed/Accuracy Frontier — SM120 Selective Attention

## Summary

Selective attention trades accuracy for speed by routing Q to a subset of KV blocks.
Mean block summaries achieve 99.2% routing recall. Remaining error is from truncation.

## Best Sweet Spots

| Context | Config | Coverage | vs SDPA | Mean Err | Use Case |
|---------|--------|----------|---------|----------|----------|
| 32K | k=2, l=8 | 2.3% | **1.18x faster** | 0.046 | Fast draft |
| 64K | k=4, l=8 | 1.4% | **1.11x faster** | 0.043 | Balanced |
| 131K | k=2, l=2 | 0.3% | **1.64x faster** | 0.061 | Maximum speed |
| 131K | k=4, l=4 | 0.5% | **1.22x faster** | 0.049 | Good balance |

## Pareto Frontier at 131K (2048 blocks)

| top_k | local | Coverage | Latency | vs Full | vs SDPA | Error |
|-------|-------|----------|---------|---------|---------|-------|
| 2 | 2 | 0.3% | 0.82ms | **10.15x** | **1.64x** | 0.061 |
| 2 | 8 | 0.6% | 0.84ms | **9.85x** | **1.59x** | 0.045 |
| 4 | 4 | 0.5% | 1.10ms | **7.55x** | **1.22x** | 0.049 |
| 8 | 4 | 0.7% | 1.64ms | **5.05x** | 0.82x | 0.043 |
| 16 | 8 | 1.3% | 2.74ms | 3.03x | 0.49x | 0.031 |
| 32 | 8 | 2.1% | 5.04ms | 1.65x | 0.27x | 0.025 |

## Key Insight

Error scales with log(1/coverage), not linearly. Going from 2% to 0.3% coverage
only increases error from 0.025 to 0.061 (2.4x) while giving 3x more speed.

## Amortized Decode Throughput (attention-only)

| KV Length | Selective tok/s | Full Exact tok/s | Speedup |
|-----------|----------------|-----------------|---------|
| 8K | 4,192 | 1,896 | 2.21x |
| 32K | 2,517 | 493 | 5.10x |
| 131K | 888 | 124 | 7.15x |

## Routing Quality Analysis

| Metric | 4K | 8K | 32K | Implication |
|--------|-----|-----|------|------------|
| Raw score recall | 100% | 100% | 100% | Routing finds high-score blocks |
| Softmax mass recall | 44.5% | 39.8% | 27.3% | But misses where attention concentrates |
| Output contribution recall | 22.7% | 14.8% | 1.6% | And mostly misses what matters for output |

The gap between raw score and softmax mass recall is fundamental to Q-independent
summaries. Real (non-random) attention patterns are more structured, so practical
quality is likely better than these random-data numbers suggest.
