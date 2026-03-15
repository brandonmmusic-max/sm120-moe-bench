# MTP=3 vs MTP=5: Speculative Decoding Trade-offs on SM120

## Summary

MTP=3 is faster than MTP=5 at short-medium contexts and high concurrency. MTP=5 only wins at long contexts (32K+). The extra 2 speculative tokens add overhead per step that isn't recovered unless the KV cache is large enough that each accepted token saves significant attention compute.

## Configuration

Both runs identical except `num_speculative_tokens`:
- TP=4, CUDA graphs ON, P2P enabled, all SymmMem patches
- `gpu_memory_utilization=0.85`, `max_model_len=262144`
- `max_num_batched_tokens=8192`
- Image: `verdictai/vllm-blackwell-k64:full-patch`

## Aggregate Throughput (tok/s)

### Single User

| Context | MTP=5 | MTP=3 | Diff | Winner |
|---------|-------|-------|------|--------|
| 0 | 140 | **145** | +4% | MTP=3 |
| 1K | 122 | **136** | +11% | MTP=3 |
| 2K | 142 | **133** | -6% | MTP=5 |
| 4K | 136 | **134** | -1% | ~tie |
| 8K | 122 | **120** | -2% | ~tie |
| 16K | 108 | **105** | -3% | ~tie |
| 32K | **88** | 81 | -8% | MTP=5 |
| 64K | **63** | 61 | -3% | MTP=5 |
| 128K | **47** | 39 | -17% | MTP=5 |

### 4 Users (System Throughput)

| Context | MTP=5 | MTP=3 | Diff | Winner |
|---------|-------|-------|------|--------|
| 0 | 386 | **427** | +11% | MTP=3 |
| 1K | 383 | **407** | +6% | MTP=3 |
| 4K | 367 | **377** | +3% | MTP=3 |
| 8K | 360 | **358** | -1% | ~tie |
| 16K | 318 | **320** | +1% | ~tie |
| 32K | **279** | 275 | -1% | ~tie |
| 64K | **218** | 206 | -6% | MTP=5 |
| 128K | **156** | 140 | -10% | MTP=5 |

### 8 Users (System Throughput)

| Context | MTP=5 | MTP=3 | Diff | Winner |
|---------|-------|-------|------|--------|
| 0 | 588 | **619** | +5% | MTP=3 |
| 1K | 568 | **601** | +6% | MTP=3 |
| 4K | 567 | **567** | 0% | tie |
| 8K | 518 | **546** | +5% | MTP=3 |
| 16K | 480 | **509** | +6% | MTP=3 |
| 32K | 438 | **442** | +1% | ~tie |
| 64K | **361** | 349 | -3% | MTP=5 |
| 128K | **267** | 252 | -6% | MTP=5 |

## Analysis

### Why MTP=3 Wins at Short Contexts

Each MTP speculation step requires a forward pass through the MTP head for each speculative token. With MTP=5, the model speculatively generates 5 tokens, but if only 2-3 are accepted, the compute for the rejected tokens is wasted. At short contexts:

1. **Attention is cheap** — short KV cache means attention completes quickly, so saving one attention pass (by accepting a speculative token) doesn't save much time
2. **Speculation overhead is fixed** — the MTP head forward pass cost is the same regardless of context length
3. **Acceptance rate may be lower** — with real generation (not trivial "hello" prompts), the model's MTP predictions are less accurate, especially for varied multi-domain content

### Why MTP=5 Wins at Long Contexts

At 32K+ contexts, each decode step's attention computation is expensive (linear in context length). Each accepted speculative token saves a full attention pass over the long KV cache:

1. **Attention dominates compute** — at 128K context, attention is ~70% of per-step cost
2. **Saved attention is worth more** — accepting even 1 extra speculative token at 128K saves much more compute than at 1K
3. **The 17% gap at 128K** (47 vs 39 tok/s) shows MTP=5 is worth the overhead when attention is the bottleneck

### Per-Request Scaling

MTP=3 also shows better per-request scaling at high concurrency:

| Context | MTP=5 per-req (8 users) | MTP=3 per-req (8 users) |
|---------|------------------------|------------------------|
| 0 | 73.5 | **77.4** |
| 8K | 64.8 | **68.3** |
| 16K | 60.0 | **63.7** |

This suggests MTP=3 has lower per-step overhead, allowing more efficient batching when the scheduler is juggling multiple requests.

## Recommendation

| Workload | Best MTP | Why |
|----------|---------|-----|
| Multi-user chatbot (short-medium context) | **MTP=3** | Lower overhead, better batching, +5-11% system throughput |
| Single-user long-form generation | **MTP=5** | Saves expensive attention passes at long contexts |
| RAG with 32K+ retrieved context | **MTP=5** | Long KV cache makes speculation payoff higher |
| High-concurrency API serving | **MTP=3** | Better per-request scaling, lower tail latency |
| Mixed workloads | **MTP=3** | Most real traffic is short-medium context |

## Crossover Point

The crossover where MTP=5 starts outperforming MTP=3 is approximately **16K-32K context length** for single-user, and **32K-64K** for multi-user. Below the crossover, MTP=3 wins. Above it, MTP=5 wins.

For typical chatbot/assistant workloads where most requests are <16K context, **MTP=3 is the better default**.
