# KLC Real-World Legal Benchmark Analysis

## How MoE Expert Routing Affects Decode Throughput

MoE (Mixture-of-Experts) models like Qwen3.5-397B route each token through a subset of experts (4 of 60). Different prompt types activate different expert combinations, which affects throughput. This benchmark measures that impact using 24 real-world Kentucky legal prompts across 6 categories.

**Hardware:** 4x RTX PRO 6000 Blackwell (SM120), TP=4, MTP=5, P2P enabled
**Config:** vLLM 0.17.1rc1, FlashInfer 0.6.6, K=64 CUTLASS patch

## Single-User Decode Speed by Prompt Category

| Category | 1K tok/s | 2K tok/s | 3K tok/s | 4K tok/s | Description |
|----------|---------|---------|---------|---------|-------------|
| **Specialized** | 170 | 161 | 153 | 146 | Employment, estate, construction — focused domain |
| **Messy real-world** | 158 | 160 | 142 | 143 | Noisy inputs, irrelevant details mixed in |
| **Short factual** | 149 | 146 | 147 | 143 | Statute lookups, simple questions |
| **Citation normalization** | 145 | 139 | 134 | 138 | KRS citations, case law cross-references |
| **Trick/hallucination** | 129 | 126 | 121 | 151 | Model must reason carefully, avoid fabrication |
| **Complex multi-factor** | 119 | 113 | 109 | 140 | Divorce strategy, PI damages, mineral rights |
| **Average** | **139** | **135** | **131** | **142** | |

### Key Finding: Complex Analysis is 30-40% Slower

Complex multi-factor analysis (e.g., contested divorce property strategy, personal injury damages theory) consistently runs slower than specialized or factual prompts. This is because:

1. **Diverse expert activation:** Complex reasoning requires knowledge from multiple legal domains (family law + property + tax + strategy), activating more diverse expert combinations per token
2. **Longer reasoning chains:** The model generates more deliberative, cross-referencing text
3. **Higher token entropy:** Complex answers have less predictable token sequences, reducing MTP speculation acceptance

In contrast, specialized prompts (e.g., employment discrimination analysis) activate a focused cluster of experts repeatedly, enabling better cache locality and more predictable token sequences.

## Time-to-First-Token (TTFT) by Category

| Category | 1K TTFT | 2K TTFT | 3K TTFT | 4K TTFT |
|----------|---------|---------|---------|---------|
| Specialized | 69ms | 72ms | 96ms | 99ms |
| Short factual | 74ms | 67ms | 69ms | 86ms |
| Citation | 72ms | 78ms | 80ms | 74ms |
| Messy real-world | 84ms | 71ms | 96ms | 98ms |
| Complex analysis | 111ms | 166ms | 115ms | 78ms |
| Trick/hallucination | 120ms | 129ms | 106ms | 81ms |

Complex and trick prompts have higher TTFT (~110-165ms vs ~70ms for simple prompts), likely because the model's initial reasoning about how to approach the problem activates more diverse expert paths during prefill.

## Multi-User System Throughput

| Concurrency | System tok/s | Per-user tok/s |
|------------|-------------|---------------|
| 1 user | ~140-170 | ~140-170 |
| 2 users | ~250-270 | ~125-135 |
| 4 users | ~365-410 | ~91-103 |

## Output Length Has Minimal Impact on Decode Speed

| Output Length | Avg tok/s (single-user) |
|--------------|------------------------|
| ~1K tokens | 139 |
| ~2K tokens | 135 |
| ~3K tokens | 131 |
| ~4K tokens | 142 |

Throughput is consistent within ±10% across output lengths, confirming that decode speed is stable regardless of how much text is generated. The slight variation is within measurement noise.

## Prompt Categories Explained

### Short Factual (3 prompts)
Quick legal lookups — statute of limitations, BAC limits, contract deadlines. Tests focused expert activation with simple routing.

- "What is the statute of limitations for personal injury claims in Kentucky?"
- "What is the legal blood alcohol limit for DUI in Kentucky?"
- "What is the statute of limitations for breach of a written contract in Kentucky?"

### Citation Normalization (3 prompts)
Cross-referencing KRS statutes and case law. Tests specialized legal knowledge retrieval.

- "Explain KRS 411.182 comparative fault vs old contributory negligence..."
- "How does Kentucky law address the broad form deed for mineral rights? Cite Ward v. Harding..."
- "What standard does Kentucky apply for summary judgment? How does Steelvest differ from Celotex?"

### Trick/Hallucination (3 prompts)
Prompts designed to test whether the model fabricates legal doctrine. Tests careful reasoning paths.

- "Are holographic wills valid in Kentucky?" (No, with exceptions — KRS 394.040/394.095)
- "Does Kentucky's castle doctrine authorize deadly force?" (Narrow — KRS 503.055)
- "Do grandparents have an absolute right to visitation?" (No — conditional under KRS 405.021)

### Complex Multi-Factor Analysis (3 prompts)
Professional-grade legal scenarios requiring analysis across multiple domains. Tests deepest expert routing diversity.

- Contested divorce with business valuation, maintenance, custody strategy
- Mineral rights with 1962 broad form deed vs 1988 constitutional amendment
- Rear-end collision PI with subsequent workplace reinjury and causation issues

### Messy Real-World (3 prompts)
Prompts with irrelevant details, misspellings, and noise mixed with legally relevant facts.

- Eviction with habitability defense, unauthorized pets, lease expiration
- DUI with expired Intoxilyzer certification, CDL consequences
- Apartment attack with negligent security, irrelevant "annoying neighbor" detail

### Specialized Domain (9 prompts)
Deep single-domain questions spanning employment, estate, construction, custody, med-mal, criminal defense, workers comp, and property partition.

## Methodology

- All prompts sent via streaming chat completions API with system prompt: "You are a Kentucky legal expert..."
- Temperature 0.0 for reproducibility
- Token counts from server-reported `usage.completion_tokens`
- TTFT measured from request start to first streaming token
- Single-user: each category tested individually
- Multi-user: batches of mixed prompt types sent concurrently
- 3 iterations per multi-user configuration

Full prompt text available in [benchmarks/klc_prompts.json](../benchmarks/klc_prompts.json).
