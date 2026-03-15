# SM120 MoE Inference Benchmark: Qwen3.5-397B on RTX PRO 6000

Reproducible benchmark suite for running Qwen3.5-397B-A17B-NVFP4 on NVIDIA RTX PRO 6000 Blackwell workstation GPUs (SM120, 99KB SMEM).

Includes a custom CUTLASS K=64 kernel fix that unblocks block-scaled MoE GEMM tiles on SM120, plus comprehensive benchmarks with real-world prompts.

## Hardware

- **GPUs:** 4x NVIDIA RTX PRO 6000 Blackwell (96GB GDDR7, SM 12.0, 99KB SMEM)
  - GPU 0, 2: Max-Q variants (300W)
  - GPU 1, 3: Full variants (600W, capped at 300W)
- **CPU:** AMD Threadripper PRO 9965WX (Shimada Peak), 24C/48T
- **RAM:** 512GB DDR5 RDIMM
- **PCIe:** Gen 5.0 x16 all slots, single NUMA node
- **OS:** Pop!_OS (Ubuntu 24.04 base), kernel param `iommu=pt`

## Software

- **Driver:** 595.45.04
- **CUDA:** 13.2
- **CUTLASS:** 4.4.1
- **vLLM:** 0.17.1rc1
- **FlashInfer:** 0.6.6
- **Model:** [nvidia/Qwen3.5-397B-A17B-NVFP4](https://huggingface.co/nvidia/Qwen3.5-397B-A17B-NVFP4)

## The Problem

SM120 (workstation Blackwell) has 99KB shared memory vs 228KB on datacenter Blackwell (B200). The CUTLASS block-scaled grouped GEMM tiles designed for B200 overflow SM120's SMEM at runtime:

```
Failed to initialize cutlass TMA WS grouped gemm
```

K=64 tiles would fit, but couldn't compile due to a TMA scale-factor layout mismatch in `sm120_blockscaled_mma_builder.inl`. This left MoE expert layers stuck on slow fallback kernels.

## The Fix

Three files patched (see [patches/](patches/)):

1. **`sm120_blockscaled_mma_builder.inl`** — Added `EffBlk_SF = min(K/SFVectorSize, Blk_SF)` to handle K=64 SF layouts
2. **`generate_kernels.py`** — Added K=64 CTA shapes for SM120 codegen
3. **`moe_gemm_template_dispatch_tma_ws.h`** — Added K=64 dispatch entries

**FlashInfer PR:** [flashinfer-ai/flashinfer#2786](https://github.com/flashinfer-ai/flashinfer/pull/2786)
**vLLM Issue:** [vllm-project/vllm#30135](https://github.com/vllm-project/vllm/issues/30135)

## Quick Start

### Pre-built Docker image

```bash
docker pull verdictai/vllm-blackwell-k64:latest

docker run -d --name vllm --gpus all --ipc host --shm-size 32g \
  -p 9200:8000 \
  -v /path/to/sehyo-qwen35-nvfp4:/model:ro \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e OMP_NUM_THREADS=6 \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  verdictai/vllm-blackwell-k64:latest \
  bash -c "exec python3 -m vllm.entrypoints.openai.api_server \
  --model /model --served-model-name qwen3.5-397b-nvfp4 \
  --host 0.0.0.0 --port 8000 --trust-remote-code \
  --tensor-parallel-size 4 --gpu-memory-utilization 0.85 \
  --max-model-len 262144 --max-num-batched-tokens 8192 \
  --enable-prefix-caching --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":5}'"
```

### Important notes

- **P2P:** If you have `iommu=pt` in kernel params, do NOT set `NCCL_P2P_DISABLE=1` — P2P gives 15-42% improvement
- **Threadripper:** Without `iommu=pt`, you MUST set `NCCL_P2P_DISABLE=1` to avoid IO_PAGE_FAULT
- **First startup:** Blackwell JIT compilation takes ~20 min on first run

## SM120-Native Flash Attention Kernel

Custom Flash Attention kernel built from scratch for SM120 (RTX PRO 6000 Blackwell). No FA3/FA4 exists for SM120 — it lacks tcgen05/TMEM, so vLLM falls back to FA2 (SM80 path via cuDNN). This kernel is SM120-native and closes the gap.

### Key Optimizations

| Optimization | TFLOPS (Sq=16K) | % of cuDNN | Registers |
|---|---|---|---|
| v2 scalar baseline | 0.3 | — | — |
| v3 MMA (fragment layout fixed) | 54 | 24% | — |
| v5 XOR swizzle | 99 | 43% | — |
| v1.0 BM128+BN64 single-stage | 117 | 51% | 247 |
| **v2.0 ldmatrix + register-P** | **190** | **83%** | 174 |
| **v2.1 + BN128 + Q-preload** | **216** | **90%** | 254 |

### Architecture

- **MMA:** `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (SM120 uses SM80-era MMA, not wgmma)
- **Tiles:** BM=128, BN=128, HD=128, 8 warps (256 threads)
- **SMEM:** Q(32KB) + K(32KB) + V(32KB) = 96KB, no P buffer (register-resident)
- **TMA:** Split-HD SWIZZLE_128B (64-elem halves, hardware swizzle)
- **Loads:** `ldmatrix.x4` for Q, `ldmatrix.x2` for K, `ldmatrix.x2.trans` for V
- **P@V:** Softmax output stays in registers — `s_acc[rh][ki*2+j][ei]` maps directly to MMA A fragment
- **Q preload:** All 8 Q fragments (32 regs) loaded before KV loop — zero Q SMEM access in hot path
- **Sync:** mbarrier phase tracking with prefetch, 1 `__syncthreads` per iteration (down from 3)

### SM120 FA vs cuDNN SDPA Benchmark

RTX PRO 6000 Blackwell (96GB, SM 12.0, 300W), BF16, non-causal, GQA Hq=32 Hkv=8:

| Sequence Length | SM120-FA | cuDNN SDPA | Ratio |
|---|---|---|---|
| 1,024 | 70 TF | 200 TF | 35% |
| 2,048 | 155 TF | 247 TF | 63% |
| 4,096 | 196 TF | 227 TF | 86% |
| 8,192 | 211 TF | 238 TF | 89% |
| **16,384** | **216 TF** | **240 TF** | **90%** |

| Config | SM120-FA | cuDNN | Ratio |
|---|---|---|---|
| B=2, Sq=2048 | 176 TF | 217 TF | 81% |
| B=4, Sq=2048 | 185 TF | 219 TF | 84% |
| MHA (Hq=Hkv=32), Sq=4096 | 189 TF | 224 TF | 85% |
| GQA 8:1 (Hq=64), Sq=2048 | 175 TF | 216 TF | 81% |

### Key Discoveries

- **MMA m16n8k16 fragment layout** (undocumented for SM120): A registers are interleaved row/row+8, NOT k/k+8. Ra0=[g,2t], Ra1=[g+8,2t], Ra2=[g,2t+8], Ra3=[g+8,2t+8]
- **ldmatrix is the single biggest optimization** — 62% speedup (117→190 TF) by replacing manual `pk()` SMEM loads with `ldmatrix.x4`/`ldmatrix.x2`
- **V needs `ldmatrix.trans`** — V's B fragment indexes rows by t (not g), requiring hardware transpose
- **FA3-style warp specialization hurts on SM120** — losing 1/8 compute to producer warp > overlap benefit
- **Single-stage > double-stage** — pipeline stall penalty exceeds memory overlap benefit on SM120
- **Register-resident P** saves 16KB SMEM round-trip but only ~5% of total SMEM loads; ldmatrix is what matters

### Selective Attention (Decode)

Block-summary routing with mean(K_block) achieves 99.2% raw score recall. Beats cuDNN SDPA at 32K+ decode context:

| Context | SM120-FA Selective | cuDNN SDPA | Speedup |
|---|---|---|---|
| 32K | 0.09ms | 0.17ms | 1.9x |
| 65K | 0.10ms | 0.34ms | 3.4x |
| 131K | 0.21ms | 0.34ms | 1.6x |

Source: [csrc/sm120_flash_attn_best_ldm_bn128.cu](https://github.com/brandonmmusic-max/sm120-moe-bench) (exact kernel), [csrc/sm120_selective_v3.cu](https://github.com/brandonmmusic-max/sm120-moe-bench) (selective)

## Benchmark Results

### Optimization Journey

| Configuration | 1-user tok/s (engine) | Change |
|--------------|----------------------|--------|
| WSL2 baseline | ~55 | — |
| Native Linux | ~119 | +116% |
| + MTP=5 + config tuning | ~134 | +13% |
| + Driver 595 + CUDA 13.2 + iommu=pt | ~142 | +6% |
| + K=64 kernel patch | ~115 | K=64 fix (P2P disabled) |
| **+ P2P enabled (NCCL_P2P_DISABLE removed)** | **~139** | **+21%** |

### Engine Throughput — Decode Matrix (server-side gen_throughput, 30s per cell)

Measured with [llm_decode_bench.py](scripts/llm_decode_bench.py) using real generation prompts.

| Context | 1 user | 2 users | 4 users | 8 users |
|---------|--------|---------|---------|---------|
| 0 | 139 | 241 | 375 | 579 |
| 1K | 134 | 236 | 382 | 584 |
| 2K | 144 | 237 | 377 | 559 |
| 4K | 135 | 228 | 376 | 542 |
| 8K | 125 | 210 | 353 | 521 |
| 16K | 109 | 200 | 324 | 491 |
| 32K | 90 | 158 | 275 | 438 |
| 64K | 67 | 120 | 214 | 354 |
| 128K | 45 | 85 | 152 | 256 |

### Prefill Speed (C=1)

| Context | TTFT | Prefill tok/s |
|---------|------|--------------|
| 8K | 0.51s | 17,425 |
| 16K | 1.00s | 17,022 |
| 32K | 2.30s | 14,508 |
| 64K | 4.14s | 16,000 |
| 128K | 9.60s | 13,716 |

### P2P Impact (before → after removing NCCL_P2P_DISABLE=1)

| Context | 1 user | 4 users | 8 users |
|---------|--------|---------|---------|
| 0 | 115→139 (+21%) | 311→375 (+21%) | 477→579 (+21%) |
| 8K | 107→125 (+17%) | 283→353 (+25%) | 424→521 (+23%) |
| 32K | 80→90 (+13%) | 207→275 (+33%) | 345→438 (+27%) |
| 64K | 57→67 (+18%) | 151→214 (+42%) | 254→354 (+39%) |
| 128K | 38→45 (+18%) | 108→152 (+41%) | 180→256 (+42%) |

### Real-World Legal Prompt Benchmark (KLC)

24 diverse Kentucky legal prompts across 6 categories, testing how MoE expert routing patterns affect decode throughput. Different prompt types activate different expert combinations, creating measurable throughput differences.

Full analysis: **[results/klc_analysis.md](results/klc_analysis.md)** | Prompts: **[benchmarks/klc_prompts.json](benchmarks/klc_prompts.json)**

**Single-user decode speed by prompt category and output length:**

| Category | 1K tok/s | 2K tok/s | 3K tok/s | 4K tok/s | Description |
|----------|---------|---------|---------|---------|-------------|
| **Specialized** | 170 | 161 | 153 | 146 | Employment, estate — focused domain experts |
| **Messy real-world** | 158 | 160 | 142 | 143 | Noisy inputs, irrelevant details to filter |
| **Short factual** | 149 | 146 | 147 | 143 | Statute lookups, simple routing |
| **Citation normalization** | 145 | 139 | 134 | 138 | KRS/case law cross-references |
| **Trick/hallucination** | 129 | 126 | 121 | 151 | Must reason carefully, avoid fabrication |
| **Complex multi-factor** | 119 | 113 | 109 | 140 | Divorce, PI damages — deepest routing |

**Key finding:** Complex multi-factor legal analysis runs **30-40% slower** than specialized prompts. This is because complex reasoning activates more diverse expert combinations per token, reducing cache locality and MTP speculation acceptance. Specialized prompts repeatedly activate focused expert clusters.

**TTFT by category:**

| Category | Avg TTFT |
|----------|---------|
| Short factual / Specialized | ~70ms |
| Citation / Messy | ~75-85ms |
| Complex / Trick | **~110-165ms** |

Complex and trick prompts have ~2x higher TTFT, likely from more diverse expert activation during prefill.

**Multi-user system throughput (mixed prompt types):**

| Concurrency | System tok/s | Per-user tok/s |
|------------|-------------|---------------|
| 1 user | ~140-170 | ~140-170 |
| 2 users | ~250-270 | ~125-135 |
| 4 users | ~365-410 | ~91-103 |

### Methodology Notes

- **Engine throughput** is measured from vLLM's `/metrics` endpoint (`vllm:avg_generation_throughput_toks_per_s`), not client-side API timing
- **API-level tok/s** (tokens received / wall time) is typically higher due to MTP speculative decoding inflating token counts with think tokens
- With thinking enabled + short prompts, API-level shows ~283 tok/s vs ~139 engine throughput
- With thinking disabled + real prompts, API-level shows ~130-136 tok/s
- All decode benchmarks use a 30-second measurement window after warmup stabilization
- Prefill benchmarks subtract baseline TTFT (0.04s) to isolate pure prefill time

## Reproducing

### 1. Run the decode benchmark

Engine throughput benchmark from [voipmonitor/llm-inference-bench](https://github.com/voipmonitor/llm-inference-bench).

```bash
pip install httpx rich
python3 scripts/llm_decode_bench.py \
  --port 9200 \
  --concurrency 1,2,4,8 \
  --contexts 0,1024,2048,4096,8192,16384,32768,65536,131072 \
  --max-tokens 2048 \
  --duration 30
```

### 2. Run the real-world legal benchmark

```bash
pip install httpx
python3 scripts/klc_real_world_bench.py
```

### 3. Apply patches manually (instead of Docker image)

```bash
docker exec -it your-container python3 /patches/apply_patches.py
```

## Key Findings

1. **K=64 kernel fix** unblocks MoE expert GEMM on SM120 — dense layers already had working tiles, MoE layers were stuck on slow fallback
2. **P2P communication** gives 15-42% improvement when `iommu=pt` is set — the old advice to use `NCCL_P2P_DISABLE=1` on Threadripper may no longer apply with driver 595+
3. **CUDA graphs are essential** — `--enforce-eager` drops decode from 139→40 tok/s (-71%). Kernel launch overhead dominates without graphs. Full analysis: [results/enforce_eager_analysis.md](results/enforce_eager_analysis.md)
4. **PCIe link width matters** — one GPU at x4 instead of x16 bottlenecks all TP allreduce operations; reseat cards if `lspci` shows degraded width
5. **Prompt complexity affects throughput** — complex multi-factor legal analysis runs 30-40% slower than specialized prompts due to MoE expert routing diversity. Full analysis: [results/klc_analysis.md](results/klc_analysis.md)
6. **MTP=3 beats MTP=5 for most workloads** — MTP=3 is 5-11% faster at short-medium contexts and high concurrency. MTP=5 only wins at 32K+ context where saving attention passes justifies extra speculation overhead. Full analysis: [results/mtp3_vs_mtp5_analysis.md](results/mtp3_vs_mtp5_analysis.md)
7. **MTP inflates API-level numbers** — engine throughput (~139 tok/s) is the honest metric; API-level with think tokens shows ~283 tok/s
8. **EP (Expert Parallelism) doesn't work on 4x96GB** — TP=2 EP=2 leaves 0 GiB for KV cache after model weights (62.5 GiB/GPU with TP=2). TP=4 is the correct config for this model on 4 GPUs

## Files

```
├── README.md
├── benchmarks/
│   ├── klc_prompts.json          # 24 real-world legal prompts with categories
│   └── methodology.md            # Detailed benchmark methodology
├── configs/
│   ├── docker_run_tp4.sh         # TP=4 launch script
│   ├── docker_run_tp2_ep2.sh     # TP=2 EP=2 launch script
│   └── env_vars.md               # Environment variable reference
├── results/
│   ├── p2p_disabled/             # Before P2P fix
│   ├── p2p_enabled/              # After P2P fix
│   ├── klc_real_world.json       # Legal prompt results
│   └── comparison.md             # Side-by-side analysis
├── scripts/
│   ├── llm_decode_bench.py       # Engine throughput benchmark
│   └── klc_real_world_bench.py   # Real-world legal benchmark
├── patches/
│   ├── sm120_blockscaled_mma_builder.inl
│   ├── generate_kernels.py
│   ├── moe_gemm_template_dispatch_tma_ws.h
│   └── apply_patches.py
└── LICENSE
```

## Acknowledgments

- **[voipmonitor/llm-inference-bench](https://github.com/voipmonitor/llm-inference-bench)** — decode throughput benchmark script used for all engine-level measurements
- **[NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)** — the block-scaled MoE GEMM builder that was patched
- **[flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)** — MoE kernel codegen and dispatch
- **[vllm-project/vllm](https://github.com/vllm-project/vllm)** — inference engine
- **[Claude Code](https://claude.com/claude-code)** — assisted with kernel development, benchmarking, and analysis

## Related

- [FlashInfer PR #2786](https://github.com/flashinfer-ai/flashinfer/pull/2786) — upstream K=64 fix
- [vLLM Issue #30135](https://github.com/vllm-project/vllm/issues/30135) — SM120 MXFP4 MoE discussion
- [NVIDIA/cutlass #2956](https://github.com/NVIDIA/cutlass/issues/2956) — FA cuteDSL for SM120 (pending)
- Docker image: `verdictai/vllm-blackwell-k64:latest`

## License

MIT
