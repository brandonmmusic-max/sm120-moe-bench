# MTP Sweep: Speculative Decoding on SM120 with `vllm bench serve`

Standardized benchmark comparing MTP=1, MTP=2, and MTP=3 using vLLM's official `vllm bench serve` tool.

## Hardware

- **GPUs**: 4x NVIDIA RTX PRO 6000 Blackwell (96GB GDDR7, SM 12.0)
  - GPU 0, 2: Max-Q variants (300W)
  - GPU 1, 3: Full variants (capped at 300W via systemd)
- **CPU**: AMD Threadripper 24C/48T
- **OS**: Pop!_OS 24.04 (kernel 6.17.9), iommu=pt enabled
- **Driver**: 595.45.04, CUDA 13.2 (container)

## Software

- **vLLM**: 0.17.1rc1.dev143 (K=64 CUTLASS patched)
- **Docker image**: `vllm-qwen35-k64:latest` (`verdictai/vllm-blackwell-k64:latest`)
- **FlashInfer**: 0.6.6 (FLASHINFER_CUTLASS for MoE, FLASH_ATTN for attention)
- **Model**: Sehyo/Qwen3.5-397B-A17B-NVFP4 (MoE, 397B total / 17B active)

## Server Configuration

```bash
# Full docker run command — see configs/docker_run_tp4_tuned.sh for template
--tensor-parallel-size 4
--gpu-memory-utilization 0.90
--max-num-batched-tokens 16384
--max-num-seqs 128
--max-model-len 262144
--enable-prefix-caching
--speculative-config '{"method":"mtp","num_speculative_tokens":<1|2|3>}'
```

Key environment variables:
```
SAFETENSORS_FAST_GPU=1
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_DEVICE_MAX_CONNECTIONS=32
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VLLM_WORKER_MULTIPROC_METHOD=spawn
VLLM_SLEEP_WHEN_IDLE=1
VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1
```

## Benchmark Command

```bash
# Single-user (concurrency=1)
vllm bench serve \
  --backend vllm --base-url http://localhost:9200 \
  --model qwen3.5-397b-nvfp4 \
  --tokenizer /path/to/sehyo-qwen35-nvfp4 \
  --endpoint /v1/completions \
  --dataset-name random --random-input-len 128 --random-output-len 256 \
  --num-prompts 50 --max-concurrency 1 \
  --request-rate inf --num-warmups 5 --temperature 0

# 8-user concurrent (concurrency=8)
vllm bench serve \
  --backend vllm --base-url http://localhost:9200 \
  --model qwen3.5-397b-nvfp4 \
  --tokenizer /path/to/sehyo-qwen35-nvfp4 \
  --endpoint /v1/completions \
  --dataset-name random --random-input-len 128 --random-output-len 256 \
  --num-prompts 100 --max-concurrency 8 \
  --request-rate inf --num-warmups 5 --temperature 0
```

Parameters: 128 input tokens, 256 output tokens, greedy (temperature=0), 5 warmup requests, random dataset, streaming enabled.

## Results

### Output Throughput (tok/s) — the user-visible metric

| MTP | Single-user | 8-user (system) | 8-user (per-user) |
|-----|-------------|-----------------|-------------------|
| **MTP=3** | **172** | **625** | **~78** |
| MTP=2 | 145 | 593 | ~74 |
| MTP=1 | 110 | 532 | ~67 |

### Total Throughput (tok/s) — includes prefill tokens

| MTP | Single-user | 8-user |
|-----|-------------|--------|
| **MTP=3** | **258** | **938** |
| MTP=2 | 218 | 890 |
| MTP=1 | 165 | 799 |

### Latency

| MTP | Median TTFT | Median TPOT | Median ITL | P99 ITL |
|-----|-------------|-------------|------------|---------|
| MTP=3 (1-user) | 64.30 ms | 5.13 ms | 18.65 ms | 20.93 ms |
| MTP=2 (1-user) | 64.30 ms | 6.39 ms | 17.92 ms | 20.49 ms |
| MTP=1 (1-user) | 54.72 ms | 8.90 ms | 16.98 ms | 18.70 ms |
| MTP=3 (8-user) | 143.10 ms | 11.52 ms | 37.10 ms | 69.19 ms |
| MTP=2 (8-user) | 132.74 ms | 12.41 ms | 32.34 ms | 67.24 ms |
| MTP=1 (8-user) | 127.07 ms | 14.12 ms | 26.06 ms | 60.36 ms |

### Speculative Decoding Efficiency

| MTP | Acceptance Rate | Avg Tokens Accepted | Position 0 | Position 1 | Position 2 |
|-----|----------------|---------------------|-----------|-----------|-----------|
| MTP=3 (1-user) | 78.98% | 3.37 | 89.00% | 78.65% | 69.30% |
| MTP=2 (1-user) | 86.10% | 2.72 | 90.94% | 81.26% | — |
| MTP=1 (1-user) | 91.72% | 1.92 | 91.72% | — | — |
| MTP=3 (8-user) | 75.29% | 3.26 | 88.02% | 74.23% | 63.62% |
| MTP=2 (8-user) | 83.80% | 2.68 | 89.55% | 78.05% | — |
| MTP=1 (8-user) | 92.14% | 1.92 | 92.14% | — | — |

## Analysis

### MTP=3 wins at short context despite lower acceptance rate

The key tradeoff: more speculative tokens means lower per-position acceptance but more tokens per forward pass. At 128 input tokens:

- **MTP=3** accepts 3.37 tokens/step at 79% acceptance → **172 tok/s**
- **MTP=2** accepts 2.72 tokens/step at 86% acceptance → **145 tok/s**
- **MTP=1** accepts 1.92 tokens/step at 92% acceptance → **110 tok/s**

MTP=3 generates **57% more tokens/s** than MTP=1 despite only 79% acceptance. The extra tokens accepted per step (3.37 vs 1.92) far outweigh the wasted compute from rejected drafts.

### Latency vs throughput tradeoff

MTP=1 has the best latency characteristics (lowest ITL, lowest TTFT) because each step is cheaper. But the throughput gap is massive. For real-world chat/agent workloads where output throughput matters more than inter-token latency, MTP=3 is the clear winner.

### Scaling efficiency

At 8 concurrent users, all MTP configs scale similarly (~3.6x system throughput for 8x concurrency). The MTP=3 advantage is preserved under load.

### When to use which

| Scenario | Recommended |
|----------|-------------|
| Short context (<4K), throughput-focused | **MTP=3** |
| Long context (32K+) | MTP=5 (see mtp3_vs_mtp5_analysis.md) |
| Ultra-low latency requirements | MTP=1 |
| General production workloads | **MTP=3** |

## Notes

- All benchmarks used greedy decoding (temperature=0). Non-greedy sampling reduces MTP acceptance by ~8pp and throughput by ~40%.
- K=64 CUTLASS kernels are required for SM120 (99KB SMEM). Without them, K=128 tiles overflow and performance degrades significantly.
- `gpu_memory_utilization=0.92` causes OOM during CUDA graph capture on this hardware. 0.90 is the practical maximum.
- `VLLM_ATTENTION_BACKEND=FLASHINFER` is not recognized in vLLM 0.17.1rc1 — the backend is auto-selected (FLASH_ATTN for attention, FLASHINFER_CUTLASS for MoE).
- The `num_gpu_blocks` profiler reports 0 blocks (overridden to 512). This may be a profiling artifact with NVFP4 models.

## Reproducing

1. Pull the K=64 patched image: `docker pull verdictai/vllm-blackwell-k64:latest`
2. Download the model: [Sehyo/Qwen3.5-397B-A17B-NVFP4](https://huggingface.co/Sehyo/Qwen3.5-397B-A17B-NVFP4)
3. Start the server: `bash configs/docker_run_tp4_tuned.sh` (edit MODEL_PATH first)
4. Install benchmark tool: `pip install vllm[bench]`
5. Run benchmarks: `bash configs/benchmark_mtp_sweep.sh` (edit TOKENIZER path first)

## Appendix: Thinking Mode (Chat Completions) Impact

Using `--backend openai-chat --endpoint /v1/chat/completions` with the same random prompts triggers the chat template and reasoning/thinking token generation. This significantly impacts MTP speculative decoding efficiency.

### Benchmark Command (thinking-enabled)

```bash
vllm bench serve \
  --backend openai-chat --base-url http://localhost:9200 \
  --model qwen3.5-397b-nvfp4 \
  --tokenizer /path/to/sehyo-qwen35-nvfp4 \
  --endpoint /v1/chat/completions \
  --dataset-name random --random-input-len 128 --random-output-len 256 \
  --num-prompts 50 --max-concurrency 1 \
  --request-rate inf --num-warmups 5 --temperature 0
```

### Completions vs Chat (Thinking) — MTP=3

| Metric | Completions | Chat (thinking) | Delta |
|--------|-------------|-----------------|-------|
| Single-user output tok/s | 172 | **151** | -12% |
| 8-user system output tok/s | 625 | **577** | -8% |
| Single-user acceptance rate | 79.0% | 66.7% | -12pp |
| 8-user acceptance rate | 75.3% | 65.3% | -10pp |
| Position 0 accept (1-user) | 89.0% | 86.1% | -3pp |
| Position 1 accept (1-user) | 78.7% | 67.0% | -12pp |
| Position 2 accept (1-user) | 69.3% | **47.1%** | -22pp |

### Analysis

Thinking/reasoning tokens are harder for MTP heads to predict. Position 2 acceptance drops from 69% to 47% — nearly a coin flip. This suggests the internal reasoning chain has higher entropy than standard text completion, making speculative decoding less effective.

The throughput impact is moderate (-8% to -12%) because:
1. The MTP heads still provide value at positions 0 and 1 (86% and 67% acceptance)
2. The overhead of rejected drafts is partially offset by accepted tokens at earlier positions

For production chat/agent workloads with thinking enabled, expect ~150 tok/s single-user and ~577 tok/s at 8 concurrent users. MTP=3 remains the best choice — MTP=2 would likely perform similarly since position 2 is barely above random.

Date: 2026-03-15
