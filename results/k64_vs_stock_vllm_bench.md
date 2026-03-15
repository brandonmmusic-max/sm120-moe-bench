# K=64 CUTLASS Kernel: Before vs After — Standardized Benchmarks

Controlled comparison of K=64 CUTLASS MoE GEMM kernels vs stock vLLM, using identical server configuration and `vllm bench serve` + `llm_decode_bench.py --no-think`.

## Hardware

- **GPUs**: 4x NVIDIA RTX PRO 6000 Blackwell (96GB GDDR7, SM 12.0)
  - GPU 0, 2: Max-Q (300W), GPU 1, 3: Full (capped 300W)
- **CPU**: AMD Threadripper 24C/48T
- **OS**: Pop!_OS 24.04, kernel 6.17.9, iommu=pt, Driver 595.45.04

## Software

- **Model**: Sehyo/Qwen3.5-397B-A17B-NVFP4 (MoE, 512 experts, 17B active)
- **vLLM**: 0.17.1rc1
- **With K=64**: `vllm-qwen35-k64:latest` (K=64 CUTLASS tiles for SM120 99KB SMEM)
- **Without K=64**: `voipmonitor/llm-pytorch-blackwell:nightly-cuda132` (stock vLLM)

## Server Configuration (identical for both)

```bash
--tensor-parallel-size 4
--gpu-memory-utilization 0.90
--max-num-batched-tokens 16384
--max-num-seqs 128
--max-model-len 262144
--num-gpu-blocks-override 1024
--enable-prefix-caching
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

## Benchmark 1: `vllm bench serve` (completions endpoint, no thinking)

```bash
vllm bench serve --backend vllm --base-url http://localhost:9200 \
  --model qwen3.5-397b-nvfp4 \
  --tokenizer /path/to/sehyo-qwen35-nvfp4 \
  --endpoint /v1/completions \
  --dataset-name random --random-input-len 128 --random-output-len 256 \
  --num-prompts <50|100> --max-concurrency <1|8> \
  --request-rate inf --num-warmups 5 --temperature 0
```

### Results — Output Throughput (tok/s)

| Concurrency | K=64 | Stock | Delta |
|-------------|------|-------|-------|
| 1 user | 171 | 161 | +6% |
| 8 users (system) | 648 | 652 | ~0% |

### Detailed Metrics

| Metric | K=64 (1-user) | Stock (1-user) | K=64 (8-user) | Stock (8-user) |
|--------|---------------|----------------|---------------|----------------|
| Output tok/s | 170.74 | 160.75 | 648.14 | 651.74 |
| Median TPOT | 5.29 ms | 5.34 ms | 10.61 ms | 10.88 ms |
| Median TTFT | 62.87 ms | 66.05 ms | 142.89 ms | 144.33 ms |
| Median ITL | 18.90 ms | 19.05 ms | 36.72 ms | 36.76 ms |
| MTP acceptance | 79.01% | 73.57% | 77.74% | 79.16% |
| Avg accepted | 3.37 | 3.21 | 3.33 | 3.37 |

## Benchmark 2: `llm_decode_bench.py --no-think` (chat endpoint, thinking disabled)

```bash
python3 scripts/llm_decode_bench.py --port 9200 \
  --model qwen3.5-397b-nvfp4 \
  --concurrency 1,2,4,8 --contexts 0 \
  --max-tokens 1024 --duration 30 --no-think
```

### Results — Aggregate Throughput (tok/s)

| Concurrency | K=64 | Stock | Delta |
|-------------|------|-------|-------|
| 1 | 144.2 | 139.2 | +4% |
| 2 | 266.5 | 230.4 | +16% |
| 4 | 408.4 | 414.1 | -1% |
| 8 | 656.7 | 611.2 | +7% |

## Analysis

### K=64 shows minimal impact with MTP=3 at short context

With MTP=3 speculative decoding enabled, the K=64 CUTLASS kernel advantage is **+4-8% at most** and within run-to-run noise for most configurations. This is because:

1. **MTP overhead dominates**: Each decode step runs 3 draft forward passes through the MTP heads. The MoE GEMM kernel (where K=64 helps) is a fraction of the total step time.
2. **Short context (128 tokens)**: At short input lengths, the MoE GEMM is not the bottleneck — attention, all-reduce, and MTP verification are comparatively larger.
3. **FlashInfer CUTLASS already autotuning**: The stock image uses FlashInfer's CUTLASS backend for MoE, which has its own autotuner that selects reasonable tile sizes at startup.

### Where K=64 matters more

Previous benchmarks (using `curl`-based tests with longer outputs) showed much larger gains because:
- Those tests measured **all output tokens including thinking/reasoning**, which inflate throughput numbers
- Longer output sequences (1K+ tokens) accumulate more MoE GEMM calls per request
- The K=64 advantage compounds with longer decode chains

### The MTP=0 baseline test (see k64_vs_stock_mtp0.md) isolates the raw kernel impact

Without speculative decoding overhead, the K=64 kernel difference should be more visible since MoE GEMM becomes a larger fraction of the decode step.

## Reproducing

1. **With K=64**: `docker pull verdictai/vllm-blackwell-k64:latest`
2. **Without K=64**: `docker pull voipmonitor/llm-pytorch-blackwell:nightly-cuda132`
3. Use identical server config (see above)
4. Install bench tool: `pip install vllm[bench]`
5. Run `vllm bench serve` commands above
6. For decode bench: `python3 scripts/llm_decode_bench.py --no-think ...`

Date: 2026-03-15
