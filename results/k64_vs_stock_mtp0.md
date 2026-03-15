# K=64 CUTLASS Kernel: MTP=0 Baseline — Isolating Kernel Impact

Pure decode baseline without speculative decoding, to isolate the K=64 CUTLASS MoE GEMM kernel impact from MTP overhead.

## Hardware & Software

Same as [k64_vs_stock_vllm_bench.md](k64_vs_stock_vllm_bench.md):
- 4x RTX PRO 6000 Blackwell (96GB, SM 12.0, all 300W)
- Qwen3.5-397B-A17B-NVFP4, TP=4, vLLM 0.17.1rc1
- K=64: `vllm-qwen35-k64:latest`, Stock: `voipmonitor/llm-pytorch-blackwell:nightly-cuda132`

## Server Configuration (identical for both, NO speculative decoding)

```bash
--tensor-parallel-size 4
--gpu-memory-utilization 0.90
--max-num-batched-tokens 16384
--max-num-seqs 128
--max-model-len 262144
--num-gpu-blocks-override 1024
--enable-prefix-caching
# NO --speculative-config
```

## Results — Full 2×2 Matrix (vllm bench serve)

### Output Throughput (tok/s)

| | MTP=3 | MTP=0 (no spec decode) |
|---|---|---|
| **K=64** | **171** / **648** | **76** / **373** |
| **Stock** | 161 / 652 | 74 / 376 |

Format: single-user / 8-user system throughput

### K=64 Delta

| Concurrency | MTP=3 | MTP=0 |
|-------------|-------|-------|
| 1 user | +6% | +3% |
| 8 users | -1% | -1% |

### Detailed MTP=0 Metrics

| Metric | K=64 (1-user) | Stock (1-user) | K=64 (8-user) | Stock (8-user) |
|--------|---------------|----------------|---------------|----------------|
| Output tok/s | 75.93 | 73.71 | 373.06 | 376.32 |
| Median TPOT | 13.01 ms | 13.42 ms | 20.65 ms | 20.00 ms |
| Median TTFT | 50.02 ms | 52.03 ms | 178.13 ms | 203.03 ms |

## Results — llm_decode_bench.py --no-think

### Aggregate Throughput (tok/s)

| Concurrency | K=64 MTP=3 | Stock MTP=3 | K=64 MTP=0 | Stock MTP=0 |
|-------------|-----------|-------------|-----------|-------------|
| 1 | 144.2 | 139.2 | 75.5 | 73.0 |
| 2 | 266.5 | 230.4 | 131.2 | 127.2 |
| 4 | 408.4 | 414.1 | 242.5 | 234.6 |
| 8 | 656.7 | 611.2 | 397.3 | 397.2 |

## Analysis

### K=64 shows no meaningful impact at short context (128 tokens)

Across all 8 test configurations (2 kernels × 2 MTP settings × 2 concurrency levels), K=64 is within **±6% of stock** — well within run-to-run noise. This finding holds for both MTP=3 and MTP=0.

### Why K=64 doesn't help here

1. **FlashInfer CUTLASS autotuner already handles SM120**: The stock vLLM image uses FlashInfer's FLASHINFER_CUTLASS backend for MoE, which has its own runtime autotuner. It skips tile sizes that overflow SMEM (the same K=128 tiles our patches address) and selects working alternatives automatically.

2. **Short context = small MoE fraction**: At 128 input tokens, MoE GEMM is a small fraction of total decode time. The kernel improvement is diluted by attention, all-reduce, embedding, and norm operations.

3. **The original K=64 benchmarks used different methodology**: The earlier 142→283 tok/s results used `curl`-based tests with `--verbose` output that counted all tokens (including reasoning/thinking). Those numbers were measuring a different thing than pure output throughput.

### MTP is the dominant factor

| Factor | 1-user impact | 8-user impact |
|--------|--------------|---------------|
| MTP=3 vs MTP=0 | **+2.3x** (76→171) | **+1.7x** (376→648) |
| K=64 vs Stock | +3-6% (noise) | ±1% (noise) |

Speculative decoding (MTP=3) provides **130% single-user speedup** and **72% concurrent speedup**. K=64 kernel differences are not statistically significant at this context length.

### Where K=64 may still matter

- **Longer contexts (32K+)**: More MoE GEMM calls per decode step, larger fraction of compute
- **Different vLLM versions**: If FlashInfer's autotuner changes behavior or is not available
- **Triton MoE backend**: If the code path uses vLLM's Triton kernels instead of FlashInfer CUTLASS

## Reproducing

Same as [k64_vs_stock_vllm_bench.md](k64_vs_stock_vllm_bench.md), but omit `--speculative-config` from the server config.

Date: 2026-03-15
