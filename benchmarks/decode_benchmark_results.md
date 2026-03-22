# SM120 Decode Benchmark Results

## Hardware
- 4x NVIDIA RTX PRO 6000 Blackwell Max-Q (96GB, 188 SMs, SM120)
- Driver 595.45.04, CUDA 13.2
- Threadripper 24C/48T

## Model
- Qwen3.5-397B-A17B-NVFP4 (lukealonso quantization)
- TP=4, MTP=3, FP8 KV cache
- max_model_len=262144

## vLLM Config
- vLLM 0.17.1rc1 (custom K=64 CUTLASS patches)
- Image: vllm-qwen35-k64:latest
- gpu_mem_util=0.90
- max_num_batched_tokens=8192, max_num_seqs=96

## Backend: FLASHINFER_CUTLASS (current)

Single-user decode, 512 output tokens, steady-state:

| Run | Tokens | Time (ms) | Tok/s |
|-----|--------|-----------|-------|
| 4 | 512 | 3554 | 144.1 |
| 5 | 512 | 3570 | 143.4 |
| 6 | 512 | 3644 | 140.5 |
| 7 | 512 | 3526 | 145.2 |
| 8 | 512 | 3661 | 139.9 |
| **Avg** | **512** | **3591** | **142.6** |

Note: First request ~58 tok/s (CUDA graph warmup), reaches steady-state by run 4.
Debug logging enabled (VLLM_LOGGING_LEVEL=DEBUG) may reduce throughput ~15%.

## Per-Layer MoE Latency (from baseline probe)
- Total: 52μs/layer (M=1 decode)
- Kernel launch overhead: 32μs (61% of total)
- GEMM1: 25μs, Activation: 8μs, GEMM2: 9μs, Reduce: 9μs

## CuteDSL Backend Attempt
- Patched FlashInfer CuteDSL gate for SM120 (0 tcgen05/TMEM references)
- Rejected by vLLM: "does not support ('standard',) activation format"
- CuteDSL uses BatchedExperts format, model needs Standard format
- Would need additional compatibility patch in vLLM
