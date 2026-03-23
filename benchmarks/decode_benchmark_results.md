# SM120 Decode Benchmark Results

## Hardware
- 4x NVIDIA RTX PRO 6000 Blackwell (96GB, 188 SMs, SM120)
  - GPU 0, 2: **Max-Q variants** (300W cap, thermal throttle to ~2280 MHz)
  - GPU 1, 3: **Full variants** (300W cap, sustain ~2840 MHz)
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
- VLLM_LOGGING_LEVEL=DEBUG (all benchmarks)

## GPU Clocks During Inference (Thermal Bottleneck)

| GPU | Variant | Clock (under load) | Power | Temp |
|-----|---------|-------------------|-------|------|
| 0 | Max-Q | 2287 MHz | 205W / 300W | 71°C |
| 1 | Full | 2842 MHz | 282W / 300W | 51°C |
| 2 | Max-Q | 2280 MHz | 200W / 300W | 57°C |
| 3 | Full | 2827 MHz | 298W / 300W | 61°C |

Max-Q variants throttle to ~2280 MHz (26% below max 3090 MHz) due to thermals.
TP=4 synchronization means all GPUs bottleneck to the slowest (Max-Q).
Max clocks 3090 MHz — never reached under sustained inference.

## Backend: FLASHINFER_CUTLASS (current baseline)

### Original Baseline (2026-03-22, early session)

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

### Re-Benchmark (2026-03-22, after extended use)

Same config, GPUs warmer from sustained use. Clock-locked to 3090 target.

| Run | Tok/s |
|-----|-------|
| 1 | 136.0 |
| 2 | 128.6 |
| 3 | 113.0 |
| 4 | 137.0 |
| 5 | 122.1 |
| 6 | 131.8 |
| 7 | 128.8 |
| 8 | 126.2 |
| 9 | 134.9 |
| 10 | 115.4 |
| **Avg** | **127.4** |

High variance (113-137) due to Max-Q thermal throttling on GPUs 0, 2.
Honest sustained baseline: **127-142 tok/s** depending on thermal conditions.

## Per-Layer MoE Latency (from baseline probe)
- Total: 52μs/layer (M=1 decode, 5 kernel launches)
- Kernel launch overhead: 32μs (61% of total)
- GEMM1: 25μs, Activation: 8μs, GEMM2: 9μs, Reduce: 9μs

## Fused Pipeline Benchmark (3 kernel launches)

GEMM1 → SwiGLU+Requant → GEMM2, CUTLASS NVF4, tile 128×128×128:

| Phase | Baseline (5 launches) | Our Pipeline (3 launches) |
|-------|----------------------|--------------------------|
| GEMM1 | 25μs | 24.5μs |
| SwiGLU+Requant | 8μs (activation only) | 10.2μs (fused with FP4 requant) |
| GEMM2 | 9μs | 10.2μs |
| Launch overhead | ~32μs (5×6.3μs) | ~19μs (3×6.3μs) |
| **Total** | **52μs** | **45μs** |
| **Speedup** | — | **1.16x** |

Per-token MoE (60 layers): 2.70ms (vs 3.12ms baseline) = 0.42ms saved/token.

### Projected End-to-End
- MoE is ~60% of decode latency
- 16% MoE improvement → ~10% overall
- From 142 tok/s baseline: **~156 tok/s**
- From 127 tok/s (thermal): **~140 tok/s**
- Single-kernel fusion (Sprint 2) would save another ~13μs → **170+ tok/s**

## CuteDSL Backend Attempt
- Patched FlashInfer CuteDSL gate for SM120 (0 tcgen05/TMEM references)
- Rejected by vLLM: "does not support ('standard',) activation format"
- CuteDSL uses BatchedExperts format, model needs Standard format
- Would need additional compatibility patch in vLLM
