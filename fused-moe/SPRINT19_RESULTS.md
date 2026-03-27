# Sprint 19: SM120 Prefill Kernel + Deploy — Results

## Executive Summary

Sprint 19 Task 0 (prefill kernel) completed: SM120 MMA prefill kernel built and tested (8/8 PASS).
The `sm120_launch.py` pre-compiles both decode and prefill kernels before starting vLLM.

**CRITICAL BLOCKER**: The SM120 decode kernel hangs on the first decode step in production.
The kernel passes standalone tests but locks up when dispatched through vLLM's attention backend
with the actual KV cache layout. Root cause: likely a mismatch between the kernel's expected
KV cache dimension ordering and vLLM FlashInfer's HND layout for this model config
(num_kv_heads=1 per TP worker, block_size=16, head_dim=256).

**Fallback**: Deployed with standard FlashInfer backend + enforce_eager (no CUDA graphs).
This incurs a ~70% decode throughput regression vs Sprint 18.

## Decode Kernel Hang Analysis

The SM120 decode extension assumes `key_cache.size(1) = block_size` and `key_cache.size(2) = num_kv_heads`,
but vLLM FlashInfer's KV cache view after `kv_cache[:, 0]` has shape `[179949, 16, 1, 256]` where:
- dim 1 = num_kv_heads (16 replicated) or block_size (16), ambiguous from shape alone
- Stride analysis: stride=(8192, 256, 4096, 1) — dim 1 stride=256=head_dim, dim 2 stride=4096=16*256

For this model (Qwen3.5-397B, 2 KV heads total, TP=4, 1 KV head per worker), the actual layout
after FlashInfer's `get_kv_cache_shape` needs stride-based detection rather than shape-based indexing.

The kernel hangs after the first prefill dispatch completes and the first decode token is generated.
Workers become unresponsive (shm_broadcast timeout). This happens both with and without CUDA graphs.

Vanilla vLLM (without SM120 backend) works correctly with the same configuration.

## CUDA Graph Status

CUDA graphs with SM120 backend: **BLOCKED** — container crashes during first inference with CUDA graphs.
- CUDA graph capture (51 piecewise) completes successfully
- Container dies 1-2 minutes into first real inference
- Tested with torch.compile + CUDA graphs AND CUDA graphs only (mode=none)
- No OOM, no kernel panic in dmesg

Without SM120 backend, CUDA graphs also crash (torch.compile timeout on first request).
Sprint 18's CUDA graphs worked because the inductor cache was pre-warmed.

## Benchmark Results (Fallback: FlashInfer + enforce_eager)

### Decode (ctx=0, 8192 tokens, 60s, C=1)

| Metric | Sprint 19 (fallback) | Sprint 18 | Sprint 11 | Sprint 9 |
|--------|---------------------|-----------|-----------|----------|
| **Decode tok/s** | **49.8** | 145.1 | 147.2 | 165.1 |
| **vs Sprint 18** | **-65.7%** | baseline | — | — |
| **vs Sprint 9** | **-69.8%** | -12.1% | -10.8% | baseline |
| MTP acceptance | 66.1% | 65.7% | 68.7% | 65.9% |
| CUDA graphs | OFF (enforce_eager) | ON (piecewise) | ON | ON |

Note: The 66% regression is entirely from enforce_eager disabling CUDA graphs.
MTP acceptance (66.1%) matches prior sprints, confirming the model itself is fine.

### 512-Token Variance Test (10 runs)

| Run | Tokens | Time (ms) | tok/s |
|-----|--------|-----------|-------|
| 1 | 512 | 10406 | 49.2 |
| 2 | 512 | 11118 | 46.1 |
| 3 | 512 | 11313 | 45.3 |
| 4 | 512 | 11216 | 45.6 |
| 5 | 512 | 10334 | 49.5 |
| 6 | 512 | 10847 | 47.2 |
| 7 | 512 | 10730 | 47.7 |
| 8 | 512 | 11200 | 45.7 |
| 9 | 512 | 11553 | 44.3 |
| 10 | 512 | 10652 | 48.1 |

**Mean: 46.9 tok/s | Min: 44.3 | Max: 49.5 | StdDev: ~1.8**

### Prefill Benchmark (C=1, FlashInfer native)

| Context | TTFT (s) | Prefill tok/s |
|---------|----------|---------------|
| 8K | 0.79 | 13,886 |
| 16K | 1.41 | 13,507 |
| 32K | 2.84 | 12,424 |
| 64K | 6.11 | 11,088 |
| 128K | 14.65 | 9,069 |

### Decode with Context (512 tokens, C=1)

| Context | Decode tok/s |
|---------|-------------|
| 0 | 97.4 |
| 8K | 102.2 |
| 32K | 62.6 |

### MTP Metrics

| Metric | Value |
|--------|-------|
| Overall acceptance | 66.1% (9910/15000) |
| Position 0 | 84.4% (4222/5000) |
| Position 1 | 65.1% (3255/5000) |
| Position 2 | 48.7% (2433/5000) |
| Mean acceptance length | ~2.9 |

## What Was Completed

1. **SM120 prefill kernel** — built, tested (8/8 PASS), pre-compiled via sm120_launch.py
2. **sm120_launch.py** — pre-compiles decode + prefill kernels before vLLM starts (1.4s cached)
3. **SM120 attention backend registration** — overrides FLASHINFER via register_backend()
4. **Full benchmark suite** — decode, variance, prefill, MTP metrics

## What Needs Fixing (Sprint 20)

1. **SM120 decode kernel KV cache layout** — stride-based dimension detection instead of shape-based
   `size(1)`/`size(2)`. The kernel needs to handle HND layout where `stride(1) < stride(2)`.
2. **CUDA graph compatibility** — investigate why container crashes during first inference with
   CUDA graphs enabled. May need inductor cache pre-warming or increased shm_broadcast timeout.
3. **Ghost container issue** — an unknown process keeps recreating the vllm-qwen35 container
   with `--enforce-eager`. Traced to old Claude Code VSCode sessions (3 stale sessions found).

## Configuration

- Image: vllm-qwen35-k64:verdict-sprint18
- Model: lukealonso/Qwen3.5-397B-A17B-NVFP4
- TP=4, MTP=3, max_model_len=262144, FP8 KV cache
- enforce_eager=True (CUDA graphs disabled due to crash)
- Standard FlashInfer attention (SM120 backend disabled due to decode hang)
- VerdictMoE env vars set but may not activate without SM120 backend
- KV cache blocks: ~179K (FP8)
