# Sprint 16: SM120 Attention CUDA Graph Fix + HND Layout Support

**Date:** 2026-03-27
**Hardware:** 4x NVIDIA RTX PRO 6000 Blackwell (96GB, SM 12.0, PCIe)
**Image:** vllm-qwen35-k64:verdict-sprint15
**Model:** Qwen3.5-397B-A17B-NVFP4

## What Changed

1. **Fixed `kv_cache.unbind(0)` crash** — KV cache is `[num_blocks, 2, ...]`, not `[2, ...]`. Changed to `kv_cache[:, 0]` / `kv_cache[:, 1]`
2. **Fixed `FIPrefill` attribute error** — Added `isinstance(TRTLLMPrefill)` check, stored metadata via `SM120MetadataBuilder.build()` override
3. **Fixed `.contiguous()` OOM** — Added `kv_block_stride` parameter to CUDA kernel to avoid 724 MiB copy during CUDA graph capture
4. **Fixed HND layout support** — vLLM uses HND layout on Blackwell (`stride_order = (0, 1, 3, 2, 4)`). Updated kernel access pattern: `blk * kv_block_stride + (kv_head * block_size + page_off) * HEAD_DIM + d`
5. **Removed MTP verify SM120 path** — `repeat_interleave` is not CUDA-graph-safe. MTP verify falls through to FlashInfer
6. **Added csrc volume mount** to `run_vllm.sh` for CUDA source updates
7. CUDA graph support: FULL_AND_PIECEWISE (PIECEWISE=49, FULL=41)
8. FP8 E4M3 KV cache + VerdictMoE + NCCL TREE_THRESHOLD=0

## Results

### CRITICAL FINDING: `verdict-sprint15` Image MTP Regression

MTP acceptance is **3.3%** — catastrophic regression from Sprint 11's 68.7%. This was verified to occur **with SM120 decode disabled** (pure FlashInfer), confirming it's a base image issue, NOT the SM120 kernel.

The `verdict-sprint15` image likely has a vLLM MTP regression or incompatibility with Qwen3.5 thinking mode.

### llm_decode_bench (ctx=0, 8192 tokens, 60s, concurrency=1)
- **Decode: 41.1 tok/s** (regression due to ~0% MTP acceptance)
- **MTP acceptance: 3.3%** (9/270 tokens)
- Per-position acceptance: pos0=7.8%, pos1=2.2%, pos2=0.0%
- Prefill 8K: 25,127 tok/s (comparable to Sprint 9's 24,923)

### Cross-Sprint Comparison
| Sprint | Config | tok/s | MTP Accept | Notes |
|--------|--------|-------|------------|-------|
| 9 | VerdictMoE, FlashInfer attn, no FP8 KV | 165.1 | 65.9% | verdict-sprint9 image |
| 11 | + NCCL fixes, reasoning parser | 147.2 | 68.7% | verdict-sprint11 image |
| 14 | + SM120 FA decode, FP8 KV | 157.1 | ~68% | verdict-sprint14 image |
| 16 prev | + SM120 MTP verify (eager, no FULL CG) | 156.7 | 64.9% | verdict-sprint15 image |
| **16 final** | **+ CUDA graph fixes, HND layout** | **41.1** | **3.3%** | **verdict-sprint15 image — MTP BROKEN** |
| **16 SM120 off** | **Pure FlashInfer (debug)** | **~43** | **~11%** | **Same image, confirms image regression** |

### CUDA Graph Status
```
Profiling CUDA graph memory: PIECEWISE=49 (largest=512), FULL=41 (largest=384)
Graph capturing finished in 48 secs, took 4.27 GiB
```

### SM120 Backend Status
```
SM120 decode kernel: enabled (head_dim=256, GQA=8:1, kv_dtype=fp8_e4m3)
SM120 Flash Decode kernel loaded successfully
VerdictMoE single fused cooperative extension compiled successfully
GPU KV cache size: 738,416 tokens
```

## Bugs Fixed (5 total)

1. `kv_cache.unbind(0)` → `kv_cache[:, 0]` — KV cache dim 0 is num_blocks, not K/V split
2. `FIPrefill has no attribute seq_lens` → isinstance check + metadata builder override
3. `.contiguous()` OOM during CUDA graph capture → `kv_block_stride` kernel parameter
4. `build()` signature mismatch → added `common_prefix_len` and `fast_build` params
5. HND layout: `(page_off * num_kv_heads + kv_head)` → `(kv_head * block_size + page_off)` for correct physical memory access

## Next Steps

1. **Root cause `verdict-sprint15` MTP regression** — likely vLLM update broke MTP token matching
2. **Re-benchmark on `verdict-sprint11` image** with Sprint 16 kernel fixes to isolate kernel performance
3. **E2E re-test TMA kernel on driver 595** — projected 2.8% improvement (from Sprint 11)
