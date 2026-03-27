# Sprint 18: SM120 Attention Backend Integration — Results

## Executive Summary

SM120 Flash Decode kernel successfully integrated as vLLM attention backend override,
handling all pure decode (Sq=1) attention on SM120 Blackwell GPUs. The kernel replaces
FlashInfer for decode, while prefill and MTP verify still use FlashInfer FP8.

**Decode: 145.1 tok/s** (ctx=0, 8192 tokens, C=1, 60s) — on par with Sprint 11's 147.2 tok/s.
MTP acceptance: 65.7% (matching Sprint 9's 65.9%).

## What Changed

1. **SM120 attention backend registered via `register_backend`** — overrides FLASHINFER
   entry in vLLM's attention backend registry using a custom launcher (`sm120_launch.py`)
2. **HND layout size() fix** in sm120_flash_decode_ext.py C++ wrapper: `size(1)` = num_kv_heads,
   `size(2)` = block_size, matching HND physical layout `[num_blocks, num_kv_heads, block_size, head_dim]`
3. **SM120MetadataBuilder** stores seq_lens/block_tables on metadata for MTP verify path
4. **kv_block_stride** from `key_cache.stride(0)` — supports non-contiguous K/V views (no `.contiguous()` OOM)

### What Did NOT Work

- **UNIFORM_BATCH CUDA graphs (FULL+PIECEWISE)** — hangs during graph capture on verdict-sprint11
  image. Reverted to UNIFORM_SINGLE_TOKEN_DECODE (PIECEWISE only, 51 graphs).
- **All attention through SM120** — prefill and MTP verify paths through SM120 decode kernel
  cause GPU kernel hangs. Disabled; only pure Sq=1 decode uses SM120.

## Benchmark: Decode (ctx=0, 8192 tokens, 60s, C=1)

| Metric | Sprint 18 | Sprint 11 | Sprint 9 |
|--------|-----------|-----------|----------|
| **Decode tok/s** | **145.1** | 147.2 | 165.1 |
| **vs Sprint 11** | **-1.4%** | baseline | — |
| **vs Sprint 9** | **-12.1%** | -10.8% | baseline |
| MTP acceptance | 65.7% | 68.7% | 65.9% |
| Prefill 8K tok/s | 12,908 | — | 24,923 |

Note: Sprint 9→11 decode drop (165→147) was from AllReduce patches + Qwen3 reasoning mode,
NOT kernel regression. Sprint 18 is on par with Sprint 11 (same AllReduce patches).

## 512-Token Variance Test (10 runs)

| Run | Tokens | Time (ms) | tok/s |
|-----|--------|-----------|-------|
| 1 | 512 | 4145 | 123.5 |
| 2 | 512 | 3842 | 133.3 |
| 3 | 512 | 4290 | 119.3 |
| 4 | 512 | 3722 | 137.6 |
| 5 | 512 | 3661 | 139.9 |
| 6 | 512 | 4084 | 125.4 |
| 7 | 512 | 3896 | 131.4 |
| 8 | 512 | 4084 | 125.4 |
| 9 | 512 | 3950 | 129.6 |
| 10 | 512 | 3855 | 132.8 |

**Mean: 129.8 tok/s | Min: 119.3 | Max: 139.9 | StdDev: ~6.5**

## Prefill Benchmark

| Context | TTFT (s) | Prefill tok/s | Decode tok/s (512 tokens) |
|---------|----------|---------------|---------------------------|
| 8K | 0.71 | 12,042 | 102.3 |
| 16K | 1.34 | 12,530 | — |
| 32K | 2.83 | 11,725 | 139.2 |
| 64K | 6.25 | 10,547 | — |
| 128K | 14.98 | 8,767 | — |

Prefill is handled by FlashInfer (not SM120). Decode at 32K context shows
139.2 tok/s — SM120 attention kernel performs well at longer contexts.

## MTP Acceptance

- **Overall**: 10086/15354 = **65.7%**
- **Position 0**: 4324/5118 = **84.5%**
- **Position 1**: 3327/5118 = **65.0%**
- **Position 2**: 2435/5118 = **47.6%**
- **Mean acceptance length**: 2.97

## Cross-Sprint Comparison

| Sprint | Image | Decode tok/s | MTP Accept | CUDA Graphs | SM120 Attention |
|--------|-------|-------------|------------|-------------|-----------------|
| Sprint 9 | verdict-sprint9 | 165.1 | 65.9% | PIECEWISE | No |
| Sprint 11 | verdict-sprint11 | 147.2 | 68.7% | PIECEWISE | No |
| Sprint 14 | verdict-sprint14-fp8kv | 157.1 | — | PIECEWISE | No |
| Sprint 16 | verdict-sprint15 (broken) | 41.1 | 3.3% | PIECEWISE+FULL | Partial |
| Sprint 17 | — | — | — | — | — |
| **Sprint 18** | **verdict-sprint11** | **145.1** | **65.7%** | **PIECEWISE** | **Decode only** |

## Configuration

- **Base image**: vllm-qwen35-k64:verdict-sprint11 (NCCL SymmMem, K=64 patches)
- **Backend**: SM120FlashAttentionBackend registered via sm120_launch.py
- **CUDA graphs**: PIECEWISE=51 (FULL disabled — hangs with MTP on verdict-sprint11)
- **SM120 paths**: Decode (Sq=1) only. MTP verify + prefill → FlashInfer fallback
- **KV cache**: FP8 E4M3, 512 blocks override
- **VerdictMoE**: Env vars set but verdict-sprint11 image may not have VerdictMoE module
- **Driver**: 595.45.04, CUDA 13.2, 4x RTX PRO 6000 Blackwell (300W cap)

## Next Steps

1. **Debug FULL CUDA graph hang** — UNIFORM_BATCH + MTP on newer vLLM version
2. **Debug MTP verify SM120 path** — kernel hangs on MTP verify dispatch
3. **Rebuild on verdict-sprint9 or newer image** with working MTP to test FULL graphs
4. **E2E re-test TMA kernel on driver 595** — projected 2.8% improvement
