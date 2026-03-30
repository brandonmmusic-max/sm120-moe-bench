# Sprint 20: Optimization Experiments

**Date:** 2026-03-29
**Hardware:** 4x NVIDIA RTX PRO 6000 Blackwell (96GB each, SM 12.0, PCIe)
**Model:** Qwen3.5-397B-A17B-NVFP4, TP=4, MTP speculative decoding

---

## Baseline

| Config | Single-user tok/s | MTP Accept | Notes |
|--------|------------------|-----------|-------|
| Sprint 9 (0.17.x) | 165.1 | 65.9% | CUTLASS + VerdictMoE |
| Sprint 20 (0.18.x, broken MTP) | 107-115 | 38-43% | build_for_drafting bug |
| **Sprint 20 BFD fix** | **153.7** | **60%+** | FlashInfer XQA, PIECEWISE CG |

## Experiments Conducted

### 1. FlashInfer GDN Decode (NEGATIVE RESULT)
- Replaced FLA Triton kernels with FlashInfer `gated_delta_rule_decode_pretranspose` for 45 GDN layers
- **Result:** No MTP improvement, throughput dropped to ~70 tok/s (gather/scatter overhead from non-contiguous mamba cache)
- **Conclusion:** GDN kernel choice irrelevant for MTP; root cause was full_attention layer mismatch

### 2. FULL Cudagraph Mode (BLOCKED)
- SM120 backend declares `UNIFORM_BATCH` → FULL CG mode enabled
- **FULL capture succeeds** (49/49 PIECEWISE + 49/49 FULL)
- **But output is garbage** — two root causes identified:
  - **GDN branch mismatch:** CUDA graph captures spec-decode kernel path during warmup; runtime non-spec batches need different path but graph replays captured kernels with stale metadata
  - **FlashInfer attention padding:** FULL mode uses `pad_attn=True` which corrupts FlashInfer's decode wrapper for the 15 full_attention layers
- **GDN fix written:** Non-spec → spec conversion in metadata builder (patch_gdn_full_cg.py)
- **With GDN fix:** 1-6% MTP acceptance (slightly better but FlashInfer padding still corrupts)
- **Conclusion:** FULL CG requires BOTH GDN fix AND an attention backend that handles padded batches (TRTLLM or fixed SM120 kernel)

### 3. SM120 Custom Attention Backend (REGRESSION)
- Registered SM120 backend via `sm120_launch.py` for the 15 full_attention layers
- **Result:** 101-120 tok/s, MTP acceptance dropped to 36-46%
- **Root cause:** SM120 decode kernel (scalar dot products) produces different numerical output than FlashInfer XQA (MMA tensor cores), degrading draft/target agreement
- FlashInfer XQA is faster AND produces better MTP acceptance

### 4. GPU Power Limits (MARGINAL)
- Raised GPU 1/3 from 300W → 450W (full variants rated 600W)
- **Result:** ~155 tok/s (vs 153.7 baseline) — marginal at single-user decode
- GPUs only draw ~200W during single-user decode; power limit matters more at high concurrency
- MTP acceptance slightly higher (73% avg vs 60% baseline) — possibly variance

### 5. NCCL SymmMem AllReduce (PARTIALLY WORKING)
- SM120 excluded from `all_reduce_utils.py` and `symm_mem.py` size tables
- **Patches created:** Added "12.0" to `SYMM_MEM_ALL_REDUCE_MAX_SIZES` and `_WORLD_SIZES_MULTIMEM`
- After patching: "Device capability not supported" → "multicast not supported" (progress!)
- Multicast requires NVLink (expected failure on PCIe)
- Unicast path may still be active — needs benchmarking

### 6. Mamba Cache Mode (ALREADY ACTIVE)
- Article reference: Google Cloud blog claims `--mamba-cache-mode=align` is critical for Qwen3.5
- **Investigation:** vLLM auto-sets `mamba_cache_mode="align"` when `enable_prefix_caching=True`
- We already had this active through all Sprint 20 runs — NOT a new optimization
- Explicit flag is redundant but harmless

### 7. NVFP4 KV Cache (BLOCKED AT C++ LEVEL)
- Patched vLLM Python config to accept `--kv-cache-dtype nvfp4` (3 files)
- **Blocked by:** C++ `reshape_and_cache_flash` kernel doesn't support nvfp4 dtype
- Fused CUDA quantize kernel written (`reshape_and_cache_nvfp4.cu`) but not yet integrated
- Would halve KV cache memory (0.5625 vs 1.0 bytes/element)

---

## Architecture Analysis

### Decode Step Time Budget (from Sprint 11 nsys profiling)
- **Compute (MoE + Attention):** ~78% of decode time
  - MoE GEMM (VerdictMoE): ~22% (17.9 μs/layer × 60 layers)
  - Attention (FlashInfer XQA): ~15-20% (15 full_attention layers)
  - GDN (FLA Triton): remaining 45 layers
- **NCCL AllReduce:** corrected from "69%" to much lower (nsys GPU kernel % ≠ wall-clock)
- **Framework overhead:** CUDA graph dispatch, metadata building

### FlashInfer XQA vs SM120 Custom Kernel
| Feature | XQA (FlashInfer) | SM120 Custom |
|---------|-----------------|--------------|
| Dot products | MMA tensor cores | Scalar + shuffle reduce |
| Pipeline | Warp-specialized (gemm0/gemm1 overlap) | Sequential |
| KV loading | Double-buffered + mbarrier | Single-buffer cp.async |
| Split-KV reduce | Fused semaphores (1 kernel) | Separate reduce kernel |
| TFLOPS | ~95% SDPA | 83% SDPA (190 TFLOPS) |

### SM120 Kernel Variants (all in sm120-fa/csrc/)
- `sm120_flash_attn_ws.cu` — Warp-specialized v1 (2 producer + 2 consumer), MMA
- `sm120_flash_attn_ws2.cu` — Warp-specialized v2 (cp.async overlap), MMA
- `sm120_flash_attn_fa3.cu` — FA3-style (1 TMA producer + 7 MMA consumers)
- `sm120_flash_decode_v2_paged.cu` — Production decode (scalar, NOT MMA)
- `sm120_flash_decode_v3_mma.cu` — NEW: MMA decode (written Sprint 20, untested)
- FA3 WS finding: "HURTS on SM120 — losing 1/8 compute > overlap benefit"

---

## Docker Images

| Image | Config | tok/s | Status |
|-------|--------|-------|--------|
| `sprint20-bfd-fix` | BFD fix, FlashInfer, PIECEWISE | **153.7** | **STABLE PRODUCTION** |
| `sprint20-full-cg` | + UNIFORM_BATCH override | garbage | GDN branch mismatch |
| `sprint20-full-cg-fix` | + GDN branch fix | garbage | FlashInfer padding issue |
| `sprint20-optimized` | + GDN fix in image | 101-120 | SM120 attn regression |
| `sprint20-full-cg-nvfp4` | + NVFP4 KV cache | crash | C++ reshape_and_cache_flash |

---

## Remaining Optimization Levers

| Lever | Expected Gain | Effort | Status |
|-------|--------------|--------|--------|
| SymmMem AllReduce (unicast) | 3-5% | Done | Patches applied, needs benchmark |
| TMA VerdictMoE (driver 595) | ~3% | 1 day | Needs interface port |
| FLA Triton GDN autotuning | 2-5% | 30 min | `num_warps=4` hardcoded |
| MMA decode kernel v3 | up to 10% on attn | Written | Needs test + integration |
| NVFP4 KV cache | capacity gain | C++ work | Kernel written, needs integration |
| EAGLE-3 speculative decoding | 50-100% | 2-3 weeks | Requires training |
| CLC persistent kernels | 5-10% | Research | SM120 supported |
| Hybrid TP-attn + EP-MoE | 8-15% | 2-3 weeks | Halves AllReduce |

---

## Key Discoveries

1. **build_for_drafting (PR 36060)** was the sole cause of MTP regression: PREFILL/DECODE attention mismatch on 15 full_attention layers
2. **FULL cudagraphs + MTP** fundamentally broken on FlashInfer (padding corruption) AND GDN layers (branch mismatch) — two independent bugs
3. **GDN kernel choice irrelevant** for MTP acceptance — both FLA Triton and FlashInfer GDN produce equivalent results
4. **SM120 custom attention is slower than FlashInfer XQA** — scalar dot products vs MMA tensor cores
5. **GPU power limits don't help single-user** — GPUs don't draw enough at batch=1
6. **mamba-cache-mode=align already auto-active** when enable_prefix_caching=True
7. **SymmMem excluded SM120** in 3 places: `supports_trtllm_attention()`, `SYMM_MEM_ALL_REDUCE_MAX_SIZES`, `_WORLD_SIZES_MULTIMEM`
