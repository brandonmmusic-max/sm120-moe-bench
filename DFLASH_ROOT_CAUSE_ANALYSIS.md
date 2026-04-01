# DFlash 0% Acceptance — Root Cause Analysis

## Summary

The v2 DFlash drafter gets 0-1.2% acceptance in vLLM despite 99.94% training accuracy. Multiple bugs were found and fixed (5 total), but the primary remaining issue is **TP sharding of embed_tokens zeroing out the mask token embedding on 3 out of 4 ranks**.

## Bug Timeline

### Bug 1: Config Layer IDs (FIXED)
`eagle_aux_hidden_state_layer_ids` was [3,15,27,39,51] instead of [2,14,26,38,50]. vLLM extracted wrong hidden state layers.

### Bug 2: CUDA Graph Context (FIXED)
dummy_run captured CUDA graphs with ctx_len=0, replaying context-less forward. Fixed with `cudagraph_runtime_mode = None` for DFlash, plus `--enforce-eager`.

### Bug 3: Causal Attention Masking (FIXED)
FlashInfer enforces causal masking. DFlash needs bidirectional attention. Fixed by bypassing to `F.scaled_dot_product_attention(q, k, v, is_causal=False)`.

### Bug 4: Context Accumulation (FIXED)
vLLM's eagle proposer only passed latest token as context. DFlash needs accumulated target hidden states. Fixed in eagle.py — context now grows across decode steps (ctx=11→28→45→...).

### Bug 5: Position IDs (FIXED)
Draft model was receiving positions=[0,0,0,...]. Fixed to generate [0..ctx+query-1].

### Bug 6: TP Embed_tokens Sharding (**CURRENT — NOT FIXED**)

With TP=4, `VocabParallelEmbedding` shards the 248320 vocab across 4 ranks (62080 each):
- Rank 0: tokens 0–62079
- Rank 1: tokens 62080–124159
- Rank 2: tokens 124160–186239
- Rank 3: tokens 186240–248319

Mask token 248070 is on **rank 3 only**. Ranks 0-2 return zeros.

**Evidence:**
- `embed mean=0.0000 std=0.0000` on ALL ranks in container logs
- `FINAL output mean=0.0000 std=0.0000` — entire draft model output is zeros
- Embedding exists on disk: `norm=0.232857`, `std=0.003639` (non-zero)
- Position 0 (real anchor tokens) achieves 20% acceptance — proves model works
- Positions 1-15 (mask token 248070) all at 0% — zeros in, zeros out

**Fix options:**
1. Replace `VocabParallelEmbedding` with replicated `nn.Embedding` for draft model (~2GB)
2. Ensure all-reduce fires after draft model's embedding lookup
3. Quick test: change mask_token_id to <62080 to confirm theory

## Current Acceptance Rates

| Position | Rate | Notes |
|----------|------|-------|
| 0 (anchor) | 20% | Real token, may land on correct rank |
| 1-15 (mask) | 0% | Token 248070 → zeros on ranks 0-2 |
| Overall | 1.2% | Only position 0 contributes |

## Previous Theory: NVFP4 Hidden State Mismatch

The SSH session's analysis (`DFLASH_QWEN35_397B_ROOT_CAUSE.md`) blamed bf16→NVFP4 hidden state distribution mismatch. While NVFP4 quantization does change hidden states, the 20% acceptance at position 0 is strong evidence the model CAN interpret NVFP4 hidden states well enough. The TP sharding bug is the more likely primary cause. NVFP4 mismatch may be a secondary factor.

## Files Modified

- `/home/brandonmusic/models/dflash-397b-v2/config.json` — fixed layer IDs
- Container `vllm-qwen35-k64:dflash-patched`:
  - `qwen3_dflash.py` — SDPA is_causal=False bypass
  - `eagle.py` — context accumulation, position ID generation
  - `eagle3_utils.py` — aux layer config reader
