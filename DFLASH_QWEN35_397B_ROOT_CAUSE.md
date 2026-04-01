# DFlash Qwen3.5-397B: Root Cause Analysis — 0% Acceptance Rate

## Summary

The DFlash v2 draft model achieves 99.94% accuracy during training but gets **0% acceptance** in vLLM. After exhaustive debugging, the root cause is a **training/serving distribution mismatch**: the model was trained on bf16 hidden states but the serving model uses NVFP4 quantization.

## Root Cause

**The draft model was trained on hidden states from `Qwen/Qwen3.5-397B-A17B` (bf16, ~800GB VRAM, trained on RunPod 7xH200), but the vLLM deployment serves `lukealonso-qwen35-nvfp4` (NVFP4 quantized, fits 4x96GB).**

NVFP4 quantization changes the intermediate hidden states at each layer. While the statistics are similar (mean~0, std~0.26), the per-dimension values differ enough that the draft model's attention layers — trained on the bf16 distribution — cannot extract meaningful predictions from the NVFP4 distribution.

## Evidence

| Check | Result | Status |
|-------|--------|--------|
| Layer IDs | [2,14,26,38,50] match training script | ✅ Fixed |
| fc+norm output | vLLM matches reference (diff = 0.0) | ✅ Correct |
| Model weights | Checksums match between disk and container | ✅ Correct |
| lm_head | Identical between training and target (diff = 0.0) | ✅ Correct |
| embed_tokens | Identical between training and target | ✅ Correct |
| Attention | SDPA is_causal=False, bypasses vLLM backend | ✅ Correct |
| RoPE | Per-token cos/sin matches reference | ✅ Correct |
| Positions | [0..ctx+query-1] matches training format | ✅ Correct |
| Context passing | hs_len=11,28,45... (verified via PROPOSE log) | ✅ Correct |
| **NVFP4 hidden states** | Reference model with NVFP4 context → wrong tokens | ❌ **Root cause** |

### Key Experiment

Running the **exact reference DFlashDraftModel.forward()** with the dumped vLLM inputs:
```
Reference model + NVFP4 context → tokens [121104, 121104, 166152, ...] ('相信自己相信自己udir...')
Reference model + zero context  → tokens [95897, 95897, ...] (garbage)
Reference model + random context → tokens [91988, 99118, ...] (garbage)
```

The target model actually generated: `<think>\n\n</think>\n\n1, 2, 3, 4, 5.`

**The draft model produces wrong predictions with ANY context — but DIFFERENT wrong predictions depending on the context, proving it IS reading the context but can't interpret it.**

## Bugs Found and Fixed Along the Way

1. **Config `eagle_aux_hidden_state_layer_ids` was [3,15,27,39,51]** — didn't match training IDs [2,14,26,38,50]. Fixed to [2,14,26,38,50].
2. **Config `dflash_config.target_layer_ids` was [3,15,27,39,51]** — same issue. Fixed to [2,14,26,38,50].
3. **CUDA graph captures DFlash forward with 0 context** — the dummy_run sets ctx_len=0, causing CUDA graphs to replay a context-less forward. Fixed by setting `cudagraph_runtime_mode = None` for DFlash.

## Fix Plan

### Option 1: Retrain on NVFP4 Hidden States (Recommended)
1. Extract hidden states from the NVFP4 model using the running vLLM server
2. Generate ~100K training samples with `output_hidden_states=True`
3. Retrain the draft model (all layers, not just fc) on NVFP4 hidden states
4. Expected: ~5-10 hours on 7xH200 RunPod pod
5. Redeploy updated model

### Option 2: Online Calibration
1. Collect (hidden_states, target_tokens) pairs from the running server
2. Fine-tune fc+hidden_norm as an adapter
3. Faster but lower quality — fine-tuning fc alone reached only 26.67% on 1 example

### Option 3: Alternative Spec Decode
- Use N-gram speculation (no trained model needed, ~1.5x speedup)
- Use MTP (Multi-Token Prediction) if Qwen3.5 supports it
- These don't have the quantization mismatch issue

## Architecture Notes

- Target model: `Qwen3_5MoeForConditionalGeneration` with 60 layers (45 linear_attention/GDN + 15 full_attention), 512 experts
- Draft model: 5 standard Qwen3 attention layers with fc projection (20480→4096)
- Captured layers [2,14,26,38,50] are all `linear_attention` (GDN) type
- Hidden states from GDN layers may have different sensitivity to weight quantization than standard attention layers

## Files Modified

- `/home/brandonmusic/models/dflash-397b-v2/config.json` — fixed layer IDs
- Container image `vllm-qwen35-k64:dflash-patched` — SDPA attention bypass, CUDA graph fix, diagnostic logging
