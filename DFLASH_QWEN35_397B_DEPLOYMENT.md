# DFlash Speculative Decoding - Qwen3.5-397B-A17B Deployment Notes

## Overview

DFlash is a novel speculative decoding method that uses a lightweight block diffusion model for drafting. This document covers deploying a custom DFlash drafter with the Qwen3.5-397B-A17B MoE model on vLLM (v0.18.x) with 4x RTX PRO 6000 Blackwell GPUs (TP=4).

## Model Architecture Compatibility

| Property | Qwen3.5-397B (target) | DFlash Drafter v2 |
|---|---|---|
| hidden_size | 4096 | 4096 |
| num_hidden_layers | 60 | 5 |
| num_attention_heads | 32 | 32 |
| num_key_value_heads | 2 | 8 |
| intermediate_size | 1024 (MoE) | 12288 |
| head_dim | 256 | 128 |
| model_type | qwen3_5_moe | qwen3 (dense) |
| quantization | NVFP4 | bfloat16 |
| layer_types | linear_attention + full_attention (hybrid Mamba) | full_attention only |
| num_experts | 512 (10 active) | N/A (dense) |

**Key**: Both models share `hidden_size=4096`, which is the critical dimension for DFlash compatibility. The drafter's `fc.weight` has shape `[4096, 20480]` = `4096 * 5` target layers concatenated.

## DFlash Config

The draft model expects hidden states from 5 specific layers of the target model:

```json
{
  "dflash_config": {
    "mask_token_id": 248070,
    "target_layer_ids": [2, 14, 26, 38, 50]
  }
}
```

These 5 layers produce a concatenated input of `4096 * 5 = 20480` dimensions, matching the drafter's `fc.weight` projection layer.

## vLLM Patches Required (v0.18.1rc1)

Three patches are needed to run DFlash with Qwen3.5-397B on vLLM v0.18.x:

### Patch 1: Eagle3 Method Delegation for Multimodal Wrapper

**File:** `vllm/model_executor/models/qwen3_5.py`

**Issue:** `Qwen3_5MoeForConditionalGeneration` (the multimodal wrapper) doesn't expose `get_eagle3_aux_hidden_state_layers()` or `set_aux_hidden_state_layers()`. These methods exist on the inner `Qwen3_5MoeForCausalLM` but the wrapper doesn't delegate.

**Error:**
```
AttributeError: 'Qwen3_5MoeForConditionalGeneration' object has no attribute
'get_eagle3_aux_hidden_state_layers'
```

**Fix:** Add delegation methods to `Qwen3_5MoeForConditionalGeneration`:
```python
class Qwen3_5MoeForConditionalGeneration(...):
    is_3d_moe_weight: bool = True

    def get_eagle3_aux_hidden_state_layers(self, method=None) -> tuple[int, ...]:
        return self.language_model.get_eagle3_aux_hidden_state_layers()

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.set_aux_hidden_state_layers(layers)
```

### Patch 2: KV Cache Page Size Unification

**File:** `vllm/v1/core/kv_cache_utils.py`

**Issue:** Qwen3.5 has hybrid Mamba+attention layers. The Mamba cache alignment sets page size to 2288 tokens, but the DFlash draft model's KV cache has a different page size. The `unify_kv_cache_spec_page_size()` function raises `NotImplementedError` when page sizes aren't divisible, and `get_uniform_page_size()` asserts exactly 1 unique page size.

**Errors:**
```
NotImplementedError: The page size of the layer is not divisible by the maximum page size.
AssertionError  # in get_uniform_page_size
```

**Fix 1 - `unify_kv_cache_spec_page_size()`:** Use LCM when page sizes aren't directly divisible:
```python
from math import gcd

# In the else branch where max_page_size % layer_page_size != 0:
_lcm = (max_page_size * layer_page_size) // gcd(max_page_size, layer_page_size)
ratio_up = _lcm // layer_page_size
new_block_size = layer_spec.block_size * ratio_up
new_spec = replace(layer_spec, block_size=new_block_size)
new_kv_cache_spec[layer_name] = new_spec
max_page_size = _lcm
```

**Fix 2 - `get_uniform_page_size()`:** Return max instead of asserting uniform:
```python
def get_uniform_page_size(kv_cache_specs):
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
    if len(page_sizes) == 1:
        return page_sizes.pop()
    return max(page_sizes)  # DFlash: target + draft may differ
```

### Patch 3: DFlash target_layer_ids Config Reading

**File:** `vllm/v1/worker/gpu_model_runner.py`

**Issue:** `_get_eagle3_aux_layers_from_config()` only checks `hf_config.eagle_aux_hidden_state_layer_ids` (EAGLE3 standard). DFlash models store layer IDs in `hf_config.dflash_config.target_layer_ids`. Without this patch, vLLM falls back to the 3-layer default `(2, 30, 57)` producing a `12288`-dim input instead of the expected `20480`-dim.

**Error:**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (17x12288 and 20480x4096)
```

**Root cause:** `4096 * 3 = 12288` (3 default layers) vs `4096 * 5 = 20480` (5 DFlash target layers).

**Fix:** Add DFlash config reading to `_get_eagle3_aux_layers_from_config()`:
```python
def _get_eagle3_aux_layers_from_config(self):
    if not (self.speculative_config and self.speculative_config.draft_model_config):
        return None
    hf_config = self.speculative_config.draft_model_config.hf_config

    # EAGLE3 standard
    if hasattr(hf_config, 'eagle_aux_hidden_state_layer_ids'):
        layer_ids = hf_config.eagle_aux_hidden_state_layer_ids
        if layer_ids and isinstance(layer_ids, (list, tuple)):
            return tuple(layer_ids)

    # DFlash standard
    if hasattr(hf_config, 'dflash_config'):
        dflash_cfg = hf_config.dflash_config
        if isinstance(dflash_cfg, dict):
            layer_ids = dflash_cfg.get('target_layer_ids')
        elif hasattr(dflash_cfg, 'target_layer_ids'):
            layer_ids = dflash_cfg.target_layer_ids
        else:
            layer_ids = None
        if layer_ids and isinstance(layer_ids, (list, tuple)):
            return tuple(layer_ids)

    return None
```

## Launch Configuration

Working launch config (after all 3 patches applied):

```bash
docker run -d \
  --name vllm-qwen35 \
  --gpus all --ipc host --shm-size 32g \
  --entrypoint bash \
  -p 9200:8000 \
  -v /path/to/lukealonso-qwen35-nvfp4:/model:ro \
  -v /path/to/dflash-397b-v2:/draft-model:ro \
  IMAGE_NAME \
  -c 'exec python3 -m vllm.entrypoints.openai.api_server \
  --model /model \
  --served-model-name qwen3.5-397b-nvfp4 \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.86 \
  --kv-cache-dtype auto \
  --max-model-len 262144 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --block-size 32 \
  -O3 \
  --no-enable-prefix-caching \
  --reasoning-parser qwen3 \
  --speculative-config '\''{"method":"dflash","model":"/draft-model","num_speculative_tokens":16}'\'''
```

### Key Config Notes

| Flag | Value | Reason |
|------|-------|--------|
| `--kv-cache-dtype auto` | bf16 | Draft model trained on bf16; fp8 KV cache causes precision mismatch |
| `--no-enable-prefix-caching` | disabled | Prefix caching triggers Mamba cache align mode which conflicts with DFlash KV cache page sizes |
| `--block-size 32` | 32 | Required to avoid Mamba page size assertion failures |
| `--speculative-config method=dflash` | dflash | Not eagle3 or mtp - vLLM resolves DFlashDraftModel architecture |
| `num_speculative_tokens: 16` | 16 | DFlash block size (parallel draft tokens per step) |

### Tradeoffs vs MTP=3

| | DFlash B=16 | MTP=3 (native) |
|---|---|---|
| Prefix caching | No | Yes |
| KV cache dtype | bf16 only | fp8 supported |
| Draft tokens/step | 16 | 3 |
| GPU memory overhead | ~1-2GB for draft model | Minimal (uses model's own MTP heads) |
| Acceptance rate | TBD (benchmarking) | ~78% greedy, ~67% thinking |
| Best for | High single-user throughput | Multi-user with shared prefixes |

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `AttributeError: ...get_eagle3_aux_hidden_state_layers` | Missing delegation in multimodal wrapper | Patch 1 |
| `NotImplementedError: page size not divisible` | Mamba + DFlash KV cache mismatch | Patch 2 |
| `AssertionError` in `get_uniform_page_size` | Multiple page sizes after unification | Patch 2 |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied (17x12288 and 20480x4096)` | Wrong aux layers (3 default vs 5 DFlash) | Patch 3 |
| `argument --speculative-config: Value method:dflash cannot be converted` | JSON quoting issue through SSH/bash | Use heredoc or file-based launch script |

## Files

| Path | Description |
|------|-------------|
| `models/dflash-397b-v2/` | Custom DFlash draft model (5 layers, hidden=4096) |
| `models/dflash-397b-v2/dflash.py` | DFlash model definition (auto_map) |
| `models/dflash-397b-v2/config.json` | Draft config with target_layer_ids |
| `fused-moe/train_dflash_397b.py` | Training script |
| `klc-linux/run_vllm_dflash.sh` | vLLM launch script (DFlash config) |
| `klc-linux/yolo_dflash_benchmark.sh` | Full benchmark + auto-start KLC |

## Date

2026-03-31 - Patches developed and tested against vLLM v0.18.1rc1.dev346+g1852b8c6d
