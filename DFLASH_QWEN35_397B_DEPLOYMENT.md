# DFlash Speculative Decoding - Qwen3.5-397B-A17B on vLLM

## Overview

DFlash is a novel speculative decoding method that uses a lightweight block diffusion model for parallel token drafting. This document covers deploying a custom 5-layer DFlash drafter with Qwen3.5-397B-A17B (NVFP4 quantized MoE, hybrid Mamba+attention architecture) on vLLM v0.18.x with 4x RTX PRO 6000 Blackwell GPUs (TP=4).

**Key result**: Five vLLM patches were required to make DFlash work with this model. All patches, the exact Docker config, and benchmark results are documented below for full reproducibility.

## Hardware

- 4x NVIDIA RTX PRO 6000 Blackwell (96GB each, SM 12.0, PCIe)
- Threadripper 24C/48T, Pop!_OS (Ubuntu 24.04 base)
- Driver: 595.45.04, CUDA: 13.2 (container)
- TP=4, all GPUs at 300W

## Model Architecture Compatibility

| Property | Qwen3.5-397B (target) | DFlash Drafter v2 |
|---|---|---|
| hidden_size | 4096 | 4096 |
| num_hidden_layers | 60 | 5 |
| num_attention_heads | 32 | 32 |
| num_key_value_heads | 2 | 8 |
| intermediate_size | 1024 (MoE per-expert) | 12288 (dense) |
| head_dim | 256 | 128 |
| model_type | qwen3_5_moe (multimodal) | qwen3 (dense, text-only) |
| quantization | NVFP4 (ModelOpt) | bfloat16 |
| layer_types | linear_attention + full_attention (hybrid Mamba/GDN) | full_attention only |
| num_experts | 512 (10 active per token) | N/A (dense) |
| mtp_num_hidden_layers | 1 (native MTP) | N/A |

**Critical compatibility dimension**: Both models share `hidden_size=4096`. The drafter's fusion layer `fc.weight` has shape `[4096, 20480]` = `4096 * 5` target aux layers concatenated.

## DFlash Config

The draft model extracts hidden states from 5 specific intermediate layers of the 60-layer target:

```json
{
  "dflash_config": {
    "mask_token_id": 248070,
    "target_layer_ids": [2, 14, 26, 38, 50]
  },
  "eagle_aux_hidden_state_layer_ids": [2, 14, 26, 38, 50]
}
```

**Important**: The `eagle_aux_hidden_state_layer_ids` field (EAGLE3 standard) MUST be present at the top level of the draft model's `config.json`. vLLM's standard code reads this field, not `dflash_config.target_layer_ids`. Without it, vLLM falls back to 3 hardcoded layers `(2, 30, 57)` causing a shape mismatch.

## vLLM Patches Required (v0.18.1rc1)

Five patches were needed. Each addresses a distinct integration gap between vLLM's speculative decoding framework and the Qwen3.5-397B hybrid Mamba/MoE architecture.

### Patch 1: Eagle3 Method Delegation for Multimodal Wrapper

**File**: `vllm/model_executor/models/qwen3_5.py`

**Problem**: The 397B resolves to `Qwen3_5MoeForConditionalGeneration` (the multimodal/vision wrapper class). This class doesn't expose `get_eagle3_aux_hidden_state_layers()` or `set_aux_hidden_state_layers()` -- these methods exist on the inner `Qwen3_5MoeForCausalLM` but the wrapper doesn't delegate.

**Error**:
```
AttributeError: 'Qwen3_5MoeForConditionalGeneration' object has no attribute 'get_eagle3_aux_hidden_state_layers'.
Did you mean: 'get_eagle3_default_aux_hidden_state_layers'?
```

**Fix**: Add delegation methods after `is_3d_moe_weight: bool = True`:
```python
class Qwen3_5MoeForConditionalGeneration(...):
    is_3d_moe_weight: bool = True

    def get_eagle3_aux_hidden_state_layers(self, method=None) -> tuple[int, ...]:
        return self.language_model.get_eagle3_aux_hidden_state_layers()

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.set_aux_hidden_state_layers(layers)
```

### Patch 2: KV Cache Page Size Unification (LCM Fallback)

**File**: `vllm/v1/core/kv_cache_utils.py`

**Problem**: Qwen3.5 has hybrid Mamba+attention layers. The Mamba cache alignment sets the attention page size to 2288 tokens. The DFlash draft model's KV cache has a different page size that isn't evenly divisible. `unify_kv_cache_spec_page_size()` raises `NotImplementedError` when sizes aren't divisible.

**Error**:
```
NotImplementedError: The page size of the layer is not divisible by the maximum page size. Cannot unify by adjusting block_size.
```

**Fix**: Use LCM (least common multiple) when page sizes don't divide:
```python
from math import gcd

# In unify_kv_cache_spec_page_size(), replace the NotImplementedError branch:
if max_page_size % layer_page_size != 0:
    _lcm = (max_page_size * layer_page_size) // gcd(max_page_size, layer_page_size)
    ratio_up = _lcm // layer_page_size
    new_block_size = layer_spec.block_size * ratio_up
    new_spec = replace(layer_spec, block_size=new_block_size)
    new_kv_cache_spec[layer_name] = new_spec
    max_page_size = _lcm
```

### Patch 3: KV Cache Uniform Page Size Assertion

**File**: `vllm/v1/core/kv_cache_utils.py`

**Problem**: `get_uniform_page_size()` asserts exactly 1 unique page size after unification. With DFlash + Mamba, there may be 2 distinct sizes even after LCM adjustment.

**Error**:
```
AssertionError  # assert len(page_sizes) == 1
```

**Fix**: Return max instead of asserting uniform:
```python
def get_uniform_page_size(kv_cache_specs):
    page_sizes = {layer.page_size_bytes for layer in kv_cache_specs}
    if len(page_sizes) == 1:
        return page_sizes.pop()
    return max(page_sizes)  # DFlash: target + draft may have different sizes
```

### Patch 4: DFlash target_layer_ids Config Reading

**Files**:
- `vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py` (primary V1 path)
- `vllm/v1/worker/gpu_model_runner.py` (legacy path)

**Problem**: `get_eagle3_aux_layers_from_config()` only checks `hf_config.eagle_aux_hidden_state_layer_ids` (EAGLE3 standard). DFlash models originally stored layer IDs in `hf_config.dflash_config.target_layer_ids`. Without this, the function returns `None` and the fallback `qwen3_5.py:get_eagle3_aux_hidden_state_layers()` returns 3 hardcoded layers `(2, num_layers//2, num_layers-3)`.

**Error**:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (17x12288 and 20480x4096)
```

**Root cause**: `3 layers * 4096 = 12288` (hardcoded fallback) vs `5 layers * 4096 = 20480` (what the drafter expects).

**Fix** (in both files):
```python
def get_eagle3_aux_layers_from_config(spec_config):
    hf_config = spec_config.draft_model_config.hf_config
    # EAGLE3 standard
    if hasattr(hf_config, 'eagle_aux_hidden_state_layer_ids'):
        layer_ids = hf_config.eagle_aux_hidden_state_layer_ids
        if layer_ids and isinstance(layer_ids, (list, tuple)):
            return tuple(layer_ids)
    # DFlash standard (fallback)
    if hasattr(hf_config, 'dflash_config'):
        dflash_cfg = hf_config.dflash_config
        layer_ids = dflash_cfg.get('target_layer_ids') if isinstance(dflash_cfg, dict) else getattr(dflash_cfg, 'target_layer_ids', None)
        if layer_ids and isinstance(layer_ids, (list, tuple)):
            return tuple(layer_ids)
    return None
```

**Better fix**: Add `eagle_aux_hidden_state_layer_ids` to the draft model's `config.json` so the standard code path works on any vLLM image without patches.

### Patch 5: DFlash Attention Positions Expansion

**File**: `vllm/model_executor/models/qwen3_dflash.py`

**Problem**: The DFlash attention layer concatenates `context_states` (aux hidden states from target model) with `hidden_states` (draft token embeddings) before the QKV projection. This creates a tensor with `num_context + num_draft` tokens. But the `positions` tensor only contains entries for `num_draft` tokens. When `rotary_emb(positions, q, k)` tries to reshape `q` using the positions count, the shapes don't match.

**Error**:
```
RuntimeError: shape '[34, -1, 128]' is invalid for input of size 41984
```

**Root cause**: `q` has 41 tokens (7 context + 34 draft) but `positions` has 34 entries. `41 * 8 heads * 128 head_dim = 41984` can't be reshaped with `num_tokens=34`.

**Fix**: Expand positions with zeros for context tokens before rotary embedding:
```python
# In Qwen3Attention.forward(), before rotary_emb call:
if num_context > 0 and positions.shape[-1] < q.shape[0]:
    ctx_positions = torch.zeros(
        num_context, dtype=positions.dtype, device=positions.device
    )
    positions = torch.cat([ctx_positions, positions], dim=0)

q, k = self.rotary_emb(positions, q, k)
```

Context tokens get position 0 (dummy) since their positional information is already encoded in the hidden states from the target model. The rotary embedding for context tokens is effectively a no-op since `q = q[num_context:]` strips them immediately after.

## V2 Model Runner Compatibility Notes

If using `VLLM_USE_V2_MODEL_RUNNER=1` (not default), two additional issues exist:

1. **V2 model runner only enables aux hidden states for `method == "eagle3"`**, not `"dflash"`. The `set_eagle3_aux_hidden_state_layers` would never be called for DFlash method.
2. **V2 speculator has `assert self.method == "eagle3"`** which would hard crash on `method == "dflash"`.

These don't affect the default V1 runner but should be fixed upstream for forward compatibility.

## Docker Launch Configuration

Exact command used (with patched image):

```bash
IMAGE="vllm-qwen35-k64:dflash-patched"
MODEL_DIR="/path/to/lukealonso-qwen35-nvfp4"
DRAFT_DIR="/path/to/dflash-397b-v2"

docker run -d \
  --name vllm-qwen35 \
  --gpus all --ipc host --shm-size 32g \
  --restart on-failure:3 \
  --entrypoint bash \
  -p 9200:8000 \
  -v "${MODEL_DIR}":/model:ro \
  -v "${DRAFT_DIR}":/draft-model:ro \
  -v /path/to/cache/vllm:/root/.cache/vllm \
  -v /path/to/cache/torch_extensions:/root/.cache/torch_extensions \
  -v /path/to/cache/triton:/root/.triton \
  -v /path/to/cache/inductor:/root/.cache/torch/inductor \
  -v /path/to/cache/flashinfer:/cache/jit/flashinfer/.cache/flashinfer \
  -v /path/to/patches/pynccl_allocator.py:/opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/pynccl_allocator.py:ro \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_NET_GDR_LEVEL=SYS \
  -e NCCL_MIN_NCHANNELS=4 \
  -e NCCL_ALLOC_P2P_NET_LL_BUFFERS=1 \
  -e NCCL_TREE_THRESHOLD=0 \
  -e VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  -e ENABLE_SM120=1 \
  -e VLLM_LOG_STATS_INTERVAL=1 \
  -e SAFETENSORS_FAST_GPU=1 \
  -e OMP_NUM_THREADS=24 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_DEVICE_MAX_CONNECTIONS=32 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e VLLM_USE_FUSED_MOE_GROUPED_TOPK=1 \
  -e VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1 \
  -e VLLM_USE_VERDICT_MOE=1 \
  -e VLLM_VERDICT_MMA=1 \
  "$IMAGE" \
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
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --mm-encoder-tp-mode data --mm-processor-cache-type shm \
  --speculative-config '"'"'{"method":"dflash","model":"/draft-model","num_speculative_tokens":16}'"'"''
```

### Key Config Flags

| Flag | Value | Reason |
|------|-------|--------|
| `--kv-cache-dtype auto` | bf16 | Draft model trained on bf16; fp8 may cause precision mismatch |
| `--no-enable-prefix-caching` | disabled | Prefix caching triggers Mamba cache align mode conflicting with DFlash page sizes |
| `--block-size 32` | 32 | Helps with Mamba page size alignment |
| `--speculative-config method=dflash` | dflash | Resolves DFlashDraftModel architecture via auto_map |
| `num_speculative_tokens: 16` | 16 | DFlash block size (parallel draft tokens per step) |
| `-O3` | max optimization | Enables CUDA graph compilation and inductor optimizations |

### Custom Kernels

The image includes custom K=64 CUTLASS kernels for SM120 (Blackwell):
- `VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8=1` - FlashInfer MoE with mixed FP4/FP8
- `VLLM_USE_VERDICT_MOE=1` / `VLLM_VERDICT_MMA=1` - VerdictMoE GEMM kernels
- `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1` - Packed recurrent decode for Mamba layers
- Custom `pynccl_allocator.py` for NCCL memory management

### Tradeoffs vs MTP=3

| | DFlash B=16 | MTP=3 (native) |
|---|---|---|
| Prefix caching | No | Yes |
| KV cache dtype | bf16 only | fp8 supported |
| Draft tokens/step | 16 | 3 |
| GPU memory overhead | ~1-2GB for draft model | Minimal (uses model's own MTP heads) |
| Best for | High single-user throughput | Multi-user with shared prefixes |

## Benchmark Results

*Benchmarked with `llm_decode_bench.py v0.3.0` -- server-side Prometheus metrics, 30s measurement per cell after dynamic warmup.*

(Results pending -- benchmark in progress)

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `AttributeError: ...get_eagle3_aux_hidden_state_layers` | Missing delegation in multimodal wrapper | Patch 1 |
| `NotImplementedError: page size not divisible` | Mamba + DFlash KV cache page mismatch | Patch 2 |
| `AssertionError` in `get_uniform_page_size` | Multiple page sizes after unification | Patch 3 |
| `RuntimeError: mat1 and mat2 shapes cannot be multiplied (Nx12288 and 20480x4096)` | Wrong aux layers (3 fallback vs 5 DFlash) | Patch 4 + add `eagle_aux_hidden_state_layer_ids` to config |
| `RuntimeError: shape '[N, -1, 128]' is invalid for input of size M` | DFlash attention positions don't cover context tokens | Patch 5 |
| `argument --speculative-config: Value method:dflash cannot be converted` | JSON quoting in bash | Use heredoc or single-quoted JSON |

## Discovery Timeline

1. **Eagle3 delegation** - Container started, immediately crashed on `get_eagle3_aux_hidden_state_layers`. The multimodal wrapper `Qwen3_5MoeForConditionalGeneration` wraps `Qwen3_5MoeForCausalLM` which has the method, but doesn't delegate.

2. **KV cache page sizes** - Got past delegation, crashed on `unify_kv_cache_spec_page_size`. Mamba layers need 2288-token pages, DFlash draft model has standard attention pages. LCM fallback + relaxed uniform assertion fixed it.

3. **Aux layer count mismatch (12288 vs 20480)** - Server booted and accepted requests, but crashed at runtime. Initial theory: 12288 = 3 * 4096 from hardcoded fallback. Confirmed: `qwen3_5.py:get_eagle3_aux_hidden_state_layers()` returns `(2, 30, 57)` (3 layers) when config lookup fails. Root cause: `eagle3_utils.py` only reads `eagle_aux_hidden_state_layer_ids`, not `dflash_config.target_layer_ids`. Fix: add the standard field to draft model config.json AND patch both code paths.

4. **Two model runner code paths** - Discovered `vllm/v1/worker/gpu/model_runner.py` (V1, active) and `vllm/v1/worker/gpu_model_runner.py` (legacy). Both need the dflash_config reading patch. The V1 path calls `eagle3_utils.set_eagle3_aux_hidden_state_layers()` at line 282.

5. **Rotary embedding positions mismatch** - After fixing aux layers, new crash: `shape '[34, -1, 128]' is invalid for input of size 41984`. DFlash attention concatenates context_states (7 tokens from 5 aux layers) with hidden_states (34 draft tokens) = 41 total, but positions only has 34 entries. Fix: prepend zero-positions for context tokens before rotary_emb.

## Files

| Path | Description |
|------|-------------|
| `models/dflash-397b-v2/` | Custom DFlash draft model (5 layers, hidden=4096, trained on bf16 397B) |
| `models/dflash-397b-v2/config.json` | Config with both `dflash_config.target_layer_ids` and `eagle_aux_hidden_state_layer_ids` |
| `models/dflash-397b-v2/dflash.py` | DFlash model definition (auto_map for vLLM) |
| `fused-moe/train_dflash_397b.py` | Training script for the DFlash drafter |
| `klc-linux/run_vllm_dflash.sh` | vLLM launch script |
| `klc-linux/yolo_dflash_benchmark.sh` | Full benchmark + auto-start KLC stack |

## Date

2026-03-31 -- Patches developed and tested against vLLM v0.18.1rc1.dev346+g1852b8c6d on 4x RTX PRO 6000 Blackwell (SM 12.0)
