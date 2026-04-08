# Bugs & Errata Found During Optimization

## Hardware Bugs

### 1. atomicAdd_system BROKEN on SM 12.0 for Cross-GPU P2P Targets
**Severity**: Critical  
**Affected**: SM 12.0 (GB202 — RTX 5090, RTX PRO 6000 Blackwell)  
**Symptom**: Multiple GPUs receive the SAME return value from `atomicAdd_system` targeting remote GPU memory via P2P BAR mapping. This is impossible for correct atomic behavior — each caller should get a unique previous value.  
**Reproduction**: Allocate memory on GPU 0, map via P2P to GPUs 1-3. All 4 GPUs call `atomicAdd_system(&counter, 1)` simultaneously. Multiple GPUs return the same value.  
**Also affects**: Host-pinned memory targets (same incorrect behavior).  
**Workaround**: Do NOT use system-scope atomics for cross-GPU synchronization on SM 12.0. Use pure posted writes + volatile flag polling instead.  
**Status**: Not reported to NVIDIA yet. Should be filed as a driver/hardware bug.  
**Found**: April 8, 2026

### 2. RTX PRO 6000 Blackwell Has 188 SMs (Not 99)
**Severity**: Medium (correctness issue in hardcoded kernels)  
**Affected**: Any code hardcoding SM count for RTX PRO 6000  
**Detail**: Early specs and documentation listed ~99 SMs. Actual hardware has 188 active SMs (near-full GB202 die). Confirmed via `cudaGetDeviceProperties.multi_processor_count = 188` on all 4 GPUs (both Max-Q and full variants).  
**Impact**: Kernels using hardcoded 99-CTA grids only use 53% of compute capacity.  
**Fix**: Always query SM count dynamically via CUDA API.  
**Found**: April 8, 2026

## vLLM Bugs / Missing Features

### 3. TreeAttention Backend Not Compatible with FP8 KV Cache
**File**: `vllm/v1/attention/backends/tree_attn.py`  
**Issue**: `supported_kv_cache_dtypes` only lists `auto/float16/bfloat16`, missing `fp8/fp8_e4m3`.  
**Fix**: Add `"fp8"` and `"fp8_e4m3"` to the list.

### 4. TreeAttention Not in SM 12.0 Backend Priority List
**File**: `vllm/platforms/cuda.py`  
**Issue**: The backend selection for `device_capability.major != 10` doesn't include `TREE_ATTN`.  
**Fix**: Add `AttentionBackendEnum.TREE_ATTN` to the priority list.

### 5. TreeAttention Missing supports_attn_type and supports_compute_capability
**File**: `vllm/v1/attention/backends/tree_attn.py`  
**Issue**: Base class defaults return False for DECODER attn type and SM 12.0 capability.  
**Fix**: Override both classmethods to return True.

### 6. Triton Unified Attention Doesn't Handle uint8 KV Cache
**File**: `vllm/v1/attention/ops/triton_unified_attention.py`  
**Issue**: FP8 KV cache stored as uint8 fails `K_load.dtype.is_fp8()` check. Triton sees uint8, not fp8e4nv.  
**Fix**: Add `elif K_load.dtype == tl.uint8:` branch with bitcast to `tl.float8e4nv`.

### 7. EagleProposer Missing self.positions for MRoPE Models  
**File**: `vllm/v1/spec_decode/eagle.py`  
**Issue**: `self.positions` only allocated in the `else` branch of MRoPE check. Models using MRoPE (like Qwen3.5) never allocate it, causing AttributeError in propose_tree().  
**Fix**: Add fallback allocation after the if/elif/else block.

### 8. propose_tree() Tuple Unpack Crash for MTP Models
**File**: `vllm/v1/spec_decode/eagle.py`  
**Issue**: `last_hidden_states, hidden_states = self.model(...)` fails for MTP models which return a single tensor, not a tuple.  
**Fix**: Check `self.model_returns_tuple()` before unpacking.

### 9. MTP Tree Pipeline Truncates 10 Tree Tokens to 3
**File**: `vllm/v1/worker/gpu_model_runner.py`  
**Issue**: `draft_token_ids_cpu` buffer allocated as `(max_reqs, num_speculative_tokens)` = `(32, 3)` instead of `(32, 10)` for tree mode. Tree tokens truncated.  
**Fix**: Use `num_draft_tokens_per_request` (total tree size) instead of `num_speculative_tokens` (tree depth). Requires changes in gpu_model_runner.py, scheduler.py, and speculative.py.  
**Status**: Patch written, not fully tested yet.

### 10. V1 Tree Rejection Sampler Missing
**File**: `vllm/v1/sample/rejection_sampler.py`  
**Issue**: Only linear chain rejection sampling implemented. No tree path walking.  
**Fix**: Created `tree_rejection_sampler.py` (418 lines) with Triton greedy kernel + Python random fallback.

## NCCL / Communication Issues

### 11. NCCL AllReduce p99 Tail Latency: 8.8ms for 8KB
**Detail**: While p50 is 13.9μs, p99 spikes to 8,779μs for small (8KB) AllReduce payloads.  
**Impact**: Catastrophic for single-user decode — a single spike adds 8.8ms to one token.  
**Likely cause**: OS/driver jitter, PCIe power management, NCCL buffer management.  
**Mitigation**: Write-based P2P AllReduce has p99 of 51.5μs (170x better than NCCL p99).

### 12. NCCL SymmMem Only Viable Fast AR on PCIe SM 12.0
**Detail**: Custom AllReduce, Torch SymmMem two_shot, FlashInfer AR, pynccl — all disabled or broken on SM 12.0. Only NCCL SymmMem works.  
**Workaround**: Write-based P2P AllReduce as replacement (8.4μs p50).
