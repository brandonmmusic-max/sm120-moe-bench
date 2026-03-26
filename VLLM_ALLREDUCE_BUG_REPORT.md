# [Bug]: All fast AllReduce paths silently disabled on PCIe Blackwell (SM 12.0)

## Environment

- **vLLM version:** 0.17.1rc1
- **Hardware:** 4x NVIDIA RTX PRO 6000 Blackwell (96GB each, SM 12.0)
- **Interconnect:** PCIe Gen5 x16, NO NVLink
- **Driver:** 595.45.04
- **CUDA:** 13.2
- **NCCL:** 2.29.7
- **OS:** Pop!_OS (Ubuntu 24.04 base)

## Bug Summary

**ALL fast AllReduce paths in vLLM are silently disabled on SM 12.0 (Blackwell PCIe GPUs).**
Every AllReduce falls through to plain PyNCCL — the slowest path. This affects all PCIe
Blackwell GPUs: RTX 5090, RTX PRO 6000, Tesla B-series without NVLink, etc.

The AllReduce+RMSNorm fusion pass is advertised as working on Blackwell:
> "Default at O2 (Hopper/Blackwell + TP > 1), 5-20% E2E Speedup"

But it silently disables itself because FlashInfer workspace creation requires NVLink multicast.

## Impact

- **5-20% potential E2E speedup lost** on all PCIe Blackwell GPUs (per vLLM's own docs)
- All 6 AllReduce paths disabled — only the slowest fallback (PyNCCL) is active
- No warning in default log level — users won't know they're running suboptimally
- Affects a large installed base: all PCIe Blackwell workstation/consumer GPUs

## Detailed Analysis

### AllReduce Dispatch Chain (all disabled on SM 12.0 PCIe)

| Path | Status | Why Disabled |
|------|--------|-------------|
| AllReduce+RMSNorm Fusion | DISABLED | FlashInfer workspace requires NVLink multicast (`SymmDeviceMemory`) |
| NCCL Symmetric Memory | DISABLED | `VLLM_USE_NCCL_SYMM_MEM=0` (default off) |
| FlashInfer AllReduce | DISABLED | Requires NVSwitch multicast |
| Custom AllReduce (IPC) | DISABLED | `is_fully_connected()` checks NVML NVLink caps — returns False on PCIe |
| Torch SymmMem | DISABLED | SM 12.0 not in `SYMM_MEM_ALL_REDUCE_MAX_SIZES`, AND `multicast_ptr == 0` check blocks two_shot |
| **PyNCCL** | **ACTIVE** | Only path left — slowest |

### Bug 1: AllReduce+RMSNorm Fusion (allreduce_rms_fusion.py)

In `AllReduceFusionPass.__init__` (~line 830), the code creates FlashInfer workspace:
```python
for workspace_init_fn in [initialize_fi_ar_workspace, initialize_fi_ar_quant_workspace]:
    try:
        workspace_init_fn(...)
    except Exception as e:
        if "multicast" in str(e).lower():
            logger.warning("AllReduce fusion pass is disabled: ...")
        return  # <-- disables the ENTIRE fusion pass
```

The FlashInfer `create_allreduce_fusion_workspace` internally calls `SymmDeviceMemory`,
which requires NVLink multicast support. On PCIe GPUs, this throws:
```
[SymmDeviceMemory] Device does not support multicasting.
```

The fusion pass catches this, logs a warning, and disables itself entirely.

**Problem:** The fusion is advertised as working on Blackwell, but silently disables itself
on any Blackwell GPU without NVLink (all PCIe variants).

**Root cause:** The fusion pattern's replacement function calls `flashinfer_trtllm_fused_allreduce_norm`,
which requires the FlashInfer workspace. There is no fallback fused kernel for non-multicast topologies.

### Bug 2: Torch SymmMem blocks two_shot on PCIe (symm_mem.py)

In `SymmMemCommunicator.__init__` (~line 106):
```python
if handle.multicast_ptr == 0:
    logger.warning("multicast operations are not supported.")
    return  # <-- disables ALL symm_mem, including two_shot
```

This check disables the ENTIRE `SymmMemCommunicator`, even though `two_shot_all_reduce_`
does NOT require multicast — it uses P2P stores which work fine on PCIe.

Additionally, `_WORLD_SIZES_MULTIMEM` and `SYMM_MEM_ALL_REDUCE_MAX_SIZES` have no entries
for SM 12.0, so even fixing the multicast check wouldn't help without adding size tables.

### Bug 3: Missing SM 12.0 entries (all_reduce_utils.py)

`CUSTOM_ALL_REDUCE_MAX_SIZES` and `SYMM_MEM_ALL_REDUCE_MAX_SIZES` only have entries for
SM 9.0 (Hopper) and SM 10.0 (Blackwell NVLink/GB200). SM 12.0 is missing entirely.

## Reproduction Steps

1. Set up any PCIe Blackwell GPU (RTX 5090, RTX PRO 6000, etc.)
2. Run vLLM with TP > 1 and compilation level O2 (default for Blackwell)
3. Check logs for:
   - `"AllReduce fusion pass is disabled"` warning
   - `"SymmMemCommunicator: Device capability 12.0 not supported"` warning
   - No log from Custom AllReduce (silently skipped by NVLink check)
4. Run nsys profile — AllReduce is 60-70% of decode time

```bash
# Quick reproduction:
python -m vllm.entrypoints.openai.api_server \
    --model <any-model> \
    --tensor-parallel-size 4 \
    2>&1 | grep -i "allreduce\|symm_mem\|disabled"
```

## Proposed Fixes

### Fix 1: Add SM 12.0 to size tables (all_reduce_utils.py)

```python
CUSTOM_ALL_REDUCE_MAX_SIZES["12.0"] = {
    2: 2 * MiB, 4: 2 * MiB, 6: 1 * MiB, 8: 1 * MiB,  # Match SM 10.0
}
SYMM_MEM_ALL_REDUCE_MAX_SIZES["12.0"] = {
    2: 8 * MiB, 4: 32 * MiB, 6: 128 * MiB, 8: 128 * MiB,  # Match SM 10.0
}
```

### Fix 2: Fix SymmMem multicast_ptr check (symm_mem.py)

```python
# Before (blocks entire communicator):
if handle.multicast_ptr == 0:
    return

# After (only blocks multimem, allows two_shot):
self.has_multicast = handle.multicast_ptr != 0
if not self.has_multicast:
    logger.info("multicast not available, using two_shot allreduce only.")

# In _WORLD_SIZES_MULTIMEM:
"12.0": [],  # No multicast on PCIe — two_shot only

# In all_reduce method, check has_multicast before using multimem
```

### Fix 3: Enable NCCL SymmMem by default on Blackwell

`VLLM_USE_NCCL_SYMM_MEM` should default to `1` (or auto-detect) when NCCL >= 2.27.3
and TP >= 4. This is architecture-agnostic and doesn't require NVLink.

### Fix 4: AllReduce+RMSNorm fusion — add non-FlashInfer fallback

The fusion pass should have a fallback path when FlashInfer multicast is unavailable:
- Use NCCL AllReduce + separate RMSNorm kernel (still benefits from graph optimization)
- Or implement a non-multicast fused kernel using P2P or NCCL symmetric memory

## Log Evidence

```
WARNING [...] AllReduce fusion pass is disabled: flashinfer workspace creation failed:
[SymmDeviceMemory] Device does not support multicasting. This is expected on GPUs
without NVSwitch (e.g., NVLink bridge-only or PCIe topologies). Falling back to
non-fused allreduce.

WARNING [...] SymmMemCommunicator: Device capability 12.0 not supported,
communicator is not available.
```

## Experimental Validation (2026-03-26)

All three FlashInfer bugs were fixed and the fusion was enabled on PCIe Blackwell:

### Additional bugs discovered during fix:
- **Bug 4 (FlashInfer):** `_mc_granularity` attribute missing in unicast path — `_allocate_unicast_buffers()`
  uses `self._mc_granularity` for `cuMemAddressReserve()` alignment, but this is only set inside
  `if enable_multicast:`. Fix: fallback to `self._alloc_granularity`.
- **Bug 5 (FlashInfer):** SM 12.0 excluded from TRT-LLM AllReduce JIT compilation (`[9, 10]` in
  `flashinfer/jit/comm.py`) and kernel arch guards (`__CUDA_ARCH__ < 1200` in `trtllm_allreduce.cuh`).

### Results:
| Configuration | Decode tok/s | Stability |
|---------------|-------------|-----------|
| NCCL (no fusion) | **151.2** | Stable |
| TRT-LLM Lamport fusion (all bugs fixed) | **130.3** (-14%) | Crashes after ~5-10 min |

**Crash mode:** `TimeoutError: RPC call to sample_tokens timed out` during extended decode generation.
The Lamport-style IPC AllReduce spin-waits on PCIe flag reads, which can deadlock under sustained load.

**Conclusion: AllReduce+RMSNorm fusion should remain DISABLED on PCIe Blackwell.**
The fusion is designed for NVLink/NVSwitch where IPC memory access is fast. On PCIe, the
cacheline-granularity loads (200ns/64B) make it fundamentally slower than NCCL's DMA-based AllReduce.

### Patch files (preserved for NVLink testing):
- `~/klc-linux/patches/flashinfer_jit_comm.py` — SM 12.0 JIT target
- `~/klc-linux/patches/trtllm_allreduce.cuh` — Removed `< 1200` arch guards
- `~/klc-linux/patches/flashinfer_mnnvl.py` — Unicast mode fixes (reverted to unconditional multicast check for PCIe safety)

## Additional Context

- NCCL AllReduce p50 latency on 4 PCIe Gen5 GPUs is ~14μs — near theoretical minimum
- The p99 latency spikes to 8.8ms for small tensors (≤16KB), which explains the high
  AllReduce % in nsys profiles
- Setting `VLLM_USE_NCCL_SYMM_MEM=1` is the quickest workaround (no code changes)
- Torch SymmMem `two_shot_all_reduce_` works on PCIe P2P and should be enabled
