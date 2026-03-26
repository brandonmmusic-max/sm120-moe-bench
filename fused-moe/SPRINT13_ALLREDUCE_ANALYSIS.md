# Sprint 13: Hybrid TP-Attention + EP-MoE AllReduce Analysis

**Date:** 2026-03-26
**Hardware:** 4x NVIDIA RTX PRO 6000 Blackwell (PCIe Gen5, NO NVLink)
**Model:** Qwen3.5-397B-A17B-NVFP4 (512 experts, top-10 routing, 60 MoE layers)

---

## Executive Summary

**Hybrid TP-attention + EP-MoE does NOT save AllReduce calls on this hardware configuration.** The hypothesis that EP eliminates the MoE AllReduce is incorrect — EP replaces TP's AllReduce with a different AllReduce (combine step), plus adds broadcast overhead for dispatch. With `dp_size=1` (our config), vLLM's `NaiveAll2AllManager` uses broadcast+AllReduce, which is **equal or worse** than pure TP.

**The only way EP saves communication is with DP-attention + EP-MoE (`dp_size > 1`), which eliminates the attention AllReduce but requires a fundamentally different serving architecture.**

---

## Why EP Does NOT Eliminate AllReduce

### The Misconception

> "With EP, each GPU has 1/4 of experts. Each GPU's output already has the full hidden state for its local experts, so no AllReduce needed."

### The Reality

With EP=4, each GPU computes outputs **only for its local 128 experts** (out of 512). A token routed to experts on GPUs 0, 1, and 3 gets partial results from each. The full output is the **weighted sum** of all partial results across all GPUs. This requires an AllReduce (or reduce-scatter).

### Communication Patterns Per MoE Layer

| Config | Dispatch | Compute | Combine | Total Collectives |
|--------|----------|---------|---------|-------------------|
| **TP=4 (current)** | None | All GPUs compute 1/4 of all experts | 1 AllReduce | **1 AllReduce** |
| **TP=4 + EP (dp=1)** | N broadcasts | Each GPU computes its 128 experts | 1 AllReduce | **N broadcasts + 1 AllReduce** |
| **DP=4 + EP=4 (TP=1)** | 1 All-to-All | Each GPU computes its 128 experts | 1 All-to-All | **2 All-to-All** |

### vLLM's NaiveAll2AllManager (dp=1 path)

Confirmed in vLLM source code and our patch file (`patch_naive_all2all.py`):

```python
# Dispatch: N sequential broadcasts (each GPU broadcasts its tokens to all others)
# Every GPU receives every token — same data as TP, NO savings

# Combine: all_reduce + slice
all_hidden_states = get_ep_group().all_reduce(hidden_states)
hidden_states = all_hidden_states[start:end, :]
```

The `is_dp_ep` flag that enables real All-to-All requires:
```python
self.is_dp_ep = (
    moe_parallel_config.dp_size > 1   # FAILS: our dp_size=1
    and moe_parallel_config.use_ep
)
```

**With dp_size=1, EP is strictly WORSE than pure TP** — same AllReduce count plus broadcast overhead.

---

## Total AllReduce Count Analysis

### Current: TP=4 (pure tensor parallel)

Per transformer layer:
- Attention: 1 AllReduce (QKV output sharded across 4 GPUs → AllReduce to sum)
- MoE: 1 AllReduce (expert output sharded across 4 GPUs → AllReduce to sum)

**Total: 60 layers × 2 AllReduce = 120 AllReduce per token**

### Hypothetical: TP=4 attention + EP=4 MoE (dp=1)

Per transformer layer:
- Attention: 1 AllReduce (unchanged)
- MoE: N broadcasts + 1 AllReduce (NaiveAll2All combine)

**Total: 60 × (1 + 1) = 120 AllReduce + 60×N broadcasts = WORSE**

### Hypothetical: DP=4 attention + EP=4 MoE (dp=4, tp=1)

Per transformer layer:
- Attention: 0 AllReduce (each GPU handles independent requests)
- MoE: 2 All-to-All (dispatch + combine)

**Total: 0 AllReduce + 60 × 2 All-to-All = 120 All-to-All**

**Problem:** DP-attention requires each GPU to hold the FULL attention weights (unsharded). Qwen3.5-397B has ~60B attention parameters — at FP4, that's ~30GB per GPU. With 96GB VRAM and ~120GB for expert weights (at FP4), this doesn't fit without model offloading.

---

## Alternative: AllReduce Overlap/Pipeline

### Concept
Overlap AllReduce of layer N with MoE compute of layer N+1. The AllReduce runs on a separate CUDA stream while the next layer's expert computation begins.

### Analysis
- vLLM does NOT currently support AllReduce overlap with MoE compute
- NCCL AllReduce at p50 = 14μs, VerdictMoE kernel = ~18μs
- Overlap would hide ~14μs behind 18μs of compute = ~77% overlap
- **Theoretical savings: ~14μs × 60 layers × 2 = 1.68ms/token → save ~1.3ms**
- **BUT:** CUDA graph capture (which vLLM uses for decode) makes stream-based overlap extremely difficult — the entire decode step is captured as a single graph

### Feasibility: LOW
Would require either:
1. Breaking CUDA graph capture to allow multi-stream execution (defeats the purpose of graphs)
2. Graph-level stream insertion (experimental, not supported in vLLM)

---

## Alternative: FP8 AllReduce (Reduce Tensor Size)

### Concept
Quantize MoE output to FP8 before AllReduce, dequantize after. Reduces AllReduce data from 8KB (BF16) to 4KB (FP8) per token.

### Analysis
- At M=1, AllReduce tensor is [1, 4096] BF16 = 8KB
- FP8 version would be [1, 4096] FP8 = 4KB
- NCCL p50 for 8KB is already ~14μs (near PCIe minimum latency)
- Halving the data to 4KB would save **<1μs** (latency-dominated, not bandwidth-dominated)
- At M=4, [4, 4096] BF16 = 32KB → FP8 = 16KB, savings ~1-2μs
- **Risk:** FP8 quantization error on the MoE output could degrade MTP acceptance

### Feasibility: LOW ROI
The tensors are so small that AllReduce is latency-bound, not bandwidth-bound. Halving size doesn't meaningfully reduce latency on PCIe.

---

## What WOULD Actually Help

### 1. Fix Remaining AllReduce Fast Paths (Medium ROI)

Sprint 11 enabled NCCL SymmMem but Torch SymmMem two_shot still hangs on SM 12.0. A PyTorch upstream fix or workaround could enable the two_shot path for the 16KB-512KB gap where PyNCCL is the only option.

**Expected savings:** Unknown, depends on whether two_shot has lower latency than PyNCCL in this range.

### 2. Reduce p99 AllReduce Tail Latency (High ROI)

Sprint 11 measured p99 = 8.8ms for 8KB AllReduces vs p50 = 14μs. These 600x spikes happen on ~1% of calls = ~1.2 per token (120 calls × 1%). Each spike adds ~8.8ms.

**If we fix the tail:** Even reducing p99 to 100μs would save ~10ms/token at the tail, which is enormous for average throughput.

**Investigation needed:** Are the spikes from NCCL scheduling, OS jitter, or PCIe contention?

### 3. NVLink Hardware (Highest ROI, Future)

On NVLink (HGX/DGX), NCCL LSA in-kernel fusion would eliminate AllReduce launch overhead entirely. EP with real All-to-All over NVLink is the production approach (DeepSeek, Meta).

**Expected savings on NVLink:** ~14μs × 120 = 1.68ms/token → 0 (fused into kernel)

---

## Conclusion

| Approach | Saves AllReduce? | Feasible on PCIe? | ROI |
|----------|-----------------|-------------------|-----|
| Hybrid TP+EP (dp=1) | No (same + broadcasts) | N/A | None |
| DP+EP (dp=4, tp=1) | Eliminates attention AR | No (VRAM doesn't fit) | N/A |
| AllReduce overlap | Hides latency | No (CUDA graphs) | Low |
| FP8 AllReduce | Reduces size, not latency | Yes but <1μs savings | Very Low |
| Fix p99 tail spikes | Reduces avg latency | Yes | **High** |
| Fix Torch SymmMem | Enables faster path | Blocked upstream | Medium |
| NVLink hardware | Enables LSA fusion | Hardware purchase | **Highest** |

**Recommendation:** The highest-ROI work on the current PCIe hardware is investigating and fixing the p99 AllReduce tail latency spikes (8.8ms vs 14μs median). This is likely a larger throughput impact than eliminating AllReduce calls entirely at p50.

---

## Appendix: vLLM EP Code References

Key files in the container:
- `fused_moe/runner/default_moe_runner.py:331` — `maybe_all_reduce_tensor_model_parallel()`: always AllReduces unless `output_is_reduced()`
- `fused_moe/runner/default_moe_runner.py:370-376` — Reduction condition: fires when `tp_size > 1 OR ep_size > 1`
- `fused_moe/modular_kernel.py:1002` — `is_dp_ep` requires `dp_size > 1 AND use_ep`
- `fused_moe/all2all_utils.py:111` — All2All dispatch only when `dp_size > 1`
- `distributed/device_communicators/all2all.py` — `NaiveAll2AllManager.combine()` uses `all_reduce + slice`
