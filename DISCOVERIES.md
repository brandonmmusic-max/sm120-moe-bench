# SM120 MoE Inference Optimization — Discoveries & Findings

## Project: Qwen3.5-397B-A17B-NVFP4 on 4x RTX PRO 6000 Blackwell
**Date**: April 7-8, 2026  
**Hardware**: 4x NVIDIA RTX PRO 6000 Blackwell (GB202, SM 12.0), 96GB GDDR7 each, PCIe 5.0 x16, Threadripper  
**Model**: Qwen3.5-397B-A17B MoE (512 experts/layer, top-10 + 1 shared, 60 layers, NVFP4)  
**Baseline**: 149 tok/s with MTP=3, 156 tok/s single-user legal prompts

---

## 1. FIRST PRINCIPLES: THE PHYSICS

### Theoretical Peak
| Metric | Value |
|--------|-------|
| Active params per token | 17B × 0.53 bytes (NVFP4+FP8 scales) = **9.2 GB** |
| Aggregate GPU bandwidth | 4 × 1.7 TB/s = **6.8 TB/s** |
| Theoretical peak (zero overhead) | 6800 / 9.2 = **739 tok/s** |
| Compute-only floor (realistic BW efficiency) | **~455 tok/s** |
| Current measured | **149-156 tok/s** |
| **Bandwidth utilization** | **~20%** |

**We are using only 20% of the hardware's capability.** The remaining 80% is overhead.

### Architecture Correction (Qwen3.5-397B)
- **512 experts per MoE layer** (not 128 as commonly assumed)
- **Top-10 routed + 1 shared expert** = 11 active per token
- Only **2% of routed experts** active per token (10/512)
- 60 layers: 15 repeating blocks of (3x DeltaNet+MoE, 1x FullAttention+MoE)
- Only **15/60 layers** have full attention (growing KV cache); 45 use DeltaNet (fixed O(1) state)
- Per-expert size in NVFP4: **~6.7 MB**
- Total experts across all layers: **30,720**

---

## 2. THE #1 BOTTLENECK: PCIe AllReduce

### Discovery: AllReduce consumes 41-50% of wall time

With TP=4, every token requires **120 AllReduce operations** (60 layers × 2: after attention + after MoE). Each AllReduce:
- Payload: 8-16 KB (hidden_size=4096 × 2 bytes BF16)
- NCCL SymmMem latency: **p50 = 14μs**, p99 = 8.7ms
- Total per token: 120 × ~20μs = **2.4 ms** out of ~5.8 ms total

**PCIe AllReduce is THE dominant bottleneck — not memory bandwidth, not compute, not KV cache.**

### Why Prior P2P AllReduce Was 4-16x Slower Than NCCL

**Critical discovery: PCIe reads vs writes are fundamentally asymmetric.**

| Operation | Latency | Mechanism |
|-----------|---------|-----------|
| PCIe READ (per cacheline) | **300-500ns** | Request-response round-trip through root complex |
| PCIe WRITE (per cacheline) | **Posted, pipelined** | Fire-and-forget, no acknowledgment needed |
| 16KB bulk reads | ~50μs | 128 sequential 400ns round-trips |
| 16KB bulk writes | ~0.3μs | Pipelined at full link bandwidth |

**Writes are ~100x faster than reads for small transfers on PCIe.**

Our prior `p2p_tree_allreduce.cu` used `cudaMemcpyPeerAsync` + read-based reduction kernels. NCCL's LL (Low Latency) protocol achieves 14μs by using **ALL WRITES**:
- GPU 0 WRITES its data to GPU 1's BAR-mapped memory (posted write)
- GPU 1 polls LOCAL flags (no remote reads)
- Zero PCIe read round-trips in the critical path

### Solution: Write-Based P2P AllReduce
- Implemented in `fused-moe/csrc/write_based_allreduce.cu`
- Uses `float4` (128-bit) stores for maximum write combining
- `__threadfence_system()` for cross-PCIe visibility (not `__threadfence()`)
- Tree-reduce for 4 GPUs: 2 parallel rounds + broadcast
- Target: **<10μs for 10KB** (competitive with NCCL's 14μs)

---

## 3. DFlash SPECULATIVE DECODING — NEGATIVE RESULT

### What We Did
- Trained a 1B DFlash draft model on **639K extracted hidden state samples** from the 397B target
- 6 epochs on vast.ai 4x B200 GPUs, ~$2,500 compute cost
- Used z-lab/Qwen3.5-9B-DFlash warmstart with FC layer reinitialized (xavier_normal)
- Implemented batched training with vectorized dense attention masks (eliminated per-sample Python loop)
- File-aware CurriculumSampler for cache-friendly data loading
- Deployed via full PR 36847 integration (13 files patched into vLLM 0.19.0)

### Results
| Metric | Value |
|--------|-------|
| Training p1 accuracy | **13.1%** (plateaued after epoch 2) |
| Deployment pos-0 acceptance | **8.6%** |
| Deployment pos-1+ acceptance | **~0%** |
| Single-user tok/s | **19.9** (7.5x SLOWER than baseline) |
| Avg accepted tokens per draft | **0.109** |

### Root Cause Analysis
1. **FC layer bottleneck**: `Linear(20480→4096)` with xavier init is a 5x compression of features from 5 target layers spanning the entire 60-layer model. A single linear layer without nonlinearity can't align features from layers with fundamentally different statistical distributions.
2. **Train-test distribution shift**: Training extracts hidden states from teacher-forced sequences; inference extracts from the model's own predictions.
3. **No block-diagonal mask in forward**: The attention mask was only used for loss masking, not in the actual forward pass. Tokens could see future context during training that doesn't exist during inference.
4. **MoE feature space complexity**: The 397B MoE model's hidden states (with expert routing) are much harder to predict than a dense 8B model's.

### Key Lesson
DFlash's cross-attention architecture works well on **dense models** (z-lab showed 60% acceptance on Qwen3-8B). On **quantized MoE models with mixed attention types**, the feature space is too complex for the independent drafter approach. The FC reinit from scratch (vs fine-tuning a pretrained FC) was likely the critical failure point.

---

## 4. MTP TREE VERIFICATION — IN PROGRESS

### Discovery: Tree Verification Infrastructure Exists in vLLM 0.19.0
- `speculative_token_tree` config parameter is supported
- `propose_tree()` method in `SpecDecodeBaseProposer` handles top-k branching
- `TreeAttentionBackend` with tree attention bias masks exists
- BUT: not wired into V1 engine, not compatible with FP8 KV cache, not compatible with MTP

### Patches Required (7 total)
1. Add `fp8`/`fp8_e4m3` to `TreeAttentionBackend.supported_kv_cache_dtypes`
2. Add `supports_attn_type()` returning True for DECODER
3. Add `supports_compute_capability()` for SM 12.0 (major >= 8)
4. Add `TREE_ATTN` to CUDA platform backend priority list for SM 12.0
5. Add uint8→fp8e4nv bitcast in `triton_unified_attention.py` for FP8 KV cache loads
6. Add `self.positions` fallback allocation for MRoPE models
7. Fix `positions` tensor shape for tree mode (1 per request, not per draft token)

### Status
- TREE_ATTN backend successfully selected: ✅
- FP8 KV cache loading: ✅ (bitcast patch)
- Tree propose integration with MTP: 🔧 (agent fixing remaining issues)

### MoE Verification Tax (Important Finding)
The tree verification agent initially concluded tree decoding is "not worth it" for MoE. **This conclusion was WRONG.** The corrected math:

At batch=1 decode (GEMV), weight reads dominate and are **the same regardless of batch size**:
- Linear MTP=3: 4 verify tokens, 9.2 GB weight reads, 2.35 expected accepted
- Tree 10 nodes: 11 verify tokens, **still 9.2 GB weight reads** (same weights loaded once)
- Tree wins because verification cost is nearly identical but acceptance is higher

The "MoE verification tax" (more expert activations) is minimal because adjacent tokens in the same context activate mostly the **same experts** (71% reuse rate from literature).

---

## 5. EXPERT SPECIALIZATION & OFFLOADING

### Gate Weight Analysis
From offline analysis of all 60 router gates `(512, 4096)`:
- Gate norms remarkably uniform: 1.428 to 1.628 range (only 14% variation)
- **SVD: rank-1 dominates** (90% variance in 1 singular value)
- Adjacent layer cosine similarity: 0.48 mean
- Expert selection driven by **input hidden state**, not gate weights

**Implication**: The rank-1 gate structure means domain-specific inputs (legal text) will **consistently activate the same expert subset** across tokens. Runtime profiling needed, not static analysis.

### Expert Offloading Potential
| Offload % | VRAM Freed | Extra KV Cache |
|-----------|------------|----------------|
| 50% | ~100 GB | ~600K-800K tokens |
| 60% | ~120 GB | ~750K-1M tokens |
| 70% | ~140 GB | ~900K-1.1M tokens |

### Expert Prediction Accuracy (From Literature)
- Adjacent-token expert reuse: **71%** (auto-regressive router)
- Cross-layer prediction: **93-97%** (ETH Zurich, Pre-Attention Expert Prediction)
- MTP-driven prefetch: draft tokens predict future expert needs for **free**

### Domain-Specific Expert Caching
From "What Gets Activated" (arXiv:2601.10159):
- Expert specialization is **real but diffuse** — no clean 1:1 expert-to-domain mapping
- A **minority of experts** per layer show strong domain preference
- **"Super Experts"** (3-5 per model): disproportionately critical, must always be GPU-resident
- For legal workload: **30-40% of experts per layer** estimated to be domain-specialized
- Pre-loading top-100 experts/layer for legal domain: estimated **85%+ cache hit rate**

---

## 6. SM120 (BLACKWELL GB202) HARDWARE FACTS

### What SM120 HAS (that SM89 did NOT)
- **FP4 MMA**: `mma.sync.aligned.m16n8k64` with E2M1 — the NVFP4 instruction
- **FP8 MMA**: `mma.sync.aligned.kind::f8f6f4.m16n8k32` — 2x throughput over BF16
- **Block-scaled MMA**: `scale_vec::4X` hardware-accelerated (CRITICAL: applies per-lane, not per-block)
- **TMA unicast**: cp.async.bulk.tensor with mbarrier
- **PDL**: griddepcontrol for kernel overlap
- **128 MB L2 cache**: huge for MoE (expert weights partially fit)
- **FP6 MMA**: E3M2 and E2M3 6-bit formats

### What SM120 DOES NOT HAVE (vs SM100 datacenter)
- **No WGMMA** — uses older mma.sync (register-to-register)
- **No TMEM** (Tensor Memory) — physically absent from GB202
- **No DSMEM** (Distributed Shared Memory)
- **No Clusters** — hardcoded 1x1x1, no cross-SM cooperation
- **No tcgen05.mma** — SM100's single-thread tensor core instruction

**Programming model**: "Ampere mma.sync + Blackwell FP4/FP6/FP8 types + TMA + PDL"

### Kernel Performance Status
- **VerdictMoE**: 17.9 μs/layer at M=1 TP=4 (5.49x CUTLASS) — at memory bandwidth floor
- **SM120 Flash Attention**: 99 TFLOPS exact, 190 TFLOPS with FP8 KV cache
- **TMA+PDL**: +13% at C=4-8 (deployed)
- **PCIe OneShot AllReduce**: +7% (deployed)

---

## 7. MEGAKERNEL: PERSISTENT FUSED TRANSFORMER FORWARD PASS

### Concept
One `cudaLaunchCooperativeKernel` that processes ALL 60 layers:
```
for layer in 0..59:
  RMSNorm → Attention (QKV GEMV + decode + O GEMV) → P2P AllReduce
  → RMSNorm → MoE Gate → Expert GEMV (top-10 of 512) → P2P AllReduce
  → Residual add
  grid.sync()
```

### Expected Performance
| Component | Current (μs/layer) | MegaKernel (μs/layer) |
|-----------|-------------------|----------------------|
| AllReduce ×2 | 40 (NCCL) | **10** (in-kernel write-based) |
| Compute | 58 | 56 |
| **Per-layer total** | **98** | **66** |
| **60 layers** | **5.9 ms** | **3.96 ms** |
| **Tokens/sec** | **~170** | **~252** (+50%) |

### Implementation Status
- `megakernel_v1.cu`: 1830 lines, all 7 TODOs completed, compiles clean on sm_120a
- Implements: RMSNorm, FP4 GEMV, MoE gate+topk, expert execution (GEMM1+SiLU+GEMM2), self-attention decode, DeltaNet linear attention, shared expert, write-based P2P AllReduce
- Uses cooperative_groups grid.sync() between layers
- 99 CTAs (1 per SM), 256 threads each

---

## 8. SPECULATIVE DECODING SURVEY (MoE-SPECIFIC)

### Methods Evaluated
| Method | Speedup (Dense) | Speedup (MoE) | Requires Training | In vLLM |
|--------|----------------|---------------|-------------------|---------|
| MTP=3 | N/A | 1.5x (current) | No (built-in) | ✅ |
| DFlash | 6.17x (8B) | ❌ 0.13x (397B) | Yes ($2500) | ✅ (patched) |
| EAGLE-3 | 6.5x | ~1.5x (Mixtral) | Yes | ✅ |
| Medusa | 2x | ~1.3x | Yes | ✅ |
| Tree MTP | 1.5x+ | TBD | No | 🔧 (patching) |
| Self-Spec (Early Exit) | 1.3x | ❌ N/A | Partial | ❌ |
| Lookahead (Jacobi) | 1.8x | ❌ Too expensive | No | ❌ |

### Why MoE Speculation is Hard
1. **Verification tax**: Verifying N speculative tokens on a 512-expert MoE may activate more unique experts than single-token AR, increasing bandwidth
2. **Expert scatter**: Top-10 of 512 experts means memory access is scattered across weight tensors
3. **However**: At M=1 GEMV, the verification cost is dominated by weight reads which are the same regardless of batch size — the "tax" is smaller than expected

### Best Path Forward
Tree-based MTP verification (no retraining, just config + patches) combined with expert offloading/prefetching and the write-based AllReduce mega kernel.

---

## 9. PATH TO 300 TOK/S

### Phase 1: Write-Based P2P AllReduce (+30%)
- Replace NCCL for small payloads (<64KB)
- Write-based protocol: posted PCIe writes + local flag polling
- Target: <10μs per AllReduce (vs NCCL's 14-20μs + launch overhead)
- 120 ops/token × 15μs saved = 1.8ms/token → **~200 tok/s**

### Phase 2: MTP Tree Verification (+20-30%)
- 10-node tree with branching factor 2 at depth 1-2
- Expected 3.5 accepted tokens/step (vs 2.35 linear)
- TREE_ATTN backend with FP8 KV cache support
- Target: **~250 tok/s**

### Phase 3: Expert Offloading + TP=2 (+40-60%)
- Profile legal domain expert activations
- LRU expert cache with MTP-driven prefetching
- TP=2 (AllReduce becomes 2-GPU peer copy: 3-5μs vs 14-20μs)
- Target: **~300+ tok/s**

---

## 10. KEY REFERENCES

### Papers
- DFlash: arXiv:2602.06036 (Block Diffusion Speculative Decoding)
- EAGLE-3: arXiv:2503.01840 (Training-Time Test)
- Sequoia: arXiv:2402.12374 (Hardware-Aware Tree Speculation)
- FlashFormer: arXiv:2505.22758 (Whole-Model Kernels)
- Pre-Attention Expert Prediction: ETH Zurich (93-97% accuracy)
- What Gets Activated: arXiv:2601.10159 (Domain/Driver Experts)
- Super Experts: arXiv:2507.23279 (Critical Expert Subset)
- MoE-SpeQ: arXiv:2511.14102 (Speculative Expert Prefetch)
- KTransformers: SOSP 2025 (CPU/GPU Hybrid MoE)
- NCCL LL Protocol: Developer Blog (Write-Based AllReduce)

### Our Code
- `fused-moe/csrc/write_based_allreduce.cu` — Write-based P2P AllReduce benchmark
- `fused-moe/csrc/megakernel_v1.cu` — MegaKernel persistent fused forward pass
- `fused-moe/csrc/write_allreduce_ext.cu` — PyTorch extension for vLLM integration
- `fused-moe/write_allreduce_ext/` — vLLM monkey-patch and test suite
- `fused-moe/csrc/verdict_fused_independent_tma.cu` — Production VerdictMoE kernel
- `fused-moe/csrc/verdict_dense_fp4.cu` — Dense FP4 GEMM kernel
- `fused-moe/csrc/p2p_tree_allreduce.cu` — Original read-based P2P AllReduce (superseded)
