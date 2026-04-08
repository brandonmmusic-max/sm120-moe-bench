# SM120 MoE Inference Optimization — Release Manifest

## Files Ready for GitHub Push

### New Files (this session)
| File | Lines | Status | Description |
|------|-------|--------|-------------|
| `DISCOVERIES.md` | ~500 | Ready | Findings and analysis document |
| `fused-moe/csrc/megakernel_v1.cu` | 1830+ | WIP | Persistent fused transformer kernel (188 SM, all phases parallelized) |
| `fused-moe/csrc/megakernel_p2p_test.cu` | ~400 | Ready | Multi-GPU P2P AllReduce test (correctness + latency) |
| `fused-moe/csrc/write_based_allreduce.cu` | 614 | WIP | Write-based P2P AllReduce benchmark (launch barrier added, deadlock fix in progress) |
| `fused-moe/csrc/write_allreduce_ext.cu` | 483 | Ready | PyTorch C++ extension for vLLM integration |
| `fused-moe/write_allreduce_ext/setup.py` | 40 | Ready | Build script (SM 12.0a) |
| `fused-moe/write_allreduce_ext/vllm_patch.py` | 450 | Ready | Monkey-patch for vLLM CudaCommunicator.all_reduce() |
| `fused-moe/write_allreduce_ext/test_write_allreduce.py` | 350 | Ready | Correctness + CUDA graph + latency tests |
| `fused-moe/write_allreduce_ext/__init__.py` | 10 | Ready | Package init |

### Modified Files
| File | Change | Status |
|------|--------|--------|
| `fused-moe/csrc/megakernel_v1.cu` | NUM_CTAS 99→188, CTA-0 phases parallelized | Done |

### Files NOT to Push (internal/temp)
- Docker images (`tree-patched-v1` through `v7`, `tree-rejection-v1/v2`, `tree-full-pipeline`)
- `/tmp/vllm-tree-fix/` (vLLM patches — these are container-only)
- Compiled binaries (`megakernel_v1`, `write_based_allreduce`)
- Training artifacts (`/root/dflash_trained_750k/` on vast.ai)

---

## Potential PRs to Upstream Projects

### PR 1: vLLM — TreeAttention FP8 KV Cache Support
**Repo**: vllm-project/vllm  
**Priority**: Medium  
**Files**: 
- `vllm/v1/attention/backends/tree_attn.py` — add fp8/fp8_e4m3 to supported_kv_cache_dtypes, add supports_attn_type for DECODER, add supports_compute_capability for SM 12.0
- `vllm/v1/attention/ops/triton_unified_attention.py` — uint8→fp8e4nv bitcast for FP8 KV cache loads (4 locations)
- `vllm/platforms/cuda.py` — add TREE_ATTN to SM 12.0 backend priority list

**Why**: TreeAttention exists in vLLM but doesn't work with FP8 KV cache or on SM 12.0 GPUs. These patches enable it.

### PR 2: vLLM — MTP Tree Verification Pipeline  
**Repo**: vllm-project/vllm  
**Priority**: High (if we get it working end-to-end)  
**Files**:
- `vllm/v1/spec_decode/eagle.py` — positions fallback for MRoPE, tuple unpack fix for MTP
- `vllm/v1/worker/gpu_model_runner.py` — num_draft_tokens_per_req for tree mode
- `vllm/v1/core/sched/scheduler.py` — num_lookahead_tokens for tree
- `vllm/config/speculative.py` — num_draft_tokens_per_request property
- `vllm/v1/sample/tree_rejection_sampler.py` — NEW: tree-aware rejection sampling

**Why**: MTP tree verification infrastructure exists in vLLM but the pipeline doesn't pass all tree tokens through. These patches complete it.

**Status**: One truncation bug remaining. NOT ready for PR yet.

### PR 3: vLLM — DFlash on Qwen3.5 MoE (Mixed Attention)
**Repo**: vllm-project/vllm  
**Priority**: Low (DFlash training results were negative)  
**Files**: The 13 files from PR 36847 (CentML/vllm dflash-attempt2) plus our EagleModelMixin patches for Qwen3.5/Qwen3Next

**Why**: DFlash works on dense models but the Qwen3.5 mixed-attention architecture needed additional patches. However, our trained model only achieved 8.6% acceptance — not useful.

**Status**: Infrastructure works, model quality insufficient. PR not recommended unless someone trains a better drafter.

### PR 4 (Community): Write-Based P2P AllReduce for PCIe Multi-GPU
**Repo**: Standalone or vllm-project/vllm  
**Priority**: High — any PCIe multi-GPU setup benefits  
**Results**: 8.4μs at 8KB (1.66x faster than NCCL SymmMem)  
**Bug report**: atomicAdd_system broken on SM 12.0 for cross-GPU targets — should be filed with NVIDIA

### NOT PR-worthy (Internal)
- MegaKernel — research prototype, architecture decision changed to hybrid approach
- Expert profiling / gate weight analysis — analysis results, not code

## Research References Found
- FlashFormer (arXiv:2505.22758): whole-model kernels, atomic barriers
- Mirage MPK (arXiv:2512.22219): worker-scheduler SMs, 1-2μs task transitions
- ParallelKittens (arXiv:2511.13940): PGL for multi-GPU, NVLink required
- Cursor Warp Decode (cursor.com/blog/warp-decode): neuron-centric MoE, 1.84x on Blackwell
- FlashMoE (NeurIPS 2025): persistent MoE + RDMA fusion

---

## Findings Worth Sharing (Blog Post / Community)

### Finding 1: PCIe Write vs Read Asymmetry for AllReduce
**Significance**: High — affects anyone doing multi-GPU inference on PCIe  
**Summary**: PCIe reads are 300-500ns per cacheline (request-response). PCIe writes are posted (fire-and-forget, pipelined). NCCL's LL protocol exploits this — all cross-GPU transfers are writes. Custom P2P AllReduce using reads is 4-16x slower than NCCL, but write-based approaches can match or beat it.  
**Measured**: 1.84μs per 4KB P2P write on PCIe 5.0 x16. In-kernel cooperative AllReduce: 7.9μs minimum for 16KB across 4 GPUs.

### Finding 2: RTX PRO 6000 Blackwell Has 188 SMs (Not 99)
**Significance**: Medium — affects kernel developers targeting this GPU  
**Summary**: Early specs and some documentation listed 99 SMs for GB202. The RTX PRO 6000 (workstation variant) has 188 active SMs, confirmed via `cudaGetDeviceProperties`. This is the near-full GB202 die.

### Finding 3: PCIe AllReduce is 41-50% of MoE Decode Wall Time
**Significance**: High — affects anyone running large MoE models on PCIe GPUs  
**Summary**: With TP=4 on Qwen3.5-397B, 120 AllReduces per token consume 2.4ms out of 5.8ms total decode time. This makes PCIe communication — not memory bandwidth, not compute — the primary bottleneck for single-user decode.

### Finding 4: MTP Tree Verification Requires 7 vLLM Patches for FP8+SM120
**Significance**: Medium — useful for vLLM community  
**Summary**: TreeAttention backend exists in vLLM 0.19.0 but doesn't work with FP8 KV cache, SM 12.0, MRoPE models, or the V1 engine's backend selector. We identified and patched all 7 blockers.

### Finding 5: DFlash on MoE — Negative Result
**Significance**: Medium — saves others from the same $2500 mistake  
**Summary**: DFlash speculative decoding achieves 6x speedup on Qwen3-8B (dense) but only 8.6% acceptance on Qwen3.5-397B (MoE). Root cause: the FC layer reinit can't learn the 397B MoE feature space. The z-lab pretrained FC was for 9B dense — different distribution entirely.

### Finding 6: MoE Expert Activation is Highly Predictable
**Significance**: Medium — supports expert offloading research  
**Summary**: Gate weight SVD shows rank-1 dominance (90% variance in 1 singular value). Adjacent-token expert reuse is 71%. Cross-layer prediction accuracy is 93-97%. Domain-specific (legal) content consistently activates the same expert subset. Pre-loading top-100 experts per layer gives estimated 85%+ cache hit rate.
