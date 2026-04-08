# What I Learned Building Custom CUDA Kernels for Blackwell RTX (SM 12.0) — Notes for the Community

Hey everyone. I've spent the last few weeks building custom CUDA kernels for inference on 4x RTX PRO 6000 Blackwell GPUs running Qwen3.5-397B-A17B-NVFP4 with vLLM. I wanted to share what I learned in case it helps anyone else pushing Blackwell desktop/workstation GPUs for LLM inference.

This isn't claiming any breakthroughs — just sharing findings, dead ends, and things I wish I'd known earlier. Some of this might save someone days of debugging.

**Hardware**: 4x NVIDIA RTX PRO 6000 Blackwell (96GB each, SM 12.0, PCIe Gen5 — NO NVLink)
**Model**: Qwen3.5-397B-A17B-NVFP4 (512 experts, top-10 routing, 60 MoE layers)
**Software**: vLLM 0.17.1, CUDA 13.2, Driver 595

---

## 1. ALL Fast AllReduce Paths Are Silently Disabled on PCIe Blackwell

This one hurt the most. vLLM advertises AllReduce+RMSNorm fusion as "5-20% E2E Speedup" on Blackwell, but on PCIe Blackwell (RTX 5090, RTX PRO 6000, etc.), **every single fast AllReduce path is disabled**:

| Path | Why Disabled on PCIe SM 12.0 |
|------|-----|
| AllReduce+RMSNorm Fusion | FlashInfer workspace needs NVLink multicast |
| NCCL Symmetric Memory | Default off (env var) |
| FlashInfer AllReduce | Requires NVSwitch |
| Custom AllReduce (IPC) | NVLink check fails |
| Torch SymmMem | SM 12.0 not in size tables |

Everything falls through to plain PyNCCL. No warnings in the logs.

**Quick fix**: `export VLLM_USE_NCCL_SYMM_MEM=1` — zero code changes, enables the NCCL symmetric memory path. Also add SM 12.0 entries to `all_reduce_utils.py` size tables.

**What about the fusion?** We fixed 3 FlashInfer bugs and 3 vLLM bugs to get it running. Result: **14% slower and crashes after 5-10 minutes on PCIe.** The Lamport IPC AllReduce uses cacheline-level loads (~200ns/64B) while NCCL uses DMA engines at full PCIe bandwidth. Don't bother with fusion on PCIe — it's designed for NVLink/NVSwitch.

---

## 2. SM120 MMA Fragment Layout Is Undocumented

Building a flash attention kernel from scratch for SM120, the `mma.sync.aligned.m16n8k16` A-register fragment layout is:
```
Ra0 = [group, 2*thread]
Ra1 = [group+8, 2*thread]
Ra2 = [group, 2*thread+8]
Ra3 = [group+8, 2*thread+8]
```

This is **NOT** what you'd guess from the GEMM documentation (which implies K/K+8 interleaving). I had to empirically discover this through probing.

Also: cross-thread softmax reduction requires 4-thread shuffle within each row group. Miss this and your output is 4x wrong. And masking invalid positions needs `-FLT_MAX`, not `0` (zero is a valid softmax score).

---

## 3. scale_vec::4X for FP4 MMA — Zero-Cost Quantization

The `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X` instruction applies 4 separate E4M3FN scale bytes per K=64 tile. The key insight: if you pack K-elements consecutively (not strided), each scale byte maps exactly to one SF_BLOCK=16 checkpoint scale.

```
Consecutive-K packing:
  Byte 0 → K[0:15]   (block 0, one scale)
  Byte 1 → K[16:31]  (block 1, one scale)
  Byte 2 → K[32:47]  (block 2, one scale)
  Byte 3 → K[48:63]  (block 3, one scale)
```

This eliminates ALL rescaling math. Strided packing gave 7.14% error and 625.9μs. Consecutive: **0.0000% error and 116.1μs (5.4x faster)**. The dot product is K-permutation invariant — reordering K positions is free.

---

## 4. __threadfence() Is MANDATORY in Atomic Barriers

If you're using atomic counter barriers instead of cooperative_groups for CUDA graph compatibility:

```cpp
__syncthreads();
__threadfence();  // WITHOUT THIS: 77,000% error
if (threadIdx.x == 0) atomicAdd(counter, 1);
while (atomicAdd(counter, 0) < expected) {}
__threadfence();
__syncthreads();
```

Without `__threadfence()`, inter-CTA memory visibility is NOT guaranteed. I was getting 77,000% error from stale global memory reads. Adding the fence (1-2μs cost) brought it down to 9.6% (same as FP4 quantization noise).

---

## 5. Runtime-Indexed Arrays → Stack Spills (Register Scalarization)

This was a 1.62x speedup from one line:

```cpp
// BAD: runtime-indexed → 208 bytes on stack → 91.6 μs
float gate_acc[MAX_M][4];
gate_acc[token_idx][i] = ...;  // token_idx is runtime variable

// GOOD: compile-time unrolled → registers → 56.7 μs
#pragma unroll
for (int t = 0; t < MAX_M_T; t++) {
    if (t < m_count) mma(gate_acc[t], ...);
}
```

When the index is a runtime variable, the compiler puts the array on the stack (local memory). `#pragma unroll` makes each iteration a compile-time constant → scalar registers.

---

## 6. uint32 SMEM Loads for FP4 Packing — 24.6% Speedup

Reading 8 FP4 nibbles via `*(uint32_t*)&smem[swizzle(offset)]` instead of 16 individual scalar nibble extractions:
- **Before**: 16 loads + 16 shift-OR ops per MMA input
- **After**: 1 uint32 load, directly feeds MMA registers
- **Result**: 116.1μs → 87.5μs (**24.6% faster**, beats CUTLASS 1.12x)

---

## 7. Persistent Kernels Don't Help for Decode-Sized MoE

We tried three approaches and all regressed:

| Approach | Result | Why |
|----------|--------|-----|
| 0-barrier persistent | 5.8x slower | Per-CTA sequential K-loads dominate (64 tiles vs 4) |
| Work-stealing | 15-26% slower | Atomic contention + barrier scaling with extra CTAs |
| cp.async pipelining | 0-13.7% slower | Kernel is bandwidth-bound, not latency-bound |

The MoE kernel at M=1 (single-token decode) does ~2 MMA operations per K-tile. There's essentially no compute to overlap with memory loads. Barriers cost ~5μs which is negligible. Sprint 9's static grid assignment with K-group splitting remains optimal.

---

## 8. TMA Performance Is Driver-Dependent

TMA (`cp.async.bulk.tensor`) results flipped across driver versions:
- **Driver 580/590**: TMA 0.6% faster standalone, **3.1% E2E regression** (mbarrier overhead)
- **Driver 595**: TMA **11% faster standalone**, E2E improvement projected

If you benchmarked TMA on an older driver and dismissed it, try again after a driver update. Also: `cp.async.bulk.tensor.shared::cta` is required on SM120 (NOT `shared::cluster`). The cluster variant produces all zeros without proper cluster launch control.

---

## 9. NCCL p99 Spikes Are GPU Context Switching, Not NCCL

We measured NCCL AllReduce latency across 500 iterations:
- **p50**: 14μs (near theoretical minimum)
- **p99**: 8,800μs (!!!)

After testing 13 different NCCL configurations (algorithms, protocols, buffers), every config showed identical spikes. Root cause: **stale background GPU processes** causing context switches.

Fix: Kill background GPU processes before starting inference. p99 dropped from 2,330μs to 22μs (106x improvement). The spikes occur every ~139 iterations (4.3ms GPU time-slice quantum) — hardware-level scheduling.

---

## 10. In-Kernel P2P AllReduce Is 4-16x Slower Than NCCL on PCIe

P2P access works on PCIe Blackwell (confirmed via `can_device_access_peer`). But in-kernel load/store P2P reads hit the SM's cache hierarchy at cacheline granularity:
- 16KB tensor: 256 cachelines × ~200ns = ~50μs (NCCL does it in ~14μs)
- 64KB tensor: 259μs (NCCL: ~16μs)

NCCL uses hardware DMA engines that transfer at full PCIe bandwidth. Kernel load/store can't compete. Don't try to fuse AllReduce into your compute kernel on PCIe.

---

## 11. `cvt.rn.satfinite.e4m3x2.f32` Byte Ordering

PTX `cvt.e4m3x2.f32 d, a, b` puts `a` in the HIGH byte and `b` in the LOW byte — **opposite of what you might expect**. This caused 35%+ byte mismatch errors until we checked empirically. Always validate PTX semantics on new architectures rather than assuming.

---

## 12. SM120 Hardware Limitations vs SM100 (Datacenter Blackwell)

If you're coming from GB200/B200 research, SM120 (desktop RTX Blackwell) is missing:
- **No WGMMA** (uses legacy `mma.sync`)
- **No TMEM** (Tensor Memory)
- **No tcgen05** instructions
- **No FlashAttention-4** (requires TMEM)
- **96KB SMEM per block** (not 228KB)
- Shared memory is `shared::cta` only (no `shared::cluster` for TMA)

Your optimization path differs significantly from datacenter Blackwell research.

---

## Repo

All the code, results, and sprint notes are at: https://github.com/brandonmmusic-max/sm120-moe-bench

FlashInfer PR (SM120 MMA fragment layout): https://github.com/flashinfer-ai/flashinfer/pull/2786

Happy to answer questions if any of this is useful. Running large MoE models on desktop Blackwell GPUs is a different beast than datacenter and I hope sharing these notes saves someone some pain.
