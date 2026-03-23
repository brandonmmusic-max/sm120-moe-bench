# SM120 Decode Profiling — REAL nsys Ground Truth

## Method
nsys profile with `--enforce-eager` (no CUDA graphs) wrapping the actual vLLM server
process. All 4 TP workers instrumented. 10-second capture window during active decode
(128 output tokens, Qwen3.5-397B-A17B-NVFP4, TP=4, MTP=3, FlashInfer CUTLASS backend).

Kernels are individually visible without CUDA graph opacity. Absolute latencies are
higher than graph mode (no launch optimization), but **proportions are accurate** —
which is all that matters for identifying the bottleneck.

## Results — GPU 0 Kernel Breakdown

| Category | Time (ms) | % | Notes |
|----------|----------|---|-------|
| **NCCL AllReduce + AllGather** | **18,217** | **68.8%** | 254μs avg per AllReduce, 70K+ calls |
| **MoE (all ops)** | **4,244** | **16.0%** | GEMM1+GEMM2+act+quant+route+strides |
| **Attention (all ops)** | **2,407** | **9.1%** | nvjet projections + delta rule + splitK |
| Other (gemvx, elementwise) | 1,439 | 5.4% | MTP, embeddings, misc |
| Normalization | 178 | 0.7% | RMSNorm + LayerNorm |
| **Total** | **26,486** | **100%** | |

## MoE Sub-Breakdown

| Kernel | Time (ms) | % of total | Avg (μs) | Count |
|--------|----------|-----------|----------|-------|
| CUTLASS GemmUniversal (GEMM1) | 1,129 | 4.3% | 34.6 | 32,596 |
| CUTLASS GemmUniversal (GEMM2) | 890 | 3.4% | 27.3 | 32,596 |
| expandInputRowsKernel (permute+quant) | 721 | 2.7% | 22.1 | 32,596 |
| doActivationKernel (SwiGLU+requant) | 637 | 2.4% | 19.6 | 32,596 |
| computeStridesTmaWarpSpecialized | 469 | 1.8% | 14.4 | 32,596 |
| topkGating (routing) | 188 | 0.7% | 5.5 | 34,216 |
| fused_moe_kernel (Triton fallback) | 131 | 0.5% | 40.4 | 3,240 |
| cvt_fp16_to_fp4 | 40 | 0.2% | 1.2 | 32,596 |
| prefix sum kernels | 103 | 0.4% | ~1.5 | ~65K |

## Attention Sub-Breakdown

| Kernel | Time (ms) | % of total |
|--------|----------|-----------|
| nvjet 320x8x64 splitK (full attn proj) | 685 | 2.5% |
| nvjet 128x8x64 splitK (linear attn proj) | 649 | 2.4% |
| WMMA bf16 GEMM (attn/MTP) | 563 | 2.1% |
| nvjet 64x8x128 splitK | 352 | 1.3% |
| fused_sigmoid_gating_delta_rule | 171 | 0.6% |
| nvjet 64x16x128 (prefill attn) | 169 | 0.6% |
| splitK reduce | 199 | 0.7% |

## NCCL Detail

| Op | Time (ms) | % | Avg (μs) | Count |
|----|----------|---|----------|-------|
| AllReduce Sum bf16 RING_LL | 17,985 | 65.9% | 254.8 | 70,593 |
| AllGather RING_LL | 233 | 0.9% | 107.7 | 2,160 |

AllReduce is called ~2× per layer × 60 layers = 120 per decode step.
At 254μs avg: 120 × 254μs = **30.5ms per decode step** in NCCL alone.
At 127 tok/s (7.87ms/token): NCCL = 30.5ms ÷ ~4 tokens/step ≈ **7.6ms/token**.
This exceeds the measured 7.87ms/token — the GPU is almost entirely NCCL-bound.

Note: In CUDA graph mode, NCCL calls are pipelined/overlapped with compute,
reducing the effective NCCL overhead. The 69% figure is the non-overlapped
proportion; the effective bottleneck in graph mode is likely 40-50%.

## Key Insights

1. **NCCL AllReduce is THE bottleneck** — 69% of GPU time (likely 40-50% effective with graph overlap)
2. **MoE kernel optimization has limited impact** — only 16% of GPU time
   - Halving MoE → saves 8% → ~10 tok/s improvement
   - SwiGLU epilogue fusion → saves 2.4% → ~3 tok/s — NOT WORTH IT
   - FP4 quant fusion → saves 2.7% → ~3.5 tok/s
3. **Attention is 9%** — smaller than expected, linear attention (delta rule) is efficient
4. **PCIe TP=4 is the wall** — NVLink would reduce AllReduce from 254μs to ~5-10μs

## Optimization Priority (Revised)

| Target | Potential savings | Effort | Worth it? |
|--------|------------------|--------|-----------|
| NCCL optimization (overlap, fusion) | 20-40% of decode | High | **YES — biggest impact** |
| Reduce TP (TP=2 if model fits) | Eliminate 50% of NCCL | Medium | Worth investigating |
| MoE kernel fusion | 8% max | High | Diminishing returns |
| SwiGLU epilogue fusion | 2.4% | High | **NO** |
| Attention optimization | 4-5% | Medium | Maybe |
