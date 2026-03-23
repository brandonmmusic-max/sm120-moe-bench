# SM120 MoE Kernel Investigation Notes

## Current Path: FlashInfer CUTLASS fused MoE (SM120 JIT)

vLLM dispatches to `FlashInferExperts` → `flashinfer.cutlass_fused_moe(backend='120')`

### Per-layer kernel breakdown (nsys trace, M=1 decode, Qwen3.5-397B TP=4):

| Kernel | Time (μs) | What |
|--------|-----------|------|
| `GemmUniversal<GroupProblemShape>` × 2 | 25 + 26 = 51 | GEMM1 + GEMM2 (grouped, 188 CTAs persistent) |
| `doActivationKernel<FP4, BF16>` | 23 | SwiGLU activation (separate kernel!) |
| `expandInputRowsKernel<BF16, FP4>` | 23 | Input permute + BF16→FP4 quantize |
| `computeStridesTmaWarpSpecialized` | 14 | TMA stride setup for grouped GEMM |
| `finalizeMoeRoutingKernel` | 8 | Output reduction + unpermute |
| `blockExpertPrefixSumKernel` | 2 | Routing prefix sum |
| `globalExpertPrefixSumKernel` | 1 | Routing global prefix |
| **Total** | **~122** | 7 kernel launches |

### Why GATED_ACTIVATION fusion is not implemented

File: `nv_internal/.../moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl` line 261:
```cpp
static_assert(FUSION == EpilogueFusion::NONE || FUSION == EpilogueFusion::FINALIZE,
              "Unimplemented fusion provided to TMA WS MoE gemm launcher");
```

All SM120 instantiations use `NONE` fusion. `GATED_ACTIVATION` would fuse SwiGLU + FP4 requant into GEMM1's epilogue, saving the 23μs activation kernel.

### TRT-LLM cubin investigation

- Cubins downloaded from NVIDIA artifact repo, target `sm_100` (SM100)
- cuobjdump confirms: `STTM tmem[...]`, `UTCOMMA.4X`, `UTMALDG` — all tcgen05/TMEM instructions
- SM120 does NOT have TMEM hardware → illegal instruction on SM120
- Host-side launcher compiles for SM120 (patched arch checks)
- Routing, activation, finalize kernels work on SM120 — only GEMM cubins fail
- `cuModuleLoad` accepts sm100 cubins on SM120 (loads without checking ISA)
- Kernel execution fails at launch with GEMM error (not illegal instruction, caught by runner)

### TRT-LLM Cubin Investigation

Cubins downloaded from NVIDIA artifact repo → `sm100f` target.
cuobjdump confirms tcgen05/TMEM instructions:
- `STTM tmem[UR7]` — Store to Tensor Memory
- `UTCOMMA.4X gdesc, gdesc, tmem, tmem, idesc, tmem` — tcgen05 MMA
- `UTMALDG.3D` — TMA load (tcgen05)
- `USETMAXREG.DEALLOC.CTAPOOL` — tcgen05 register management

SM120 does NOT have TMEM → these instructions cause illegal instruction errors.
`cuModuleLoad` accepts the cubins (loads without ISA check) but execution fails.
Host-side launcher patched to accept SM120 (arch check bypass) — routing, activation,
and finalize kernels work. Only GEMM cubins fail.

The TRT-LLM path also only supports **MxE2m1 (MXF4)** weights with BF16 activation,
not **E2m1 (NVF4)** which our Qwen3.5 model uses. Format mismatch independent of ISA.

## Optimization Targets (in order of impact)

1. **GATED_ACTIVATION epilogue fusion** — save 23μs activation kernel
   - REQUIRES weight interleaving: [gate_0, up_0, gate_1, up_1, ...]
   - GEMM1 currently runs N=512 (gate+up combined), gate[i] and up[i] in different tiles
   - Interleaving puts paired values in same CTA tile, enabling epilogue fusion
   - `static_assert` in launcher enforces NONE/FINALIZE only — needs new instantiations
2. **Input quant fusion** — save 23μs expandInputRows kernel
   - Already fused with permutation (BF16→FP4 + routing in one kernel)
   - Kernel pads ALL 512 experts (128×512=65K ops) even with only 10 active
   - Patched launcher to use `min(experts, tokens)` for grid size
3. **TMA stride precomputation** — save some of 14μs compute strides
4. **FINALIZE fusion on GEMM2** — already instantiated, autotuner may select it

## Key Code Locations (in vLLM container)

```
flashinfer/data/csrc/fused_moe/cutlass_backend/
  flashinfer_cutlass_fused_moe_binding.cu   — Python↔C++ binding, NeedQuant selection
  cutlass_fused_moe_kernels.cuh             — expandInputRows, doActivation, runMoe

nv_internal/tensorrt_llm/kernels/cutlass_kernels/
  include/moe_kernels.h                     — CutlassMoeFCRunner class
  include/moe_gemm_kernels.h                — EpilogueFusion enum, GEMM runner
  moe_gemm/moe_gemm_template_dispatch.h     — SM120 dispatch, tile config selection
  moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl  — INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM

trtllm_fused_moe_kernel_launcher.cu         — TRT-LLM fused runner (cubin-based)
trtllm_fused_moe_runner.cu                  — TRT-LLM GEMM dispatch

flashinfer/jit/fused_moe.py                 — JIT spec (SM120 compile flags)
flashinfer/fused_moe/core.py                — Oracle dispatch, MoERunner
vllm/.../fused_moe/flashinfer_cutlass_moe.py  — FlashInferExperts class
vllm/.../fused_moe/oracle/nvfp4.py          — Backend priority selection
```
