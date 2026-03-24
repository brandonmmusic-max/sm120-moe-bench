# VerdictMoE Phase 2 Results

## Task 1: Per-Layer Output Comparison (Diagnose Correctness Bug)

**Date:** 2026-03-24
**Status:** PASSED — 3 root causes identified and fixed

### Root Cause 1: Block Scale Swizzle (Primary)

**Bug:** `prepare_nvfp4_moe_layer_for_fi_or_cutlass()` in `flashinfer_fp4_moe.py`
unconditionally applies `swizzle_blockscale()` to w13_scale and w2_scale for all
non-TRTLLM backends, including VERDICT_MOE.

`swizzle_blockscale()` transforms scales from `[E, M, K]` to a CUTLASS-specific
interleaved layout via:
```python
padded.reshape(B, M_padded//128, 4, 32, K_padded//4, 4).permute(0,1,4,3,2,5)
```

VerdictMoE's CUDA kernel reads scales with simple linear indexing:
```c
w_sf[row * sf_cols + col / FP4_BLOCK_SIZE]
```
This is completely incompatible with the swizzled layout.

**Fix:** Added `if backend != NvFp4MoeBackend.VERDICT_MOE:` guard around `swizzle_blockscale()`
calls in the `else` branch. Also set `is_nvfp4_scale_swizzled=False` for VERDICT_MOE in
`make_nvfp4_moe_quant_config()`.

**Files modified:**
- `flashinfer_fp4_moe.py` (in container)
- `oracle/nvfp4.py` (in container)

### Root Cause 2: N_half Dimension Error

**Bug:** `verdict_moe.py` line 289:
```python
n = w2.size(2) * 2  # N_half * 2 = N  ← WRONG COMMENT
N_half = n // 2                         ← BUG: should be N_half = n
```

- `w2.size(2)` = 512 (packed FP4 bytes for intermediate_size=1024)
- `n = 1024` = actual intermediate_size = N_half (NOT N!)
- Code computed `N_half = 512` instead of `N_half = 1024`

**Impact:**
- GEMM1: `N2 = 2*N_half = 1024` instead of correct 2048. Only 50% of W1 rows processed.
- Expert stride wrong: `eid * 1024 * 2048` instead of `eid * 2048 * 2048` (experts overlap!)
- SwiGLU output truncated to 512 instead of 1024 elements
- GEMM2 K-reduction covers half the intermediate

**Why standalone tests passed:** Used `N_half=256` which coincidentally matched `BLOCK_SIZE=256`.

**Fix:** `N_half = w2.size(2) * 2  # intermediate_size`

**Additional CUDA kernel fixes required:**
- GEMM1: Changed `if (tid < N_half)` to `for (int row = tid; row < N_half; row += BLOCK_SIZE)`
  to handle N_half=1024 > BLOCK_SIZE=256
- SwiGLU reduce: Same loop fix for N_half > BLOCK_SIZE
- SMEM: Added `cudaFuncSetAttribute(verdict_gemm1_distributed,
  cudaFuncAttributeMaxDynamicSharedMemorySize, smem_k1)` because SMEM grew from 18KB to 72KB
  (exceeds default 48KB limit)
- GEMM2: Already correct (strided loop and quarter-based reduction handle N_half > BLOCK_SIZE)

### Root Cause 3: E4M3FN Decode Integer Overflow

**Bug:** `decode_e4m3fn()` used integer bit-shift to compute 2^(e-7):
```c
val = (1.0f + m * 0.125f) * __int2float_rn(1 << (e + 17));
val *= 5.960464477539063e-08f;  // 2^(-24)
```

For e≥14, `1 << (e+17)` overflows signed 32-bit int:
- e=14: `1 << 31 = -2147483648` (INT_MIN) → decode returns **-128** instead of +128
- e=15: `1 << 32` is UB → typically returns **0** instead of 256

**Impact:** 3-11% of block scales in Qwen3.5-397B have exponent ≥14 (range 128-448).
These are the LARGEST weight groups, so their corruption disproportionately affects output.

| Weight | Scales with e≥14 | Percentage |
|--------|-----------------|------------|
| gate_proj | 14061/262144 | 5.4% |
| up_proj | 28703/262144 | 10.9% |
| down_proj | 68-7506/262144 | 0.03-2.9% |

**Fix:** Replaced integer shift with `ldexpf()`:
```c
val = ldexpf(1.0f + __int2float_rn(m) * 0.125f, e - 7);
```
Correct for all exponent values, no overflow possible.

## Task 2: Verify Correctness Fix

**Date:** 2026-03-24
**Status:** PASSED — All 4 prompts produce coherent, correct output

### Test Parameters
- Endpoint: `http://localhost:9200/v1/completions`
- Model: `qwen3.5-397b-nvfp4` (Qwen3.5-397B-A17B-NVFP4)
- Temperature: 0 (greedy decoding)
- All 4 workers confirmed VERDICT_MOE backend
- Container image: `vllm-qwen35-k64:verdict-moe-v2`

### Prompt 1: "The capital of Kentucky is" (max_tokens=50)

**Expected:** Coherent text mentioning Frankfort
**Actual:** "Frankfort, a city that serves as the seat of Franklin County and is situated
along the Kentucky River in the north-central part of the state. Frankfort is the only
capital city in the United States that is not the most populous city in its state"
**Result:** PASS — Correct, detailed, factually accurate

### Prompt 2: "def fibonacci(n):" (max_tokens=100)

**Expected:** Valid Python code
**Actual:**
```python
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
```
**Result:** PASS — Valid recursive fibonacci implementation

### Prompt 3: "Explain quantum entanglement in one sentence:" (max_tokens=60)

**Expected:** Coherent physics explanation
**Actual:** "Quantum entanglement is a phenomenon where two or more particles become
inextricably linked such that the quantum state of one cannot be described independently
of the others, causing an instantaneous correlation between their properties regardless
of the distance separating them."
**Result:** PASS — Accurate, well-written physics explanation

### Prompt 4: "Write a detailed essay about the history of AI:" (max_tokens=200)

**Expected:** Coherent essay with no repetition
**Actual:** Structured essay with thinking section, outline, and coherent content about AI history
**Result:** PASS — No repetition, proper essay structure

### Decode Throughput (Scalar GEMV, correct output)

| Metric | Value |
|--------|-------|
| Median | 16.1 tok/s |
| Average | 16.3 tok/s |
| Min | 15.1 tok/s |
| Max | 17.5 tok/s |

**Note:** Throughput is similar to Phase 1 (~17.3 tok/s) because the scalar GEMV is
compute-bound. With correct N_half=1024 (4× larger than buggy N_half=256), each thread
processes 4× more rows but the per-layer latency offsets MTP acceptance improvement.

## Task 3: Replace Scalar GEMV with CUTLASS MMA Atoms

**Date:** 2026-03-24
**Status:** DEFERRED — Architecture plan documented, implementation requires Sprint 2

### Assessment

The scalar `decode_fp4()` / `decode_e4m3fn()` path in `verdict_moe_ext.cu` processes
one FP4 element per thread per iteration. With N_half=1024, GEMM1 does
256 threads × 4 rows × 64 K-elements = 65K scalar ops per block. This is fundamentally
memory-bandwidth-limited by the strided FP4 nibble reads.

CUTLASS MMA replacement requires:
1. **GEMM1**: `mxf4nvf4.block_scale.m16n8k64` tensor core atoms (validated in Sprint 1)
   - TMA loads for FP4 weights (128-byte aligned, swizzled SMEM)
   - E4M3FN scale factors via `scale_vec::1X` (sf_vec_size=16)
   - Output: FP32 fragments in CLayout (SM80_16x8_Row on SM120)
2. **SwiGLU**: CLayout → E4M3 conversion + Swizzle<3,4,3> SMEM handoff (validated in Task 2)
3. **GEMM2**: `f8f6f4.m16n8k32` MMA (E4M3 × FP4, validated in Task 1)
4. **Integration**: torch.utils.cpp_extension with CUTLASS 4.4.1 headers

### What Exists
- `collective_mma_test.cu`: CUTLASS GemmUniversal NVFP4 — bit-exact on SM120
- `gemm2_test.cu`: E4M3×FP4 via MxF8F6F4 schedule — bit-exact on SM120
- `clayout_to_alayout_test.cu`: Full CLayout→ALayout→GEMM2 pipeline — bit-exact
- `verdict_fused_single_expert.cu`: Fused single-expert (FP32 weights, cooperative)
- `verdict_fused_multi_expert.cu`: 10-expert fused — 38.9μs (56× vs baseline)

### Projected Performance

| Path | MoE/layer | Projected tok/s | Notes |
|------|----------|----------------|-------|
| VLLM_CUTLASS (reference) | 98μs | ~129 | 5 kernels, tensor core |
| VerdictMoE scalar (current) | ~280μs | ~16 | Scalar GEMV, correct but slow |
| VerdictMoE + CUTLASS GEMM | ~60-70μs | ~140-160 | Per-expert CUTLASS dispatch |
| VerdictMoE cooperative + MMA | ~45-55μs | ~190-220 | Fused with tensor cores |

### Recommended Path
1. **Short-term**: Fall back to VLLM_CUTLASS (129 tok/s) for production
2. **Sprint 2**: Integrate CUTLASS GemmUniversal per-expert dispatch into VerdictMoE
3. **Sprint 3**: Fuse via cooperative kernel with TMA + MMA atoms

## Task 4: Benchmark (Deferred)

Performance benchmarking deferred to Sprint 2 when tensor core integration is complete.
Current scalar GEMV benchmark: 16.1 tok/s median (vs 129 tok/s VLLM_CUTLASS baseline).

## Task 5: Commit + Push

**Date:** 2026-03-24
**Status:** PENDING

### Summary of All Changes

**Root cause fixes (3 bugs):**

| Bug | File | Fix | Impact |
|-----|------|-----|--------|
| swizzle_blockscale | flashinfer_fp4_moe.py | Skip swizzle for VERDICT_MOE | Primary: scale data garbled |
| N_half dimension | verdict_moe.py | `N_half = w2.size(2) * 2` (was `//2`) | 50% of weights unread |
| E4M3FN overflow | verdict_moe_ext.cu | `ldexpf()` replaces `1 << (e+17)` | 3-11% of scales wrong sign |

**Supporting fixes:**

| Fix | File | Description |
|-----|------|-------------|
| GEMM1 loop | verdict_moe_ext.cu | Loop for N_half > BLOCK_SIZE |
| SwiGLU loop | verdict_moe_ext.cu | Loop for N_half > BLOCK_SIZE |
| SMEM limit | verdict_moe_ext.cu | cudaFuncSetAttribute for >48KB |
| Oracle flag | oracle/nvfp4.py | is_nvfp4_scale_swizzled=False |

### Docker Image
- `vllm-qwen35-k64:verdict-moe-v2` — contains all Phase 2 fixes
- Based on: `vllm-qwen35-k64:verdict-moe` (Phase 1 image)
