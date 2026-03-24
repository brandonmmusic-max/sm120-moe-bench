# VerdictGemm Fused MoE Kernel — Results (2026-03-24)

## Summary

Built and validated SM120-native CUDA kernels for fused MoE computation targeting
Qwen3.5-397B decode on RTX PRO 6000 Blackwell (4× TP, 188 SMs, 99KB SMEM).

## Task 1: GEMM2 Standalone Validation

### FP4×FP4 (NVF4) — PASSED

**File:** `fused-moe/csrc/gemm2_correctness_test.cu`

| Metric | Value |
|---|---|
| Shape | [128, 256] × [256, 4096] → [128, 4096] |
| Elements | 524,288 |
| Normalized error | **0.0000%** |
| Within 2% | **524,288/524,288 (100%)** |
| Schedule | KernelTmaWarpSpecializedNvf4Sm120 |

**Bit-exact** with CUTLASS host reference using NVF4 block-scaled FP4.

### E4M3×FP4 (MXF8F6F4) — BLOCKED

**File:** `fused-moe/csrc/gemm2_e4m3_fp4_test.cu`

Attempted mixed-precision FP8×FP4 via `KernelTmaWarpSpecializedMxf8f6f4Sm120`.

**Findings:**
- CUTLASS `CollectiveBuilder` successfully matches `mx_float8_t<float_e4m3_t>` × `mx_float4_t<float_e2m1_t>`
- Alignment requirement for MXF8F6F4 FP4 operand: **512 bits (128 elements)**, not the standard 128 bits
- **Bug in CUTLASS `fp4_shift_B`** (`cute/atom/mma_traits_sm120.hpp:256`): lambda takes `RegisterTypeB&` but `cute::transform` passes a tensor view, causing type mismatch
- This is a genuine CUTLASS library bug in the MXF8F6F4 path for SM120

**Decision:** Use validated FP4×FP4 (NVF4) path. Fusion savings from eliminating GMEM round-trips don't depend on intermediate precision.

## Task 2: CLayout → ALayout SMEM Handoff

### CLayout Discovery — CORRECTED

**File:** `fused-moe/csrc/clayout_to_alayout_test.cu`

The Phase 1c "LANDMARK" discovery was **partially wrong**. The claim that "d[1] and d[3] are always zero" was an artifact of the v3 kernel's test setup (only half the A matrix populated).

**Actual SM120 FP4 MMA CLayout (`mxf4nvf4.block_scale.scale_vec::2X.m16n8k64`):**

```
mg  = lane_id / 8     (0-3, four M-groups)
col = lane_id % 8     (0-7, eight output columns)

d[0] = C[mg,    col]     → rows 0-3
d[1] = C[mg+8,  col]     → rows 8-11
d[2] = C[mg+4,  col]     → rows 4-7
d[3] = C[mg+12, col]     → rows 12-15
```

**Validated with:**
- All-ones test: 0.0000% error, 128/128 (100%)
- Row-dependent pattern: 0.0000% error, 128/128 (100%)

### UE8M0 Scale Factor Bias — CONFIRMED

**UE8M0 bias = 127** (not 128 as assumed in some code):
- `0x7F` (127) → scale = 2^(127-127) = 1.0 (but gives combined 0.5 due to double-application)
- `0x80` (128) → scale = 2^(128-127) = 2.0 (combined scale = 2.0)

Empirically verified: all-ones A×B with K=64 and SF=0x80 produces output=128 (= 64 × 2.0).

### SMEM Layout — PARTIAL

- **Row ordering is correct** (validated with row-dependent test)
- **K-dimension byte redistribution is NOT simple row-major** — ldmatrix applies a hardware-defined permutation within rows
- For uniform K values, this doesn't matter → test passes
- For varying K values, byte scrambling causes errors

### CLayout → SMEM Write Pattern — VALIDATED

```
g = lane_id / 4,  t = lane_id % 4

Pack d[0] (low nibble) + d[2] (high nibble) → byte at (row=2g, byte_col=t)
Pack d[1] (low nibble) + d[3] (high nibble) → byte at (row=2g+1, byte_col=t)

Each thread writes exactly 2 bytes — no cross-thread conflicts.
```

## Task 3: Single-Expert Fused Pipeline

### 3-Kernel Pipeline — VALIDATED

**Files:**
- `fused-moe/csrc/fused_gemm1_swiglu_gemm2_test.cu` — Correctness test
- `fused-moe/csrc/pipeline_benchmark.cu` — Latency benchmark

| Component | Latency (μs) | Notes |
|---|---|---|
| GEMM1 [128, 4096] × [4096, 512] | 24.5 | CUTLASS NvF4Sm120 |
| SwiGLU + Requant | 10.2 | Custom kernel, FP32→FP4 |
| GEMM2 [128, 256] × [256, 4096] | 9.2 | CUTLASS NvF4Sm120 |
| **Total** | **44.0** | **3 kernel launches** |

### Comparison

| Path | Launches | Latency (μs) | vs Baseline |
|---|---|---|---|
| FlashInfer CUTLASS | 7 | 130 | 1.00x |
| VLLM_CUTLASS | 5 | 98 | 1.33x |
| **VerdictGemm pipeline** | **3** | **44** | **2.95x** |
| Compute floor (GEMMs only) | 2 | 33.7 | 3.86x |

## Tasks 4-5: Multi-Expert + vLLM Integration

### Status: Blocked on ldmatrix K-Dimension Mapping

For true single-kernel multi-expert grid (640 CTAs), needs ldmatrix.b4x16_p64 byte permutation mapping.

**Path forward options:**
1. Empirical byte probing — write unique values to each byte, observe ldmatrix output
2. CUTLASS SmemCopyAtom — use SM100_SU4_DU8x16_x4_LDSM_N for layout
3. GMEM intermediate — write to GMEM, let TMA handle (adds ~2μs)
4. Use existing `cutlass_fp4_moe_mm` grouped GEMM for multi-expert

## Key Technical Discoveries

### 1. SM120 FP4 MMA CLayout is NOT SM80_16x8_Row

SM120 uses: d[0]=C[mg,col], d[1]=C[mg+8,col], d[2]=C[mg+4,col], d[3]=C[mg+12,col]
where mg=lane/8 (0-3), col=lane%8 (0-7)

### 2. CUTLASS MXF8F6F4 Path Has a Bug on SM120

`fp4_shift_B` const-correctness issue prevents mixed E4M3×FP4 GEMM.

### 3. NVF4 UE8M0 Bias is 127, Not 128

Scale = 2^(exp - 127). Code using bias=128 produces 2x errors.

### 4. ldmatrix.b4x16_p64 Has Non-Trivial K-Dimension Byte Permutation

Only affects correctness when K-dimension values vary within a row.

## Files Created/Modified

```
fused-moe/csrc/
  gemm2_correctness_test.cu        — GEMM2 FP4×FP4 validation (bit-exact)
  gemm2_e4m3_fp4_test.cu           — GEMM2 E4M3×FP4 attempt (CUTLASS bug)
  clayout_to_alayout_test.cu       — CLayout mapping + SMEM handoff test
  fused_gemm1_swiglu_gemm2_test.cu — Full 3-kernel pipeline validation
  pipeline_benchmark.cu            — Latency benchmark (44μs per expert)
```
