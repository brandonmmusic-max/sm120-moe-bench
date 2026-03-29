# SM120 NVFP4 KV Cache Decode Kernel

NVFP4 (E2M1 + E4M3FN per-block scales) KV cache support for the SM120 Flash Attention decode kernel, targeting 4x NVIDIA RTX PRO 6000 Blackwell GPUs (96GB, SM 12.0, PCIe) running Qwen3.5-397B-A17B-NVFP4 with vLLM.

## Motivation

The standard FP8 E4M3 KV cache uses 1 byte per element. NVFP4 uses 0.5625 bytes per element (0.5B data + 0.0625B scale overhead), yielding **~1.78x cache capacity** at the same GPU memory. For a model like Qwen3.5-397B with HEAD_DIM=256, this translates to significantly longer context windows or higher batch sizes.

| KV Cache Format | Bytes/Element | Cache for 728K tokens (4x96GB) | Relative |
|-----------------|--------------|-------------------------------|----------|
| BF16            | 2.0          | ~364K tokens                  | 1.0x     |
| FP8 E4M3        | 1.0          | ~728K tokens                  | 2.0x     |
| **NVFP4**       | **0.5625**   | **~1,294K tokens**            | **3.55x**|

## NVFP4 Format Specification

### FP4 E2M1 (4-bit: 1 sign + 2 exponent + 1 mantissa)

The data values use the E2M1 format with bias=1:

```
bit[3]   = sign (0=positive, 1=negative)
bit[2:1] = exponent (e), bias = 1
bit[0]   = mantissa (m)
```

**Dequantization formula:**

For exponent e and mantissa m:

- **Subnormal (e=0):** value = (-1)^s * m * 2^(-1) = (-1)^s * m * 0.5
- **Normal (e>0):** value = (-1)^s * (1 + m * 2^(-1)) * 2^(e-1) = (-1)^s * (2+m) * 2^(e-2)

This produces exactly 8 magnitude levels:

| Nibble (bin) | e | m | Formula | Value |
|-------------|---|---|---------|-------|
| 0000 | 0 | 0 | 0 * 0.5 | 0.0 |
| 0001 | 0 | 1 | 1 * 0.5 | 0.5 |
| 0010 | 1 | 0 | (2+0) * 2^(-1) | 1.0 |
| 0011 | 1 | 1 | (2+1) * 2^(-1) | 1.5 |
| 0100 | 2 | 0 | (2+0) * 2^0 | 2.0 |
| 0101 | 2 | 1 | (2+1) * 2^0 | 3.0 |
| 0110 | 3 | 0 | (2+0) * 2^1 | 4.0 |
| 0111 | 3 | 1 | (2+1) * 2^1 | 6.0 |

With sign bit: 16 total values in {0, +/-0.5, +/-1.0, +/-1.5, +/-2.0, +/-3.0, +/-4.0, +/-6.0}.

**Byte packing:** Two FP4 values per byte. Low nibble (bits 3:0) = even-indexed element, high nibble (bits 7:4) = odd-indexed element. This matches the NVIDIA NVFP4 standard.

### E4M3FN Block Scale (8-bit: 1 sign + 4 exponent + 3 mantissa)

Each group of `FP4_BLOCK_SIZE=16` consecutive FP4 values shares one E4M3FN block scale byte.

**Dequantization formula** (bias=7):

- **NaN (e=15, m=7):** mapped to 0.0 (E4M3FN has no infinity, NaN treated as zero)
- **Subnormal (e=0):** value = (-1)^s * m * 2^(-9) = (-1)^s * m / 512
- **Normal (e>0):** value = (-1)^s * (1 + m/8) * 2^(e-7)

**CRITICAL: Uses `ldexpf()`, NOT integer shift.** The expression `1 << (e+17)` overflows a 32-bit integer for e >= 14, producing sign flips and zeros. This bug affected 3-11% of block scales in Qwen3.5-397B (5.4% gate_proj, 10.9% up_proj) and was the third of three critical bugs found during VerdictMoE development. See `PHASE2_RESULTS.md` for the full debugging history.

**Range:** Positive values from 2^(-9) (subnormal min, ~0.00195) to 448.0 (normal max, e=15,m=6). Block scales are always positive (sign bit unused for KV cache scales).

### Two-Level Scale Architecture

For data whose magnitude exceeds the E4M3FN block scale range (max representable = 448 * 6 = 2688), a per-tensor pre-normalization scale is applied:

```
real_value = decode_fp4(nibble) * decode_e4m3fn(block_scale) * tensor_scale
```

- **tensor_scale = 1.0** for normal-range data (attention K/V values in the range ~0.001-10.0)
- **tensor_scale > 1.0** when global max_abs > 80% of max_representable (2688 * 0.8 = 2150)

The K tensor scale is absorbed into the softmax scale (one multiply, not N), and the V tensor scale is applied at output normalization.

### Packed Cache Layout

Each KV cache row (one position, one head) stores:

```
[fp4_data: HEAD_DIM/2 bytes] [block_scales: HEAD_DIM/16 bytes]
```

For HEAD_DIM=256:
- fp4_data: 128 bytes (256 elements / 2 per byte)
- block_scales: 16 bytes (256 elements / 16 per group)
- **Total: 144 bytes per row** (vs 256 for FP8, 512 for BF16)
- **packed_dim = 9 * HEAD_DIM / 16**

## Kernel Architecture

### Phase 1: Tiled Split-KV Partial Attention

```
Grid:  (num_splits, batch_size, num_q_heads)
Block: HEAD_DIM threads (256 for Qwen3.5-397B)
SMEM:  q(1024B) + fp4_data(16384B) + scale_bytes(2048B) + scale_floats(8192B) + p(528B) = ~28KB
```

For each tile of BLOCK_KV=128 KV positions:

1. **Load K tile** — cp.async 16B vectorized loads from paged cache to split SMEM regions (FP4 data + scales separately)
2. **Pre-decode scales** — All 256 threads cooperatively decode 2048 scale bytes to float32 in SMEM (8 decodes/thread, negligible cost)
3. **Q@K^T dot product** — Each thread computes partial dot product over D_PER_THREAD=128 dimensions with on-the-fly FP4 dequant:
   ```
   for each scale group (8 groups of 16 elements):
       scale = scale_float_smem[group_idx]
       for each pair in group (8 pairs):
           byte = fp4_smem[byte_idx]
           v0 = decode_fp4(byte & 0xF) * scale  // low nibble
           v1 = decode_fp4(byte >> 4) * scale   // high nibble
           dot += q[d] * v0 + q[d+1] * v1
   ```
4. **Shuffle reduction** — R-thread groups reduce partial dots via `__shfl_xor_sync`
5. **Online softmax** — Thread 0 computes tile softmax in log-base-2 domain (overlapped with V load via cp.async)
6. **P@V accumulation** — Each thread (one per output dimension) accumulates weighted V values with on-the-fly FP4 dequant

### Phase 2: Split Reduction

Standard log-sum-exp merge across splits (identical to BF16/FP8 kernel). v_tensor_scale applied at normalization.

### Key Design Decisions

1. **Separate SMEM regions for data and scales** (not interleaved): Enables 16B-aligned cp.async loads for both. Scale region is exactly 16 bytes per row for HD=256, fitting one cp.async chunk per row.

2. **Pre-decoded float scales in SMEM**: Avoids calling decode_e4m3fn (branch + ldexpf) in the hot QK/PV loops. 2048 scale decodes across 256 threads = 8 per thread. The 8KB extra SMEM is well within the ~28KB total (SM120 supports up to 228KB L1/SMEM).

3. **Register-only FP4 decode** (no `__constant__` LUT): decode_fp4 uses register arithmetic (branches + multiplies). `__constant__` memory serializes divergent warp access (up to 32 serial reads per warp), which is catastrophic for inner-loop LUT lookups where threads access different entries. See `feedback_constant_mem_lut.md`.

4. **`#pragma unroll` on all inner loops**: Ensures accumulators stay in registers, not stack. Runtime-indexed arrays force stack allocation on GPUs because registers aren't addressable by runtime index. See `feedback_register_scalarization.md`.

## Quantization: Optimal Scale Search

The quantization function doesn't just pick the nearest E4M3FN value to `max_abs / 6.0`. It performs an **optimal scale search**:

For each block of 16 values:
1. Compute initial estimate: `target_scale = max_abs / 6.0`
2. Find the 7 nearest E4M3FN values to this estimate
3. For each candidate scale:
   a. Decode the E4M3FN byte to float (the actual scale that will be used)
   b. Quantize all 16 values: `normalized[i] = value[i] / decoded_scale`
   c. Round each to nearest FP4 magnitude
   d. Dequantize back: `roundtrip[i] = nearest_fp4[i] * decoded_scale`
   e. Compute MSE: `sum((value[i] - roundtrip[i])^2) / 16`
4. Select the candidate with minimum MSE

This reduces quantization error by finding the E4M3FN scale that, after the rounding introduced by E4M3FN encoding, produces the lowest total block error. The naive "nearest E4M3FN to target" approach can land on a scale that's slightly too large or small, wasting FP4 range.

## Theoretical Quantization Error Analysis

### FP4 Quantization Noise Model

For a single FP4 element with magnitude range [0, 6], the quantization step sizes are non-uniform:

| Range | Nearest FP4 values | Max step | Max error |
|-------|-------------------|----------|-----------|
| [0, 0.25] | 0, 0.5 | 0.5 | 0.25 |
| [0.25, 0.75] | 0.5, 1.0 | 0.5 | 0.25 |
| [0.75, 1.25] | 1.0, 1.5 | 0.5 | 0.25 |
| [1.25, 1.75] | 1.5, 2.0 | 0.5 | 0.25 |
| [1.75, 2.5] | 2.0, 3.0 | 1.0 | 0.50 |
| [2.5, 3.5] | 3.0, 4.0 | 1.0 | 0.50 |
| [3.5, 5.0] | 4.0, 6.0 | 2.0 | 1.00 |
| [5.0, 6.0] | 6.0 | 1.0 | 1.00 |

For uniform random data in [-6, 6] (post block-scale normalization), the expected quantization error variance:

```
E[error^2] = integral of (x - round_fp4(x))^2 * pdf(x) dx
```

For each range with step size s, the error is approximately uniformly distributed in [-s/2, s/2], giving variance s^2/12. Weighted by the fraction of data in each range:

```
E[error^2] ≈ (4/12)*0.021 + (4/12)*0.083 + (4/12)*0.333
           ≈ 0.007 + 0.028 + 0.111 = 0.146
Signal variance = E[x^2] for uniform [-6,6] = 12
NMSE = 0.146/12 ≈ 0.012
NRMSE = sqrt(0.012) ≈ 0.110 = 11.0%
```

### Attention Output Error (K + V both quantized)

The attention computation `softmax(Q @ K^T / sqrt(d)) @ V` has two error sources:

1. **K quantization** → perturbed attention weights via softmax
2. **V quantization** → perturbed output values

For small perturbations, the errors combine approximately in quadrature:

```
total_NRMSE ≈ sqrt(NRMSE_K_effect^2 + NRMSE_V^2)
```

Since attention weights are a softmax of dot products (not linear in K), the K error is amplified by the softmax temperature. Our measured total NRMSE of 8-9% (vs theoretical single-element 11%) is consistent with the softmax concentrating mass on a few positions, effectively reducing the K error contribution.

### Error Degradation at Extreme Magnitudes

When input data scale increases, the intra-block variance (measured as coefficient of variation within each 16-element block) increases proportionally for Gaussian data, but the FP4 resolution stays fixed at 8 magnitude levels. This causes:

- **Scale 0.1 (typical):** Block max_abs ~ 0.47, scale ~ 0.08, smallest step = 0.04. Most values representable with ~10% relative error.
- **Scale 100 (extreme):** Block max_abs ~ 468, scale ~ 78, smallest step = 39. Values below ~20 quantize to 0 (100% error for those elements).

This is a fundamental limitation of per-block quantization with 4 bits, not a kernel bug. The kernel produces mathematically exact results (kernel_L2 = 0.000000) at all magnitudes.

## Test Results

### Kernel Correctness (error decomposition methodology)

The test separates kernel error from quantization error:
1. Quantize K, V to NVFP4
2. Run FP4 kernel → `output_fp4`
3. Dequantize K, V back to BF16
4. Run BF16 kernel on dequantized data → `output_dequant`
5. Run BF16 kernel on original data → `output_ref`
6. **Kernel error** = `||output_fp4 - output_dequant|| / ||output_dequant||` (should be ~0)
7. **Quantization error** = `||output_dequant - output_ref|| / ||output_ref||`
8. **Total error** = `||output_fp4 - output_ref|| / ||output_ref||`

Results (HEAD_DIM=256, GQA 8:1, block_size=16):

| seq_len | Kernel L2 | Kernel Cosine | Quant L2 | Total L2 | Total Cosine |
|---------|-----------|---------------|----------|----------|--------------|
| 1       | 0.000000  | 1.000000      | 0.0841   | 0.0841   | 0.9965       |
| 64      | 0.000000  | 1.000000      | 0.0863   | 0.0863   | 0.9963       |
| 512     | 0.000000  | 1.000000      | 0.0914   | 0.0914   | 0.9958       |
| 4,096   | 0.000000  | 1.000000      | 0.0848   | 0.0848   | 0.9964       |
| 32,768  | 0.000000  | 1.000000      | 0.0814   | 0.0814   | 0.9967       |

**Kernel L2 = 0.000000 and Kernel Cosine = 1.000000 at all sequence lengths.** The FP4 kernel is mathematically identical to the BF16 kernel operating on dequantized data.

### Extreme Value Tests (kernel correctness at all magnitudes)

| Scale | kv_max | Kernel L2 | Kernel Cos | Total L2 | Total Cos |
|-------|--------|-----------|------------|----------|-----------|
| 0.001 | 0.005  | 0.000000  | 1.000000   | 0.2825   | 0.9602    |
| 0.01  | 0.047  | 0.000000  | 1.000000   | 0.1012   | 0.9949    |
| 0.1   | 0.47   | 0.000000  | 1.000000   | 0.0851   | 0.9964    |
| 1.0   | 4.69   | 0.000000  | 1.000000   | 0.0825   | 0.9966    |
| 10.0  | 47.0   | 0.000000  | 1.000000   | 0.1232   | 0.9925    |
| 100.0 | 468    | 0.000000  | 1.000000   | 0.4356   | 0.9072    |
| 400.0 | 1,872  | 0.000000  | 1.000000   | 0.8963   | 0.5839    |

**Kernel error = 0.000000 at ALL magnitudes**, including values 4x beyond E4M3FN max (448). Total error degradation at extreme scales is entirely from quantization granularity (8 magnitude levels in FP4), not from the kernel.

### Performance (RTX PRO 6000 Blackwell, single-seq decode)

| seq_len | BF16 (us) | FP4 (us) | Speedup | Cache Savings |
|---------|-----------|----------|---------|---------------|
| 64      | 10.1      | 14.4     | 0.70x   | 3.56x         |
| 512     | 12.7      | 15.8     | 0.80x   | 3.56x         |
| 4,096   | 42.8      | 54.4     | 0.79x   | 3.56x         |

FP4 is 20-30% slower than BF16 due to dequantization ALU overhead (decode_fp4 + scale multiply in inner loops). However, the 3.56x cache memory savings enables proportionally longer context or higher batch sizes, which is the primary goal.

## Provenance

### FP4 Dequantization Functions

`decode_fp4()`, `decode_e4m3fn()`, and `read_nvfp4()` originate from VerdictMoE (`verdict_moe_ext.cu`), where they were battle-tested at 165.1 tok/s with Qwen3.5-397B in production. The E4M3FN decode function went through three phases of correctness fixes:

1. **Phase 1:** Block scale swizzle — CUTLASS-specific swizzle applied unconditionally to block scales, causing corruption for non-CUTLASS backends.
2. **Phase 2:** N_half dimension error — `n // 2` instead of `n`, causing only 50% of rows to be processed.
3. **Phase 3:** Integer overflow at e>=14 — `1 << (e+17)` overflows int32 for exponents 14-15, affecting 3-11% of block scales in Qwen3.5-397B. Fixed with `ldexpf()`.

Full history in `PHASE2_RESULTS.md`.

## Files

| File | Description |
|------|-------------|
| `csrc/sm120_flash_decode_v2_paged_fp4.cu` | NVFP4 decode kernel (CUDA) |
| `csrc/sm120_flash_decode_v2_paged.cu` | BF16/FP8 decode kernel (reference) |
| `sm120_flash_decode_ext.py` | Python extension + NVFP4 quantizer |
| `sm120_vllm_backend.py` | vLLM backend with FP4 routing |
| `test_fp4_kv_decode.py` | Correctness + performance tests |

## Usage

```bash
# Run full test suite
python test_fp4_kv_decode.py --verbose --seq-lens 1,64,512,4096

# With performance benchmarks
python test_fp4_kv_decode.py --verbose --perf --seq-lens 64,512,4096,32768
```

## Hardware

- 4x NVIDIA RTX PRO 6000 Blackwell (96GB each, SM 12.0)
- Driver 595.45.04, CUDA 13.2
- PyTorch 2.11.0+cu130
