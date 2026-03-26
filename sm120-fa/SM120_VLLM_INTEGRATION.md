# SM120 Flash Attention — vLLM Integration

## Status: Correctness Proven, MMA Upgrade Needed for Performance

### What's Done

1. **Paged KV cache decode kernel** (`csrc/sm120_flash_decode_paged.cu`)
   - Split-KV design with adaptive parallelism (1-32 splits)
   - Supports paged KV cache (block_table lookup) — matches vLLM's layout
   - GQA support (arbitrary q:kv head ratio, tested at 16:1 and 4:1)
   - HEAD_DIM=128 and HEAD_DIM=256 (Qwen3.5's unusual 256)
   - BF16 Q/KV/output
   - Pre-allocated workspace for CUDA graph compatibility
   - Variable sequence lengths per request

2. **PyTorch extension** (`sm120_flash_decode_ext.py`)
   - JIT-compiled torch extension
   - `SM120FlashDecodeWorkspace` for pre-allocated buffers
   - Clean Python API matching vLLM conventions

3. **vLLM backend** (`sm120_vllm_backend.py`)
   - `SM120FlashAttentionBackend` subclasses `FlashAttentionBackend`
   - Intercepts decode (max_query_len=1) → SM120 kernel
   - Falls through to FA for prefill, cascade, DCP, FP8 KV
   - Ready for `register_backend()` integration

### Correctness Results

All configs pass at < 0.025% max error, > 0.9999 cosine similarity:

```
head_dim=256, q_heads=8, kv_heads=2, block_size=16
  batch=1, seq_lens=[64]:     max_err=0.000244, cos_sim=0.999998
  batch=1, seq_lens=[16384]:  max_err=0.000015, cos_sim=0.999997
  batch=8, variable lengths:  max_err=0.000122, cos_sim=0.999998
```

### Performance (Current Scalar Kernel)

| Config | SM120 Paged (us) | FlashInfer Single (us) | Ratio |
|--------|----------------:|----------------------:|------:|
| 1K ctx | 154 | 23 | 6.8x slower |
| 4K ctx | 293 | 23 | 12.8x slower |
| 16K ctx | 569 | 25 | 23x slower |
| 32K ctx | 584 | 34 | 17x slower |

**Root cause**: Scalar dot products (1 element/thread) vs FlashInfer's MMA tensor cores.
FlashInfer single_decode is non-paged (contiguous KV), so the comparison is slightly unfair.

### Key Finding: FlashInfer Gap on SM120

FlashInfer's `BatchDecodeWithPagedKVCache` **crashes** on SM120 for:
- head_dim=256 with any GQA ratio (group_size=4, 8, 16 all fail)
- Error: `Unsupported group_size` in batch_decode.cu

This means our kernel is the **only paged decode option** for Qwen3.5 on SM120.
vLLM works around this by using the varlen/prefill path for decode.

### MMA Upgrade Path (Next Sprint)

To close the 7-23x performance gap:

1. **FlashDecoding with MMA** (recommended):
   - Pad Q from 1→16 rows (MMA_M=16)
   - Use BN=128 KV tile (process 128 KV tokens per MMA iteration)
   - QK^T: MMA (16×HD) × (HD×128) → 16×128, extract row 0
   - PV: MMA (16×128) × (128×HD) → 16×HD, extract row 0
   - Wastes 15/16 compute but MMA is 20x faster → net 1.25x faster
   - Add paged KV addressing to global→SMEM copy

2. **Required changes to v4 kernel**:
   - Replace TMA loads with ldglobal + block_table lookup
   - Set BM=16 (minimum MMA tile)
   - Add split-KV grid launch for parallelism
   - Add online softmax cross-CTA reduction

3. **FP8 KV cache support** (for full vLLM integration):
   - Modify SMEM load to dequantize FP8→BF16
   - Apply per-head quantization scales (k_scale, v_scale)

### Deployment Instructions

```bash
# 1. Copy files to container
docker cp sm120-fa/csrc/sm120_flash_decode_paged.cu vllm-qwen35:/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/sm120_csrc/
docker cp sm120-fa/sm120_flash_decode_ext.py vllm-qwen35:/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/
docker cp sm120-fa/sm120_vllm_backend.py vllm-qwen35:/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/

# 2. Register backend (add to vLLM startup)
# In Python:
from vllm.v1.attention.backends.registry import register_backend, AttentionBackendEnum
register_backend(AttentionBackendEnum.FLASH_ATTN,
                 "vllm.v1.attention.backends.sm120_vllm_backend.SM120FlashAttentionBackend")

# 3. Or use as standalone:
from sm120_flash_decode_ext import sm120_flash_decode_paged
output = sm120_flash_decode_paged(query, key_cache, value_cache, block_table, seq_lens)
```

### Files

| File | Description |
|------|-------------|
| `csrc/sm120_flash_decode_paged.cu` | CUDA kernel (split-KV, paged, GQA) |
| `sm120_flash_decode_ext.py` | PyTorch extension wrapper |
| `sm120_vllm_backend.py` | vLLM backend (subclass FlashAttentionBackend) |
| `tests/test_flash_decode_paged.py` | Correctness + benchmark tests |
| `benchmarks/bench_decode_vs_flashinfer.py` | FlashInfer comparison benchmark |
