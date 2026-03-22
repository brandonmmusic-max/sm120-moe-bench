#!/usr/bin/env python3
"""
VerdictMoE: Single-tile FP4 GEMM using CuteDSL on SM120.

Computes one 128×128×128 tile: C[128,128] = A[128,128] @ B[128,128]^T
with NVF4 block-16 E4M3FN scale factors.

This is the minimal kernel to validate the CuteDSL MMA pipeline
on SM120 before building the full fused MoE kernel.
"""

import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu.warp.mma import Field as WarpField

from test_gemm_cutedsl import nvfp4_quantize, nvfp4_dequantize, reference_gemm


# Constants
SF_VEC = 16
TILE_M, TILE_N, TILE_K = 128, 128, SF_VEC * 8  # 128
NUM_MMA_WARPS = 8
TMA_WARP = NUM_MMA_WARPS  # warp 8
THREADS = (NUM_MMA_WARPS + 1) * 32  # 288


class SingleTileGemm:
    """Single-tile NVFP4 GEMM using CuteDSL."""

    def __init__(self):
        self.sf_vec_size = SF_VEC
        self.tile_shape_mnk = (TILE_M, TILE_N, TILE_K)
        self.a_dtype = cutlass.Float4E2M1FN
        self.b_dtype = cutlass.Float4E2M1FN
        self.sf_dtype = cutlass.Float8E4M3FN
        self.acc_dtype = cutlass.Float32
        self.c_dtype = cutlass.BFloat16

        self.tiled_mma = None  # Created in JIT context

    @cute.jit
    def run(
        self,
        a_fp4: cute.Tensor,     # [M, K/2] uint8 packed FP4
        b_fp4: cute.Tensor,     # [N, K/2] uint8 packed FP4
        sfa: cute.Tensor,       # [M, K/SF_VEC] E4M3FN
        sfb: cute.Tensor,       # [N, K/SF_VEC] E4M3FN
        c: cute.Tensor,         # [M, N] BF16 output
        stream: cuda.CUstream,
    ):
        """Launch single-tile GEMM."""
        # Create MMA op and tiled MMA inside JIT context
        mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(
            self.a_dtype, self.acc_dtype, self.sf_dtype
        )
        atom_layout = cute.make_layout((4, 2, 1))
        permutation_mnk = sm120_utils.get_permutation_mnk(
            self.tile_shape_mnk, self.sf_vec_size, False
        )
        tiled_mma = cute.make_tiled_mma(
            mma_op, atom_layout, permutation_mnk=permutation_mnk
        )
        mma_atom = cute.make_mma_atom(mma_op)

        # SMEM layouts
        a_smem_atom = blockscaled_utils.sm120_smem_layout_atom_A(
            self.a_dtype, self.tile_shape_mnk[2]
        )
        b_smem_atom = blockscaled_utils.sm120_smem_layout_atom_B(
            self.b_dtype, self.tile_shape_mnk[2]
        )
        a_smem_layout = cute.tile_to_shape(
            a_smem_atom, (self.tile_shape_mnk[0], self.tile_shape_mnk[2])
        )
        b_smem_layout = cute.tile_to_shape(
            b_smem_atom, (self.tile_shape_mnk[1], self.tile_shape_mnk[2])
        )

        sfa_smem_layout = blockscaled_utils.sm120_make_smem_layout_sfa(
            tiled_mma, self.tile_shape_mnk, self.sf_vec_size, 1
        )
        sfb_smem_layout = blockscaled_utils.sm120_make_smem_layout_sfb(
            tiled_mma, self.tile_shape_mnk, self.sf_vec_size, 1
        )

        # TMA descriptors
        # ... (need to set up TMA for A, B, SFA, SFB)

        # For now, just print that we got this far
        print(f"SingleTileGemm: tiled_mma created")
        print(f"  a_smem_layout: {a_smem_layout}")
        print(f"  sfa_smem_layout: {sfa_smem_layout}")


def test():
    print("SingleTileGemm: testing JIT compilation...")
    M, N, K = 128, 128, 128

    torch.manual_seed(42)
    a_f32 = torch.randn(M, K, device="cuda") * 0.5
    b_f32 = torch.randn(N, K, device="cuda") * 0.5

    # Quantize
    a_packed, a_sf = nvfp4_quantize(a_f32, sf_vec_size=SF_VEC)
    b_packed, b_sf = nvfp4_quantize(b_f32, sf_vec_size=SF_VEC)

    # Reference
    a_deq = nvfp4_dequantize(a_packed, a_sf, sf_vec_size=SF_VEC)
    b_deq = nvfp4_dequantize(b_packed, b_sf, sf_vec_size=SF_VEC)
    c_ref = reference_gemm(a_deq, b_deq)
    print(f"Quantized ref[0,0:4]: {c_ref[0, :4].tolist()}")

    # Output
    c_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")

    # Try to instantiate
    stg = SingleTileGemm()

    stream_ptr = torch.cuda.current_stream().cuda_stream
    stream = cuda.CUstream(stream_ptr)

    try:
        stg.run(
            cute.from_dlpack(a_packed.contiguous()),
            cute.from_dlpack(b_packed.contiguous()),
            cute.from_dlpack(a_sf.contiguous()),
            cute.from_dlpack(b_sf.contiguous()),
            cute.from_dlpack(c_out),
            stream,
        )
        print("JIT compilation successful!")
    except Exception as e:
        print(f"JIT error: {e}")


if __name__ == "__main__":
    test()
