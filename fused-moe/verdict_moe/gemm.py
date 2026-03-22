"""
VerdictMoE Dense GEMM — SM120-native block-scaled FP4 GEMM using CuteDSL.

This is our own implementation, informed by but not copying b12x's architecture.

Key design decisions specific to our use case:
  - NVF4 format: sf_vec_size=16, E4M3FN scale factors
  - MmaMXF4NVF4Op atom with atom_layout=(4,2,1) → m64 n16 k64 per atom group
  - 8 MMA warps + 1 TMA warp = 288 threads
  - Tile: 128×128×128 (tile_k = sf_vec_size * 8 = 128 for NVF4)
  - Cluster: (1,1,1) — SM120 doesn't support multi-cluster

Sprint 1: Get this producing correct FC1 output.
Sprint 2: Add SplitK for M=1 (the differentiator).
"""

from __future__ import annotations
from typing import Tuple, Optional, Callable

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cuda.bindings.driver as cuda
import torch

from cutlass.cute.nvgpu.warp.mma import Field as WarpField


class VerdictGemm:
    """SM120 block-scaled FP4 GEMM kernel using CuteDSL.

    Computes: C = alpha * (A × SFA) @ (B × SFB)
    Where A, B are FP4 (E2M1), SFA/SFB are E4M3FN block-16 scale factors.
    """

    def __init__(
        self,
        sf_vec_size: int = 16,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        enable_pdl: bool = True,
    ):
        # NVF4: E4M3FN scales, sf_vec=16
        self.sf_vec_size = sf_vec_size
        self.tile_k = sf_vec_size * 8  # 128 for NVF4
        self.tile_shape_mnk = (mma_tiler_mn[0], mma_tiler_mn[1], self.tile_k)
        self.cluster_shape = (1, 1, 1)
        self.enable_pdl = enable_pdl

        # Thread config: 8 MMA warps + 1 TMA warp
        self.num_mma_warps = 8
        self.tma_warp_id = self.num_mma_warps
        self.threads_per_warp = 32
        self.threads_per_cta = (self.num_mma_warps + 1) * self.threads_per_warp

        # Dtypes
        self.a_dtype = cutlass.Float4E2M1FN
        self.b_dtype = cutlass.Float4E2M1FN
        self.sf_dtype = cutlass.Float8E4M3FN  # NVF4 uses E4M3, not UE8M0
        self.acc_dtype = cutlass.Float32
        self.c_dtype = cutlass.BFloat16

        # SMEM capacity
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")

        # Pipeline barriers
        self.mma_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.num_mma_warps * self.threads_per_warp,
        )

        # MMA config (instantiated lazily in JIT context)
        self.atom_shape = (4, 2, 1)
        mma_m, mma_n, mma_k = 16, 8, 64
        self.num_m_tiles = self.tile_shape_mnk[0] // (mma_m * self.atom_shape[0])
        self.num_n_tiles = self.tile_shape_mnk[1] // (mma_n * self.atom_shape[1])
        self.num_k_blocks = self.tile_shape_mnk[2] // mma_k

        # These are created inside @cute.jit context
        self.tiled_mma = None
        self.mma_atom = None

    def get_info(self) -> dict:
        """Return kernel configuration info."""
        return {
            "tile_shape": self.tile_shape_mnk,
            "sf_vec_size": self.sf_vec_size,
            "threads_per_cta": self.threads_per_cta,
            "num_mma_warps": self.num_mma_warps,
            "num_m_tiles": self.num_m_tiles,
            "num_n_tiles": self.num_n_tiles,
            "num_k_blocks": self.num_k_blocks,
            "smem_capacity": self.smem_capacity,
        }


def test_verdict_gemm():
    """Quick test: instantiate VerdictGemm, print config."""
    vg = VerdictGemm(sf_vec_size=16)
    info = vg.get_info()
    print("VerdictGemm Configuration:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print(f"\n  MMA atom: MmaMXF4NVF4Op")
    print(f"  SF dtype: Float8E4M3FN (NVF4)")
    print(f"  A/B dtype: Float4E2M1FN")
    print(f"  Acc dtype: Float32")
    print(f"  Output dtype: BFloat16")


if __name__ == "__main__":
    test_verdict_gemm()
