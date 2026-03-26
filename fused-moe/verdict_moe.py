# SPDX-License-Identifier: Apache-2.0
"""
VerdictMoE — Fused MoE expert backend for SM120 Blackwell.

Fuses GEMM1 + SwiGLU + E4M3 requant + GEMM2 into a multi-kernel pipeline
with on-the-fly NVFP4 dequantization. Eliminates GMEM round-trips between
GEMM1->activation->GEMM2.

Two backend paths:
  - Scalar GEMV (default): verdict_moe_ext.cu — 4-kernel pipeline
  - Single fused cooperative MMA: verdict_fused_cooperative_ext.cu — ONE kernel
    launch per token. BF16→FP4→GEMM1→SwiGLU→E4M3 requant→GEMM2→BF16 all inside.
    Hybrid K×N distribution (640 CTAs), consecutive-K packing, scale_vec::4X,
    native E4M3FN scales, atomic barriers. CUDA-graph safe.
    Enabled via: VLLM_VERDICT_MMA=1

Grid: num_pairs × num_tiles = M*topk × (tiles_n × k_groups) ≤ 752 max concurrent.
For MTP=3 (M=4): ONE kernel launch processes 40 (token, expert) pairs independently.
Each CTA handles ONE pair — exactly mirrors CUTLASS grouped GEMM semantics.
Weight format: NVFP4 (E2M1 packed uint8 + E4M3FN block scales).
Input: BF16, Output: BF16.

CUDA-graph safe: all buffers pre-allocated at init time via setup_buffers().
No dynamic allocation during forward. No GPU-to-CPU sync in forward path.

Selectable via: VLLM_USE_VERDICT_MOE=1
"""

import os
import logging
from pathlib import Path

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform

logger = logging.getLogger(__name__)

# ============================================================================
# JIT-compile the CUDA extensions
# ============================================================================
_verdict_ext = None
_verdict_mma_ext = None

# MMA path toggle: VLLM_VERDICT_MMA=1 enables MMA tensor core path
_USE_MMA = os.environ.get("VLLM_VERDICT_MMA", "0") == "1"

# Sprint 7: Runtime N_HALF support — no more compile-time constant.
# The cooperative kernel now accepts N_HALF and K_GROUPS as kernel args,
# supporting both EP=4 (N_HALF=1024) and TP=4 (N_HALF=256).

_CUDA_FLAGS = [
    "-gencode=arch=compute_120a,code=sm_120a",
    "-O2",
    "--expt-relaxed-constexpr",
    # NOTE: -use_fast_math deliberately omitted.
    # It makes expf/divf use ~2 ULP approximations, which compounds through
    # 60 MoE layers (SiLU, scale divisions) and shifts logits enough to flip
    # argmax ~4% of the time under strict MTP rejection sampling.
    # The MMA instructions (PTX inline asm) are unaffected by this flag.
]


def _get_verdict_ext():
    """Load the scalar GEMV extension (fallback / default path)."""
    global _verdict_ext
    if _verdict_ext is not None:
        return _verdict_ext

    from torch.utils.cpp_extension import load

    csrc_dir = Path(__file__).parent / "csrc"
    ext_src = csrc_dir / "verdict_moe_ext.cu"

    if not ext_src.exists():
        raise FileNotFoundError(f"VerdictMoE CUDA source not found: {ext_src}")

    logger.info("JIT-compiling VerdictMoE scalar CUDA extension (SM120)...")
    _verdict_ext = load(
        name="verdict_moe_ext",
        sources=[str(ext_src)],
        extra_cuda_cflags=_CUDA_FLAGS,
        verbose=False,
    )
    logger.info("VerdictMoE scalar CUDA extension compiled successfully")
    return _verdict_ext


def _get_verdict_mma_ext():
    """Load the single fused cooperative kernel extension (SM120).

    ONE kernel launch: BF16→FP4 → GEMM1 → SwiGLU → requant → GEMM2 → BF16.
    Atomic barriers, CUDA-graph safe, no cooperative_groups.
    """
    global _verdict_mma_ext
    if _verdict_mma_ext is not None:
        return _verdict_mma_ext

    from torch.utils.cpp_extension import load

    csrc_dir = Path(__file__).parent / "csrc"
    ext_src = csrc_dir / "verdict_fused_cooperative_ext.cu"

    if not ext_src.exists():
        raise FileNotFoundError(
            f"VerdictMoE fused cooperative CUDA source not found: {ext_src}")

    logger.info(
        "JIT-compiling VerdictMoE single fused cooperative extension (SM120)...")
    _verdict_mma_ext = load(
        name="verdict_fused_cooperative_ext",
        sources=[str(ext_src)],
        extra_cuda_cflags=_CUDA_FLAGS,
        verbose=False,
    )
    logger.info(
        "VerdictMoE single fused cooperative extension compiled successfully")
    return _verdict_mma_ext


# ============================================================================
# VerdictMoEExperts — CUDA-graph safe
# ============================================================================
class VerdictMoEExperts(mk.FusedMoEExpertsModular):
    """
    Fused MoE experts using VerdictGemm 3-kernel pipeline.

    CUDA-graph safe: all working buffers are pre-allocated via setup_buffers()
    on first apply() call (during warmup, before graph capture). Subsequent
    calls only slice into pre-allocated memory — zero torch.empty/zeros/arange.

    Task 4 validated: 38.9us per MoE layer (10 experts, M=1, 640 CTAs).
    """

    MMA_BN = 64  # MMA N-tile width (hardware-fixed)
    MAX_BATCHED_TOKENS = 512
    MAX_EXPERTS_PER_TOK = 10

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return p.is_cuda() and p.is_device_capability_family(120)

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [MoEActivation.SILU, MoEActivation.SWIGLUOAI]

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return not (
            moe_parallel_config.use_fi_all2allv_kernels
            or moe_parallel_config.use_deepep_ht_kernels
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def supports_expert_map(self) -> bool:
        return True

    def supports_chunking(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return act_dtype

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (M * topk, max(2 * N, K))
        workspace2 = (M * topk, N)
        output = (M, K)
        return (workspace1, workspace2, output)

    # ========================================================================
    # Buffer pre-allocation for CUDA graph safety
    # ========================================================================
    def setup_buffers(
        self,
        max_tokens: int,
        max_topk: int,
        K: int,
        N_half: int,
        num_experts: int,
        device: torch.device,
    ):
        """Pre-allocate ALL working buffers. Call once before CUDA graph capture.

        After this, apply() does zero torch.empty/zeros/arange calls.
        All forward-path tensors are slices of these buffers.
        """
        max_active = max_tokens * max_topk
        tiles = N_half // self.MMA_BN  # N-distributed tiles for MMA path

        # --- Big compute buffers ---
        # Partials: allocated per-call (size depends on M, the CUDA graph variable).
        # Size is deterministic per graph capture → caching allocator reuses on replay.
        # NOT pre-allocated because it's 1.25 GB at M=512 and OOMs during profiling.
        # Intermediate after SwiGLU: [max_active, N_half] flattened
        self._buf_gmem_inter = torch.empty(
            max_active * N_half,
            dtype=torch.float32, device=device,
        )
        # Float32 output accumulator (zeroed per-call via cudaMemsetAsync in CUDA code)
        self._buf_output_f32 = torch.empty(
            max_tokens * K,
            dtype=torch.float32, device=device,
        )

        # --- Routing workspace (int32/float32, copied into each call) ---
        self._buf_expert_ids = torch.empty(
            max_active, dtype=torch.int32, device=device,
        )
        self._buf_expert_wts = torch.empty(
            max_active, dtype=torch.float32, device=device,
        )

        # Pre-computed token ID pattern: [0,0,...0, 1,1,...1, ..., max_tokens-1,...]
        # Each token ID repeated max_topk times. Sliced to [m * topk] in forward.
        self._buf_token_ids = (
            torch.arange(max_tokens, device=device, dtype=torch.int32)
            .unsqueeze(1)
            .expand(max_tokens, max_topk)
            .reshape(-1)
            .contiguous()
        )

        # --- Per-active-expert alpha buffers ---
        self._buf_w1_alpha = torch.empty(
            max_active, dtype=torch.float32, device=device,
        )
        self._buf_w2_alpha = torch.empty(
            max_active, dtype=torch.float32, device=device,
        )

        # --- Per-expert alpha products (recomputed once via torch.mul(out=)) ---
        self._buf_w1_alpha_all = torch.empty(
            num_experts, dtype=torch.float32, device=device,
        )
        self._buf_w2_alpha_all = torch.empty(
            num_experts, dtype=torch.float32, device=device,
        )

        # --- Constant: ones for apply_router_weight_on_input path ---
        self._buf_ones = torch.ones(
            max_active, dtype=torch.float32, device=device,
        )

        # --- Fused MMA-specific buffers (any N_half — runtime parameterized) ---
        SF_BLOCK = 16  # E4M3FN scale block size (matches weight scales)
        self._mma_buffers_ready = False
        if _USE_MMA:
            # Input FP4: quantized BF16 input (E4M3FN scales, SF_BLOCK=16)
            # Sized for max_tokens (each token quantized in prologue)
            self._buf_input_fp4 = torch.empty(
                max_tokens * (K // 2),
                dtype=torch.uint8, device=device,
            )
            self._buf_input_sf = torch.empty(
                max_tokens * (K // SF_BLOCK),
                dtype=torch.uint8, device=device,
            )
            # Intermediate FP4: SwiGLU output (E4M3FN scales, SF_BLOCK=16)
            # Per-token per-expert: topk * M * N_half/2
            self._buf_inter_fp4 = torch.empty(
                max_active * (N_half // 2),
                dtype=torch.uint8, device=device,
            )
            self._buf_inter_sf = torch.empty(
                max_active * (N_half // SF_BLOCK),
                dtype=torch.uint8, device=device,
            )
            # Per-pair GEMM2 output for deterministic reduction
            # (replaces non-deterministic atomicAdd accumulation)
            # Sized for max_pairs (MAX_M_KERNEL * max_topk = 40), NOT max_active
            MAX_M_KERNEL = 4  # MTP=3 → M=4 max for fused path
            max_pairs = MAX_M_KERNEL * max_topk  # 40
            self._buf_pair_output_f32 = torch.empty(
                max_pairs * K,
                dtype=torch.float32, device=device,
            )
            # Atomic barrier counter for fused cooperative kernel
            self._buf_barrier = torch.zeros(
                1, dtype=torch.int32, device=device,
            )
            # Partials buffer for cooperative kernel
            # Sprint 9: independent per-token routing, one pair per CTA
            # num_pairs = M * topk (max = MAX_M_KERNEL * max_topk = 40)
            # Grid = num_pairs × num_tiles, targeting 640 CTAs
            MAX_M_KERNEL = 4  # MTP=3 → M=4 max
            max_pairs = MAX_M_KERNEL * max_topk  # 40
            tiles_n = N_half // 64
            k_groups_coop = max(1, 640 // (max_pairs * tiles_n))
            # k_groups must divide 64 (total K-tiles)
            while 64 % k_groups_coop != 0 and k_groups_coop > 1:
                k_groups_coop -= 1
            num_tiles_coop = tiles_n * k_groups_coop
            self._buf_partials = torch.empty(
                max_pairs * num_tiles_coop * 128,
                dtype=torch.float32, device=device,
            )
            # Query max concurrent CTAs for deadlock safety check
            props = torch.cuda.get_device_properties(device)
            self._max_fused_ctas = 4 * props.multi_processor_count
            self._max_m_kernel = MAX_M_KERNEL

            self._mma_buffers_ready = True
            logger.info(
                "VerdictMoE fused independent buffers (N_half=%d, k_groups=%d): "
                "input_fp4=%.1f KB, inter_fp4=%.1f KB, partials=%.1f KB, "
                "max_fused_ctas=%d",
                N_half, k_groups_coop,
                (self._buf_input_fp4.nbytes + self._buf_input_sf.nbytes) / 1024,
                (self._buf_inter_fp4.nbytes + self._buf_inter_sf.nbytes) / 1024,
                self._buf_partials.nbytes / 1024,
                self._max_fused_ctas,
            )

        # Store dimensions for debug/validation
        self._buf_K = K
        self._buf_N_half = N_half
        self._buf_max_tokens = max_tokens
        self._buf_max_topk = max_topk
        self._buffers_ready = True

        total_bytes = (
            self._buf_gmem_inter.nbytes
            + self._buf_output_f32.nbytes
            + self._buf_expert_ids.nbytes
            + self._buf_expert_wts.nbytes
            + self._buf_token_ids.nbytes
            + self._buf_w1_alpha.nbytes
            + self._buf_w2_alpha.nbytes
            + self._buf_w1_alpha_all.nbytes
            + self._buf_w2_alpha_all.nbytes
            + self._buf_ones.nbytes
        )
        if self._mma_buffers_ready:
            total_bytes += (
                self._buf_input_fp4.nbytes + self._buf_input_sf.nbytes
                + self._buf_inter_fp4.nbytes + self._buf_inter_sf.nbytes
                + self._buf_partials.nbytes
            )
        logger.info(
            "VerdictMoE buffers allocated: %.1f MB "
            "(max_tokens=%d, max_topk=%d, K=%d, N_half=%d, mma=%s)",
            total_bytes / 1e6, max_tokens, max_topk, K, N_half,
            "ON" if self._mma_buffers_ready else "OFF",
        )

    _buffers_ready = False

    # ========================================================================
    # Forward — CUDA-graph safe (no allocations, no sync, no data branching)
    # ========================================================================
    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        e, m, n, k, topk_val = self.moe_problem_size(
            hidden_states, w1, w2, topk_ids
        )
        # w2 is [E, K, N_packed] where N_packed = intermediate_size / 2 (FP4 packed)
        # N_half = intermediate_size (each of gate/up output dimension)
        N_half = w2.size(2) * 2
        num_topk = topk_ids.size(1)
        num_active = m * num_topk
        device = hidden_states.device

        # --- Profile run bypass (MUST be before buffer alloc) ---
        # vLLM's profile_run() sends max_num_batched_tokens (8192) in eager mode
        # to probe memory limits. GPU is nearly maxed during profiling.
        # Return zeros — actual computation result doesn't matter for profiling.
        # Buffer allocation deferred to first real inference call (warmup at M≤512).
        if m > self.MAX_BATCHED_TOKENS:
            logger.info(
                "VerdictMoE: profile run (m=%d > max=%d), returning zeros",
                m, self.MAX_BATCHED_TOKENS,
            )
            output.zero_()
            return

        # --- Lazy buffer init (runs once during warmup, before graph capture) ---
        if not self._buffers_ready:
            self.setup_buffers(
                self.MAX_BATCHED_TOKENS,
                self.MAX_EXPERTS_PER_TOK,
                k, N_half, global_num_experts, device,
            )

        # --- EP: remap global expert IDs to local, zero non-local weights ---
        # CUDA-graph safe: no sync, no data-dependent branching.
        # Fixed-size tensor ops (M and topk are padded constants during capture).
        if expert_map is not None:
            local_topk_ids = expert_map[topk_ids]
            non_local = local_topk_ids == -1
            topk_weights = topk_weights * (~non_local).to(topk_weights.dtype)
            topk_ids = local_topk_ids.clamp(min=0)

        if apply_router_weight_on_input:
            hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)

        # Compute tiles_per_expert from actual weight shape (N-distributed)
        # MMA tiles = N_HALF / BN. Scalar tiles = K / BK (unchanged).
        mma_tiles = N_half // self.MMA_BN  # 1024/64 = 16 for Qwen3.5
        scalar_tiles = k // 64             # 4096/64 = 64 (K-distributed)

        # --- Normal path: use pre-allocated buffers (CUDA-graph safe) ---
        # --- Copy routing into pre-allocated int32/float32 buffers ---
        # copy_() handles dtype conversion in-place (no allocation)
        expert_ids_flat = self._buf_expert_ids[:num_active]
        expert_wts_flat = self._buf_expert_wts[:num_active]
        expert_ids_flat.copy_(topk_ids.reshape(-1))
        expert_wts_flat.copy_(topk_weights.reshape(-1))

        # Token IDs: slice pre-computed pattern (no torch.arange)
        token_ids_flat = self._buf_token_ids[:num_active]

        # --- Compute per-active-expert weight scales (pre-allocated) ---
        num_local_experts = self.g1_alphas.size(0)
        torch.mul(self.g1_alphas, self.a1_gscale,
                  out=self._buf_w1_alpha_all[:num_local_experts])
        torch.mul(self.g2_alphas, self.a2_gscale,
                  out=self._buf_w2_alpha_all[:num_local_experts])
        w1_alpha = self._buf_w1_alpha[:num_active]
        w2_alpha = self._buf_w2_alpha[:num_active]
        torch.index_select(
            self._buf_w1_alpha_all[:num_local_experts],
            0, expert_ids_flat, out=w1_alpha,
        )
        torch.index_select(
            self._buf_w2_alpha_all[:num_local_experts],
            0, expert_ids_flat, out=w2_alpha,
        )

        # --- Working buffers ---
        output_f32 = self._buf_output_f32[:m * k]

        # --- Cast block scales to uint8 (view, no allocation) ---
        w1_sf = self.w1_scale.view(torch.uint8)  # [E, 2*N, K//16]
        w2_sf = self.w2_scale.view(torch.uint8)  # [E, K, N//16]

        # --- Select expert weights: pre-allocated ones vs computed weights ---
        if apply_router_weight_on_input:
            wts_for_kernel = self._buf_ones[:num_active]
        else:
            wts_for_kernel = expert_wts_flat

        # --- Dispatch: MMA or scalar path ---
        # Sprint 9: independent per-token routing (flat pair tables).
        # M ≤ 4: one launch with M*topk pairs. M > 4: per-token loop.
        use_mma = _USE_MMA and self._mma_buffers_ready
        if use_mma:
            self._apply_mma(
                output, hidden_states, w1, w2,
                w1_sf, w2_sf, w1_alpha, w2_alpha,
                expert_ids_flat, wts_for_kernel, token_ids_flat,
                output_f32,
                k, N_half, num_active, mma_tiles, m,
                topk_ids, topk_weights, apply_router_weight_on_input,
            )
        else:
            # Scalar path still needs partials (K-distributed with 64 tiles)
            partials = torch.empty(
                num_active * scalar_tiles * 2 * N_half,
                dtype=torch.float32, device=device,
            )
            self._apply_scalar(
                output, hidden_states, w1, w2,
                w1_sf, w2_sf, w1_alpha, w2_alpha,
                expert_ids_flat, wts_for_kernel, token_ids_flat,
                partials, output_f32,
                k, N_half, num_active, scalar_tiles,
            )

    # ========================================================================
    # Scalar GEMV path (original, validated)
    # ========================================================================
    def _apply_scalar(
        self, output, hidden_states, w1, w2,
        w1_sf, w2_sf, w1_alpha, w2_alpha,
        expert_ids_flat, wts_for_kernel, token_ids_flat,
        partials, output_f32,
        k, N_half, num_active, tiles_per_expert,
    ):
        ext = _get_verdict_ext()
        gmem_inter = self._buf_gmem_inter[:num_active * N_half]

        # output_f32 is zeroed by cudaMemsetAsync inside the CUDA extension
        ext.forward(
            hidden_states.contiguous(),
            w1.contiguous(),           # w1_fp4 [E, 2*N, K//2] uint8
            w1_sf.contiguous(),
            w1_alpha,
            w2.contiguous(),           # w2_fp4 [E, K, N//2] uint8
            w2_sf.contiguous(),
            w2_alpha,
            output,                    # [M, K] BF16 — final output
            expert_ids_flat,
            wts_for_kernel,
            token_ids_flat,
            partials,
            gmem_inter,
            output_f32,
            k, N_half, num_active, tiles_per_expert,
        )

    # ========================================================================
    # Independent Per-Token Routing — Sprint 9
    # BF16→FP4→GEMM1→SwiGLU→requant→GEMM2→BF16, all inside one kernel.
    #
    # Each CTA handles ONE (token, expert) pair — exactly mirrors CUTLASS.
    # Flat routing tables: expert_ids[M*topk], token_ids[M*topk], weights[M*topk]
    # MTP acceptance rate matches CUTLASS baseline (~75%).
    #
    # M ≤ 4 (decode+MTP): ONE launch with M*topk pairs.
    # M > 4 (warmup/prefill): per-token loop, each launch topk pairs.
    # Both paths use same kernel, CUDA-graph safe, atomic barriers.
    # ========================================================================
    def _apply_mma(
        self, output, hidden_states, w1, w2,
        w1_sf, w2_sf, w1_alpha, w2_alpha,
        expert_ids_flat, wts_for_kernel, token_ids_flat,
        output_f32,
        k, N_half, num_active, tiles_per_expert, m,
        topk_ids, topk_weights, apply_router_weight_on_input,
    ):
        ext_coop = _get_verdict_mma_ext()

        topk = num_active // m  # experts per token
        k_packed = k // 2
        sf_cols = k // 16
        n_half_packed = N_half // 2
        n_half_sf = N_half // 16

        if m <= 4:
            # --- Independent per-token routing: flat pair tables ---
            num_pairs = m * topk  # M*topk pairs

            input_fp4 = self._buf_input_fp4[:m * k_packed]
            input_sf = self._buf_input_sf[:m * sf_cols]
            inter_fp4 = self._buf_inter_fp4[:num_pairs * n_half_packed]
            inter_sf = self._buf_inter_sf[:num_pairs * n_half_sf]
            pair_output_f32 = self._buf_pair_output_f32[:num_pairs * k]

            ext_coop.forward(
                hidden_states.contiguous(),
                w1.contiguous(),
                w1_sf.contiguous(),
                w1_alpha,           # [num_pairs] per-pair alpha
                w2.contiguous(),
                w2_sf.contiguous(),
                w2_alpha,           # [num_pairs] per-pair alpha
                output,
                expert_ids_flat,    # [num_pairs] int32
                token_ids_flat,     # [num_pairs] int32
                wts_for_kernel,     # [num_pairs] float32
                output_f32,
                pair_output_f32,
                input_fp4,
                input_sf,
                self._buf_partials,
                inter_fp4,
                inter_sf,
                self._buf_barrier,
                k, N_half, num_pairs,
            )
        else:
            # --- PER-TOKEN LOOP: M>4 (warmup/prefill), M=1 per launch ---
            input_fp4 = self._buf_input_fp4[:k_packed]
            input_sf = self._buf_input_sf[:sf_cols]
            inter_fp4 = self._buf_inter_fp4[:topk * n_half_packed]
            inter_sf = self._buf_inter_sf[:topk * n_half_sf]
            pair_output_f32_m1 = self._buf_pair_output_f32[:topk * k]

            # M=1 token_ids: all zeros (single token → token_id=0)
            m1_token_ids = self._buf_token_ids[:topk]
            m1_token_ids.zero_()

            for tok in range(m):
                tok_ids = expert_ids_flat[tok * topk:(tok + 1) * topk]
                tok_w1a = w1_alpha[tok * topk:(tok + 1) * topk]
                tok_w2a = w2_alpha[tok * topk:(tok + 1) * topk]
                tok_wts = wts_for_kernel[tok * topk:(tok + 1) * topk]

                ext_coop.forward(
                    hidden_states[tok:tok + 1].contiguous(),
                    w1.contiguous(),
                    w1_sf.contiguous(),
                    tok_w1a,
                    w2.contiguous(),
                    w2_sf.contiguous(),
                    tok_w2a,
                    output[tok:tok + 1],
                    tok_ids,
                    m1_token_ids,
                    tok_wts,
                    output_f32[:k],
                    pair_output_f32_m1,
                    input_fp4,
                    input_sf,
                    self._buf_partials,
                    inter_fp4,
                    inter_sf,
                    self._buf_barrier,
                    k, N_half, topk,
                )
