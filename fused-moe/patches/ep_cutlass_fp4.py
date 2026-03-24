#!/usr/bin/env python3
"""
Patch CutlassExpertsFp4 to support expert_map for EP mode.

Changes:
1. _supports_parallel_config: remove ep_size == 1 restriction
2. supports_expert_map: return True
3. apply(): use moe_permute when expert_map is provided
4. run_cutlass_moe_fp4(): add expert_map parameter, remap topk_ids
"""
import sys

TARGET = "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/cutlass_moe.py"

with open(TARGET, 'r') as f:
    content = f.read()

# Patch 1: _supports_parallel_config — allow EP
old1 = """    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # CutlassExpertsFp4 does not support expert map, which is
        # needed for STANDARD activation format kernels in EP mode.
        return moe_parallel_config.ep_size == 1"""

new1 = """    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # EP mode supported via expert_map remapping in apply()
        # Only reject unsupported all2all backends
        return not (
            moe_parallel_config.use_fi_all2allv_kernels
            or moe_parallel_config.use_deepep_ht_kernels
        )"""

# Patch 2: supports_expert_map — enable
old2 = """    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()"""

new2 = """    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()"""

# Patch 3: apply() — handle expert_map
old3 = """    def apply(
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
        a1q_scale: torch.Tensor | None,  # unused
        a2_scale: torch.Tensor | None,  # unused
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        e, m, n, k, _ = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        n = w2.shape[2] * 2

        run_cutlass_moe_fp4(
            output=output,
            a=hidden_states,
            a1_gscale=self.a1_gscale,
            w1_fp4=w1,
            w1_blockscale=self.w1_scale,
            w1_alphas=self.g1_alphas,
            a2_gscale=self.a2_gscale,
            w2_fp4=w2,
            w2_blockscale=self.w2_scale,
            w2_alphas=self.g2_alphas,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            workspace13=workspace13,
            workspace2=workspace2,
            m=m,
            n=n,
            k=k,
            e=e,
            device=hidden_states.device,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )"""

new3 = """    def apply(
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
        a1q_scale: torch.Tensor | None,  # unused
        a2_scale: torch.Tensor | None,  # unused
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        e, m, n, k, _ = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        n = w2.shape[2] * 2
        # For EP: e = local_num_experts, w1/w2 already sliced by vLLM
        # expert_map remaps global topk_ids to local indices

        run_cutlass_moe_fp4(
            output=output,
            a=hidden_states,
            a1_gscale=self.a1_gscale,
            w1_fp4=w1,
            w1_blockscale=self.w1_scale,
            w1_alphas=self.g1_alphas,
            a2_gscale=self.a2_gscale,
            w2_fp4=w2,
            w2_blockscale=self.w2_scale,
            w2_alphas=self.g2_alphas,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            workspace13=workspace13,
            workspace2=workspace2,
            m=m,
            n=n,
            k=k,
            e=e,
            device=hidden_states.device,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            global_num_experts=global_num_experts,
        )"""

# Patch 4: run_cutlass_moe_fp4 — add expert_map parameter
old4 = """def run_cutlass_moe_fp4(
    output: torch.Tensor,
    a: torch.Tensor,
    a1_gscale: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    a2_gscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    m: int,
    n: int,
    k: int,
    e: int,
    device: torch.device,
    apply_router_weight_on_input: bool = False,
) -> None:"""

new4 = """def run_cutlass_moe_fp4(
    output: torch.Tensor,
    a: torch.Tensor,
    a1_gscale: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    a2_gscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    m: int,
    n: int,
    k: int,
    e: int,
    device: torch.device,
    apply_router_weight_on_input: bool = False,
    expert_map: torch.Tensor | None = None,
    global_num_experts: int = -1,
) -> None:"""

# Patch 5: Inside run_cutlass_moe_fp4, remap topk_ids when expert_map is provided
# Insert after the assertion block, before expert_offsets allocation
old5 = """    expert_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)"""

new5 = """    # EP support: remap global expert IDs to local IDs
    if expert_map is not None:
        topk_ids = expert_map[topk_ids.long()].to(torch.int32)
        # Tokens routed to non-local experts will have topk_ids == -1
        # get_cutlass_moe_mm_data handles this by assigning them to no expert

    expert_offsets = torch.empty((e + 1), dtype=torch.int32, device=device)"""

# Apply patches
patches = [
    (old1, new1, "_supports_parallel_config"),
    (old2, new2, "supports_expert_map"),
    (old3, new3, "apply()"),
    (old4, new4, "run_cutlass_moe_fp4 signature"),
    (old5, new5, "expert_map remapping"),
]

for old, new, name in patches:
    if old in content:
        content = content.replace(old, new, 1)
        print(f"✓ Patched: {name}")
    else:
        print(f"✗ FAILED: {name} — pattern not found")
        # Try to find approximate match
        key_line = old.split('\n')[0].strip()
        if key_line in content:
            print(f"  (first line found, but full block doesn't match)")

with open(TARGET, 'w') as f:
    f.write(content)

print("\nDone. Restart vLLM to test with --enable-expert-parallel")
