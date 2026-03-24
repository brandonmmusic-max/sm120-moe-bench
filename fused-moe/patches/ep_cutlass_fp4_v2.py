#!/usr/bin/env python3
"""
Patch CutlassExpertsFp4 for EP (Expert Parallel) mode on SM120.

Changes:
1. _supports_parallel_config: allow ep_size > 1
2. supports_expert_map: return True
3. apply(): remap topk_ids via expert_map, zero non-local weights

This is the minimal patch — the run_cutlass_moe_fp4 function signature
stays unchanged. All remapping happens in apply().
"""

TARGET = "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/cutlass_moe.py"

with open(TARGET, 'r') as f:
    content = f.read()

# ═════════════════════════════════════════════════════════════════════
# Patch 1: _supports_parallel_config — allow EP
# ═════════════════════════════════════════════════════════════════════
old1 = """    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # CutlassExpertsFp4 does not support expert map, which is
        # needed for STANDARD activation format kernels in EP mode.
        return moe_parallel_config.ep_size == 1"""

new1 = """    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # EP mode supported — expert_map remapping in apply()
        return not (
            moe_parallel_config.use_fi_all2allv_kernels
            or moe_parallel_config.use_deepep_ht_kernels
        )"""

# ═════════════════════════════════════════════════════════════════════
# Patch 2: supports_expert_map — enable
# ═════════════════════════════════════════════════════════════════════
old2 = """    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()"""

new2 = """    def supports_expert_map(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()"""

# ═════════════════════════════════════════════════════════════════════
# Patch 3: apply() — remap expert_map before calling run_cutlass_moe_fp4
# ═════════════════════════════════════════════════════════════════════
# Find the apply method and add expert_map remapping logic
old3 = """        e, m, n, k, _ = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
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

new3 = """        # EP: remap global expert IDs to local IDs via expert_map
        if expert_map is not None:
            local_topk_ids = expert_map[topk_ids.long()]
            # Non-local experts have local_id == -1 -> clamp to 0
            non_local_mask = local_topk_ids < 0
            local_topk_ids = local_topk_ids.clamp(min=0).to(topk_ids.dtype)
            # Zero weights for non-local experts so they don't contribute
            local_topk_weights = topk_weights.clone()
            local_topk_weights[non_local_mask] = 0.0
            topk_ids = local_topk_ids
            topk_weights = local_topk_weights

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

# ═════════════════════════════════════════════════════════════════════
# Apply all patches
# ═════════════════════════════════════════════════════════════════════
patches = [
    (old1, new1, "_supports_parallel_config"),
    (old2, new2, "supports_expert_map"),
    (old3, new3, "apply() expert_map remapping"),
]

ok = True
for old, new, name in patches:
    if old in content:
        content = content.replace(old, new, 1)
        print(f"✓ Patched: {name}")
    else:
        print(f"✗ FAILED: {name}")
        # Debug: show first line
        key_line = old.strip().split('\n')[0].strip()
        if key_line in content:
            print(f"  First line found but full block doesn't match")
        else:
            print(f"  First line NOT found: {key_line[:80]}")
        ok = False

if ok:
    with open(TARGET, 'w') as f:
        f.write(content)
    print("\n✓ All patches applied. CutlassExpertsFp4 now supports EP mode.")
else:
    print("\n✗ Some patches failed — file NOT modified.")
    exit(1)
