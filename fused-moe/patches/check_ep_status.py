#!/usr/bin/env python3
"""
Diagnostic: Check EP configuration status in running vLLM.
Run with: docker exec vllm-ep python3 /patches/check_ep_status.py
"""
import sys
sys.path.insert(0, '/opt/venv/lib/python3.12/site-packages')

try:
    # Check if the patch was applied
    from vllm.model_executor.layers.fused_moe.cutlass_moe import CutlassExpertsFp4

    print("=== CutlassExpertsFp4 EP Status ===")

    # Check supports_expert_map
    inst = CutlassExpertsFp4.__new__(CutlassExpertsFp4)
    print(f"  supports_expert_map(): {inst.supports_expert_map()}")

    # Check _supports_parallel_config
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
    test_config = FusedMoEParallelConfig(
        tp_size=1, tp_rank=0,
        pcp_size=1, pcp_rank=0,
        dp_size=1, dp_rank=0,
        ep_size=4, ep_rank=0,
        sp_size=1,
        use_ep=True,
    )
    print(f"  _supports_parallel_config(ep_size=4): {CutlassExpertsFp4._supports_parallel_config(test_config)}")

    # Check apply method source for expert_map handling
    import inspect
    src = inspect.getsource(CutlassExpertsFp4.apply)
    has_remap = 'expert_map[topk_ids' in src or 'non_local_mask' in src
    print(f"  apply() has expert_map remapping: {has_remap}")

    print("\n=== All2All Manager ===")
    try:
        from vllm.distributed.device_communicators.all2all import NaiveAll2AllManager
        src = inspect.getsource(NaiveAll2AllManager.combine)
        has_rs = 'reduce_scatter' in src
        print(f"  NaiveAll2AllManager.combine() uses reduce_scatter: {has_rs}")
    except Exception as e:
        print(f"  Error checking all2all: {e}")

    print("\n✓ EP patch verification complete")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
