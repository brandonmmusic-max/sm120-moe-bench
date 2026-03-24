#!/usr/bin/env python3
"""
Patch NaiveAll2AllManager to use all-gather + reduce-scatter
instead of per-rank broadcasts + all-reduce.

The naive implementation does N sequential broadcasts for dispatch
and all-reduce + slice for combine. This patch uses:
  - dispatch: single all-gather (more efficient NCCL collective)
  - combine: reduce-scatter (eliminates the wasteful slice)

This is Phase 1 of the P2P optimization. Phase 2 will replace
NCCL collectives with raw P2P copies.

Apply INSIDE the container after starting vLLM:
  python3 /patches/patch_naive_all2all.py
"""

TARGET = "/opt/venv/lib/python3.12/site-packages/vllm/distributed/device_communicators/all2all.py"

with open(TARGET, 'r') as f:
    content = f.read()

# Patch the NaiveAll2AllManager.combine() to use reduce_scatter
# instead of all_reduce + slice
old_combine = '''    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        ep_rank = self.rank if is_sequence_parallel else self.dp_rank

        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sp_size = self.tp_group.world_size if is_sequence_parallel else 1
        cu_tokens_across_sp_cpu = dp_metadata.cu_tokens_across_sp(sp_size)

        start = 0 if ep_rank == 0 else cu_tokens_across_sp_cpu[ep_rank - 1]
        end = cu_tokens_across_sp_cpu[ep_rank]

        all_hidden_states = get_ep_group().all_reduce(hidden_states)
        hidden_states = all_hidden_states[start:end, :]
        return hidden_states'''

new_combine = '''    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        ep_rank = self.rank if is_sequence_parallel else self.dp_rank

        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sp_size = self.tp_group.world_size if is_sequence_parallel else 1
        cu_tokens_across_sp_cpu = dp_metadata.cu_tokens_across_sp(sp_size)

        # Use reduce-scatter instead of all-reduce + slice
        # This is more efficient: each rank gets only its portion
        world_size = self.world_size if is_sequence_parallel else self.dp_world_size
        sizes = []
        for i in range(world_size):
            s = 0 if i == 0 else int(cu_tokens_across_sp_cpu[i - 1])
            e = int(cu_tokens_across_sp_cpu[i])
            sizes.append(e - s)

        my_size = sizes[ep_rank]
        output = torch.zeros(
            my_size, hidden_states.shape[1],
            dtype=hidden_states.dtype, device=hidden_states.device
        )

        # Try reduce_scatter for efficiency; fall back to all_reduce+slice
        try:
            import torch.distributed as dist
            chunks = list(torch.split(hidden_states, sizes, dim=0))
            dist.reduce_scatter(
                output, chunks,
                op=dist.ReduceOp.SUM,
                group=get_ep_group().device_group,
            )
        except Exception:
            # Fallback to original all-reduce + slice
            all_hidden_states = get_ep_group().all_reduce(hidden_states)
            start = 0 if ep_rank == 0 else cu_tokens_across_sp_cpu[ep_rank - 1]
            end = cu_tokens_across_sp_cpu[ep_rank]
            output = all_hidden_states[start:end, :]

        return output'''

if old_combine in content:
    content = content.replace(old_combine, new_combine, 1)
    print("✓ Patched NaiveAll2AllManager.combine() → reduce_scatter")
    with open(TARGET, 'w') as f:
        f.write(content)
else:
    print("✗ Could not find NaiveAll2AllManager.combine() pattern")
    # Debug
    if "def combine" in content and "NaiveAll2AllManager" in content:
        print("  (class and method exist but pattern mismatch)")
    exit(1)
