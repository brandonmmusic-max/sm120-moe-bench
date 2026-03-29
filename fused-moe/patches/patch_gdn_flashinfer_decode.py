#!/usr/bin/env python3
"""
Sprint 20: Wire FlashInfer 0.6.7 GDN decode kernels into vLLM's gdn_linear_attn.py

Replaces vLLM's FLA Triton kernels with FlashInfer's native GDN decode kernels
for all three decode paths:

1. Non-spec decode (packed): gated_delta_rule_decode_pretranspose (gather/scatter)
2. Spec/MTP decode: Sequential per-position gated_delta_rule_decode_pretranspose
   (matches FLA's per-position-slot state management)
3. Non-spec decode in mixed batch: gated_delta_rule_decode_pretranspose (gather/scatter)

Uses gather/scatter pattern instead of pool indexing because vLLM's mamba cache
uses as_strided with non-contiguous inter-slot strides (mamba page layout has
conv_state + ssm_state interleaved per slot).

Controlled by env var VLLM_GDN_DECODE_BACKEND:
  - "flashinfer" (default on SM90+): Use FlashInfer GDN decode kernels
  - "triton": Use original FLA Triton kernels (fallback)
  - "auto": FlashInfer on SM90+, Triton otherwise
"""

import sys

TARGET = "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn_linear_attn.py"

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

def patch():
    src = read_file(TARGET)
    original = src

    # =========================================================================
    # PATCH 1: Add imports and backend selection function after existing imports
    # =========================================================================
    import_block = """import os

_USE_FLASHINFER_GDN_DECODE = None


def _should_use_flashinfer_gdn_decode():
    \"\"\"Check if FlashInfer GDN decode backend should be used.\"\"\"
    global _USE_FLASHINFER_GDN_DECODE
    if _USE_FLASHINFER_GDN_DECODE is not None:
        return _USE_FLASHINFER_GDN_DECODE

    backend = os.environ.get("VLLM_GDN_DECODE_BACKEND", "auto").lower()
    if backend == "flashinfer":
        _USE_FLASHINFER_GDN_DECODE = True
    elif backend == "triton":
        _USE_FLASHINFER_GDN_DECODE = False
    else:  # auto
        _USE_FLASHINFER_GDN_DECODE = (
            current_platform.is_cuda()
            and current_platform.is_device_capability(90)
        )

    if _USE_FLASHINFER_GDN_DECODE:
        logger.info_once(
            "Using FlashInfer GDN decode backend "
            "(set VLLM_GDN_DECODE_BACKEND=triton to disable)",
            scope="local",
        )
    else:
        logger.info_once("Using FLA Triton GDN decode backend", scope="local")
    return _USE_FLASHINFER_GDN_DECODE


def _get_fi_gdn_decode():
    \"\"\"Lazy import FlashInfer GDN decode function.\"\"\"
    from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
    return gated_delta_rule_decode_pretranspose

"""

    # Insert after the logger = init_logger(__name__) line
    marker = 'logger = init_logger(__name__)\n'
    if marker not in src:
        print("ERROR: Could not find logger marker")
        sys.exit(1)
    src = src.replace(marker, marker + '\n' + import_block)

    # =========================================================================
    # PATCH 2: Replace _forward_core_decode_non_spec to use FlashInfer
    # =========================================================================
    old_method_start = "    def _forward_core_decode_non_spec("
    idx_start = src.find(old_method_start)
    if idx_start == -1:
        print("ERROR: Could not find _forward_core_decode_non_spec")
        sys.exit(1)

    # Find next top-level function
    idx_after_start = idx_start + len(old_method_start)
    next_def = src.find("\ndef ", idx_after_start)
    if next_def == -1:
        next_def = len(src)

    new_method = """    def _forward_core_decode_non_spec(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        attn_metadata: GDNAttentionMetadata,
    ):
        \"\"\"
        Core attention computation with a packed non-spec decode fast path.
        Dispatches to FlashInfer GDN decode when available.
        \"\"\"
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
        self_kv_cache = self.kv_cache
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        mixed_qkv_non_spec = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],
            validate_data=False,
        )

        if _should_use_flashinfer_gdn_decode():
            return self._fi_decode_non_spec(
                mixed_qkv_non_spec, b, a, core_attn_out,
                ssm_state, non_spec_state_indices_tensor[:num_actual_tokens],
                num_actual_tokens,
            )

        # Original FLA Triton path
        out_buf = core_attn_out[:num_actual_tokens].unsqueeze(1)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv_non_spec,
            a=a,
            b=b,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            scale=self.head_k_dim**-0.5,
            initial_state=ssm_state,
            out=out_buf,
            ssm_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],
            use_qk_l2norm_in_kernel=True,
        )
        return

    def _fi_decode_non_spec(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        ssm_state: torch.Tensor,
        state_indices: torch.Tensor,
        num_tokens: int,
    ):
        \"\"\"FlashInfer GDN decode for non-speculative single-token decode.

        Uses gather/scatter pattern because vLLM's mamba cache has
        non-contiguous inter-slot strides (mamba page layout).
        \"\"\"
        fi_decode = _get_fi_gdn_decode()

        B = num_tokens
        key_dim_tp = self.key_dim // self.tp_size
        value_dim_tp = self.value_dim // self.tp_size
        num_k_heads_tp = self.num_k_heads // self.tp_size
        num_v_heads_tp = self.num_v_heads // self.tp_size

        # Split packed mixed_qkv into q, k, v
        q, k, v = torch.split(
            mixed_qkv, [key_dim_tp, key_dim_tp, value_dim_tp], dim=-1
        )

        # Reshape to FlashInfer expected format: [B, 1, H, D]
        q = q.view(B, 1, num_k_heads_tp, self.head_k_dim).contiguous()
        k = k.view(B, 1, num_k_heads_tp, self.head_k_dim).contiguous()
        v = v.view(B, 1, num_v_heads_tp, self.head_v_dim).contiguous()

        # Reshape a, b to [B, 1, HV]
        a_fi = a.unsqueeze(1)  # [B, 1, HV]
        b_fi = b.unsqueeze(1)  # [B, 1, HV]

        # Gather states from strided pool into contiguous buffer
        # ssm_state: [pool_size, HV, V, K] (strided in dim 0)
        # gathered: [B, HV, V, K] (contiguous)
        idx = state_indices.long()
        gathered = ssm_state[idx].contiguous()  # [B, HV, V, K]

        # Pre-allocate output [B, 1, HV, V]
        output = torch.empty(
            B, 1, num_v_heads_tp, self.head_v_dim,
            dtype=mixed_qkv.dtype, device=mixed_qkv.device,
        )

        # detach nn.Parameters to avoid dlpack gradient error
        A_log = self.A_log.detach()
        dt_bias = self.dt_bias.detach()

        out, updated = fi_decode(
            q=q, k=k, v=v,
            state=gathered,  # direct per-batch state, NOT pool
            A_log=A_log,
            a=a_fi,
            dt_bias=dt_bias,
            b=b_fi,
            scale=self.head_k_dim ** -0.5,
            output=output,
            use_qk_l2norm=True,
            initial_state=None,
            initial_state_indices=None,
        )

        # Scatter updated states back to strided pool
        ssm_state[idx] = updated

        # Write to core_attn_out: [B, HV, V]
        core_attn_out[:num_tokens] = out.squeeze(1)
        return

    def _fi_decode_spec(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ssm_state: torch.Tensor,
        spec_state_indices: torch.Tensor,
        spec_query_start_loc: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        num_spec_decodes: int,
    ):
        \"\"\"FlashInfer GDN decode for speculative/MTP multi-token decode.

        Processes T tokens per sequence using sequential per-position calls
        to gated_delta_rule_decode_pretranspose. Uses gather/scatter per
        position to handle vLLM's per-position-slot state management and
        non-contiguous mamba cache.

        State flow matches the FLA kernel exactly:
        - t=0: read from ssm_state[spec_indices[i, num_accepted[i]-1]]
               write to ssm_state[spec_indices[i, 0]]
        - t>0: read from ssm_state[spec_indices[i, t-1]]
               write to ssm_state[spec_indices[i, t]]
        \"\"\"
        fi_decode = _get_fi_gdn_decode()

        B = num_spec_decodes
        device = query.device
        dtype = query.dtype

        # query/key shape: [1, total_spec_tokens, H, K]
        # value shape: [1, total_spec_tokens, HV, V]
        total_tokens = query.shape[1]
        T = total_tokens // B if B > 0 else 0

        if T == 0 or B == 0:
            return torch.zeros(
                1, 0, value.shape[2], value.shape[3],
                dtype=dtype, device=device,
            ), ssm_state

        # Reshape from [1, B*T, H, D] to [B, T, H, D]
        q = query.squeeze(0).view(B, T, query.shape[2], query.shape[3]).contiguous()
        k = key.squeeze(0).view(B, T, key.shape[2], key.shape[3]).contiguous()
        v = value.squeeze(0).view(B, T, value.shape[2], value.shape[3]).contiguous()

        num_v_heads_tp = value.shape[2]

        # a, b: [num_actual_tokens, HV] -> [B, T, HV]
        a_fi = a[:B * T].view(B, T, -1)
        b_fi = b[:B * T].view(B, T, -1)

        # Output buffer [B, T, HV, V]
        output = torch.empty(
            B, T, num_v_heads_tp, self.head_v_dim,
            dtype=dtype, device=device,
        )

        # detach nn.Parameters to avoid dlpack gradient error
        A_log = self.A_log.detach()
        dt_bias = self.dt_bias.detach()

        # Get initial read slots from previous round's accepted position
        batch_range = torch.arange(B, device=device)
        accepted_idx = (num_accepted_tokens[:B] - 1).clamp(min=0)
        prev_read_slots = spec_state_indices[batch_range, accepted_idx].long()

        # Process T positions sequentially with gather/scatter
        for t in range(T):
            write_slots = spec_state_indices[:B, t].long()

            # Gather state from previous position
            gathered = ssm_state[prev_read_slots].contiguous()  # [B, HV, V, K]

            # Extract per-position tensors [B, 1, H, D]
            q_t = q[:, t : t + 1].contiguous()
            k_t = k[:, t : t + 1].contiguous()
            v_t = v[:, t : t + 1].contiguous()
            a_t = a_fi[:, t : t + 1].contiguous()  # [B, 1, HV]
            b_t = b_fi[:, t : t + 1].contiguous()  # [B, 1, HV]

            out_t = torch.empty(
                B, 1, num_v_heads_tp, self.head_v_dim,
                dtype=dtype, device=device,
            )

            out_t, updated = fi_decode(
                q=q_t, k=k_t, v=v_t,
                state=gathered,  # gathered contiguous state
                A_log=A_log,
                a=a_t,
                dt_bias=dt_bias,
                b=b_t,
                scale=self.head_k_dim ** -0.5,
                output=out_t,
                use_qk_l2norm=True,
                initial_state=None,
                initial_state_indices=None,
            )

            # Scatter updated state to write slots
            valid = write_slots >= 0
            if valid.all():
                ssm_state[write_slots] = updated
            elif valid.any():
                ssm_state[write_slots[valid]] = updated[valid]

            output[:, t : t + 1] = out_t
            prev_read_slots = write_slots

        # Reshape output to [1, B*T, HV, V] to match FLA kernel output format
        output = output.view(1, B * T, num_v_heads_tp, self.head_v_dim)
        return output, ssm_state

"""

    # Replace the method
    src = src[:idx_start] + new_method + src[next_def:]

    # =========================================================================
    # PATCH 3: Modify _forward_core spec path (section 2.1) to dispatch
    # =========================================================================
    old_spec_call = """        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[
                        : attn_metadata.num_spec_decodes + 1
                    ],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            )
        else:
            core_attn_out_spec, last_recurrent_state = None, None"""

    new_spec_call = """        # 2.1: Process the multi-query part
        if spec_sequence_masks is not None:
            if _should_use_flashinfer_gdn_decode():
                core_attn_out_spec, last_recurrent_state = (
                    self._fi_decode_spec(
                        query=query_spec,
                        key=key_spec,
                        value=value_spec,
                        a=a,
                        b=b,
                        ssm_state=ssm_state,
                        spec_state_indices=spec_state_indices_tensor,
                        spec_query_start_loc=spec_query_start_loc[
                            : attn_metadata.num_spec_decodes + 1
                        ],
                        num_accepted_tokens=num_accepted_tokens,
                        num_spec_decodes=attn_metadata.num_spec_decodes,
                    )
                )
            else:
                core_attn_out_spec, last_recurrent_state = (
                    fused_sigmoid_gating_delta_rule_update(
                        A_log=self.A_log,
                        a=a,
                        b=b,
                        dt_bias=self.dt_bias,
                        q=query_spec,
                        k=key_spec,
                        v=value_spec,
                        initial_state=ssm_state,
                        inplace_final_state=True,
                        cu_seqlens=spec_query_start_loc[
                            : attn_metadata.num_spec_decodes + 1
                        ],
                        ssm_state_indices=spec_state_indices_tensor,
                        num_accepted_tokens=num_accepted_tokens,
                        use_qk_l2norm_in_kernel=True,
                    )
                )
        else:
            core_attn_out_spec, last_recurrent_state = None, None"""

    if old_spec_call not in src:
        print("ERROR: Could not find spec decode call block")
        sys.exit(1)
    src = src.replace(old_spec_call, new_spec_call)

    # =========================================================================
    # PATCH 4: Modify _forward_core non-spec decode path (section 2.2)
    # =========================================================================
    old_nonspec_decode = """        elif attn_metadata.num_decodes > 0:
            core_attn_out_non_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[
                        : attn_metadata.num_decodes + 1
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                )
            )"""

    new_nonspec_decode = """        elif attn_metadata.num_decodes > 0:
            if _should_use_flashinfer_gdn_decode():
                core_attn_out_non_spec = self._fi_decode_single_from_qkv(
                    query=query_non_spec,
                    key=key_non_spec,
                    value=value_non_spec,
                    a=a,
                    b=b,
                    ssm_state=ssm_state,
                    state_indices=non_spec_state_indices_tensor,
                    num_decodes=attn_metadata.num_decodes,
                )
                last_recurrent_state = ssm_state
            else:
                core_attn_out_non_spec, last_recurrent_state = (
                    fused_sigmoid_gating_delta_rule_update(
                        A_log=self.A_log,
                        a=a,
                        b=b,
                        dt_bias=self.dt_bias,
                        q=query_non_spec,
                        k=key_non_spec,
                        v=value_non_spec,
                        initial_state=ssm_state,
                        inplace_final_state=True,
                        cu_seqlens=non_spec_query_start_loc[
                            : attn_metadata.num_decodes + 1
                        ],
                        ssm_state_indices=non_spec_state_indices_tensor,
                        use_qk_l2norm_in_kernel=True,
                    )
                )"""

    if old_nonspec_decode not in src:
        print("ERROR: Could not find non-spec decode call block")
        sys.exit(1)
    src = src.replace(old_nonspec_decode, new_nonspec_decode)

    # =========================================================================
    # PATCH 5: Add _fi_decode_single_from_qkv method (for non-spec in mixed)
    # =========================================================================
    fi_single_method = """
    def _fi_decode_single_from_qkv(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ssm_state: torch.Tensor,
        state_indices: torch.Tensor,
        num_decodes: int,
    ):
        \"\"\"FlashInfer GDN decode for non-spec single-token decode (pre-split QKV).

        Used when QKV are already split and reshaped by rearrange_mixed_qkv.
        query/key: [1, num_decodes, H, K], value: [1, num_decodes, HV, V]

        Uses gather/scatter for non-contiguous mamba cache.
        \"\"\"
        fi_decode = _get_fi_gdn_decode()

        B = num_decodes
        dtype = query.dtype
        device = query.device

        # Reshape from [1, B, H, D] to [B, 1, H, D]
        q = query.squeeze(0).unsqueeze(1).contiguous()  # [B, 1, H, K]
        k = key.squeeze(0).unsqueeze(1).contiguous()    # [B, 1, H, K]
        v = value.squeeze(0).unsqueeze(1).contiguous()   # [B, 1, HV, V]

        num_v_heads_tp = value.shape[2]

        # a, b: [num_tokens, HV] -> [B, 1, HV]
        a_fi = a[:B].unsqueeze(1)
        b_fi = b[:B].unsqueeze(1)

        # Gather states from strided pool
        idx = state_indices[:B].long()
        gathered = ssm_state[idx].contiguous()  # [B, HV, V, K]

        output = torch.empty(
            B, 1, num_v_heads_tp, self.head_v_dim,
            dtype=dtype, device=device,
        )

        # detach nn.Parameters to avoid dlpack gradient error
        A_log = self.A_log.detach()
        dt_bias = self.dt_bias.detach()

        out, updated = fi_decode(
            q=q, k=k, v=v,
            state=gathered,
            A_log=A_log,
            a=a_fi,
            dt_bias=dt_bias,
            b=b_fi,
            scale=self.head_k_dim ** -0.5,
            output=output,
            use_qk_l2norm=True,
            initial_state=None,
            initial_state_indices=None,
        )

        # Scatter updated states back
        ssm_state[idx] = updated

        # Return in [1, B, HV, V] format to match FLA output
        return out.transpose(0, 1).unsqueeze(0)  # [1, B, HV, V]

"""

    # Insert before gdn_attention_core function
    gdn_core_marker = "\ndef gdn_attention_core("
    if gdn_core_marker not in src:
        print("ERROR: Could not find gdn_attention_core marker")
        sys.exit(1)
    src = src.replace(gdn_core_marker, fi_single_method + gdn_core_marker)

    # =========================================================================
    # Verify patches applied
    # =========================================================================
    checks = [
        ("_should_use_flashinfer_gdn_decode", "Patch 1 (imports)"),
        ("_fi_decode_non_spec", "Patch 2 (non-spec decode)"),
        ("_fi_decode_spec", "Patch 3 (spec decode)"),
        ("_fi_decode_single_from_qkv", "Patch 5 (single from qkv)"),
        ("gathered = ssm_state[idx].contiguous()", "Gather/scatter pattern"),
        ("ssm_state[idx] = updated", "Scatter-back pattern"),
    ]
    for needle, name in checks:
        if needle not in src:
            print(f"ERROR: {name} not applied")
            sys.exit(1)

    write_file(TARGET, src)
    print(f"SUCCESS: All patches applied to {TARGET}")
    print(f"  Original: {len(original)} chars")
    print(f"  Patched:  {len(src)} chars")
    print(f"  Delta:    +{len(src) - len(original)} chars")
    print(f"  Key change: gather/scatter pattern (no pool indexing)")


if __name__ == "__main__":
    patch()
