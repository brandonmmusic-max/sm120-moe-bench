#!/usr/bin/env python3
"""
Patch vLLM's NVFP4 oracle to add VerdictMoE backend.

Usage (inside vLLM container):
    python3 /workspace/patches/nvfp4_oracle_verdict.py

This adds:
1. VERDICT_MOE enum to NvFp4MoeBackend
2. backend_to_kernel_cls mapping for VERDICT_MOE
3. VLLM_USE_VERDICT_MOE=1 env var check in select_nvfp4_moe_backend
"""

import sys
import os

ORACLE_PATH = "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py"
VERDICT_MOE_PATH = "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/verdict_moe.py"
VERDICT_CSRC_DIR = "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/csrc"


def patch_oracle():
    with open(ORACLE_PATH, "r") as f:
        content = f.read()

    if "VERDICT_MOE" in content:
        print("Oracle already patched with VERDICT_MOE")
        return

    # 1. Add VERDICT_MOE to enum
    content = content.replace(
        '    MARLIN = "MARLIN"',
        '    MARLIN = "MARLIN"\n    VERDICT_MOE = "VERDICT_MOE"',
    )

    # 2. Add to backend_to_kernel_cls
    verdict_block = '''
    elif backend == NvFp4MoeBackend.VERDICT_MOE:
        from vllm.model_executor.layers.fused_moe.verdict_moe import (
            VerdictMoEExperts,
        )

        return [VerdictMoEExperts]
'''
    content = content.replace(
        '    elif backend == NvFp4MoeBackend.MARLIN:',
        verdict_block + '    elif backend == NvFp4MoeBackend.MARLIN:',
    )

    # 3. Add VLLM_USE_VERDICT_MOE check before the main selection loop
    verdict_env_check = '''
    # VerdictMoE: fused 3-kernel pipeline with FP4 dequant (SM120 only)
    if os.environ.get("VLLM_USE_VERDICT_MOE", "0") == "1":
        backend = NvFp4MoeBackend.VERDICT_MOE
        return _return_or_raise(
            backend, config, weight_key, activation_key, activation_format
        )

'''
    # Insert before the VLLM_TEST_FORCE_FP8_MARLIN check
    content = content.replace(
        '    if envs.VLLM_TEST_FORCE_FP8_MARLIN:',
        verdict_env_check + '    if envs.VLLM_TEST_FORCE_FP8_MARLIN:',
    )

    # 4. Add 'import os' if not already present
    if 'import os\n' not in content:
        content = 'import os\n' + content

    # 5. Add to map_nvfp4_backend
    content = content.replace(
        '        "marlin": NvFp4MoeBackend.MARLIN,',
        '        "marlin": NvFp4MoeBackend.MARLIN,\n        "verdict_moe": NvFp4MoeBackend.VERDICT_MOE,',
    )

    with open(ORACLE_PATH, "w") as f:
        f.write(content)

    print(f"Patched oracle: {ORACLE_PATH}")
    print("  - Added VERDICT_MOE to NvFp4MoeBackend enum")
    print("  - Added backend_to_kernel_cls mapping")
    print("  - Added VLLM_USE_VERDICT_MOE=1 env check")
    print("  - Added to map_nvfp4_backend")


def install_verdict_moe():
    """Copy VerdictMoE files into the vLLM package."""
    import shutil

    # Source files (mounted from host)
    src_base = "/workspace/fused-moe"
    verdict_moe_src = os.path.join(src_base, "verdict_moe.py")
    csrc_ext = os.path.join(src_base, "csrc", "verdict_moe_ext.cu")

    # Copy Python module
    if os.path.exists(verdict_moe_src):
        shutil.copy2(verdict_moe_src, VERDICT_MOE_PATH)
        print(f"Installed: {VERDICT_MOE_PATH}")

    # Copy CUDA source
    os.makedirs(VERDICT_CSRC_DIR, exist_ok=True)
    if os.path.exists(csrc_ext):
        shutil.copy2(csrc_ext, os.path.join(VERDICT_CSRC_DIR, "verdict_moe_ext.cu"))
        print(f"Installed: {VERDICT_CSRC_DIR}/verdict_moe_ext.cu")


if __name__ == "__main__":
    install_verdict_moe()
    patch_oracle()
    print("\nDone! Set VLLM_USE_VERDICT_MOE=1 to enable VerdictMoE backend.")
