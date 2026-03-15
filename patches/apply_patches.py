#!/usr/bin/env python3
"""Apply all SM120 NVFP4 optimizations to FlashInfer inside the vLLM Docker container.

Usage (inside container):
    docker exec -it vllm-qwen35 python3 /patches/apply_patches.py
"""

FI = "/usr/local/lib/python3.12/dist-packages/flashinfer"

# PATCH 1: compute_120f suffix for SM120 JIT kernels
p = f"{FI}/compilation_context.py"
with open(p) as f: c = f.read()
if "major == 12:" not in c:
    old = "                    if major >= 9:"
    new = '''                    if major == 12:
                        try:
                            from flashinfer.jit.cpp_ext import is_cuda_version_at_least
                            if is_cuda_version_at_least("13.0"):
                                minor = str(minor) + "f"
                            else:
                                minor = str(minor) + "a"
                        except Exception:
                            minor = str(minor) + "a"
                    elif major >= 9:'''
    c = c.replace(old, new, 1)
    with open(p, "w") as f: f.write(c)
    print("1. compute_120f APPLIED")
else:
    print("1. compute_120f already present")

# PATCH 2: K=64 tile validation gate
p = f"{FI}/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch_tma_ws.h"
with open(p) as f: c = f.read()
old = "(TileM == 256 && TileN == 128 && TileK == 128);"
new = """(TileM == 256 && TileN == 128 && TileK == 128) ||
         (TileM == 128 && TileN == 128 && TileK == 64) ||
         (TileM == 128 && TileN == 256 && TileK == 64) ||
         (TileM == 256 && TileN == 128 && TileK == 64);"""
if "(TileK == 64)" not in c:
    c = c.replace(old, new, 1)
    with open(p, "w") as f: f.write(c)
    print("2. K=64 validation APPLIED")
else:
    print("2. K=64 validation already present")

# PATCH 3: K=64 codegen instantiation
p = f"{FI}/jit/gemm/cutlass/generate_kernels.py"
with open(p) as f: c = f.read()
if "[128, 128, 64]," not in c:
    c = c.replace("        [128, 128, 128],", "        [128, 128, 64],\n        [128, 128, 128],", 1)
    with open(p, "w") as f: f.write(c)
    print("3. K=64 codegen APPLIED")
else:
    print("3. K=64 codegen already present")

# PATCH 4: StageCount<2> for SM120 SMEM overflow fix
p = f"{FI}/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl"
with open(p) as f: c = f.read()
old = "StageCountAutoCarveout, KernelSchedule>::CollectiveOp;"
new = "std::conditional_t<IsSM120, cutlass::gemm::collective::StageCount<2>, StageCountAutoCarveout>, KernelSchedule>::CollectiveOp;"
if "StageCount<2>" not in c:
    c = c.replace(old, new, 1)
    with open(p, "w") as f: f.write(c)
    print("4. StageCount<2> APPLIED")
else:
    print("4. StageCount<2> already present")

print("\nAll patches complete! Clear cache: rm -rf ~/.cache/flashinfer/")
