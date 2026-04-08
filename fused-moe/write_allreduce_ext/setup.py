"""
Build script for write_allreduce_ext — P2P AllReduce PyTorch extension.

Build (standalone):
    cd /home/brandonmusic/sm120-moe-bench/fused-moe/write_allreduce_ext
    pip install -e .

Build (inside Docker):
    pip install /path/to/write_allreduce_ext/

The extension compiles for SM 12.0a (Blackwell GB202).
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Path to the CUDA source relative to this setup.py
CSRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "csrc")

setup(
    name="write_allreduce_ext",
    version="0.1.0",
    description="Write-based P2P AllReduce for vLLM TP=4 on PCIe Blackwell GPUs",
    ext_modules=[
        CUDAExtension(
            name="write_allreduce_ext",
            sources=[
                os.path.join(CSRC_DIR, "write_allreduce_ext.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_120a,code=sm_120a",
                    "--expt-relaxed-constexpr",
                    "-std=c++17",
                    # Needed for __threadfence_system to work correctly
                    "-Xptxas", "-dlcm=cg",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
