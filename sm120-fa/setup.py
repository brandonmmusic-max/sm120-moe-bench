"""
Build SM120 Flash Attention as a PyTorch extension.

Usage:
    pip install -e .
    # or
    python setup.py build_ext --inplace
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="sm120_flash_attn",
    version="0.1.0",
    description="Flash Attention for SM120 (RTX PRO 6000 Blackwell)",
    ext_modules=[
        CUDAExtension(
            name="sm120_flash_attn",
            sources=[
                "csrc/torch_binding.cpp",
                "csrc/sm120_flash_attn.cu",
                "csrc/sm120_selective_cache.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-gencode=arch=compute_120,code=sm_120",
                    "--use_fast_math",
                    "-lineinfo",
                    "--ptxas-options=-v",  # Show register usage
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
