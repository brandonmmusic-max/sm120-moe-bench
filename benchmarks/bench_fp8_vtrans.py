"""
Benchmark SM120 Flash Attention FP8 with pre-transposed V.

Standalone test: builds and benchmarks the FP8 V^T kernel.
Compares against BF16 v4 and torch SDPA.
"""

import torch
import torch.nn.functional as F
import time
import subprocess
import os
import sys

# Build the FP8 V^T kernel as a separate extension
from torch.utils.cpp_extension import load

def build_fp8_kernel():
    """Build FP8 V^T kernel as a standalone extension."""
    src_dir = os.path.join(os.path.dirname(__file__), '..', 'csrc')
    return load(
        name="sm120_fa_fp8_vtrans",
        sources=[os.path.join(src_dir, 'sm120_flash_attn_fp8_vtrans.cu')],
        extra_cuda_cflags=[
            '-O3', '-std=c++17',
            '-gencode=arch=compute_120a,code=sm_120a',
            '--use_fast_math', '-lineinfo',
            '--ptxas-options=-v',
        ],
        extra_ldflags=['-lcuda'],
        verbose=True,
    )


def benchmark_one(func, warmup=5, iters=20):
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        func()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    times.sort()
    return times[len(times) // 2]


def compute_tflops(batch, heads, seq_q, seq_kv, head_dim, time_ms):
    flops = 4.0 * batch * heads * seq_q * seq_kv * head_dim
    return flops / (time_ms * 1e-3) / 1e12


def main():
    print("Building FP8 V^T kernel...")
    # The kernel doesn't have a torch binding yet, so we need to add one.
    # For now, let's just verify compilation and build a simple C wrapper.

    # Actually, the kernel uses extern "C" interface, not pybind11.
    # We need a torch binding. Let's create a temporary one.

    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'csrc')

    # Create a temporary binding file
    binding_code = '''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

extern "C" void sm120_flash_attn_fp8_vtrans_forward(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    int batch, int Hq, int Hkv, int Sq, int Skv, int hd, cudaStream_t stream
);

std::vector<torch::Tensor> fp8_vtrans_fwd(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool return_lse
) {
    TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kBFloat16 && Q.is_contiguous());
    TORCH_CHECK(K.is_cuda() && K.dtype() == torch::kBFloat16 && K.is_contiguous());
    TORCH_CHECK(V.is_cuda() && V.dtype() == torch::kBFloat16 && V.is_contiguous());

    const int batch = Q.size(0), Hq = Q.size(1), Sq = Q.size(2), hd = Q.size(3);
    const int Hkv = K.size(1), Skv = K.size(2);

    auto O = torch::empty_like(Q);
    torch::Tensor L;
    float* L_ptr = nullptr;
    if (return_lse) {
        L = torch::empty({batch * Hq, Sq}, torch::dtype(torch::kFloat32).device(Q.device()));
        L_ptr = L.data_ptr<float>();
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sm120_flash_attn_fp8_vtrans_forward(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
        L_ptr,
        batch, Hq, Hkv, Sq, Skv, hd, stream
    );

    if (return_lse) return {O, L.reshape({batch, Hq, Sq})};
    return {O};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fp8_vtrans_fwd, "FP8 V^T Flash Attention",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("return_lse") = false);
}
'''

    binding_path = os.path.join(src_dir, 'fp8_vtrans_binding.cpp')
    with open(binding_path, 'w') as f:
        f.write(binding_code)

    try:
        mod = load(
            name="sm120_fa_fp8_vtrans",
            sources=[
                binding_path,
                os.path.join(src_dir, 'sm120_flash_attn_fp8_vtrans.cu'),
            ],
            extra_cuda_cflags=[
                '-O3', '-std=c++17',
                '-gencode=arch=compute_120a,code=sm_120a',
                '--use_fast_math', '-lineinfo',
            ],
            extra_ldflags=['-lcuda'],
            verbose=True,
        )
    except Exception as e:
        print(f"Build failed: {e}")
        return

    # Also try to load the BF16 v4 kernel for comparison
    try:
        import sm120_flash_attn as bf16_mod
        have_bf16 = True
    except ImportError:
        have_bf16 = False
        print("BF16 v4 kernel not available for comparison")

    device = "cuda"
    dtype = torch.bfloat16
    B, Hq, Hkv, D = 1, 32, 8, 128

    # Correctness test first
    print("\n=== Correctness Test ===")
    seq = 256
    Q = torch.randn(B, Hq, seq, D, device=device, dtype=dtype)
    K = torch.randn(B, Hkv, seq, D, device=device, dtype=dtype)
    V = torch.randn(B, Hkv, seq, D, device=device, dtype=dtype)

    # Reference: torch SDPA
    kv_repeat = Hq // Hkv
    Ke = K.repeat_interleave(kv_repeat, dim=1)
    Ve = V.repeat_interleave(kv_repeat, dim=1)
    ref = F.scaled_dot_product_attention(Q, Ke, Ve)

    # FP8 V^T kernel
    out = mod.forward(Q, K, V, False)[0]

    # FP8 has lower precision, so use relaxed tolerance
    max_err = (out.float() - ref.float()).abs().max().item()
    mean_err = (out.float() - ref.float()).abs().mean().item()
    cos_sim = F.cosine_similarity(out.float().reshape(-1, D), ref.float().reshape(-1, D), dim=1).mean().item()
    print(f"Max error: {max_err:.6f}, Mean error: {mean_err:.6f}, Cosine sim: {cos_sim:.6f}")
    if cos_sim > 0.99:
        print("PASS (cosine sim > 0.99)")
    else:
        print("FAIL — checking if kernel produces non-zero output...")
        print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
        print(f"Ref range: [{ref.min().item():.4f}, {ref.max().item():.4f}]")

    # Performance benchmark
    print("\n=== Performance Benchmark ===")
    seq_lengths = [512, 1024, 2048, 4096, 8192]

    header = f"{'Seq':>6s} | {'FP8 VT':>10s} {'TF':>7s}"
    if have_bf16:
        header += f" | {'BF16 v4':>10s} {'TF':>7s} | {'Ratio':>6s}"
    header += f" | {'SDPA':>10s} {'TF':>7s} | {'vs SDPA':>7s}"
    print(header)
    print("-" * len(header))

    for seq in seq_lengths:
        Q = torch.randn(B, Hq, seq, D, device=device, dtype=dtype)
        K = torch.randn(B, Hkv, seq, D, device=device, dtype=dtype)
        V = torch.randn(B, Hkv, seq, D, device=device, dtype=dtype)
        Ke = K.repeat_interleave(kv_repeat, dim=1)
        Ve = V.repeat_interleave(kv_repeat, dim=1)

        # FP8 V^T
        t_fp8 = benchmark_one(lambda: mod.forward(Q, K, V, False))
        tf_fp8 = compute_tflops(B, Hq, seq, seq, D, t_fp8)

        line = f"{seq:>6d} | {t_fp8:>7.2f} ms {tf_fp8:>6.1f}"

        # BF16 v4
        if have_bf16:
            t_bf16 = benchmark_one(lambda: bf16_mod.forward(Q, K, V, False))
            tf_bf16 = compute_tflops(B, Hq, seq, seq, D, t_bf16)
            ratio = tf_fp8 / tf_bf16 if tf_bf16 > 0 else 0
            line += f" | {t_bf16:>7.2f} ms {tf_bf16:>6.1f} | {ratio:>5.2f}x"

        # SDPA
        t_sdpa = benchmark_one(lambda: F.scaled_dot_product_attention(Q, Ke, Ve))
        tf_sdpa = compute_tflops(B, Hq, seq, seq, D, t_sdpa)
        vs_sdpa = t_sdpa / t_fp8
        line += f" | {t_sdpa:>7.2f} ms {tf_sdpa:>6.1f} | {vs_sdpa:>6.2f}x"

        print(line)


if __name__ == "__main__":
    main()
