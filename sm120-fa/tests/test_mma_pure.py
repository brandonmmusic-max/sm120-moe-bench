"""Test pure MMA (no ldmatrix) to validate fragment layout."""
import torch
import ctypes
import subprocess

# Build
subprocess.run(
    "nvcc -O3 -gencode=arch=compute_120,code=sm_120 "
    "--use_fast_math -lineinfo --ptxas-options=-v "
    "-shared -o /tmp/sm120-fa/debug_mma_pure.so "
    "/tmp/sm120-fa/csrc/debug_mma_pure.cu "
    "--compiler-options '-fPIC'",
    shell=True, check=True
)

lib = ctypes.CDLL("/tmp/sm120-fa/debug_mma_pure.so")

torch.manual_seed(42)
# A: [16, 16] row-major
A = torch.randn(16, 16, dtype=torch.bfloat16, device="cuda")
# K: [8, 16] → K^T: [16, 8] row-major
K = torch.randn(8, 16, dtype=torch.bfloat16, device="cuda")
KT = K.T.contiguous()  # [16, 8] row-major

C = torch.zeros(16, 8, dtype=torch.float32, device="cuda")

lib.run_pure_mma(
    ctypes.c_void_p(A.data_ptr()),
    ctypes.c_void_p(KT.data_ptr()),
    ctypes.c_void_p(C.data_ptr()),
    ctypes.c_void_p(0)
)
torch.cuda.synchronize()

ref = A.float() @ K.float().T  # [16,16] @ [16,8] = [16,8]
abs_err = (C.cpu() - ref.cpu()).abs()
max_err = abs_err.max().item()
mean_err = abs_err.mean().item()

print(f"Pure MMA test (no ldmatrix):")
print(f"  Max error:  {max_err:.6f}")
print(f"  Mean error: {mean_err:.6f}")

if max_err < 0.2:
    print("  PASS")
else:
    print(f"  FAIL")
    print(f"\nC[:4]:\n{C.cpu()[:4]}")
    print(f"ref[:4]:\n{ref.cpu()[:4]}")
