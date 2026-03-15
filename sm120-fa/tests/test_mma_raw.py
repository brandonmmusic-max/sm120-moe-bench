"""
Raw MMA fragment analysis.

Test 1: All-ones → every fragment should be 16.0
Test 2: Basis-vector Q → fragments select specific K rows
Test 3: Derive writeback mapping from known values
"""
import torch
import ctypes
import subprocess

# Build
subprocess.run(
    "nvcc -O3 -gencode=arch=compute_120,code=sm_120 "
    "--use_fast_math -lineinfo --ptxas-options=-v "
    "-shared -o /tmp/sm120-fa/debug_mma_raw.so "
    "/tmp/sm120-fa/csrc/debug_mma_raw.cu "
    "--compiler-options '-fPIC'",
    shell=True, check=True, capture_output=True
)
lib = ctypes.CDLL("/tmp/sm120-fa/debug_mma_raw.so")


def run_mma(A, B):
    """Run MMA and return raw [32, 4] fragment dump."""
    raw = torch.zeros(32 * 4, dtype=torch.float32, device="cuda")
    lib.run_raw_mma(
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_void_p(B.data_ptr()),
        ctypes.c_void_p(raw.data_ptr()),
        ctypes.c_void_p(0)
    )
    torch.cuda.synchronize()
    return raw.cpu().reshape(32, 4)


# =========================================================================
# Test 1: All ones → C = A @ B^T = 16.0 everywhere
# =========================================================================
print("=" * 60)
print("TEST 1: All ones (C should be 16.0 everywhere)")
print("=" * 60)

A = torch.ones(16, 16, dtype=torch.bfloat16, device="cuda")
# B_in is K^T stored as [K=16, N=8] row-major
B = torch.ones(16, 8, dtype=torch.bfloat16, device="cuda")

frags = run_mma(A, B)

all_16 = (frags - 16.0).abs().max().item() < 0.1
print(f"  All fragments == 16.0: {all_16}")
if not all_16:
    unique_vals = sorted(set(frags.flatten().tolist()))
    print(f"  Unique values: {unique_vals[:10]}")
    print(f"  Fragment dump (first 8 lanes):")
    for lane in range(8):
        print(f"    Lane {lane}: {frags[lane].tolist()}")

# =========================================================================
# Test 2: Selector test — Q row 0 = [1,0,0,...], K row 0 = [0.5, 0.5, ...]
# C[0, :] should be K[0, :] scaled by Q, others zero
# =========================================================================
print()
print("=" * 60)
print("TEST 2: Basis vector Q (row 0 only)")
print("=" * 60)

# Q: only row 0 has value 1.0 in column 0
A2 = torch.zeros(16, 16, dtype=torch.bfloat16, device="cuda")
A2[0, 0] = 1.0  # Q[0, 0] = 1.0

# K^T [16, 8]: only row 0 (k=0) has values
B2 = torch.zeros(16, 8, dtype=torch.bfloat16, device="cuda")
for n in range(8):
    B2[0, n] = float(n + 1) * 0.125  # K^T[0, n] = (n+1)/8

# Expected: C[0, n] = Q[0, 0] * K^T[0, n] = (n+1)/8
# C[m>0, :] = 0
ref = A2.float() @ B2.float()  # Not K^T again — B is already K^T
# Wait: A is [16,16], B is [16,8]. MMA computes C = A @ B (where B is col-major).
# So C[m,n] = sum_k A[m,k] * B[k,n]
# With A[0,0]=1, B[0,n]=(n+1)/8: C[0,n] = (n+1)/8

print(f"  Reference C[0, :] = {ref[0].tolist()}")
print(f"  Reference C[1, :] = {ref[1].tolist()} (should be zeros)")

frags2 = run_mma(A2, B2)

# Find which lanes/regs have non-zero values
print(f"  Non-zero fragments:")
for lane in range(32):
    for reg in range(4):
        val = frags2[lane, reg].item()
        if abs(val) > 0.001:
            print(f"    Lane {lane:2d}, reg {reg}: {val:.6f}")

# =========================================================================
# Test 3: Derive (row, col) mapping from structured test
# =========================================================================
print()
print("=" * 60)
print("TEST 3: Derive writeback mapping")
print("=" * 60)

# Use A = identity-like, B = column indicators
# A[m, k] = m (each row is constant = row index)
# B[k, n] = 1 only for k=0
# C[m, n] = A[m, 0] * B[0, n] = m * B[0, n]

A3 = torch.zeros(16, 16, dtype=torch.bfloat16, device="cuda")
for m in range(16):
    A3[m, 0] = float(m)

B3 = torch.zeros(16, 8, dtype=torch.bfloat16, device="cuda")
for n in range(8):
    B3[0, n] = float(n + 1)

# C[m, n] = m * (n+1)
ref3 = A3.float() @ B3.float()
print(f"  Reference C[0, :] = {ref3[0].tolist()}")
print(f"  Reference C[1, :] = {ref3[1].tolist()}")
print(f"  Reference C[2, :] = {ref3[2].tolist()}")

frags3 = run_mma(A3, B3)

# Derive mapping: for each non-zero fragment, figure out which (m, n) it corresponds to
print(f"\n  Derived mapping (lane, reg) → (row, col):")
for lane in range(32):
    for reg in range(4):
        val = frags3[lane, reg].item()
        if abs(val) > 0.001:
            # val = m * (n+1), so find m and n
            # Try each (m, n) combination
            found = False
            for m in range(16):
                for n in range(8):
                    expected = float(m) * float(n + 1)
                    if abs(val - expected) < 0.5 and expected != 0:
                        print(f"    Lane {lane:2d}, reg {reg}: val={val:8.2f} → row={m:2d}, col={n}")
                        found = True
                        break
                if found:
                    break
            if not found and abs(val) > 0.001:
                print(f"    Lane {lane:2d}, reg {reg}: val={val:8.2f} → UNKNOWN")
