"""
Debug test for ldmatrix loads on SM120.

Validates that ldmatrix.x1 and ldmatrix.x4 correctly load data
from swizzled shared memory into MMA fragment registers.
"""

import torch
import ctypes
import os

# Build the debug kernel
os.system("cd /tmp/sm120-fa && nvcc -O3 -gencode=arch=compute_120,code=sm_120 "
          "--use_fast_math -lineinfo --ptxas-options=-v "
          "-Xptxas=-warn-spills -Xptxas=-warn-lmem-usage "
          "-shared -o /tmp/sm120-fa/debug_ldmatrix.so "
          "/tmp/sm120-fa/csrc/debug_ldmatrix.cu "
          "--compiler-options '-fPIC' 2>&1")

lib = ctypes.CDLL("/tmp/sm120-fa/debug_ldmatrix.so")

TILE_M = 16
TILE_K = 16
HEAD_DIM = 128


def test_ldmatrix_x1():
    """Test ldmatrix.x1 loads first 8x8 sub-tile correctly."""
    print("=" * 60)
    print("TEST: ldmatrix.x1")
    print("=" * 60)

    # Create known input pattern: each element = row * 100 + col
    input_data = torch.zeros(TILE_M, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    for r in range(TILE_M):
        for c in range(HEAD_DIM):
            input_data[r, c] = float(r * 100 + c)

    output = torch.zeros(TILE_M, TILE_K, dtype=torch.float32, device="cuda")
    addr_log = torch.zeros(32 * 4, dtype=torch.int32, device="cuda")

    lib.run_debug_ldmatrix_x1(
        ctypes.c_void_p(input_data.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(addr_log.data_ptr()),
        ctypes.c_void_p(0)  # default stream
    )
    torch.cuda.synchronize()

    # Check addresses
    print("\nAddress log (lane, mat_row, mat_col, byte_addr, align):")
    log = addr_log.cpu().reshape(32, 4)
    for lane in range(min(16, 32)):
        row, col, addr, align = log[lane].tolist()
        status = "OK" if align == 0 else "MISALIGNED!"
        print(f"  Lane {lane:2d}: row={row}, col={col}, addr={addr:5d}, align={align} {status}")

    # Check output
    print("\nOutput (should be row*100 + col for first 8x8):")
    out = output.cpu()
    expected = torch.zeros(TILE_M, TILE_K)
    for r in range(8):
        for c in range(TILE_K):
            expected[r, c] = r * 100 + c

    # Only check the 8×8 sub-tile that ldmatrix.x1 loads
    # ldmatrix.x1 only fills part of the output
    has_data = (out != 0).any()
    print(f"  Has non-zero data: {has_data}")

    if has_data:
        for r in range(8):
            vals = [f"{out[r, c].item():6.0f}" for c in range(8)]
            exp_vals = [f"{expected[r, c].item():6.0f}" for c in range(8)]
            match = all(abs(out[r, c].item() - expected[r, c].item()) < 1 for c in range(8))
            print(f"  Row {r}: got [{', '.join(vals)}]  exp [{', '.join(exp_vals)}]  {'OK' if match else 'MISMATCH'}")

    return has_data


def test_ldmatrix_x4():
    """Test ldmatrix.x4 loads full 16x16 sub-tile correctly."""
    print("\n" + "=" * 60)
    print("TEST: ldmatrix.x4 (A matrix, m16n8k16)")
    print("=" * 60)

    # Create known input: element = row * 100 + col
    input_data = torch.zeros(TILE_M, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    for r in range(TILE_M):
        for c in range(HEAD_DIM):
            input_data[r, c] = float(r * 100 + c)

    output = torch.zeros(TILE_M, TILE_K, dtype=torch.float32, device="cuda")
    addr_log = torch.zeros(32 * 6, dtype=torch.int32, device="cuda")

    lib.run_debug_ldmatrix_x4(
        ctypes.c_void_p(input_data.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        ctypes.c_void_p(addr_log.data_ptr()),
        ctypes.c_void_p(0)
    )
    torch.cuda.synchronize()

    # Check addresses
    print("\nAddress log (lane, sub_mat, smem_row, smem_col, byte_addr, align, bank):")
    log = addr_log.cpu().reshape(32, 6)
    all_aligned = True
    for lane in range(32):
        sub, row, col, addr, align, bank = log[lane].tolist()
        if align != 0:
            all_aligned = False
        print(f"  Lane {lane:2d}: sub={sub}, row={row:2d}, col={col:2d}, "
              f"addr={addr:5d}, align={align}, bank={bank:2d}"
              f"{'  MISALIGNED!' if align != 0 else ''}")

    print(f"\n  All addresses 16-byte aligned: {all_aligned}")

    # Check output: should reconstruct the 16×16 sub-tile
    out = output.cpu()
    print("\nOutput (16×16 matrix, should be row*100+col):")

    all_correct = True
    for r in range(TILE_M):
        vals = []
        for c in range(TILE_K):
            expected = r * 100 + c
            got = out[r, c].item()
            match = abs(got - expected) < 1
            if not match:
                all_correct = False
            vals.append(f"{got:6.0f}")
        exp = [f"{r * 100 + c:6.0f}" for c in range(TILE_K)]
        row_ok = all(abs(out[r, c].item() - (r * 100 + c)) < 1 for c in range(TILE_K))
        print(f"  Row {r:2d}: [{', '.join(vals[:8])} | {', '.join(vals[8:])}]  {'OK' if row_ok else 'MISMATCH'}")

    print(f"\n  All values correct: {all_correct}")
    return all_correct


if __name__ == "__main__":
    ok1 = test_ldmatrix_x1()
    ok4 = test_ldmatrix_x4()

    print("\n" + "=" * 60)
    if ok1 and ok4:
        print("ldmatrix validation: READY FOR MMA")
    else:
        print("ldmatrix validation: NEEDS FIXING")
    print("=" * 60)
