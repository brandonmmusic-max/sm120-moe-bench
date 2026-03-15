"""
Reverse-engineer TMA SWIZZLE_128B pattern.

Creates known data where element [row, col] = row * 1000 + col.
Loads via TMA with SWIZZLE_128B.
Reads SMEM linearly to see where each element ended up.
Derives the swizzle formula.
"""
import torch
import ctypes
import subprocess
import struct

# Build probe kernel
subprocess.run(
    "nvcc -O3 -gencode=arch=compute_120a,code=sm_120a "
    "--ptxas-options=-v -shared "
    "-o /tmp/sm120-fa/tma_swizzle_probe.so "
    "/tmp/sm120-fa/csrc/tma_swizzle_probe.cu "
    "--compiler-options '-fPIC' -lcuda",
    shell=True, check=True, capture_output=True
)

lib = ctypes.CDLL("/tmp/sm120-fa/tma_swizzle_probe.so")
cuda_drv = ctypes.CDLL("libcuda.so")

HEAD_DIM = 128
BLOCK_N = 16

# Create input data: element [row, col] = row * 1000 + col
data = torch.zeros(BLOCK_N, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
for r in range(BLOCK_N):
    for c in range(HEAD_DIM):
        data[r, c] = float(r * 1000 + c)

# Create TMA descriptor with SWIZZLE_128B
desc = (ctypes.c_uint64 * 16)()  # CUtensorMap is 16 uint64s

dims = (ctypes.c_uint64 * 2)(HEAD_DIM, BLOCK_N)
strides = (ctypes.c_uint64 * 1)(HEAD_DIM * 2)  # bytes
box = (ctypes.c_uint32 * 2)(HEAD_DIM, BLOCK_N)
elem_strides = (ctypes.c_uint32 * 2)(1, 1)

err = cuda_drv.cuTensorMapEncodeTiled(
    ctypes.byref(desc),
    1,   # CU_TENSOR_MAP_DATA_TYPE_UINT16 (bfloat16 = 2 bytes = uint16)
    2,   # rank
    ctypes.c_void_p(data.data_ptr()),
    dims,
    strides,
    box,
    elem_strides,
    0,   # CU_TENSOR_MAP_INTERLEAVE_NONE
    3,   # CU_TENSOR_MAP_SWIZZLE_128B
    0,   # CU_TENSOR_MAP_L2_PROMOTION_NONE
    0,   # CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
)
print(f"cuTensorMapEncodeTiled: err={err}")

# Copy descriptor to device
d_desc = torch.zeros(16, dtype=torch.int64, device="cuda")
for i in range(16):
    d_desc[i] = desc[i]

# Output buffer
output = torch.zeros(BLOCK_N * HEAD_DIM, dtype=torch.float32, device="cuda")

lib.run_swizzle_probe(
    ctypes.c_void_p(d_desc.data_ptr()),
    ctypes.c_void_p(output.data_ptr()),
    ctypes.c_void_p(0)
)
torch.cuda.synchronize()

# Analyze the swizzle pattern
out = output.cpu().numpy()
print(f"\nSMEM layout after TMA SWIZZLE_128B:")
print(f"Expected: element [row, col] = row*1000 + col")
print()

# For each SMEM position i, determine which (row, col) it contains
swizzle_map = {}  # smem_idx → (row, col)
for i in range(BLOCK_N * HEAD_DIM):
    val = out[i]
    if val == 0 and i > 0:
        continue
    row = int(round(val)) // 1000
    col = int(round(val)) % 1000
    if 0 <= row < BLOCK_N and 0 <= col < HEAD_DIM:
        smem_idx = i
        logical_row = row
        logical_col = col
        # Linear address for this element would be row * HEAD_DIM + col
        linear = row * HEAD_DIM + col
        swizzle_map[smem_idx] = (logical_row, logical_col, linear)

# Print first 32 entries
print("SMEM idx → (row, col, linear_addr) | XOR = smem_idx ^ linear")
for i in range(min(256, len(swizzle_map))):
    if i in swizzle_map:
        r, c, lin = swizzle_map[i]
        xor_val = i ^ lin
        byte_xor = (i * 2) ^ (lin * 2)
        print(f"  [{i:4d}] → ({r:2d}, {c:3d}) linear={lin:5d} XOR={xor_val:5d} byte_XOR={byte_xor:5d}")

# Derive the swizzle formula
print("\n\nDeriving swizzle formula...")
print("Checking: does smem_byte_addr = linear_byte_addr ^ f(row)?")
for row in range(min(8, BLOCK_N)):
    # Find entries for this row
    row_entries = [(idx, r, c, lin) for idx, (r, c, lin) in swizzle_map.items() if r == row]
    if row_entries:
        # Check what f(row) is
        xors = set()
        for idx, r, c, lin in row_entries[:8]:
            byte_idx = idx * 2
            byte_lin = lin * 2
            xors.add(byte_idx ^ byte_lin)
        if len(xors) == 1:
            f_row = xors.pop()
            print(f"  row={row}: f(row) = {f_row} = 0x{f_row:04x} = bits {bin(f_row)}")
        else:
            print(f"  row={row}: multiple XOR values: {sorted(xors)[:5]}")
            # Check element-level
            for idx, r, c, lin in row_entries[:4]:
                print(f"    [{idx}] → ({r},{c}) lin={lin} elem_xor={idx^lin}")
