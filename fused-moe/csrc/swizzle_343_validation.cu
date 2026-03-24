/**
 * Sprint 4 Task 1: Swizzle<3,4,3> SMEM Layout Validation
 *
 * Tests the complete pipeline:
 *   Swizzled SMEM write -> ldmatrix (standard m8n8.x4.b16) -> MMA m16n8k64 -> FP32 output
 *
 * The SMEM layout is derived from the CuTe ALayout/BLayout in mma_traits_sm120.hpp:
 *   ALayout: (T32,V32) -> (M16,K64)  Shape<(4,8),(8,2,2)> Stride<(128,1),(16,8,512)>
 *   BLayout: (T32,V16) -> (N8,K64)   Shape<(4,8),(8,2)>   Stride<(64,1),(8,256)>
 *
 * For SM75_U32x4_LDSM_N: thread tid reads SMEM[tid*16..tid*16+15] = 4 uint32 registers.
 * Data must be pre-arranged in SMEM so that each thread's 16 bytes match the ALayout mapping.
 *
 * Build:
 *   nvcc -std=c++17 -O2 -arch=sm_120a -rdc=true -o swizzle_343_validation swizzle_343_validation.cu
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

// ============================================================================
// Constants
// ============================================================================
static constexpr int BM = 16;
static constexpr int BN = 8;
static constexpr int BK = 64;
static constexpr int SF_BLOCK = 32;

// SMEM sizes (packed FP4, no P64 padding)
static constexpr int SMEM_A = BM * (BK / 2);   // 16 * 32 = 512 bytes
static constexpr int SMEM_B = BN * (BK / 2);    // 8 * 32 = 256 bytes

// E2M1 value table (unsigned magnitudes)
static const float E2M1_TABLE[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// ============================================================================
// Swizzle<3,4,3> function
// ============================================================================
// From CUTLASS cute/swizzle.hpp:
//   BBits=3, MBase=4, SShift=3
//   yyy_msk = 0b111 << 7 = bits[9:7]
//   zzz_msk = 0b111 << 4 = bits[6:4]
//   apply(offset) = offset ^ shiftr(offset & yyy_msk, msk_sft)
//                 = offset ^ ((offset >> 3) & 0x70)
//
// Self-inverse: apply(apply(x)) == x
__device__ __host__ __forceinline__ uint32_t swizzle_343(uint32_t byte_offset) {
    return byte_offset ^ ((byte_offset >> 3) & 0x70u);
}

// ============================================================================
// ALayout inverse mapping: (M, K) -> SMEM byte address + nibble position
// ============================================================================
// ALayout: flat = t0*128 + t1 + v0*16 + v1*8 + v2*512
//   where flat = M*64 + K
//   t0 = tid%4 in [0,4), t1 = tid/4 in [0,8)
//   v0 in [0,8), v1 in [0,2), v2 in [0,2)
//   tid = t0 + t1*4,  val = v0 + v1*8 + v2*16
//   SMEM byte = tid*16 + val/2,  nibble = val%2 (0=low, 1=high)
struct SmemAddr {
    int byte_addr;
    int nibble;  // 0 = low 4 bits, 1 = high 4 bits
};

__device__ __host__ __forceinline__ SmemAddr a_layout_inv(int m, int k) {
    int flat = m * 64 + k;
    int v2 = flat / 512;
    int rem = flat - v2 * 512;
    int t0 = rem / 128;
    rem -= t0 * 128;
    int t1 = rem % 8;
    int rest = rem / 8;
    int v1 = rest % 2;
    int v0 = rest / 2;

    int tid = t0 + t1 * 4;
    int val = v0 + v1 * 8 + v2 * 16;
    return {tid * 16 + val / 2, val % 2};
}

// ============================================================================
// BLayout inverse mapping: (N, K) -> SMEM byte address + nibble position
// ============================================================================
// BLayout: flat = t0*64 + t1 + v0*8 + v1*256
//   where flat = N*64 + K
//   t0 in [0,4), t1 in [0,8), v0 in [0,8), v1 in [0,2)
//   tid = t0 + t1*4,  val = v0 + v1*8
//   For ldmatrix x2: 16 unique rows (SM75_U32x2_LDSM_N)
//     SrcLayout: Shape<(16,2),128> Stride<(128,0),1>
//     thread tid reads from SMEM byte (tid%16)*16
//   BUT: we use x2 which means 2 registers per thread, 256 bytes total.
//   Threads 0-15 provide unique addresses, threads 16-31 duplicate.
//
// For simplicity with x2 ldmatrix: B SMEM is 256 bytes.
// Thread (tid%16) reads from byte offset (tid%16)*16.
// But the register mapping must account for the x2 broadcast pattern.
//
// Alternative: for B we can use direct register packing (proven correct)
// and only validate A with ldmatrix.
__device__ __host__ __forceinline__ SmemAddr b_layout_inv(int n, int k) {
    int flat = n * 64 + k;
    int v1 = flat / 256;
    int rem = flat - v1 * 256;
    int t0 = rem / 64;
    rem -= t0 * 64;
    int t1 = rem % 8;
    int v0 = rem / 8;

    int tid = t0 + t1 * 4;
    int val = v0 + v1 * 8;

    // For ldmatrix x2: SMEM address uses (tid%16)*16 as base
    // Thread tid's read position in B SMEM = (tid%16)*16
    // Each thread reads 16 bytes = 2 uint32 registers = 16 nibbles
    // The B SMEM mapping: thread tid%16 maps to a specific data arrangement
    int smem_byte = (tid % 16) * 16 + val / 2;
    return {smem_byte, val % 2};
}

// ============================================================================
// Helper: write a nibble to SMEM with optional swizzle
// ============================================================================
__device__ __host__ __forceinline__
void write_nibble(uint8_t* smem, int byte_addr, int nibble_pos, uint8_t value, bool use_swizzle) {
    int phys_addr = use_swizzle ? (int)swizzle_343((uint32_t)byte_addr) : byte_addr;
    if (nibble_pos == 0)
        smem[phys_addr] = (smem[phys_addr] & 0xF0) | (value & 0x0F);
    else
        smem[phys_addr] = (smem[phys_addr] & 0x0F) | ((value & 0x0F) << 4);
}

// ============================================================================
// PTX helpers
// ============================================================================
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Standard ldmatrix m8n8.x4.b16 (SM75_U32x4_LDSM_N)
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&d)[4], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(addr));
}

// Standard ldmatrix m8n8.x2.b16 (SM75_U32x2_LDSM_N)
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&d)[2], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(d[0]), "=r"(d[1])
        : "r"(addr));
}

// FP4 b4x16_p64 ldmatrix variants (MXF8F6F4 path, for comparison)
__device__ __forceinline__ void ldmatrix_b4_x4(uint32_t (&d)[4], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_b4_x2(uint32_t (&d)[2], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 {%0,%1}, [%2];\n"
        : "=r"(d[0]), "=r"(d[1])
        : "r"(addr));
}

// MMA: mxf4nvf4 block_scale scale_vec::2X m16n8k64 ue8m0
__device__ __forceinline__ void mma_nvf4(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
    const float (&c)[4], uint32_t sfa, uint32_t sfb)
{
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},"
        "{%4,%5,%6,%7},"
        "{%8,%9},"
        "{%10,%11,%12,%13},"
        "{%14},{%15,%16},"
        "{%17},{%18,%19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
          "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

// ============================================================================
// Manual register packing (proven correct from analysis)
// ============================================================================
__device__ __forceinline__ uint32_t get_nibble(const uint8_t* data, int k) {
    uint8_t byte = data[k / 2];
    return (k & 1) ? ((byte >> 4) & 0xFu) : (byte & 0xFu);
}

// Pack A for M=1 (only row 0 has data, rest zero)
// s_A: [16, 32] row-major packed FP4
__device__ __forceinline__ void pack_a_manual(
    uint32_t (&a)[4], const uint8_t* s_A, int lane_id)
{
    a[0] = a[1] = a[2] = a[3] = 0;
    if (lane_id % 4 == 0) {
        int t1 = lane_id / 4;
        a[0] = get_nibble(s_A, t1)
             | (get_nibble(s_A, t1 + 16) << 4)
             | (get_nibble(s_A, t1 + 32) << 8)
             | (get_nibble(s_A, t1 + 48) << 12);
        a[1] = get_nibble(s_A, t1 + 8)
             | (get_nibble(s_A, t1 + 24) << 4)
             | (get_nibble(s_A, t1 + 40) << 8)
             | (get_nibble(s_A, t1 + 56) << 12);
    }
}

// Pack B: s_B_row0 points to first of 8 N-rows, each 32 bytes
__device__ __forceinline__ void pack_b_manual(
    uint32_t (&b)[2], const uint8_t* s_B, int lane_id)
{
    int t0 = lane_id % 4;
    int t1 = lane_id / 4;
    const uint8_t* row_n0 = s_B + t0 * (BK / 2);
    const uint8_t* row_n4 = s_B + (t0 + 4) * (BK / 2);
    b[0] = b[1] = 0;
    #pragma unroll
    for (int vi = 0; vi < 8; vi++) {
        int k = t1 + vi * 8;
        b[0] |= get_nibble(row_n0, k) << (vi * 4);
        b[1] |= get_nibble(row_n4, k) << (vi * 4);
    }
}

// ============================================================================
// Host: FP4 quantization with UE8M0 scales (bias=127)
// ============================================================================
void host_quantize_fp4(const float* data, int numel, uint8_t* packed, uint8_t* sf) {
    int num_blocks = (numel + SF_BLOCK - 1) / SF_BLOCK;
    memset(packed, 0, (numel + 1) / 2);

    for (int b = 0; b < num_blocks; b++) {
        int start = b * SF_BLOCK;
        int end = start + SF_BLOCK;
        if (end > numel) end = numel;

        float bmax = 0;
        for (int i = start; i < end; i++)
            bmax = fmaxf(bmax, fabsf(data[i]));

        float scale = bmax / 6.0f;
        if (scale < 1e-30f) scale = 1e-30f;

        int exp_val = 127 + (int)ceilf(log2f(scale));
        exp_val = (exp_val < 1) ? 1 : (exp_val > 254) ? 254 : exp_val;
        sf[b] = (uint8_t)exp_val;
        float actual_scale = powf(2.0f, (float)(exp_val - 127));

        for (int i = start; i < end; i++) {
            float scaled = data[i] / actual_scale;
            float av = fabsf(scaled);
            int sign = (scaled < 0) ? 1 : 0;
            int idx = 0;
            float best = av;
            for (int j = 1; j < 8; j++) {
                float diff = fabsf(av - E2M1_TABLE[j]);
                if (diff < best) { best = diff; idx = j; }
            }
            uint8_t fp4 = (uint8_t)((sign << 3) | idx);
            int byte_idx = i / 2;
            if (i % 2 == 0)
                packed[byte_idx] = (packed[byte_idx] & 0xF0) | fp4;
            else
                packed[byte_idx] = (packed[byte_idx] & 0x0F) | (fp4 << 4);
        }
    }
}

float host_dequant(const uint8_t* packed, const uint8_t* sf, int idx) {
    int byte_idx = idx / 2;
    uint8_t nib = (idx & 1) ? (packed[byte_idx] >> 4) : (packed[byte_idx] & 0xF);
    int sign = (nib >> 3) & 1;
    int mag = nib & 7;
    float val = E2M1_TABLE[mag];
    float scale = powf(2.0f, (float)((int)sf[idx / SF_BLOCK] - 127));
    return sign ? -val * scale : val * scale;
}

// ============================================================================
// Host: FP32 reference for single MMA tile (M=16, N=8, K=64)
// ============================================================================
void host_mma_reference(
    const uint8_t* a_fp4, const uint8_t* a_sf,   // A: [M=16, K=64] packed
    const uint8_t* b_fp4, const uint8_t* b_sf,   // B: [N=8, K=64] packed
    float* output)                                 // output: [M=16, N=8]
{
    for (int m = 0; m < BM; m++) {
        for (int n = 0; n < BN; n++) {
            float sum = 0;
            for (int k = 0; k < BK; k++) {
                float a_val = host_dequant(a_fp4 + m * (BK/2), a_sf + m * (BK/SF_BLOCK), k);
                float b_val = host_dequant(b_fp4 + n * (BK/2), b_sf + n * (BK/SF_BLOCK), k);
                sum += a_val * b_val;
            }
            output[m * BN + n] = sum;
        }
    }
}

// ============================================================================
// KERNEL: Test MMA with multiple SMEM loading strategies
// ============================================================================
// Test 0: Manual register packing (baseline, known correct)
// Test 1: ldmatrix m8n8.x4.b16 with ALayout-ordered SMEM (no swizzle)
// Test 2: ldmatrix m8n8.x4.b16 with ALayout-ordered SMEM + Swizzle<3,4,3>
// Test 3: ldmatrix b4x16_p64 with P64-padded SMEM + Swizzle<3,4,3>
//
// For all tests, B uses manual packing (proven correct) to isolate A loading.
__global__ void __launch_bounds__(32, 1)
test_mma_loading(
    const uint8_t* __restrict__ g_a_fp4,    // [16, 32] row-major packed
    const uint8_t* __restrict__ g_a_sf,     // [16, 2] UE8M0 scales
    const uint8_t* __restrict__ g_b_fp4,    // [8, 32] row-major packed
    const uint8_t* __restrict__ g_b_sf,     // [8, 2] UE8M0 scales
    float* __restrict__ results,            // [4 tests * 4 accum] output
    uint32_t* __restrict__ reg_dump)        // [4 tests * 32 threads * 4 regs] debug
{
    int lid = threadIdx.x;

    // SMEM: 4 different A layouts + 1 B layout
    __shared__ __align__(1024) uint8_t s_A_rowmaj[SMEM_A];      // Test 0: row-major
    __shared__ __align__(1024) uint8_t s_A_alayout[SMEM_A];     // Test 1: ALayout order
    __shared__ __align__(1024) uint8_t s_A_swizzled[SMEM_A];    // Test 2: ALayout + swizzle
    __shared__ __align__(1024) uint8_t s_A_p64[SMEM_A * 2];     // Test 3: P64 padded + swizzle
    __shared__ __align__(1024) uint8_t s_B[SMEM_B];             // B: row-major
    __shared__ __align__(16)   uint8_t s_SFA[BM * (BK / SF_BLOCK)];  // 16*2 = 32 bytes
    __shared__ __align__(16)   uint8_t s_SFB[BN * (BK / SF_BLOCK)];  // 8*2 = 16 bytes

    // --- Load data to SMEM ---
    // Row-major A for manual packing
    for (int i = lid; i < SMEM_A; i += 32)
        s_A_rowmaj[i] = g_a_fp4[i];

    // Row-major B
    for (int i = lid; i < SMEM_B; i += 32)
        s_B[i] = g_b_fp4[i];

    // Scale factors
    if (lid < BM * (BK / SF_BLOCK))
        s_SFA[lid] = g_a_sf[lid];
    if (lid < BN * (BK / SF_BLOCK))
        s_SFB[lid] = g_b_sf[lid];

    // Zero the layout-ordered buffers
    for (int i = lid; i < SMEM_A; i += 32) {
        s_A_alayout[i] = 0;
        s_A_swizzled[i] = 0;
    }
    for (int i = lid; i < SMEM_A * 2; i += 32)
        s_A_p64[i] = 0;

    __syncthreads();

    // --- Build ALayout-ordered SMEM (single thread for correctness) ---
    if (lid == 0) {
        for (int m = 0; m < BM; m++) {
            for (int k = 0; k < BK; k++) {
                // Get FP4 nibble from row-major source
                int src_byte = m * (BK / 2) + k / 2;
                uint8_t src = g_a_fp4[src_byte];
                uint8_t nibble = (k & 1) ? ((src >> 4) & 0xF) : (src & 0xF);

                // Compute ALayout SMEM position
                SmemAddr addr = a_layout_inv(m, k);

                // Test 1: ALayout order, no swizzle
                write_nibble(s_A_alayout, addr.byte_addr, addr.nibble, nibble, false);

                // Test 2: ALayout order + Swizzle<3,4,3>
                write_nibble(s_A_swizzled, addr.byte_addr, addr.nibble, nibble, true);
            }
        }

        // Test 3: P64-padded layout
        // P64: each group of 16 FP4 elements = 8 data bytes + 8 padding bytes = 16 bytes
        // For A (M=16, K=64): 16 rows × 4 groups × 16 bytes = 1024 bytes
        // Layout: row m, group g -> offset = m*64 + g*16, data at bytes 0-7, pad at 8-15
        for (int m = 0; m < BM; m++) {
            for (int k = 0; k < BK; k++) {
                int src_byte = m * (BK / 2) + k / 2;
                uint8_t src = g_a_fp4[src_byte];
                uint8_t nibble = (k & 1) ? ((src >> 4) & 0xF) : (src & 0xF);

                // P64 group: 16 FP4 elements per group
                int group = k / 16;
                int k_in_group = k % 16;
                int data_byte = k_in_group / 2;  // 0..7 within group
                int nib_pos = k_in_group % 2;

                // P64 SMEM offset: row * 64 + group * 16 + data_byte
                int p64_offset = m * 64 + group * 16 + data_byte;

                // Apply Swizzle<3,4,3> to the P64 offset
                int swizzled = (int)swizzle_343((uint32_t)p64_offset);

                if (nib_pos == 0)
                    s_A_p64[swizzled] = (s_A_p64[swizzled] & 0xF0) | (nibble & 0x0F);
                else
                    s_A_p64[swizzled] = (s_A_p64[swizzled] & 0x0F) | ((nibble & 0x0F) << 4);
            }
        }
    }
    __syncthreads();

    // --- Pack B registers (manual, all tests use same B) ---
    uint32_t b_regs[2];
    pack_b_manual(b_regs, s_B, lid);

    // SFA: for M=1, only row 0 matters -> 2 UE8M0 bytes
    // For full M=16 test, each thread needs its own row's SF
    // Use row 0's SF for simplicity (matches M=1 decode case)
    uint32_t sfa = (uint32_t)((uint16_t)s_SFA[0] | ((uint16_t)s_SFA[1] << 8));

    // SFB: per-N-column (lanes 0-3 use same SF)
    int sf_n = lid % 4;  // N-column: 0-3 for b[0], 4-7 for b[1]
    // Actually for scale_vec::2X, the SFB applies to 32 K elements per column
    // Each N-column has 2 SF bytes -> pack into uint16
    uint32_t sfb = (uint32_t)((uint16_t)s_SFB[sf_n * 2] |
                              ((uint16_t)s_SFB[sf_n * 2 + 1] << 8));

    // ====================================================================
    // TEST 0: Manual packing (baseline)
    // ====================================================================
    {
        uint32_t a_regs[4];
        pack_a_manual(a_regs, s_A_rowmaj, lid);
        float acc[4] = {0, 0, 0, 0};
        mma_nvf4(acc, a_regs, b_regs, acc, sfa, sfb);

        // Store register dump
        for (int r = 0; r < 4; r++)
            reg_dump[0 * 32 * 4 + lid * 4 + r] = a_regs[r];

        // CLayout SM80_16x8_Row: (T32,V4) -> (M16,N8)
        // Thread t: t0=t%4, t1=t/4
        // d[0]: M = t0*2 + 2*(0/2) = t0*2, N = t1 + 8*(0%2) = t1
        //   Wait, SM80_16x8_Row specifics...
        // For M=1 input: only threads with t0=0 have non-zero M=0 output
        // d[0] at those threads: M=0, N=t1
        if (lid % 4 == 0) {
            int n_col = lid / 4;
            results[0 * 8 + n_col] = acc[0];
        }
    }
    __syncthreads();

    // ====================================================================
    // TEST 1: ldmatrix m8n8.x4.b16, ALayout-ordered SMEM, no swizzle
    // ====================================================================
    {
        uint32_t a_regs[4];
        uint32_t addr = smem_u32(&s_A_alayout[lid * 16]);
        ldmatrix_x4(a_regs, addr);
        float acc[4] = {0, 0, 0, 0};
        mma_nvf4(acc, a_regs, b_regs, acc, sfa, sfb);

        for (int r = 0; r < 4; r++)
            reg_dump[1 * 32 * 4 + lid * 4 + r] = a_regs[r];

        if (lid % 4 == 0) {
            int n_col = lid / 4;
            results[1 * 8 + n_col] = acc[0];
        }
    }
    __syncthreads();

    // ====================================================================
    // TEST 2: ldmatrix m8n8.x4.b16, ALayout-ordered SMEM + Swizzle<3,4,3>
    // ====================================================================
    {
        uint32_t a_regs[4];
        // Read from swizzled address: both write and read use swizzle
        uint32_t base_addr = lid * 16;
        uint32_t swiz_addr = swizzle_343(base_addr);
        ldmatrix_x4(a_regs, smem_u32(&s_A_swizzled[swiz_addr]));
        float acc[4] = {0, 0, 0, 0};
        mma_nvf4(acc, a_regs, b_regs, acc, sfa, sfb);

        for (int r = 0; r < 4; r++)
            reg_dump[2 * 32 * 4 + lid * 4 + r] = a_regs[r];

        if (lid % 4 == 0) {
            int n_col = lid / 4;
            results[2 * 8 + n_col] = acc[0];
        }
    }
    __syncthreads();

    // ====================================================================
    // TEST 3: ldmatrix b4x16_p64, P64-padded SMEM + Swizzle<3,4,3>
    // ====================================================================
    {
        uint32_t a_regs[4];
        // P64: each thread reads 16 bytes (8 data + 8 pad) from s_A_p64
        // Address: thread reads from swizzled offset
        uint32_t p64_base = lid * 16;  // 32 threads * 16 = 512 bytes (half of 1024)
        uint32_t p64_swiz = swizzle_343(p64_base);
        ldmatrix_b4_x4(a_regs, smem_u32(&s_A_p64[p64_swiz]));
        float acc[4] = {0, 0, 0, 0};
        mma_nvf4(acc, a_regs, b_regs, acc, sfa, sfb);

        for (int r = 0; r < 4; r++)
            reg_dump[3 * 32 * 4 + lid * 4 + r] = a_regs[r];

        if (lid % 4 == 0) {
            int n_col = lid / 4;
            results[3 * 8 + n_col] = acc[0];
        }
    }
}

// ============================================================================
// KERNEL: Probe ldmatrix data routing (fills SMEM with address pattern)
// ============================================================================
__global__ void probe_ldmatrix_routing(uint32_t* output) {
    __shared__ __align__(1024) uint8_t smem[512];

    int lid = threadIdx.x;

    // Fill SMEM: each byte = its address (0x00..0xFF, wraps at 256)
    for (int i = lid; i < 512; i += 32)
        smem[i] = (uint8_t)(i & 0xFF);
    __syncthreads();

    // Standard ldmatrix x4: thread reads from lid*16
    uint32_t regs[4];
    ldmatrix_x4(regs, smem_u32(&smem[lid * 16]));

    // Dump registers
    for (int r = 0; r < 4; r++)
        output[lid * 4 + r] = regs[r];
}

// ============================================================================
// Main: run tests and compare
// ============================================================================
int main() {
    printf("=== Sprint 4 Task 1: Swizzle<3,4,3> SMEM Layout Validation ===\n\n");

    // ----------------------------------------------------------------
    // Step 1: Probe ldmatrix data routing
    // ----------------------------------------------------------------
    printf("--- Step 1: Probing ldmatrix m8n8.x4.b16 data routing ---\n");
    {
        uint32_t* d_out;
        cudaMalloc(&d_out, 32 * 4 * sizeof(uint32_t));

        probe_ldmatrix_routing<<<1, 32>>>(d_out);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("PROBE FAILED: %s\n", cudaGetErrorString(err));
            printf("(ldmatrix m8n8.x4.b16 may not be supported on this GPU)\n\n");
        } else {
            uint32_t h_out[128];
            cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);

            printf("Thread 0 registers (SMEM[0..15]):\n");
            for (int r = 0; r < 4; r++) {
                printf("  reg[%d] = 0x%08X  bytes: ", r, h_out[r]);
                for (int b = 0; b < 4; b++)
                    printf("%02X ", (h_out[r] >> (b*8)) & 0xFF);
                printf("\n");
            }

            // Check if ldmatrix does identity mapping (each thread keeps its bytes)
            bool identity = true;
            for (int t = 0; t < 32 && identity; t++) {
                for (int r = 0; r < 4; r++) {
                    uint32_t expected = 0;
                    for (int b = 0; b < 4; b++) {
                        int smem_byte = t * 16 + r * 4 + b;
                        expected |= ((uint32_t)(smem_byte & 0xFF)) << (b * 8);
                    }
                    if (h_out[t * 4 + r] != expected) {
                        identity = false;
                        printf("  Thread %d reg[%d]: got 0x%08X, expected 0x%08X (identity)\n",
                               t, r, h_out[t * 4 + r], expected);
                    }
                }
            }
            printf("ldmatrix identity mapping: %s\n", identity ? "YES" : "NO (has transpose)");

            // If not identity, dump the full routing table
            if (!identity) {
                printf("\nFull routing for first 8 threads:\n");
                for (int t = 0; t < 8; t++) {
                    printf("  T%d: ", t);
                    for (int r = 0; r < 4; r++) {
                        printf("r%d=[", r);
                        for (int b = 0; b < 4; b++) {
                            uint8_t val = (h_out[t * 4 + r] >> (b * 8)) & 0xFF;
                            printf("%d", val);
                            if (b < 3) printf(",");
                        }
                        printf("] ");
                    }
                    printf("\n");
                }
            }
        }
        cudaFree(d_out);
    }
    printf("\n");

    // ----------------------------------------------------------------
    // Step 2: Create test data
    // ----------------------------------------------------------------
    printf("--- Step 2: Creating test data ---\n");

    // Test case: A = all 1.0 (M=1, row 0 only), B = all 1.0
    // With UE8M0 scale = 0x7F (2^0 = 1.0), expected: 1.0 * 1.0 * 64 = 64.0
    const int a_bytes = BM * (BK / 2);  // 512
    const int b_bytes = BN * (BK / 2);  // 256
    const int a_sf_bytes = BM * (BK / SF_BLOCK);  // 32
    const int b_sf_bytes = BN * (BK / SF_BLOCK);  // 16

    // Generate float data, quantize
    float* h_a_float = new float[BM * BK]();  // zeros
    float* h_b_float = new float[BN * BK];

    // A: row 0 = all 1.0, rest = 0
    for (int k = 0; k < BK; k++)
        h_a_float[0 * BK + k] = 1.0f;

    // B: all 1.0
    for (int i = 0; i < BN * BK; i++)
        h_b_float[i] = 1.0f;

    uint8_t* h_a_fp4 = new uint8_t[a_bytes]();
    uint8_t* h_a_sf = new uint8_t[a_sf_bytes]();
    uint8_t* h_b_fp4 = new uint8_t[b_bytes]();
    uint8_t* h_b_sf = new uint8_t[b_sf_bytes]();

    for (int m = 0; m < BM; m++)
        host_quantize_fp4(&h_a_float[m * BK], BK, &h_a_fp4[m * (BK/2)], &h_a_sf[m * (BK/SF_BLOCK)]);
    for (int n = 0; n < BN; n++)
        host_quantize_fp4(&h_b_float[n * BK], BK, &h_b_fp4[n * (BK/2)], &h_b_sf[n * (BK/SF_BLOCK)]);

    // FP32 reference
    float h_ref[BM * BN];
    host_mma_reference(h_a_fp4, h_a_sf, h_b_fp4, h_b_sf, h_ref);

    printf("FP32 reference (M=0 row, all 1.0 inputs):\n");
    printf("  C[0,0..7] = ");
    for (int n = 0; n < BN; n++) printf("%.1f ", h_ref[0 * BN + n]);
    printf("\n");

    // Verify quantization
    printf("  A[0,0] dequant = %.3f, A[0,1] = %.3f\n",
           host_dequant(h_a_fp4, h_a_sf, 0),
           host_dequant(h_a_fp4, h_a_sf, 1));
    printf("  B[0,0] dequant = %.3f, B[0,1] = %.3f\n",
           host_dequant(h_b_fp4, h_b_sf, 0),
           host_dequant(h_b_fp4, h_b_sf, 1));
    printf("  A SF[0]=%d SF[1]=%d, B SF[0]=%d SF[1]=%d\n",
           h_a_sf[0], h_a_sf[1], h_b_sf[0], h_b_sf[1]);

    // ----------------------------------------------------------------
    // Step 3: Run MMA tests
    // ----------------------------------------------------------------
    printf("\n--- Step 3: Running MMA loading tests ---\n");

    uint8_t *d_a_fp4, *d_a_sf, *d_b_fp4, *d_b_sf;
    float* d_results;
    uint32_t* d_reg_dump;

    cudaMalloc(&d_a_fp4, a_bytes);
    cudaMalloc(&d_a_sf, a_sf_bytes);
    cudaMalloc(&d_b_fp4, b_bytes);
    cudaMalloc(&d_b_sf, b_sf_bytes);
    cudaMalloc(&d_results, 4 * 8 * sizeof(float));
    cudaMalloc(&d_reg_dump, 4 * 32 * 4 * sizeof(uint32_t));

    cudaMemcpy(d_a_fp4, h_a_fp4, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_sf, h_a_sf, a_sf_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_fp4, h_b_fp4, b_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_sf, h_b_sf, b_sf_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_results, 0, 4 * 8 * sizeof(float));
    cudaMemset(d_reg_dump, 0, 4 * 32 * 4 * sizeof(uint32_t));

    int smem_size = SMEM_A * 4 + SMEM_A * 2 + SMEM_B + 32 + 16 + 256;
    cudaFuncSetAttribute(test_mma_loading,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    test_mma_loading<<<1, 32>>>(d_a_fp4, d_a_sf, d_b_fp4, d_b_sf, d_results, d_reg_dump);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("KERNEL FAILED: %s\n", cudaGetErrorString(err));
        // Try without the ldmatrix tests (they may not be supported)
        printf("Note: Standard ldmatrix m8n8.x4.b16 may require specific SM arch\n");
    }

    float h_results[32];
    uint32_t h_reg_dump[4 * 32 * 4];
    cudaMemcpy(h_results, d_results, sizeof(h_results), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_reg_dump, d_reg_dump, sizeof(h_reg_dump), cudaMemcpyDeviceToHost);

    const char* test_names[] = {
        "Manual pack (baseline)",
        "ldmatrix x4 + ALayout SMEM (no swizzle)",
        "ldmatrix x4 + ALayout SMEM + Swizzle<3,4,3>",
        "ldmatrix b4x16_p64 + P64 SMEM + Swizzle<3,4,3>"
    };

    for (int test = 0; test < 4; test++) {
        printf("\nTest %d: %s\n", test, test_names[test]);
        printf("  MMA output C[0,n] = ");
        for (int n = 0; n < 8; n++)
            printf("%.1f ", h_results[test * 8 + n]);
        printf("\n");
        printf("  Reference  C[0,n] = ");
        for (int n = 0; n < 8; n++)
            printf("%.1f ", h_ref[0 * BN + n]);
        printf("\n");

        // Check match
        bool match = true;
        float max_err = 0;
        for (int n = 0; n < 8; n++) {
            float err_val = fabsf(h_results[test * 8 + n] - h_ref[0 * BN + n]);
            if (err_val > max_err) max_err = err_val;
            if (err_val > 1e-3f) match = false;
        }
        printf("  %s (max_err=%.4f)\n", match ? "PASS" : "FAIL", max_err);

        // Dump thread 0 registers for debugging
        printf("  Thread 0 A regs: ");
        for (int r = 0; r < 4; r++)
            printf("0x%08X ", h_reg_dump[test * 32 * 4 + 0 * 4 + r]);
        printf("\n");
    }

    // ----------------------------------------------------------------
    // Step 4: Test with random data
    // ----------------------------------------------------------------
    printf("\n--- Step 4: Random data test ---\n");

    srand(42);
    for (int i = 0; i < BM * BK; i++)
        h_a_float[i] = 0;  // Keep M=1 (only row 0)
    for (int k = 0; k < BK; k++)
        h_a_float[0 * BK + k] = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
    for (int i = 0; i < BN * BK; i++)
        h_b_float[i] = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;

    // Re-quantize
    memset(h_a_fp4, 0, a_bytes);
    memset(h_b_fp4, 0, b_bytes);
    for (int m = 0; m < BM; m++)
        host_quantize_fp4(&h_a_float[m * BK], BK, &h_a_fp4[m * (BK/2)], &h_a_sf[m * (BK/SF_BLOCK)]);
    for (int n = 0; n < BN; n++)
        host_quantize_fp4(&h_b_float[n * BK], BK, &h_b_fp4[n * (BK/2)], &h_b_sf[n * (BK/SF_BLOCK)]);

    host_mma_reference(h_a_fp4, h_a_sf, h_b_fp4, h_b_sf, h_ref);

    cudaMemcpy(d_a_fp4, h_a_fp4, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_sf, h_a_sf, a_sf_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_fp4, h_b_fp4, b_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_sf, h_b_sf, b_sf_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_results, 0, 4 * 8 * sizeof(float));
    cudaMemset(d_reg_dump, 0, 4 * 32 * 4 * sizeof(uint32_t));

    test_mma_loading<<<1, 32>>>(d_a_fp4, d_a_sf, d_b_fp4, d_b_sf, d_results, d_reg_dump);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Random test FAILED: %s\n", cudaGetErrorString(err));
    } else {
        cudaMemcpy(h_results, d_results, 4 * 8 * sizeof(float), cudaMemcpyDeviceToHost);

        printf("Random data reference C[0,0..7]: ");
        for (int n = 0; n < 8; n++) printf("%.2f ", h_ref[0 * BN + n]);
        printf("\n");

        for (int test = 0; test < 4; test++) {
            bool match = true;
            float max_err = 0;
            for (int n = 0; n < 8; n++) {
                float e = fabsf(h_results[test * 8 + n] - h_ref[0 * BN + n]);
                if (e > max_err) max_err = e;
                if (e > 0.5f) match = false;  // FP4 tolerance
            }
            printf("  Test %d (%s): %s (max_err=%.3f)\n",
                   test, test_names[test], match ? "PASS" : "FAIL", max_err);
        }
    }

    // Cleanup
    delete[] h_a_float; delete[] h_b_float;
    delete[] h_a_fp4; delete[] h_a_sf;
    delete[] h_b_fp4; delete[] h_b_sf;
    cudaFree(d_a_fp4); cudaFree(d_a_sf);
    cudaFree(d_b_fp4); cudaFree(d_b_sf);
    cudaFree(d_results); cudaFree(d_reg_dump);

    printf("\n=== Validation Complete ===\n");
    return 0;
}
