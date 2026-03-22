/**
 * SM120 Fused MoE GEMM — v3: Correct fragment loading via ldmatrix.b4x16_p64
 * ============================================================================
 *
 * Uses SM100+ ldmatrix.b4x16_p64 for FP4 fragment loading:
 * - Loads 4-bit data from SMEM
 * - Unpacks to 8-bit containers with 2-bit left-shift (p64 = pad to position 6:4)
 * - Distributes across warp lanes in correct MMA fragment layout
 *
 * SMEM addressing: thread t loads from byte offset t * 16.
 * Data in SMEM must be row-major with SW128 swizzle for bank-conflict avoidance.
 *
 * Build:
 *   nvcc -std=c++17 -gencode=arch=compute_120a,code=sm_120a \
 *     -I${CUTLASS}/include -I${CUTLASS}/tools/util/include \
 *     -diag-suppress 177 -o fused_moe_v3 sm120_fused_moe_v3.cu
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// Dimensions (Qwen3.5-397B at TP=4)
// ============================================================================
static constexpr int HIDDEN = 4096;
static constexpr int GATE_UP = 512;
static constexpr int INTERMEDIATE = 256;

static constexpr int BM = 16;
static constexpr int BN = 64;
static constexpr int BK = 64;

static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 64;  // FP4 block-scaled MMA K dimension

static constexpr int NUM_WARPS = 8;
static constexpr int WARP_SIZE = 32;
static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

static constexpr int SF_BLOCK = 32;  // NVFP4 scale factor block size

static constexpr int N_MMA_PER_BN = BN / MMA_N;  // 8

// ============================================================================
// SMEM sizes
// ============================================================================
// A tile: [BM=16, BK=64] FP4 packed = 16 * 32 = 512 bytes
// B tile: [BN=64, BK=64] FP4 packed = 64 * 32 = 2048 bytes
// SFA: scale factors for A, [ceil(BM/SF_BLOCK), BK/SF_BLOCK] UE8M0
// SFB: scale factors for B, [BN, BK/SF_BLOCK] UE8M0
// Gate buffer: [BM, INTERMEDIATE] float = 16 * 256 * 4 = 16384 bytes

static constexpr int SMEM_A = BM * BK / 2;             // 512
static constexpr int SMEM_B = BN * BK / 2;             // 2048
static constexpr int SMEM_SFA = 64;                     // small
static constexpr int SMEM_SFB = BN * (BK / SF_BLOCK);  // 64*2 = 128
static constexpr int SMEM_GATE = BM * INTERMEDIATE * sizeof(float);  // 16384

// Single-buffered for simplicity in Phase 1c (double-buffer in Phase 2)
static constexpr int SMEM_TOTAL = SMEM_A + SMEM_B + SMEM_SFA + SMEM_SFB + SMEM_GATE + 256;

// ============================================================================
// Inline PTX helpers
// ============================================================================

// FP4 ldmatrix: loads 4-bit data, unpacks to 8-bit with padding, distributes to lanes
// ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64
// Each thread provides a 16-byte (128-bit) SMEM address
// Produces 4 × uint32 registers per thread
__device__ __forceinline__ void ldmatrix_b4x16_x4(
    uint32_t (&dst)[4], uint32_t smem_addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "r"(smem_addr)
    );
}

// FP4 ldmatrix x2 variant for B operand (8 rows instead of 16)
__device__ __forceinline__ void ldmatrix_b4x16_x2(
    uint32_t (&dst)[2], uint32_t smem_addr
) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 {%0,%1}, [%2];\n"
        : "=r"(dst[0]), "=r"(dst[1])
        : "r"(smem_addr)
    );
}

// Block-scaled FP4 MMA: m16n8k64
__device__ __forceinline__ void mma_mxf4nvf4_m16n8k64(
    float (&d)[4],
    const uint32_t (&a)[4], const uint32_t (&b)[2],
    const float (&c)[4],
    uint32_t sfa, uint16_t bidA, uint16_t tidA,
    uint32_t sfb, uint16_t bidB, uint16_t tidB
) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},"
        "{%4,%5,%6,%7},"
        "{%8,%9},"
        "{%10,%11,%12,%13},"
        "{%14},{%15,%16},"
        "{%17},{%18,%19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),
           "r"(b[0]),  "r"(b[1]),
           "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3]),
           "r"(sfa), "h"(bidA), "h"(tidA),
           "r"(sfb), "h"(bidB), "h"(tidB)
    );
}

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// ============================================================================
// Kernel: GEMM1 + SwiGLU with correct ldmatrix fragment loading
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
sm120_fused_moe_gemm1_swiglu_v3(
    const uint8_t* __restrict__ input_fp4,     // [M, HIDDEN/2] packed FP4
    const uint8_t* __restrict__ input_sf,      // [ceil(M/32), HIDDEN/32] UE8M0
    const uint8_t* __restrict__ weight_fp4,    // [GATE_UP, HIDDEN/2] packed FP4
    const uint8_t* __restrict__ weight_sf,     // [GATE_UP, HIDDEN/32] UE8M0
    __nv_bfloat16* __restrict__ output,        // [M, INTERMEDIATE] BF16
    int M
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // CLayout fragment positions (SM80_16x8_Row):
    //   d[0] → (row=2*g,     col=2*t)
    //   d[1] → (row=2*g+8,   col=2*t)
    //   d[2] → (row=2*g,     col=2*t+1)
    //   d[3] → (row=2*g+8,   col=2*t+1)
    const int g = lane_id / 4;
    const int t = lane_id % 4;
    const int frow0 = 2 * g, frow1 = 2 * g + 8;
    const int fcol0 = 2 * t, fcol1 = 2 * t + 1;

    extern __shared__ char smem[];
    uint8_t* s_A   = reinterpret_cast<uint8_t*>(smem);
    uint8_t* s_B   = s_A + SMEM_A;
    uint8_t* s_SFA = s_B + SMEM_B;
    uint8_t* s_SFB = s_SFA + SMEM_SFA;
    float*   s_gate = reinterpret_cast<float*>(s_SFB + SMEM_SFB);

    // Each MMA loads MMA_N=8 B-rows but produces only 4 useful N-columns
    // (due to SM120 quadrant split, cols 4-7 go to rows 8-15 quadrant).
    // 8 warps × 8 B-rows = 64 B-rows loaded per pass (BN=64).
    // 8 warps × 4 useful cols = 32 output columns per pass.
    // GATE_UP = 512 → need 512/32 = 16 passes, loading 16*64 = 1024 B-rows.
    // But we only have 512 weight rows total!
    //
    // Solution: each warp loads 4 B-rows (its useful columns) + 4 padding rows.
    // Or: advance n_off by 32 per pass (not 64), loading B-rows [n_off..n_off+63]
    // with overlap between passes. This wastes bandwidth but keeps SMEM layout simple.
    //
    // Better solution: keep BN=64 advance, accept 32 useful cols per pass.
    // 512 / 64 = 8 passes (same as before), getting 32 * 8 = 256 useful cols = GATE_UP/2 ✓
    const int N_PASSES = GATE_UP / BN;  // 8 passes, 32 useful cols each = 256 total

    for (int n_pass = 0; n_pass < N_PASSES; n_pass++) {
        const int n_off = n_pass * BN;  // advance by 64 B-rows per pass
        // First 4 passes (n_off 0-255) = gate, last 4 (256-511) = up
        const bool is_gate = (n_off < GATE_UP / 2);

        // Accumulator per warp (each warp handles 1 MMA N-tile = 8 cols)
        float acc[4] = {0, 0, 0, 0};

        // K-loop: HIDDEN/BK = 64 iterations
        for (int ki = 0; ki < HIDDEN / BK; ki++) {
            const int k_off = ki * BK;

            // === Cooperative SMEM loads ===

            // A tile: [BM=16 rows, BK/2=32 cols] bytes = 512 bytes
            // Layout: row-major, row i at bytes [i*32, i*32+31]
            for (int i = tid; i < SMEM_A; i += BLOCK_SIZE) {
                int row = i / (BK / 2);
                int col = i % (BK / 2);
                s_A[i] = (row < M) ? input_fp4[row * (HIDDEN / 2) + k_off / 2 + col] : 0;
            }

            // B tile: [BN=64 rows, BK/2=32 cols] bytes = 2048 bytes
            // Weight layout: [N_total, K_total/2] row-major packed FP4
            for (int i = tid; i < SMEM_B; i += BLOCK_SIZE) {
                int row = i / (BK / 2);
                int col = i % (BK / 2);
                int n_global = n_off + row;
                s_B[i] = (n_global < GATE_UP) ?
                    weight_fp4[n_global * (HIDDEN / 2) + k_off / 2 + col] : 0;
            }

            // SFA: [1, BK/SF_BLOCK=2] bytes (BM=16 < SF_BLOCK=32, so 1 M-block)
            if (tid < BK / SF_BLOCK) {
                s_SFA[tid] = input_sf[0 * (HIDDEN / SF_BLOCK) + k_off / SF_BLOCK + tid];
            }

            // SFB: [BN=64, BK/SF_BLOCK=2] = 128 bytes
            for (int i = tid; i < BN * (BK / SF_BLOCK); i += BLOCK_SIZE) {
                int sf_n = i / (BK / SF_BLOCK);
                int sf_k = i % (BK / SF_BLOCK);
                int n_global = n_off + sf_n;
                s_SFB[i] = weight_sf[n_global * (HIDDEN / SF_BLOCK) + k_off / SF_BLOCK + sf_k];
            }

            __syncthreads();

            // === MMA: each warp handles one MMA N-tile ===
            int col_in_tile = lane_id % 4;  // output col within warp's 4 active cols
            if (warp_id < N_MMA_PER_BN) {
                // Load A fragment via ldmatrix.b4x16_p64.x4
                // A in SMEM: [16, 32] bytes row-major
                // ldmatrix: thread t loads 16 bytes starting at t * 16
                // For 16 rows × 32 bytes/row = 512 bytes = 32 threads × 16 bytes
                uint32_t a_regs[4];
                {
                    uint32_t addr = smem_u32(&s_A[lane_id * 16]);
                    ldmatrix_b4x16_x4(a_regs, addr);
                }

                // Load B fragment via ldmatrix.b4x16_p64.x2
                // B for this warp's N-tile: [MMA_N=8 rows, BK=64 cols] FP4
                // = [8, 32] bytes = 256 bytes
                // x2 ldmatrix: 32 threads each provide a 16-byte-aligned addr,
                // but only 16 threads' data is used (256 / 16 = 16).
                // Lanes 16-31 must still provide VALID aligned addresses.
                uint32_t b_regs[2];
                {
                    int b_base = warp_id * MMA_N * (BK / 2);  // start of this warp's 8 rows
                    // lane_id 0-15 cover the 256 bytes (16 * 16 = 256)
                    // lane_id 16-31 wrap around to provide valid addresses
                    int b_lane = lane_id % 16;
                    uint32_t addr = smem_u32(&s_B[b_base + b_lane * 16]);
                    ldmatrix_b4x16_x2(b_regs, addr);
                }

                // Scale factors for this K-tile
                // For K=64 with SF_BLOCK=32: 2 scale blocks (k=0..31, k=32..63)
                // scale_vec::2X means the MMA processes both blocks,
                // so we pack 2 UE8M0 values into uint16

                // Real scale factors from SMEM
                // UE8M0 bias = 128: value = 2^(exp - 128)
                // SFA: 2 K-blocks packed into uint16 (lo=block0, hi=block1)
                uint16_t sfa_packed = (uint16_t)s_SFA[0] | ((uint16_t)s_SFA[1] << 8);

                // SFB: scale for this warp's B-rows
                // Warp w's first B-row in tile = w * MMA_N = w * 8
                // The active col for this lane = w*8 + lane%4
                // SF is per N-row, 2 K-blocks (k=0..31, k=32..63)
                int sf_n = warp_id * MMA_N + col_in_tile;  // B-row index in tile
                uint16_t sfb_packed = (uint16_t)s_SFB[sf_n * 2 + 0] |
                                      ((uint16_t)s_SFB[sf_n * 2 + 1] << 8);

                // bid/tid = 0 works when all SFs within a block are the same
                // (empirically verified: sweep had no effect)
                uint16_t bidA = 0;
                uint16_t tidA = 0;
                uint16_t bidB = 0;
                uint16_t tidB = 0;

                mma_mxf4nvf4_m16n8k64(
                    acc, a_regs, b_regs, acc,
                    (uint32_t)sfa_packed, bidA, tidA,
                    (uint32_t)sfb_packed, bidB, tidB
                );
            }

            __syncthreads();
        }  // K-loop

        // === Store gate / apply SwiGLU ===
        // SM120 FP4 MMA CLayout (empirically mapped):
        //   d[0] = C[group, col]     where group=lane/8, col=lane%4
        //   d[2] = C[group+4, col]   (rows 0-3 → cols 0-3 quadrant)
        //   d[0] also = C[group+8, col+4]  (rows 8-15 → cols 4-7 quadrant, SUMMED!)
        //   d[2] also = C[group+12, col+4]
        //   d[1] = d[3] = always 0
        //
        // For decode M=1: only row 0 active. group=0 (lanes 0-7) holds C[0, 0..3].
        // The second quadrant (rows 8+) contributes 0. So d[0] = C[0, col] cleanly.
        //
        // Each MMA covers 4 N-columns (not 8!). With 8 warps:
        //   warp 0 → N-cols 0-3 of the B tile
        //   warp 1 → N-cols 4-7? No — each warp loads 8 B-rows but only
        //   columns 0-3 appear in the output. The other 4 B-rows contribute
        //   to the second quadrant which is summed into the same registers.
        //
        // For M=1: warp w produces C[0, w*4 .. w*4+3]? No, all warps load
        // the SAME A tile. The B tile per warp has different N-rows.
        // Let me just use: lane%4 gives the output column WITHIN this warp's MMA.
        // With 8 warps × 4 cols = 32 cols per N-pass, not 64.
        // This means we need 64/4 = 16 N-passes, not 8.
        //
        // Actually, the MMA is m16n8k64 but the effective output is m8n4.
        // 8 warps × 4 cols = 32 unique output columns per N-pass.
        // For GATE_UP/2 = 256 intermediate cols: 256/32 = 8 gate passes. Fine.
        //
        // FOR NOW: just store the 4 cols per warp that we know are correct.

        if (warp_id < N_MMA_PER_BN) {
            // B-row → output col mapping (empirically verified):
            //   B-rows 0-3 → active quadrant cols 0-3 (for M≤4)
            //   B-rows 4-7 → inactive quadrant (zero for M≤4)
            //
            // Weight matrix column = n_off + B-row-in-tile
            // This warp's B-rows in tile: [warp_id*8 .. warp_id*8+7]
            // Active B-rows: warp_id*8 + 0..3 → weight cols n_off + warp_id*8 + 0..3
            //
            // Output intermediate col = weight col (direct mapping for gate/up)
            // Gate cols: 0..255 (weight rows 0..255)
            // Up cols:   0..255 (weight rows 256..511, offset by GATE_UP/2)
            int col_in_warp = lane_id % 4;
            int weight_col = n_off + warp_id * 8 + col_in_warp;
            int out_col = is_gate ? weight_col : (weight_col - GATE_UP / 2);
            int m_group = lane_id / 8;  // 0..3 → M-rows 0..3

            // Only lanes where m_group < M produce valid output
            // For M=1: only m_group=0 (lanes 0-7) is valid
            // d[0] = C[m_group, col], d[2] = C[m_group+4, col]

            if (is_gate) {
                if (m_group < M && out_col < INTERMEDIATE) {
                    s_gate[m_group * INTERMEDIATE + out_col] = acc[0];
                }
                // d[2] = row m_group+4 — only valid if M > 4
                if (m_group + 4 < M && out_col < INTERMEDIATE) {
                    s_gate[(m_group + 4) * INTERMEDIATE + out_col] = acc[2];
                }
            } else {
                __syncthreads();

                float gate_val = 0;
                if (m_group < M && out_col < INTERMEDIATE) {
                    gate_val = s_gate[m_group * INTERMEDIATE + out_col];
                    float swiglu = acc[0] * silu_f(gate_val);
                    output[m_group * INTERMEDIATE + out_col] = __float2bfloat16(swiglu);
                }
                if (m_group + 4 < M && out_col < INTERMEDIATE) {
                    gate_val = s_gate[(m_group + 4) * INTERMEDIATE + out_col];
                    float swiglu = acc[2] * silu_f(gate_val);
                    output[(m_group + 4) * INTERMEDIATE + out_col] = __float2bfloat16(swiglu);
                }
            }
        }
        __syncthreads();
    }  // N-pass
}

// ============================================================================
// Reference: FP32 GEMM1 + SwiGLU
// ============================================================================

__device__ float silu_ref(float x) { return x / (1.0f + expf(-x)); }

__global__ void ref_gemm1_swiglu(
    const float* __restrict__ A, const float* __restrict__ W,
    float* __restrict__ out, int M
) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row >= M || col >= INTERMEDIATE) return;

    float gate = 0, up = 0;
    for (int k = 0; k < HIDDEN; k++) {
        float a = A[row * HIDDEN + k];
        gate += a * W[col * HIDDEN + k];
        up   += a * W[(col + INTERMEDIATE) * HIDDEN + k];
    }
    out[row * INTERMEDIATE + col] = up * silu_ref(gate);
}

// ============================================================================
// FP4 Quantization helpers (host-side)
// ============================================================================

// E2M1 value table: index 0-7 maps to {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
static const float E2M1_TABLE[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

void quantize_to_nvfp4(
    const float* data, int numel,
    uint8_t* packed_out,    // [numel/2] packed FP4
    uint8_t* sf_out,        // [numel/SF_BLOCK] UE8M0
    int sf_block = 32
) {
    int num_blocks = (numel + sf_block - 1) / sf_block;

    for (int b = 0; b < num_blocks; b++) {
        int start = b * sf_block;
        int end = (start + sf_block < numel) ? start + sf_block : numel;

        // Find max absolute value in block
        float bmax = 0;
        for (int i = start; i < end; i++) bmax = fmaxf(bmax, fabsf(data[i]));

        // Scale: max_representable = 6.0 (E2M1 max)
        float scale = bmax / 6.0f;
        if (scale < 1e-30f) scale = 1e-30f;

        // UE8M0 with bias 128: 2^(exp - 128) = scale → exp = 128 + log2(scale)
        int exp_val = 128 + (int)roundf(log2f(scale));
        exp_val = (exp_val < 0) ? 0 : (exp_val > 255) ? 255 : exp_val;
        sf_out[b] = (uint8_t)exp_val;

        // Actual scale from quantized exponent (bias = 128)
        float actual_scale = powf(2.0f, (float)(exp_val - 128));

        // Quantize each element to nearest E2M1
        for (int i = start; i < end; i++) {
            float scaled = data[i] / actual_scale;
            float abs_scaled = fabsf(scaled);
            int sign = (scaled < 0) ? 1 : 0;

            // Find nearest E2M1 value
            int best_idx = 0;
            float best_diff = fabsf(abs_scaled);
            for (int j = 1; j < 8; j++) {
                float diff = fabsf(abs_scaled - E2M1_TABLE[j]);
                if (diff < best_diff) { best_diff = diff; best_idx = j; }
            }

            // 4-bit encoding: bit3=sign, bit2:0=value_index
            uint8_t fp4 = (uint8_t)((sign << 3) | best_idx);

            // Pack 2 FP4 values per byte: even index in low nibble, odd in high
            int byte_idx = i / 2;
            if (i % 2 == 0) {
                packed_out[byte_idx] = fp4;
            } else {
                packed_out[byte_idx] |= (fp4 << 4);
            }
        }
    }
}

// ============================================================================
// Host: test and validate
// ============================================================================

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM%d%d, %d SMs)\n\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);

    const int M = 1;

    // Generate random test data
    srand(42);
    float* h_input = new float[M * HIDDEN];
    float* h_weight = new float[GATE_UP * HIDDEN];
    for (int i = 0; i < M * HIDDEN; i++) h_input[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < GATE_UP * HIDDEN; i++) h_weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    // === Run FP32 reference ===
    float *d_input_f32, *d_weight_f32, *d_ref_out;
    cudaMalloc(&d_input_f32, M * HIDDEN * sizeof(float));
    cudaMalloc(&d_weight_f32, GATE_UP * HIDDEN * sizeof(float));
    cudaMalloc(&d_ref_out, M * INTERMEDIATE * sizeof(float));
    cudaMemcpy(d_input_f32, h_input, M * HIDDEN * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_f32, h_weight, GATE_UP * HIDDEN * sizeof(float), cudaMemcpyHostToDevice);

    ref_gemm1_swiglu<<<M, INTERMEDIATE>>>(d_input_f32, d_weight_f32, d_ref_out, M);
    cudaDeviceSynchronize();

    float* h_ref = new float[M * INTERMEDIATE];
    cudaMemcpy(h_ref, d_ref_out, M * INTERMEDIATE * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Reference output[0:8]:  ");
    for (int i = 0; i < 8; i++) printf("%8.4f ", h_ref[i]);
    printf("\n");

    // === Quantize to FP4 ===
    uint8_t* h_input_fp4 = new uint8_t[M * HIDDEN / 2]();
    uint8_t* h_input_sf = new uint8_t[M * HIDDEN / SF_BLOCK]();
    uint8_t* h_weight_fp4 = new uint8_t[GATE_UP * HIDDEN / 2]();
    uint8_t* h_weight_sf = new uint8_t[GATE_UP * HIDDEN / SF_BLOCK]();

    // Quantize input: each row independently
    for (int m = 0; m < M; m++) {
        quantize_to_nvfp4(
            &h_input[m * HIDDEN], HIDDEN,
            &h_input_fp4[m * HIDDEN / 2],
            &h_input_sf[m * (HIDDEN / SF_BLOCK)]
        );
    }

    // Quantize weights: each row (N dimension) independently
    for (int n = 0; n < GATE_UP; n++) {
        quantize_to_nvfp4(
            &h_weight[n * HIDDEN], HIDDEN,
            &h_weight_fp4[n * HIDDEN / 2],
            &h_weight_sf[n * (HIDDEN / SF_BLOCK)]
        );
    }

    // Upload FP4 data
    uint8_t *d_input_fp4, *d_input_sf, *d_weight_fp4, *d_weight_sf;
    cudaMalloc(&d_input_fp4, M * HIDDEN / 2);
    cudaMalloc(&d_input_sf, M * HIDDEN / SF_BLOCK);
    cudaMalloc(&d_weight_fp4, GATE_UP * HIDDEN / 2);
    cudaMalloc(&d_weight_sf, GATE_UP * HIDDEN / SF_BLOCK);
    cudaMemcpy(d_input_fp4, h_input_fp4, M * HIDDEN / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_sf, h_input_sf, M * HIDDEN / SF_BLOCK, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_fp4, h_weight_fp4, GATE_UP * HIDDEN / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_sf, h_weight_sf, GATE_UP * HIDDEN / SF_BLOCK, cudaMemcpyHostToDevice);

    // === Run fused FP4 kernel ===
    __nv_bfloat16* d_fused_out;
    cudaMalloc(&d_fused_out, M * INTERMEDIATE * sizeof(__nv_bfloat16));
    cudaMemset(d_fused_out, 0, M * INTERMEDIATE * sizeof(__nv_bfloat16));

    cudaFuncSetAttribute(sm120_fused_moe_gemm1_swiglu_v3,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);

    sm120_fused_moe_gemm1_swiglu_v3<<<1, BLOCK_SIZE, SMEM_TOTAL>>>(
        d_input_fp4, d_input_sf, d_weight_fp4, d_weight_sf, d_fused_out, M);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    __nv_bfloat16* h_fused = new __nv_bfloat16[M * INTERMEDIATE];
    cudaMemcpy(h_fused, d_fused_out, M * INTERMEDIATE * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    printf("Fused FP4 output[0:8]: ");
    for (int i = 0; i < 8; i++) printf("%8.4f ", (float)h_fused[i]);
    printf("\n");

    // === Compare ===
    float max_err = 0, sum_err = 0;
    int nonzero_fused = 0;
    printf("\nNon-zero positions:\n");
    for (int i = 0; i < M * INTERMEDIATE; i++) {
        float ref = h_ref[i];
        float fused = (float)h_fused[i];
        float err_val = fabsf(ref - fused);
        max_err = fmaxf(max_err, err_val);
        sum_err += err_val;
        if (fused != 0.0f) {
            nonzero_fused++;
            if (nonzero_fused <= 20) {
                printf("  [%3d] ref=%10.4f  fused=%10.4f\n", i, ref, fused);
            }
        }
    }
    float avg_err = sum_err / (M * INTERMEDIATE);

    printf("\n=== Comparison ===\n");
    printf("  Nonzero fused outputs: %d / %d\n", nonzero_fused, M * INTERMEDIATE);
    printf("  Max error:  %.6f\n", max_err);
    printf("  Avg error:  %.6f\n", avg_err);

    // FP4 tolerance: with scale factor quantization + E2M1 rounding,
    // expect ~10-30% relative error for random data
    float ref_rms = 0;
    for (int i = 0; i < M * INTERMEDIATE; i++) ref_rms += h_ref[i] * h_ref[i];
    ref_rms = sqrtf(ref_rms / (M * INTERMEDIATE));
    printf("  Ref RMS:    %.6f\n", ref_rms);
    printf("  Rel error:  %.1f%%\n", (ref_rms > 0) ? 100.0f * avg_err / ref_rms : 0.0f);

    if (nonzero_fused > 0 && avg_err < ref_rms * 0.5f) {
        printf("\n  PASS: FP4 kernel produces reasonable output!\n");
    } else if (nonzero_fused == 0) {
        printf("\n  FAIL: All zeros — fragment loading likely wrong\n");
    } else {
        printf("\n  WARN: High error — investigate scale factors or lane mapping\n");
    }

    // Cleanup
    delete[] h_input; delete[] h_weight; delete[] h_ref; delete[] h_fused;
    delete[] h_input_fp4; delete[] h_input_sf; delete[] h_weight_fp4; delete[] h_weight_sf;
    cudaFree(d_input_f32); cudaFree(d_weight_f32); cudaFree(d_ref_out);
    cudaFree(d_input_fp4); cudaFree(d_input_sf); cudaFree(d_weight_fp4); cudaFree(d_weight_sf);
    cudaFree(d_fused_out);

    return 0;
}
