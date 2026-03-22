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

// Two A tiles (q1 + q2) + B + SF + gate buffer
static constexpr int SMEM_TOTAL = 2 * SMEM_A + SMEM_B + SMEM_SFA + SMEM_SFB + SMEM_GATE + 256;

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
    // Two A tiles: A_q1 (token at row 0) and A_q2 (token at row 8)
    uint8_t* s_A1  = reinterpret_cast<uint8_t*>(smem);               // 512 bytes
    uint8_t* s_A2  = s_A1 + SMEM_A;                                   // 512 bytes
    uint8_t* s_B   = s_A2 + SMEM_A;                                   // 2048 bytes
    uint8_t* s_SFA = s_B + SMEM_B;                                    // 64 bytes
    uint8_t* s_SFB = s_SFA + SMEM_SFA;                                // 128 bytes
    float*   s_gate = reinterpret_cast<float*>(s_SFB + SMEM_SFB);     // 16384 bytes

    // 8 N-passes: 4 gate (weight rows 0-255) + 4 up (weight rows 256-511)
    // Each pass: BN=64 B-rows, 8 warps × 8 MMA cols = 64 output cols
    // Using dual-quadrant MMA (2 MMAs per K-iter) to get all 8 cols per warp
    const int N_PASSES = GATE_UP / BN;  // 8

    for (int n_pass = 0; n_pass < N_PASSES; n_pass++) {
        const int n_off = n_pass * BN;
        const bool is_gate = (n_off < GATE_UP / 2);

        // Two accumulators: q1 for B-rows 0-3 (cols 0-3), q2 for B-rows 4-7 (cols 4-7)
        float acc_q1[4] = {0, 0, 0, 0};
        float acc_q2[4] = {0, 0, 0, 0};

        for (int ki = 0; ki < HIDDEN / BK; ki++) {
            const int k_off = ki * BK;

            // Load A_q1: token at row 0 (rows 1-15 zero)
            for (int i = tid; i < SMEM_A; i += BLOCK_SIZE) {
                int row = i / (BK / 2), col = i % (BK / 2);
                s_A1[i] = (row < M) ? input_fp4[row * (HIDDEN / 2) + k_off / 2 + col] : 0;
            }
            // Load A_q2: token at row 8 (rows 0-7 zero, row 8 = token data)
            for (int i = tid; i < SMEM_A; i += BLOCK_SIZE) {
                int row = i / (BK / 2), col = i % (BK / 2);
                int src_row = row - 8;  // row 8 maps to token 0
                s_A2[i] = (src_row >= 0 && src_row < M) ?
                    input_fp4[src_row * (HIDDEN / 2) + k_off / 2 + col] : 0;
            }
            // Load B: [64 rows, 32 bytes/row]
            for (int i = tid; i < SMEM_B; i += BLOCK_SIZE) {
                int row = i / (BK / 2), col = i % (BK / 2);
                s_B[i] = weight_fp4[(n_off + row) * (HIDDEN / 2) + k_off / 2 + col];
            }
            // SFA
            if (tid < BK / SF_BLOCK)
                s_SFA[tid] = input_sf[k_off / SF_BLOCK + tid];
            // SFB
            for (int i = tid; i < BN * (BK / SF_BLOCK); i += BLOCK_SIZE) {
                int sf_n = i / (BK / SF_BLOCK), sf_k = i % (BK / SF_BLOCK);
                s_SFB[i] = weight_sf[(n_off + sf_n) * (HIDDEN / SF_BLOCK) + k_off / SF_BLOCK + sf_k];
            }

            __syncthreads();

            if (warp_id < N_MMA_PER_BN) {
                // Load B fragment (same for both quadrant passes)
                uint32_t b_regs[2];
                {
                    int b_base = warp_id * MMA_N * (BK / 2);
                    uint32_t addr = smem_u32(&s_B[b_base + (lane_id % 16) * 16]);
                    ldmatrix_b4x16_x2(b_regs, addr);
                }

                uint16_t sfa_packed = (uint16_t)s_SFA[0] | ((uint16_t)s_SFA[1] << 8);
                int sf_n = warp_id * MMA_N + (lane_id % 4);
                uint16_t sfb_packed = (uint16_t)s_SFB[sf_n * 2] | ((uint16_t)s_SFB[sf_n * 2 + 1] << 8);

                // MMA pass 1: A_q1 (token at row 0) → cols 0-3
                uint32_t a1[4];
                ldmatrix_b4x16_x4(a1, smem_u32(&s_A1[lane_id * 16]));
                mma_mxf4nvf4_m16n8k64(acc_q1, a1, b_regs, acc_q1,
                    (uint32_t)sfa_packed, 0, 0, (uint32_t)sfb_packed, 0, 0);

                // MMA pass 2: A_q2 (token at row 8) → cols 4-7
                uint32_t a2[4];
                ldmatrix_b4x16_x4(a2, smem_u32(&s_A2[lane_id * 16]));

                // SFB for cols 4-7: B-rows 4-7 within this warp's tile
                int sf_n2 = warp_id * MMA_N + 4 + (lane_id % 4);
                uint16_t sfb2 = (uint16_t)s_SFB[sf_n2 * 2] | ((uint16_t)s_SFB[sf_n2 * 2 + 1] << 8);
                mma_mxf4nvf4_m16n8k64(acc_q2, a2, b_regs, acc_q2,
                    (uint32_t)sfa_packed, 0, 0, (uint32_t)sfb2, 0, 0);
            }
            __syncthreads();
        }

        // Output: each warp has 8 output columns (4 from q1, 4 from q2)
        if (warp_id < N_MMA_PER_BN) {
            int c = lane_id % 4;
            int m_group = lane_id / 8;
            // q1: cols 0-3 → weight cols n_off + warp*8 + c
            int col_q1 = is_gate ? (n_off + warp_id * 8 + c) : (n_off + warp_id * 8 + c - GATE_UP / 2);
            // q2: cols 4-7 → weight cols n_off + warp*8 + 4 + c
            int col_q2 = is_gate ? (n_off + warp_id * 8 + 4 + c) : (n_off + warp_id * 8 + 4 + c - GATE_UP / 2);

            if (is_gate) {
                if (m_group < M && col_q1 < INTERMEDIATE)
                    s_gate[m_group * INTERMEDIATE + col_q1] = acc_q1[0];
                if (m_group + 4 < M && col_q1 < INTERMEDIATE)
                    s_gate[(m_group + 4) * INTERMEDIATE + col_q1] = acc_q1[2];
                if (m_group < M && col_q2 < INTERMEDIATE)
                    s_gate[m_group * INTERMEDIATE + col_q2] = acc_q2[0];
                if (m_group + 4 < M && col_q2 < INTERMEDIATE)
                    s_gate[(m_group + 4) * INTERMEDIATE + col_q2] = acc_q2[2];
            } else {
                __syncthreads();
                // SwiGLU: up * silu(gate) for both quadrants
                if (m_group < M && col_q1 < INTERMEDIATE) {
                    float gv = s_gate[m_group * INTERMEDIATE + col_q1];
                    output[m_group * INTERMEDIATE + col_q1] = __float2bfloat16(acc_q1[0] * silu_f(gv));
                }
                if (m_group + 4 < M && col_q1 < INTERMEDIATE) {
                    float gv = s_gate[(m_group + 4) * INTERMEDIATE + col_q1];
                    output[(m_group + 4) * INTERMEDIATE + col_q1] = __float2bfloat16(acc_q1[2] * silu_f(gv));
                }
                if (m_group < M && col_q2 < INTERMEDIATE) {
                    float gv = s_gate[m_group * INTERMEDIATE + col_q2];
                    output[m_group * INTERMEDIATE + col_q2] = __float2bfloat16(acc_q2[0] * silu_f(gv));
                }
                if (m_group + 4 < M && col_q2 < INTERMEDIATE) {
                    float gv = s_gate[(m_group + 4) * INTERMEDIATE + col_q2];
                    output[(m_group + 4) * INTERMEDIATE + col_q2] = __float2bfloat16(acc_q2[2] * silu_f(gv));
                }
            }
        }
        __syncthreads();
    }
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
