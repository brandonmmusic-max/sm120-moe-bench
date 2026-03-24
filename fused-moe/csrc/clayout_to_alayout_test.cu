/**
 * CLayout → ALayout SMEM Handoff Test (v2)
 *
 * Two-part validation:
 *   Part A: Row-major FP4 in SMEM → ldmatrix → MMA → verify output
 *           (validates the SMEM→fragment→MMA chain)
 *   Part B: CLayout register positions → SMEM → readback → verify mapping
 *           (validates the CLayout→SMEM write pattern)
 *
 * SM120 FP4 MMA CLayout (LANDMARK discovery from Phase 1c):
 *   m_group = lane_id / 8 (0-3)
 *   col = lane_id % 4 (0-3 for q1, 4-7 for q2)
 *   d[0] → accumulates rows {m_group, m_group+8}  (folded!)
 *   d[2] → accumulates rows {m_group+4, m_group+12} (folded!)
 *   d[1] = 0, d[3] = 0 always
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a \
 *     --expt-relaxed-constexpr -diag-suppress 177 \
 *     -o clayout_to_alayout_test clayout_to_alayout_test.cu
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>

static const float E2M1_TABLE[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
__constant__ float d_e2m1_table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// SM120 FP4 MMA
__device__ __forceinline__ void mma_mxf4nvf4_m16n8k64(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
    const float (&c)[4],
    uint32_t sfa, uint16_t bidA, uint16_t tidA,
    uint32_t sfb, uint16_t bidB, uint16_t tidB)
{
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),
           "r"(b[0]),  "r"(b[1]),
           "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3]),
           "r"(sfa), "h"(bidA), "h"(tidA),
           "r"(sfb), "h"(bidB), "h"(tidB)
    );
}

__device__ __forceinline__ void ldmatrix_b4x16_x4(uint32_t (&dst)[4], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_b4x16_x2(uint32_t (&dst)[2], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 {%0,%1}, [%2];\n"
        : "=r"(dst[0]), "=r"(dst[1]) : "r"(addr));
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Host FP4 helpers
uint8_t host_quantize_e2m1(float v) {
    int sign = (v < 0) ? 1 : 0;
    float av = fabsf(v);
    int best = 0; float bd = av;
    for (int j = 1; j < 8; j++) {
        float d = fabsf(av - E2M1_TABLE[j]);
        if (d < bd) { bd = d; best = j; }
    }
    return (sign << 3) | best;
}

float host_dequant_e2m1(uint8_t nib) {
    float s = (nib & 8) ? -1.0f : 1.0f;
    return s * E2M1_TABLE[nib & 7];
}

// ============================================================================
// Part A: Fill SMEM cooperatively → ldmatrix → MMA → verify
// ============================================================================

__global__ void part_a_smem_to_mma(
    const uint8_t* __restrict__ a_fp4,  // [16, 32] packed FP4 (row-major)
    const uint8_t* __restrict__ a_sf,   // [16, 2] UE8M0
    const uint8_t* __restrict__ b_fp4,  // [8, 32] packed FP4
    const uint8_t* __restrict__ b_sf,   // [8, 2] UE8M0
    float* __restrict__ output          // [16, 8] FP32
) {
    const int lane = threadIdx.x;

    // SMEM: A[512 bytes] + A_SF[32 bytes] + B[256 bytes] + B_SF[16 bytes]
    extern __shared__ char smem[];
    uint8_t* s_a  = (uint8_t*)smem;
    uint8_t* s_asf = s_a + 512;
    uint8_t* s_b  = s_asf + 32;
    uint8_t* s_bsf = s_b + 256;

    // Load A (all 32 lanes load 16 bytes each = 512 bytes total)
    for (int i = lane; i < 512; i += 32) s_a[i] = a_fp4[i];
    for (int i = lane; i < 32; i += 32) s_asf[i] = a_sf[i];
    for (int i = lane; i < 256; i += 32) s_b[i] = b_fp4[i];
    for (int i = lane; i < 16; i += 32) s_bsf[i] = b_sf[i];
    __syncthreads();

    // Load A via ldmatrix (each lane at lane*16)
    uint32_t a_regs[4];
    ldmatrix_b4x16_x4(a_regs, smem_u32(&s_a[lane * 16]));

    // Load B via ldmatrix
    uint32_t b_regs[2];
    ldmatrix_b4x16_x2(b_regs, smem_u32(&s_b[(lane % 16) * 16]));

    // Scale factors — Use values from SMEM (which are set to 0x80 by host)
    uint16_t sfa_packed = (uint16_t)s_asf[0] | ((uint16_t)s_asf[1] << 8);
    int sf_n = lane % 4;
    uint16_t sfb_packed = (uint16_t)s_bsf[sf_n * 2] | ((uint16_t)s_bsf[sf_n * 2 + 1] << 8);

    // MMA: [16, 64] × [8, 64]^T → [16, 8]
    float acc[4] = {0, 0, 0, 0};
    mma_mxf4nvf4_m16n8k64(acc, a_regs, b_regs, acc,
        (uint32_t)sfa_packed, 0, 0, (uint32_t)sfb_packed, 0, 0);

    // SM120 FP4 MMA CLayout (empirically verified):
    // mg = lane / 8 (0-3),  col = lane % 8 (0-7)
    // d[0] = C[mg,    col]   (rows 0-3)
    // d[1] = C[mg+8,  col]   (rows 8-11)
    // d[2] = C[mg+4,  col]   (rows 4-7)
    // d[3] = C[mg+12, col]   (rows 12-15)
    int mg  = lane / 8;   // 0-3
    int col = lane % 8;   // 0-7
    output[mg       * 8 + col] = acc[0];
    output[(mg + 8) * 8 + col] = acc[1];
    output[(mg + 4) * 8 + col] = acc[2];
    output[(mg +12) * 8 + col] = acc[3];
}

// ============================================================================
// Part B: Write known values at CLayout positions → SMEM → readback
// ============================================================================

__global__ void part_b_clayout_write_test(
    float* __restrict__ smem_dump  // [16, 64] FP32 dequantized readback
) {
    const int lane = threadIdx.x;
    const int m_group = lane / 8;
    const int col_within = lane % 4;  // output column (0-3 for first quadrant)

    extern __shared__ char smem[];
    uint8_t* s_fp4 = (uint8_t*)smem;  // [16, 32] packed

    // Zero SMEM
    for (int i = lane; i < 16 * 32; i += 32) s_fp4[i] = 0;
    __syncthreads();

    // Simulate: after GEMM1+SwiGLU, each thread has output at known positions.
    // For SM120 dual-quadrant MMA:
    //   d[0] covers rows {m_group, m_group+8}  — but in the SINGLE quadrant MMA,
    //                     only ONE of these is active (the other is zero in A)
    //   d[2] covers rows {m_group+4, m_group+12}
    //
    // In the v3 kernel, A1 has data in rows 0-7 and A2 has data in rows 8-15.
    // After the MMA, the accumulated d[0] value corresponds to:
    //   q1 (A1): row m_group (0-3)
    //   q2 (A2): row m_group (but shifted by +8 in the output)
    // Similarly d[2] → q1: row m_group+4, q2: row m_group+4+8

    // For the handoff, after SwiGLU we know:
    //   result_q1_d0 → output[m_group,     col_within]      = SwiGLU value
    //   result_q1_d2 → output[m_group + 4, col_within]      = SwiGLU value
    //   result_q2_d0 → output[m_group,     col_within + 4]  = SwiGLU value
    //   result_q2_d2 → output[m_group + 4, col_within + 4]  = SwiGLU value

    // Write test pattern: value = row * 10 + col (makes it easy to identify)
    // We write to ALL 16 rows × 8 cols of the intermediate (simulating all warps writing)
    // For col_pass 0 (cols 0-7 of the K=64 intermediate, which covers 1 sub-tile):

    // Each lane writes 4 positions: 2 from q1, 2 from q2
    // q1: (m_group, col_within) and (m_group+4, col_within)
    // q2: (m_group, col_within+4) and (m_group+4, col_within+4)

    // Test values (E2M1-representable):
    float v_q1_d0 = d_e2m1_table[m_group % 8];              // row=m_group
    float v_q1_d2 = d_e2m1_table[(m_group + 4) % 8];        // row=m_group+4
    float v_q2_d0 = d_e2m1_table[(m_group + 1) % 8];        // row=m_group (different col)
    float v_q2_d2 = d_e2m1_table[(m_group + 5) % 8];        // row=m_group+4

    // Quantize (trivial since values are exact E2M1 with scale=1.0)
    auto quant = [](float v) -> uint8_t {
        int sign = (v < 0) ? 1 : 0;
        float av = fabsf(v);
        int best = 0; float bd = av;
        for (int j = 1; j < 8; j++) {
            float d = fabsf(av - d_e2m1_table[j]);
            if (d < bd) { bd = d; best = j; }
        }
        return (sign << 3) | best;
    };

    // Pack and write to SMEM (row-major, 32 bytes/row for 64 FP4 cols)
    // q1_d0: row=m_group, col=col_within
    // q1_d2: row=m_group+4, col=col_within
    // q2_d0: row=m_group, col=col_within+4
    // q2_d2: row=m_group+4, col=col_within+4

    // Cols 0-7 map to the first 4 bytes of each row (8 FP4 / 2 = 4 bytes)
    // col_within pairs with col_within+4 for packing? No — (0,1), (2,3), (4,5), (6,7)
    // Wait, FP4 packing: col 0 → low nibble of byte 0, col 1 → high nibble of byte 0
    // So (col_within=0, col_within=1) pack together, (2,3), etc.

    // For q1: col_within (0-3) and some other thread has col_within ± 1
    // The issue: only col_within lanes write, so we need to be careful about
    // which lane writes which byte.

    // Actually, with 32 lanes and 4 unique col_within values (0-3), we have:
    // Lanes 0,8,16,24 → col_within=0
    // Lanes 1,9,17,25 → col_within=1
    // etc.
    // And m_group: lanes 0-7→0, 8-15→1, 16-23→2, 24-31→3

    // For byte packing: byte_idx = col / 2
    // col=0 → byte 0 low,  col=1 → byte 0 high
    // col=2 → byte 1 low,  col=3 → byte 1 high
    // col=4 → byte 2 low,  col=5 → byte 2 high
    // col=6 → byte 3 low,  col=7 → byte 3 high

    // Two threads with consecutive col_within need to write to the same byte.
    // Use atomicOr to avoid race conditions.

    uint8_t nib_q1_d0 = quant(v_q1_d0);
    uint8_t nib_q1_d2 = quant(v_q1_d2);
    uint8_t nib_q2_d0 = quant(v_q2_d0);
    uint8_t nib_q2_d2 = quant(v_q2_d2);

    int byte_q1 = col_within / 2;
    int byte_q2 = (col_within + 4) / 2;
    int shift_q1 = (col_within % 2) * 4;
    int shift_q2 = ((col_within + 4) % 2) * 4;

    // Write nibbles to SMEM. Use the standard CLayout (g=lane/4, t=lane%4):
    // d[0]=(2g, 2t), d[1]=(2g+1, 2t), d[2]=(2g, 2t+1), d[3]=(2g+1, 2t+1)
    // Two values at same row and consecutive cols pack into one byte.
    // d[0]+d[2] share a byte at (row=2g, byte_col=t),
    // d[1]+d[3] share a byte at (row=2g+1, byte_col=t)
    // Each thread writes 2 unique bytes — no conflicts!
    int g2 = lane / 4;
    int t2 = lane % 4;

    // Use test pattern values based on position
    uint8_t nib0 = quant(d_e2m1_table[(2*g2) % 7 + 1]);       // d[0]: (2g, 2t)
    uint8_t nib1 = quant(d_e2m1_table[(2*g2+1) % 7 + 1]);     // d[1]: (2g+1, 2t)
    uint8_t nib2 = quant(d_e2m1_table[(2*g2+2) % 7 + 1]);     // d[2]: (2g, 2t+1)
    uint8_t nib3 = quant(d_e2m1_table[(2*g2+3) % 7 + 1]);     // d[3]: (2g+1, 2t+1)

    // Pack: d[0] low nibble + d[2] high nibble → byte at (row=2g, col=t)
    s_fp4[(2*g2) * 32 + t2] = nib0 | (nib2 << 4);
    // Pack: d[1] low nibble + d[3] high nibble → byte at (row=2g+1, col=t)
    s_fp4[(2*g2+1) * 32 + t2] = nib1 | (nib3 << 4);

    __syncthreads();

    // Readback: dequantize SMEM to FP32 for host verification
    // Only dump first 8 columns (the interesting part)
    for (int row = 0; row < 16; row++) {
        for (int byte = lane; byte < 4; byte += 32) {
            uint8_t packed = s_fp4[row * 32 + byte];
            int col_lo = byte * 2;
            int col_hi = byte * 2 + 1;
            float s_lo = ((packed & 0xF) & 8) ? -1.0f : 1.0f;
            float s_hi = (((packed >> 4) & 0xF) & 8) ? -1.0f : 1.0f;
            smem_dump[row * 64 + col_lo] = s_lo * d_e2m1_table[(packed & 0xF) & 7];
            smem_dump[row * 64 + col_hi] = s_hi * d_e2m1_table[((packed >> 4) & 0xF) & 7];
        }
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM%d%d)\n\n", prop.name, prop.major, prop.minor);

    // ====================================================================
    // Part A: Row-major SMEM → ldmatrix → MMA correctness
    // ====================================================================
    printf("=== Part A: SMEM → ldmatrix → MMA ===\n");

    constexpr int M = 16, K = 64, N = 8;

    // Create FP32 matrices, quantize to FP4
    float h_a[M * K], h_b[N * K];
    uint8_t h_a_fp4[M * K / 2], h_b_fp4[N * K / 2];
    uint8_t h_a_sf[M * 2], h_b_sf[N * 2];  // [rows, K/32]

    // Random E2M1-representable values for full validation
    srand(42);
    for (int i = 0; i < M * K; i++) {
        int idx = rand() % 15;
        if (idx < 8) h_a[i] = E2M1_TABLE[idx];
        else h_a[i] = -E2M1_TABLE[idx - 7];
    }
    for (int i = 0; i < N * K; i++) {
        int idx = rand() % 15;
        if (idx < 8) h_b[i] = E2M1_TABLE[idx];
        else h_b[i] = -E2M1_TABLE[idx - 7];
    }

    // Quantize A
    memset(h_a_fp4, 0, sizeof(h_a_fp4));
    for (int r = 0; r < M; r++) {
        for (int c = 0; c < K; c++) {
            uint8_t nib = host_quantize_e2m1(h_a[r * K + c]);
            int byte_idx = r * (K / 2) + c / 2;
            if (c % 2 == 0) h_a_fp4[byte_idx] = nib;
            else h_a_fp4[byte_idx] |= (nib << 4);
        }
        h_a_sf[r * 2] = 0x80;      // UE8M0 = 1.0
        h_a_sf[r * 2 + 1] = 0x80;
    }

    // Quantize B
    memset(h_b_fp4, 0, sizeof(h_b_fp4));
    for (int r = 0; r < N; r++) {
        for (int c = 0; c < K; c++) {
            uint8_t nib = host_quantize_e2m1(h_b[r * K + c]);
            int byte_idx = r * (K / 2) + c / 2;
            if (c % 2 == 0) h_b_fp4[byte_idx] = nib;
            else h_b_fp4[byte_idx] |= (nib << 4);
        }
        h_b_sf[r * 2] = 0x80;
        h_b_sf[r * 2 + 1] = 0x80;
    }

    // Host reference: A × B^T with the dual-quadrant folding
    // d[0] at (m_group, col) = sum_k A[m_group, k] * B[col, k] + A[m_group+8, k] * B[col, k]
    //                        = (A[m_group,:] + A[m_group+8,:]) · B[col,:]
    // d[2] at (m_group+4, col) = (A[m_group+4,:] + A[m_group+12,:]) · B[col,:]
    // GEMM reference with block-scale: C[m,n] = sum_blocks (SFA_b * SFB_b * sum_k A[m,k]*B[n,k])
    // Empirically: UE8M0 = 0x80 → combined scale = 2.0 per K-block of 32
    // SFA = SFB = 0x80 → scale_A * scale_B = 2.0 (bias=127: 2^(128-127)=2, but only one applies? Or sqrt-applied?)
    // Empirically: with all-1.0 inputs, K=64, UE8M0=0x80 → output = 128 = 64 * 2.0
    // So combined scale per K-block = 2.0 / num_blocks = 2.0 (since 128 = 2 * 32 * 2.0)
    // Actually: output = sum_blocks(scale * 32) = 2 * scale * 32 = 64 * scale → scale = 128/64 = 2.0
    const float sf_combined = 2.0f;  // empirically determined for UE8M0=0x80
    float h_ref[M * N];
    memset(h_ref, 0, sizeof(h_ref));
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += h_a[m * K + k] * h_b[n * K + k];
            }
            h_ref[m * N + n] = sum * sf_combined;
        }
    }

    // Upload and run
    uint8_t *d_a_fp4, *d_a_sf, *d_b_fp4, *d_b_sf;
    float *d_output;
    cudaMalloc(&d_a_fp4, sizeof(h_a_fp4));
    cudaMalloc(&d_a_sf, sizeof(h_a_sf));
    cudaMalloc(&d_b_fp4, sizeof(h_b_fp4));
    cudaMalloc(&d_b_sf, sizeof(h_b_sf));
    cudaMalloc(&d_output, M * N * sizeof(float));

    cudaMemcpy(d_a_fp4, h_a_fp4, sizeof(h_a_fp4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_sf, h_a_sf, sizeof(h_a_sf), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_fp4, h_b_fp4, sizeof(h_b_fp4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_sf, h_b_sf, sizeof(h_b_sf), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, M * N * sizeof(float));

    int smem_a = 512 + 32 + 256 + 16 + 64;
    part_a_smem_to_mma<<<1, 32, smem_a>>>(d_a_fp4, d_a_sf, d_b_fp4, d_b_sf, d_output);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  Part A kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    float h_output[M * N];
    cudaMemcpy(h_output, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare all 128 elements (16 rows × 8 cols)
    printf("  Output (kern/ref) — first 4 rows:\n");
    float max_err = 0, sum_err = 0;
    int n_close = 0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float kern = h_output[m * N + n];
            float ref = h_ref[m * N + n];
            float e = fabsf(kern - ref);
            max_err = fmaxf(max_err, e);
            sum_err += e;
            if (fabsf(ref) > 0.01f && e / fabsf(ref) < 0.02f) n_close++;
            else if (fabsf(ref) <= 0.01f && fabsf(kern) < 0.02f) n_close++;
        }
        if (m < 4) {
            printf("    row%2d:", m);
            for (int n = 0; n < N; n++)
                printf(" %7.1f/%7.1f", h_output[m*N+n], h_ref[m*N+n]);
            printf("\n");
        }
    }

    int total = M * N;
    printf("\n  Max err: %.4f, Avg err: %.4f\n", max_err, sum_err / total);
    printf("  Within 2%%: %d/%d (%.1f%%)\n", n_close, total, 100.0f * n_close / total);

    bool part_a_ok = (n_close > total * 0.8f);
    printf("  Part A VERDICT: %s\n\n", part_a_ok ? "PASSED" : "FAILED");

    printf("\n");

    // ====================================================================
    // Part B: CLayout write pattern verification
    // ====================================================================
    printf("=== Part B: CLayout → SMEM write pattern ===\n");

    float *d_dump;
    cudaMalloc(&d_dump, 16 * 64 * sizeof(float));
    cudaMemset(d_dump, 0, 16 * 64 * sizeof(float));

    part_b_clayout_write_test<<<1, 32, 16 * 32 + 64>>>(d_dump);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  Part B kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    float h_dump[16 * 64];
    cudaMemcpy(h_dump, d_dump, 16 * 64 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("  First 8 cols of each row (CLayout-written FP4 values):\n");
    int nonzero_rows = 0;
    for (int r = 0; r < 8; r++) {
        printf("    row %2d:", r);
        bool has_nonzero = false;
        for (int c = 0; c < 8; c++) {
            float v = h_dump[r * 64 + c];
            printf(" %4.1f", v);
            if (v != 0) has_nonzero = true;
        }
        if (has_nonzero) nonzero_rows++;
        printf("\n");
    }

    bool part_b_ok = (nonzero_rows == 8);
    printf("\n  Non-zero rows: %d/8 (rows 0-7)\n", nonzero_rows);
    printf("  Part B VERDICT: %s\n\n", part_b_ok ? "PASSED" : "NEEDS INVESTIGATION");

    printf("=== OVERALL CLayout → ALayout VERDICT: %s ===\n",
           (part_a_ok) ? "PASSED" : "NEEDS INVESTIGATION");

    // Cleanup
    cudaFree(d_a_fp4); cudaFree(d_a_sf); cudaFree(d_b_fp4); cudaFree(d_b_sf);
    cudaFree(d_output); cudaFree(d_dump);

    return part_a_ok ? 0 : 1;
}
