/**
 * CLayout Mapping Probe v2 — Isolate individual C[m,n] elements
 * ==============================================================
 *
 * A = all zeros except row r (all 1.0)
 * B = all zeros except row c (all 1.0)
 * Only C[r,c] is non-zero → whichever (lane, reg_idx) lights up IS the mapping
 *
 * Tests (r,c) pairs: sweep all 16 M-rows × 8 N-cols = 128 combinations
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 64;
static constexpr int WARP_SIZE = 32;

static constexpr int SMEM_A = MMA_M * MMA_K / 2;  // 512 bytes
static constexpr int SMEM_B = MMA_N * MMA_K / 2;   // 256 bytes
static constexpr int SMEM_TOTAL = SMEM_A + SMEM_B + 256;

__device__ __forceinline__ void ldmatrix_b4x16_x4(
    uint32_t (&dst)[4], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_b4x16_x2(
    uint32_t (&dst)[2], uint32_t addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 {%0,%1}, [%2];\n"
        : "=r"(dst[0]), "=r"(dst[1]) : "r"(addr));
}

__device__ __forceinline__ uint32_t to_smem(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__global__ void clayout_probe(
    float* __restrict__ output,  // [32 lanes × 4 regs]
    int active_row,              // which M-row of A is non-zero
    int active_col               // which N-row of B is non-zero
) {
    const int lane = threadIdx.x;
    if (lane >= WARP_SIZE) return;

    extern __shared__ char smem[];
    uint8_t* s_A = reinterpret_cast<uint8_t*>(smem);
    uint8_t* s_B = s_A + SMEM_A;

    // Zero everything
    for (int i = lane; i < SMEM_A + SMEM_B; i += WARP_SIZE) {
        reinterpret_cast<uint8_t*>(smem)[i] = 0;
    }
    __syncwarp();

    // Set active_row in A to all 1.0 (E2M1 index 2 = 0b0010)
    // A layout: [16 rows, 32 bytes/row] = 512 bytes
    // Row r starts at byte r*32, spans 32 bytes (64 FP4 values)
    // E2M1 1.0 packed: 0x22 per byte (two 1.0 values)
    if (active_row < MMA_M) {
        int row_start = active_row * (MMA_K / 2);  // row * 32
        for (int i = lane; i < MMA_K / 2; i += WARP_SIZE) {
            s_A[row_start + i] = 0x22;  // 1.0 | 1.0
        }
    }

    // Set active_col in B to all 1.0
    // B layout: [8 rows, 32 bytes/row] = 256 bytes
    if (active_col < MMA_N) {
        int row_start = active_col * (MMA_K / 2);
        for (int i = lane; i < MMA_K / 2; i += WARP_SIZE) {
            s_B[row_start + i] = 0x22;
        }
    }
    __syncwarp();

    // Load fragments
    uint32_t a_regs[4];
    ldmatrix_b4x16_x4(a_regs, to_smem(&s_A[lane * 16]));

    uint32_t b_regs[2];
    ldmatrix_b4x16_x2(b_regs, to_smem(&s_B[(lane % 16) * 16]));

    // MMA with SF=1.0
    float acc[4] = {0, 0, 0, 0};
    uint16_t sf = 0x7F7F;  // both K-blocks = 1.0
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
        :  "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]),
           "r"(b_regs[0]), "r"(b_regs[1]),
           "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f),
           "r"((uint32_t)sf), "h"((uint16_t)0), "h"((uint16_t)0),
           "r"((uint32_t)sf), "h"((uint16_t)0), "h"((uint16_t)0));

    output[lane * 4 + 0] = acc[0];
    output[lane * 4 + 1] = acc[1];
    output[lane * 4 + 2] = acc[2];
    output[lane * 4 + 3] = acc[3];
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("SM%d%d CLayout Mapping Probe\n", prop.major, prop.minor);
    printf("MMA: mxf4nvf4.block_scale.scale_vec::2X.m16n8k64\n\n");

    float* d_out;
    cudaMalloc(&d_out, 128 * sizeof(float));
    float h_out[128];

    // Map: for each (M-row, N-col), find which (lane, reg) holds the result
    printf("=== Full CLayout Map: C[r,c] → (lane, reg_idx) ===\n");
    printf("%-6s", "r\\c");
    for (int c = 0; c < MMA_N; c++) printf("  col%-2d       ", c);
    printf("\n");

    // Store the mapping
    int map_lane[MMA_M][MMA_N];
    int map_reg[MMA_M][MMA_N];
    memset(map_lane, -1, sizeof(map_lane));
    memset(map_reg, -1, sizeof(map_reg));

    for (int r = 0; r < MMA_M; r++) {
        printf("row%2d:", r);
        for (int c = 0; c < MMA_N; c++) {
            clayout_probe<<<1, WARP_SIZE, SMEM_TOTAL>>>(d_out, r, c);
            cudaDeviceSynchronize();
            cudaMemcpy(h_out, d_out, 128 * sizeof(float), cudaMemcpyDeviceToHost);

            // Find which (lane, reg) is non-zero
            int found_lane = -1, found_reg = -1;
            float found_val = 0;
            int num_nonzero = 0;
            for (int l = 0; l < 32; l++) {
                for (int ri = 0; ri < 4; ri++) {
                    if (h_out[l * 4 + ri] != 0.0f) {
                        if (found_lane == -1) {
                            found_lane = l;
                            found_reg = ri;
                            found_val = h_out[l * 4 + ri];
                        }
                        num_nonzero++;
                    }
                }
            }

            if (num_nonzero == 1) {
                printf(" L%02d.d%d=%4.0f", found_lane, found_reg, found_val);
            } else if (num_nonzero > 1) {
                // Multiple non-zeros — print first and count
                printf(" L%02d.d%d*%d ", found_lane, found_reg, num_nonzero);
            } else {
                printf("     ZERO    ");
            }

            map_lane[r][c] = found_lane;
            map_reg[r][c] = found_reg;
        }
        printf("\n");
    }

    // Summary
    printf("\n=== Mapping Summary ===\n");
    printf("Format: C[row,col] → lane.d[reg]\n\n");
    for (int r = 0; r < MMA_M; r++) {
        for (int c = 0; c < MMA_N; c++) {
            if (map_lane[r][c] >= 0) {
                printf("C[%2d,%d] → lane %2d . d[%d]\n", r, c, map_lane[r][c], map_reg[r][c]);
            }
        }
    }

    // Check d[0] vs d[1] duplication
    printf("\n=== d[0] vs d[1] duplication check ===\n");
    int d0_count = 0, d1_count = 0, d2_count = 0, d3_count = 0;
    for (int r = 0; r < MMA_M; r++) {
        for (int c = 0; c < MMA_N; c++) {
            if (map_reg[r][c] == 0) d0_count++;
            if (map_reg[r][c] == 1) d1_count++;
            if (map_reg[r][c] == 2) d2_count++;
            if (map_reg[r][c] == 3) d3_count++;
        }
    }
    printf("d[0]: %d elements, d[1]: %d elements, d[2]: %d elements, d[3]: %d elements\n",
           d0_count, d1_count, d2_count, d3_count);
    printf("Total unique mappings: %d (expect 128 for m16n8)\n",
           d0_count + d1_count + d2_count + d3_count);

    cudaFree(d_out);
    return 0;
}
