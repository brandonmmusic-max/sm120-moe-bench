/**
 * SM120 Fused MoE GEMM Kernel — Phase 1c: Single-Expert GEMM1 + SwiGLU
 * ======================================================================
 *
 * Computes gate_up projection + SwiGLU activation for one expert:
 *   GEMM1: [M, K] × [K, N_gate_up] → [M, N_gate_up]   (FP4 × FP4, block-scaled)
 *   SwiGLU: gate * silu(up) → [M, N_intermediate]
 *
 * For Qwen3.5-397B at TP=4:
 *   M = 1-8 (decode), K = 4096, N_gate_up = 512 (256 gate + 256 up)
 *
 * MMA: mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64
 *   - A[4×uint32] = 16 rows × 64 cols FP4 (8 bytes/row × 16 rows = 128B)
 *   - B[2×uint32] = 8 cols × 64 rows FP4 (8 bytes/col × 8 cols = 64B)
 *   - SFA, SFB = UE8M0 block-scale registers (block_size=32 for NVFP4)
 *   - C/D[4×float] = FP32 accumulator
 *
 * This is Phase 1c — single expert, validates MMA + SwiGLU fusion.
 * Phase 2 adds GEMM2 consuming output from SMEM.
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// ============================================================================
// Dimensions (hardcoded for Qwen3.5-397B at TP=4, Phase 1)
// ============================================================================

#define HIDDEN_SIZE 4096      // K dimension for GEMM1
#define GATE_UP_SIZE 512      // N dimension for GEMM1 (256 gate + 256 up)
#define INTERMEDIATE_SIZE 256 // After SwiGLU: gate * silu(up)

// Tile sizes
#define BM 16       // M-tile (covers M=1 decode with padding)
#define BN 64       // N-tile for GEMM1 (8 passes to cover 512)
#define BK 64       // K-tile = MMA K dimension

// MMA dimensions
#define MMA_M 16
#define MMA_N 8
#define MMA_K 64

// Block config
#define NUM_WARPS 8
#define WARP_SIZE 32
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE)

// FP4 packing: 2 values per byte
#define FP4_PACK 2

// Scale factor: block-32 for NVFP4 (scale_vec::2X means SF covers 32 elements)
#define SF_BLOCK_SIZE 32

// ============================================================================
// PTX helpers
// ============================================================================

// FP4 block-scaled MMA: m16n8k64, E2M1 × E2M1 → F32, with UE8M0 block scales
// scale_vec::2X = block size 32
__device__ __forceinline__ void mma_fp4_blockscaled(
    float (&d)[4],
    const uint32_t (&a)[4],
    const uint32_t (&b)[2],
    const float (&c)[4],
    uint32_t sfa, uint16_t bidA, uint16_t tidA,
    uint32_t sfb, uint16_t bidB, uint16_t tidB
) {
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13},"
        "{%14},"
        "{%15, %16},"
        "{%17},"
        "{%18, %19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),
           "r"(b[0]),  "r"(b[1]),
           "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3]),
           "r"(sfa), "h"(bidA), "h"(tidA),
           "r"(sfb), "h"(bidB), "h"(tidB)
    );
}

// FP8×FP4 MMA for GEMM2: m16n8k32, E4M3 × E2M1 → F32 (no block scale)
__device__ __forceinline__ void mma_fp8xfp4(
    float (&d)[4],
    const uint32_t (&a)[4],
    const uint32_t (&b)[2],
    const float (&c)[4]
) {
    asm volatile(
        "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e2m1.f32 "
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),
           "r"(b[0]),  "r"(b[1]),
           "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3])
    );
}

// SiLU activation: x * sigmoid(x)
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// ============================================================================
// SMEM layout
// ============================================================================

// GEMM1 weight tile: BK × BN FP4 packed = 64 × 64 / 2 = 2048 bytes
// Double-buffered: 2 × 2048 = 4096 bytes
#define WEIGHT_TILE_BYTES (BK * BN / FP4_PACK)
#define WEIGHT_SMEM_BYTES (2 * WEIGHT_TILE_BYTES)

// Scale factors for weights: (BK / SF_BLOCK_SIZE) × BN = 2 × 64 = 128 bytes per stage
#define WEIGHT_SF_BYTES (2 * (BK / SF_BLOCK_SIZE) * BN)

// Input tile: BM × BK FP4 packed = 16 × 64 / 2 = 512 bytes
// Double-buffered: 1024 bytes
#define INPUT_TILE_BYTES (BM * BK / FP4_PACK)
#define INPUT_SMEM_BYTES (2 * INPUT_TILE_BYTES)

// Input scale factors: (BM / SF_BLOCK_SIZE) × (BK / SF_BLOCK_SIZE) — small
#define INPUT_SF_BYTES (2 * 64)  // generous estimate

// Gate buffer: BM × (GATE_UP_SIZE/2) × sizeof(float) = 16 × 256 × 4 = 16384 bytes
// Store gate results in FP32 for SwiGLU precision
#define GATE_BUF_BYTES (BM * INTERMEDIATE_SIZE * sizeof(float))

// Total SMEM
#define TOTAL_SMEM_BYTES (WEIGHT_SMEM_BYTES + WEIGHT_SF_BYTES + INPUT_SMEM_BYTES + INPUT_SF_BYTES + GATE_BUF_BYTES)
// = 4096 + 256 + 1024 + 128 + 16384 = ~22KB — well within 99KB

// ============================================================================
// Reference kernel: GEMM1 + SwiGLU using standard loads (no TMA yet)
// ============================================================================
// Phase 1c starts with cp.async loads to validate the MMA + SwiGLU logic.
// TMA loads will be added in Phase 2 optimization.

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
sm120_fused_moe_gemm1_swiglu(
    // Input activations: [M, HIDDEN_SIZE] packed FP4 (uint8, 2 values/byte)
    const uint8_t* __restrict__ input,
    // Input scale factors: [M, HIDDEN_SIZE/SF_BLOCK_SIZE] UE8M0
    const uint8_t* __restrict__ input_sf,
    // Gate-up weights: [GATE_UP_SIZE, HIDDEN_SIZE] packed FP4 (N-major = col-major in TN)
    const uint8_t* __restrict__ gate_up_weight,
    // Weight scale factors: [GATE_UP_SIZE, HIDDEN_SIZE/SF_BLOCK_SIZE] UE8M0
    const uint8_t* __restrict__ gate_up_sf,
    // Output: [M, INTERMEDIATE_SIZE] BF16 (SwiGLU result)
    __nv_bfloat16* __restrict__ output,
    int M  // actual number of tokens (≤ BM)
) {
    // Thread identification
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // This CTA processes one N-tile pass of GEMM1.
    // blockIdx.x selects which N-tile (0..GATE_UP_SIZE/BN-1 = 0..7)
    const int n_tile = blockIdx.x;
    const int n_offset = n_tile * BN;  // Starting N column

    // ====================================================================
    // Shared memory allocation
    // ====================================================================
    extern __shared__ char smem[];

    // Weight tile: [BK, BN] FP4 packed, double-buffered
    uint8_t* w_smem = reinterpret_cast<uint8_t*>(smem);
    // Weight SF: [BK/SF_BLOCK_SIZE, BN] UE8M0, double-buffered
    uint8_t* w_sf_smem = w_smem + WEIGHT_SMEM_BYTES;
    // Input tile: [BM, BK] FP4 packed, double-buffered
    uint8_t* a_smem = w_sf_smem + WEIGHT_SF_BYTES;
    // Input SF
    uint8_t* a_sf_smem = a_smem + INPUT_SMEM_BYTES;
    // Gate buffer: [BM, INTERMEDIATE_SIZE] FP32 — persistent across N-passes
    float* gate_buf = reinterpret_cast<float*>(a_sf_smem + INPUT_SF_BYTES);

    // ====================================================================
    // GEMM1: [BM, HIDDEN_SIZE] × [HIDDEN_SIZE, BN] → [BM, BN]
    // Tile over K in steps of BK=64
    // ====================================================================

    // Accumulators for this N-tile: each thread holds part of [BM, BN]
    // MMA m16n8k64: 16 rows × 8 cols per MMA
    // With BN=64, we need BN/MMA_N = 64/8 = 8 MMA tiles in N
    // With BM=16, we need BM/MMA_M = 1 MMA tile in M
    // Each thread accumulates: 1 (M-tiles) × 8 (N-tiles) × 4 (regs/MMA) = 32 FP32 regs
    const int N_MMA_TILES = BN / MMA_N;  // 8
    float acc[N_MMA_TILES][4];

    // Zero accumulators
    #pragma unroll
    for (int nt = 0; nt < N_MMA_TILES; nt++) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            acc[nt][r] = 0.0f;
        }
    }

    // K-loop: HIDDEN_SIZE / BK = 4096 / 64 = 64 iterations
    const int num_k_iters = HIDDEN_SIZE / BK;

    for (int ki = 0; ki < num_k_iters; ki++) {
        const int k_offset = ki * BK;
        const int stage = ki & 1;  // Double-buffer stage

        // --- Load weight tile [BK=64, BN=64] FP4 packed into SMEM ---
        // Total bytes: 64 × 64 / 2 = 2048 bytes = 128 × 16B lines
        // 256 threads: each loads 2048/256 = 8 bytes
        {
            uint8_t* w_dst = w_smem + stage * WEIGHT_TILE_BYTES;
            const int bytes_per_thread = WEIGHT_TILE_BYTES / BLOCK_SIZE;
            const int offset = tid * bytes_per_thread;
            if (offset + bytes_per_thread <= WEIGHT_TILE_BYTES) {
                // Weight layout: [N, K] packed FP4, we load a [BK, BN] tile
                // Row = k within tile, Col = n within tile
                // Global address: weight[(n_offset + col), (k_offset + row)] packed
                // For simplicity in Phase 1, load contiguously and let the MMA
                // instruction handle the element mapping
                const uint8_t* w_src = gate_up_weight +
                    (n_offset * HIDDEN_SIZE / FP4_PACK) + // N-row offset
                    (k_offset / FP4_PACK);                // K-col offset

                // Simple copy (Phase 1 — will optimize with TMA in Phase 2)
                #pragma unroll
                for (int b = 0; b < bytes_per_thread; b++) {
                    int flat = offset + b;
                    int row = flat / (BN / FP4_PACK);  // K dimension
                    int col = flat % (BN / FP4_PACK);  // N dimension packed
                    w_dst[flat] = w_src[row * (HIDDEN_SIZE / FP4_PACK) + col]; // TODO: fix striding
                }
            }
        }

        // --- Load input tile [BM=16, BK=64] FP4 packed into SMEM ---
        {
            uint8_t* a_dst = a_smem + stage * INPUT_TILE_BYTES;
            const int total_bytes = INPUT_TILE_BYTES;  // 512 bytes
            if (tid < total_bytes) {
                int row = tid / (BK / FP4_PACK);
                int col = tid % (BK / FP4_PACK);
                a_dst[tid] = input[row * (HIDDEN_SIZE / FP4_PACK) + (k_offset / FP4_PACK) + col];
            }
        }

        // --- Load scale factors ---
        // Weight SF: [BK/SF_BLOCK_SIZE, BN] = [2, 64] = 128 bytes
        {
            uint8_t* sf_dst = w_sf_smem + stage * (BK / SF_BLOCK_SIZE * BN);
            if (tid < (BK / SF_BLOCK_SIZE) * BN) {
                int sf_k = tid / BN;
                int sf_n = tid % BN;
                sf_dst[tid] = gate_up_sf[(n_offset + sf_n) * (HIDDEN_SIZE / SF_BLOCK_SIZE) +
                                         (k_offset / SF_BLOCK_SIZE) + sf_k];
            }
        }

        // Input SF: [BM, BK/SF_BLOCK_SIZE]
        {
            uint8_t* sf_dst = a_sf_smem + stage * (BM * BK / SF_BLOCK_SIZE);
            int sf_size = BM * (BK / SF_BLOCK_SIZE);  // 16 × 2 = 32
            if (tid < sf_size) {
                int sf_m = tid / (BK / SF_BLOCK_SIZE);
                int sf_k = tid % (BK / SF_BLOCK_SIZE);
                sf_dst[tid] = input_sf[sf_m * (HIDDEN_SIZE / SF_BLOCK_SIZE) +
                                       (k_offset / SF_BLOCK_SIZE) + sf_k];
            }
        }

        __syncthreads();

        // --- MMA computation ---
        // Each warp computes a portion of the [BM=16, BN=64] output
        // With 8 warps and BN/MMA_N=8 N-tiles: each warp does 1 N-tile
        // (or we can have warps share — for Phase 1, 1 warp per N-tile)

        if (warp_id < N_MMA_TILES) {
            int nt = warp_id;  // This warp's N-tile

            // Load A operand from SMEM (input)
            uint32_t a_regs[4];
            {
                // A layout in SMEM: [BM=16, BK=64] FP4 packed
                // MMA A: 16 rows × 64 cols E2M1 = 16×32 bytes
                // Thread mapping: lane_id determines which row/col pair
                uint8_t* a_ptr = a_smem + stage * INPUT_TILE_BYTES;
                // Simple load: each thread grabs 4 × uint32 = 16 bytes
                // TODO: proper MMA fragment layout
                int base = lane_id * 4;
                if (base + 3 < INPUT_TILE_BYTES / 4) {
                    a_regs[0] = reinterpret_cast<uint32_t*>(a_ptr)[base];
                    a_regs[1] = reinterpret_cast<uint32_t*>(a_ptr)[base + 1];
                    a_regs[2] = reinterpret_cast<uint32_t*>(a_ptr)[base + 2];
                    a_regs[3] = reinterpret_cast<uint32_t*>(a_ptr)[base + 3];
                }
            }

            // Load B operand from SMEM (weight)
            uint32_t b_regs[2];
            {
                uint8_t* w_ptr = w_smem + stage * WEIGHT_TILE_BYTES;
                int n_local = nt * MMA_N;  // Local N offset within tile
                // B: 8 cols × 64 rows E2M1 = 8×32 bytes = 256 bytes
                int base = (n_local * BK / FP4_PACK) / 4 + lane_id * 2;
                b_regs[0] = reinterpret_cast<uint32_t*>(w_ptr)[base];
                b_regs[1] = reinterpret_cast<uint32_t*>(w_ptr)[base + 1];
            }

            // Load scale factors into registers
            uint8_t* w_sf = w_sf_smem + stage * (BK / SF_BLOCK_SIZE * BN);
            uint8_t* a_sf = a_sf_smem + stage * (BM * BK / SF_BLOCK_SIZE);

            // Scale factor A: per block of 32 elements along K
            // Scale factor B: per block of 32 elements along K for each N column
            uint32_t sfa = a_sf[0];  // Simplified — proper indexing in Phase 2
            uint32_t sfb = w_sf[nt * MMA_N];  // Simplified

            uint16_t bidA = 0, tidA = 0;
            uint16_t bidB = 0, tidB = 0;

            // Execute MMA
            mma_fp4_blockscaled(acc[nt], a_regs, b_regs, acc[nt],
                               sfa, bidA, tidA, sfb, bidB, tidB);
        }

        __syncthreads();
    }

    // ====================================================================
    // Store results or apply SwiGLU
    // ====================================================================

    // After all K iterations, acc[nt][4] holds the GEMM1 result for this N-tile.
    // n_tile 0..3 = gate columns (N=0..255)
    // n_tile 4..7 = up columns (N=256..511)

    const bool is_gate = (n_tile < (GATE_UP_SIZE / 2) / BN);  // tiles 0..3
    const int intermediate_col = n_offset - (is_gate ? 0 : GATE_UP_SIZE / 2);

    if (is_gate) {
        // Store gate result to SMEM gate_buf for SwiGLU later
        // Each warp writes its N-tile portion
        if (warp_id < N_MMA_TILES) {
            // MMA output fragment: d[0..3] map to specific (row, col) positions
            // For m16n8: d0 = (group*2, 0..1), d1 = (group*2+8, 0..1),
            //            d2 = (group*2, 2..3), d3 = (group*2+8, 2..3)
            // Thread group g = lane_id / 4, sub-position t = lane_id % 4
            int g = lane_id / 4;
            int t = lane_id % 4;

            // Row positions for this thread
            int row0 = g * 2;
            int row1 = g * 2 + 8;
            // Col positions
            int col0 = warp_id * MMA_N + t * 2;
            int col1 = col0 + 1;

            if (row0 < M) {
                gate_buf[row0 * INTERMEDIATE_SIZE + intermediate_col + col0] = acc[warp_id][0];
                if (col1 < BN) gate_buf[row0 * INTERMEDIATE_SIZE + intermediate_col + col1] = acc[warp_id][2];
            }
            if (row1 < M) {
                gate_buf[row1 * INTERMEDIATE_SIZE + intermediate_col + col0] = acc[warp_id][1];
                if (col1 < BN) gate_buf[row1 * INTERMEDIATE_SIZE + intermediate_col + col1] = acc[warp_id][3];
            }
        }
    } else {
        // This is an "up" tile — apply SwiGLU with corresponding gate from gate_buf
        __syncthreads();  // Ensure gate_buf is written

        if (warp_id < N_MMA_TILES) {
            int g = lane_id / 4;
            int t = lane_id % 4;

            int row0 = g * 2;
            int row1 = g * 2 + 8;
            int col0 = warp_id * MMA_N + t * 2;
            int col1 = col0 + 1;

            // Load gate values from SMEM
            float gate0 = (row0 < M) ? gate_buf[row0 * INTERMEDIATE_SIZE + intermediate_col + col0] : 0.0f;
            float gate1 = (row1 < M) ? gate_buf[row1 * INTERMEDIATE_SIZE + intermediate_col + col0] : 0.0f;
            float gate2 = (row0 < M && col1 < BN) ? gate_buf[row0 * INTERMEDIATE_SIZE + intermediate_col + col1] : 0.0f;
            float gate3 = (row1 < M && col1 < BN) ? gate_buf[row1 * INTERMEDIATE_SIZE + intermediate_col + col1] : 0.0f;

            // SwiGLU: output = up * silu(gate)
            float out0 = acc[warp_id][0] * silu(gate0);
            float out1 = acc[warp_id][1] * silu(gate1);
            float out2 = acc[warp_id][2] * silu(gate2);
            float out3 = acc[warp_id][3] * silu(gate3);

            // Write to output as BF16
            if (row0 < M) {
                output[row0 * INTERMEDIATE_SIZE + intermediate_col + col0] = __float2bfloat16(out0);
                if (col1 < BN) output[row0 * INTERMEDIATE_SIZE + intermediate_col + col1] = __float2bfloat16(out2);
            }
            if (row1 < M) {
                output[row1 * INTERMEDIATE_SIZE + intermediate_col + col0] = __float2bfloat16(out1);
                if (col1 < BN) output[row1 * INTERMEDIATE_SIZE + intermediate_col + col1] = __float2bfloat16(out3);
            }
        }
    }
}

// ============================================================================
// Host launcher
// ============================================================================

void launch_fused_moe_gemm1_swiglu(
    const uint8_t* input,      // [M, HIDDEN_SIZE/2] FP4 packed
    const uint8_t* input_sf,   // [M, HIDDEN_SIZE/SF_BLOCK_SIZE] UE8M0
    const uint8_t* gate_up_w,  // [GATE_UP_SIZE, HIDDEN_SIZE/2] FP4 packed
    const uint8_t* gate_up_sf, // [GATE_UP_SIZE, HIDDEN_SIZE/SF_BLOCK_SIZE] UE8M0
    __nv_bfloat16* output,     // [M, INTERMEDIATE_SIZE] BF16
    int M,
    cudaStream_t stream
) {
    // Grid: 8 N-tiles (4 gate + 4 up), 1 block per N-tile
    // Phase 1: gate tiles store to SMEM, up tiles read + SwiGLU
    // NOTE: This requires gate tiles to run before up tiles, which grid-level
    // ordering doesn't guarantee. Phase 2 will fix this with a single-CTA design.

    // For Phase 1 correctness test: run gate and up in separate launches
    dim3 grid_gate(GATE_UP_SIZE / 2 / BN);  // 4 blocks for gate
    dim3 grid_up(GATE_UP_SIZE / 2 / BN);    // 4 blocks for up
    dim3 block(BLOCK_SIZE);

    int smem_bytes = TOTAL_SMEM_BYTES;

    // Set max dynamic SMEM
    cudaFuncSetAttribute(sm120_fused_moe_gemm1_swiglu,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);

    // Launch gate tiles (n_tile 0..3)
    sm120_fused_moe_gemm1_swiglu<<<grid_gate, block, smem_bytes, stream>>>(
        input, input_sf, gate_up_w, gate_up_sf, output, M);

    // Launch up tiles (n_tile 4..7) — reads gate from SMEM... but SMEM is per-CTA!
    // BUG: gate_buf in SMEM is not shared across CTAs.
    // This design needs revision — see note below.
}

// ============================================================================
// NOTE: The multi-CTA gate/up split has a fundamental issue: SMEM is per-CTA,
// so gate results can't be shared across CTAs via SMEM. Solutions:
//
// Option A: Single CTA does ALL 8 N-passes sequentially (gate then up).
//           Simple, correct, but serializes N-tiles.
//
// Option B: Gate results go through global memory (temp buffer).
//           Fast if L2-resident, but adds GMEM traffic.
//
// Option C: Single CTA with N-tiled loop (our Phase 2 design).
//           Gate passes store to SMEM, up passes read from same SMEM.
//
// For Phase 1c, we implement Option A: single CTA, sequential N-passes.
// ============================================================================

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
sm120_fused_moe_gemm1_swiglu_single_cta(
    const uint8_t* __restrict__ input,
    const uint8_t* __restrict__ input_sf,
    const uint8_t* __restrict__ gate_up_weight,
    const uint8_t* __restrict__ gate_up_sf,
    __nv_bfloat16* __restrict__ output,
    int M
) {
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    extern __shared__ char smem[];

    // SMEM partitions
    uint8_t* w_tile = reinterpret_cast<uint8_t*>(smem);                      // Weight tile [BK, BN]
    uint8_t* a_tile = w_tile + WEIGHT_TILE_BYTES;                            // Input tile [BM, BK]
    float*   gate_buf = reinterpret_cast<float*>(a_tile + INPUT_TILE_BYTES); // Gate results [BM, 256]

    const int N_PASSES = GATE_UP_SIZE / BN;  // 8 passes (4 gate + 4 up)

    // ================================================================
    // N-pass loop: first 4 passes = gate, next 4 = up
    // ================================================================
    for (int n_pass = 0; n_pass < N_PASSES; n_pass++) {
        const int n_offset = n_pass * BN;
        const bool is_gate = (n_pass < N_PASSES / 2);

        // Per-N-tile accumulators: 8 MMA N-tiles within BN=64
        float acc[N_MMA_TILES][4];
        #pragma unroll
        for (int nt = 0; nt < BN / MMA_N; nt++) {
            acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
        }

        // K-loop
        for (int ki = 0; ki < HIDDEN_SIZE / BK; ki++) {
            const int k_offset = ki * BK;

            // --- Cooperative load: weight tile [BK=64, BN=64] FP4 → SMEM ---
            {
                const int total = WEIGHT_TILE_BYTES;  // 2048 bytes
                for (int i = tid; i < total; i += BLOCK_SIZE) {
                    int k_local = i / (BN / FP4_PACK);
                    int n_local = i % (BN / FP4_PACK);
                    // Weight layout: [N_total, K_total] FP4 packed (row-major in N)
                    int global_n = n_offset + n_local * FP4_PACK;  // unpacked N index
                    int global_k = k_offset + k_local;
                    // Each byte = 2 FP4 values along N dimension
                    w_tile[i] = gate_up_weight[(n_offset + n_local) * (HIDDEN_SIZE / FP4_PACK) +
                                               k_offset / FP4_PACK + k_local / FP4_PACK];
                    // TODO: fix exact indexing for FP4 packed layout
                }
            }

            // --- Cooperative load: input tile [BM=16, BK=64] FP4 → SMEM ---
            {
                const int total = INPUT_TILE_BYTES;  // 512 bytes
                for (int i = tid; i < total; i += BLOCK_SIZE) {
                    a_tile[i] = input[i / (BK / FP4_PACK) * (HIDDEN_SIZE / FP4_PACK) +
                                      k_offset / FP4_PACK + i % (BK / FP4_PACK)];
                }
            }

            __syncthreads();

            // --- MMA: each warp handles one N-tile (8 warps = 8 N-tiles) ---
            if (warp_id < BN / MMA_N) {
                int nt = warp_id;

                // Load A fragment from SMEM
                // MMA m16n8k64 E2M1: A is 16×64 FP4 = 16×32 bytes
                // Each thread loads 4 uint32 = 16 bytes
                uint32_t a_regs[4];
                {
                    uint32_t* a32 = reinterpret_cast<uint32_t*>(a_tile);
                    int idx = lane_id * 4;
                    a_regs[0] = a32[idx + 0];
                    a_regs[1] = a32[idx + 1];
                    a_regs[2] = a32[idx + 2];
                    a_regs[3] = a32[idx + 3];
                }

                // Load B fragment from SMEM
                // MMA m16n8k64 E2M1: B is 8×64 FP4 = 8×32 bytes = 256 bytes total
                uint32_t b_regs[2];
                {
                    uint32_t* w32 = reinterpret_cast<uint32_t*>(w_tile);
                    int b_offset = nt * (MMA_N * BK / FP4_PACK / 4);
                    b_regs[0] = w32[b_offset + lane_id * 2];
                    b_regs[1] = w32[b_offset + lane_id * 2 + 1];
                }

                // Scale factors (simplified for Phase 1 — proper SF indexing in Phase 2)
                uint32_t sfa = 0x3F;  // 1.0 in UE8M0 (exponent bias)
                uint32_t sfb = 0x3F;
                uint16_t bidA = 0, tidA = (uint16_t)lane_id;
                uint16_t bidB = 0, tidB = (uint16_t)lane_id;

                mma_fp4_blockscaled(acc[nt], a_regs, b_regs, acc[nt],
                                   sfa, bidA, tidA, sfb, bidB, tidB);
            }

            __syncthreads();
        }  // end K-loop

        // --- Store results ---
        if (warp_id < BN / MMA_N) {
            int nt = warp_id;
            int g = lane_id / 4;
            int t = lane_id % 4;
            int row0 = g * 2;
            int row1 = g * 2 + 8;
            int col_base = nt * MMA_N + t * 2;

            if (is_gate) {
                // Store gate to SMEM buffer
                int buf_col = n_offset - 0 + col_base;  // offset within 256-wide gate
                if (row0 < M && buf_col < INTERMEDIATE_SIZE) {
                    gate_buf[row0 * INTERMEDIATE_SIZE + buf_col] = acc[nt][0];
                    gate_buf[row0 * INTERMEDIATE_SIZE + buf_col + 1] = acc[nt][2];
                }
                if (row1 < M && buf_col < INTERMEDIATE_SIZE) {
                    gate_buf[row1 * INTERMEDIATE_SIZE + buf_col] = acc[nt][1];
                    gate_buf[row1 * INTERMEDIATE_SIZE + buf_col + 1] = acc[nt][3];
                }
            } else {
                // Up pass: apply SwiGLU with gate from SMEM
                int buf_col = n_offset - GATE_UP_SIZE / 2 + col_base;

                float g0 = (row0 < M && buf_col < INTERMEDIATE_SIZE) ?
                    gate_buf[row0 * INTERMEDIATE_SIZE + buf_col] : 0.0f;
                float g1 = (row1 < M && buf_col < INTERMEDIATE_SIZE) ?
                    gate_buf[row1 * INTERMEDIATE_SIZE + buf_col] : 0.0f;
                float g2 = (row0 < M && buf_col + 1 < INTERMEDIATE_SIZE) ?
                    gate_buf[row0 * INTERMEDIATE_SIZE + buf_col + 1] : 0.0f;
                float g3 = (row1 < M && buf_col + 1 < INTERMEDIATE_SIZE) ?
                    gate_buf[row1 * INTERMEDIATE_SIZE + buf_col + 1] : 0.0f;

                // SwiGLU: output = up * silu(gate)
                float o0 = acc[nt][0] * silu(g0);
                float o1 = acc[nt][1] * silu(g1);
                float o2 = acc[nt][2] * silu(g2);
                float o3 = acc[nt][3] * silu(g3);

                // Write BF16 output
                if (row0 < M && buf_col < INTERMEDIATE_SIZE) {
                    output[row0 * INTERMEDIATE_SIZE + buf_col] = __float2bfloat16(o0);
                    output[row0 * INTERMEDIATE_SIZE + buf_col + 1] = __float2bfloat16(o2);
                }
                if (row1 < M && buf_col < INTERMEDIATE_SIZE) {
                    output[row1 * INTERMEDIATE_SIZE + buf_col] = __float2bfloat16(o1);
                    output[row1 * INTERMEDIATE_SIZE + buf_col + 1] = __float2bfloat16(o3);
                }
            }
        }

        __syncthreads();  // Ensure gate_buf is complete before up passes read it
    }  // end N-pass loop
}

// ============================================================================
// Phase 1c Host Launcher (single CTA)
// ============================================================================

extern "C"
void launch_fused_moe_gemm1_swiglu_v2(
    const uint8_t* input,
    const uint8_t* input_sf,
    const uint8_t* gate_up_w,
    const uint8_t* gate_up_sf,
    __nv_bfloat16* output,
    int M,
    cudaStream_t stream
) {
    dim3 grid(1);  // Single CTA
    dim3 block(BLOCK_SIZE);
    int smem_bytes = WEIGHT_TILE_BYTES + INPUT_TILE_BYTES + GATE_BUF_BYTES + 1024;

    cudaFuncSetAttribute(sm120_fused_moe_gemm1_swiglu_single_cta,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_bytes);

    sm120_fused_moe_gemm1_swiglu_single_cta<<<grid, block, smem_bytes, stream>>>(
        input, input_sf, gate_up_w, gate_up_sf, output, M);
}
