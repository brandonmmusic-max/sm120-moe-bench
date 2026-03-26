/**
 * VerdictMoE Fused Cooperative Extension for vLLM — SM120 Blackwell
 * Sprint 9: Independent Per-Token Routing
 *
 * ONE kernel launch per forward: BF16→FP4 → GEMM1 → SwiGLU → E4M3 requant → GEMM2 → BF16
 *
 * Key change from Sprint 7/8:
 *   - Each CTA handles ONE (token, expert) pair — no shared/union routing
 *   - Flat pair tables: expert_ids[num_pairs], token_ids[num_pairs], weights[num_pairs]
 *   - Exactly mirrors CUTLASS grouped GEMM semantics
 *   - MTP acceptance rate matches baseline (~75%)
 *
 * Grid: num_pairs × num_tiles CTAs (always ~640 via adaptive k_groups)
 *   TP=4 M=1: 10×64=640, TP=4 M=4: 40×16=640
 *   EP=4 M=1: 10×64=640, EP=4 M=4: 40×16=640
 *
 * Atomic barriers (no cooperative_groups, no -rdc=true). CUDA-graph safe.
 * Consecutive-K packing, scale_vec::4X with native E4M3FN, vectorized uint32.
 *
 * Build: torch JIT with -gencode=arch=compute_120a,code=sm_120a -O2
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// ============================================================================
// Compile-time Constants
// ============================================================================
constexpr int HIDDEN      = 4096;
constexpr int BN = 64, BK = 64;
constexpr int SF_BLOCK    = 16;
constexpr int SF_PER_K    = BK / SF_BLOCK;       // 4
constexpr int NUM_WARPS   = 8;
constexpr int WARP_SIZE   = 32;
constexpr int BLOCK_SIZE  = NUM_WARPS * WARP_SIZE;  // 256
constexpr int K_PACKED    = HIDDEN / 2;              // 2048
constexpr int SF_COLS_W1  = HIDDEN / SF_BLOCK;       // 256
constexpr int SMEM_B      = BN * (BK / 2);           // 2048
constexpr int SMEM_SFB    = BN * SF_PER_K;            // 256
constexpr int PARTIALS_PER_CTA = 2 * BN;              // 128

// ============================================================================
// Device Helpers
// ============================================================================
__device__ __forceinline__ float d_e4m3fn_decode(uint8_t x) {
    int s = (x >> 7) & 1, e = (x >> 3) & 0xF, m = x & 7;
    float val;
    if (e == 0) val = ldexpf((float)m, -9);
    else if (e == 15 && m == 7) val = 0.0f;
    else val = ldexpf((float)(8 + m), e - 10);
    return s ? -val : val;
}

__device__ __forceinline__ uint8_t d_e4m3fn_encode(float val) {
    uint16_t packed;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
        : "=h"(packed) : "f"(val), "f"(0.0f));
    return (uint8_t)((packed >> 8) & 0xFF);
}

__device__ __forceinline__ float d_silu(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ uint8_t d_quantize_e2m1(float value) {
    float av = fabsf(value);
    int sign = (value < 0.0f) ? 1 : 0, idx;
    if      (av < 0.25f) idx = 0; else if (av < 0.75f) idx = 1;
    else if (av < 1.25f) idx = 2; else if (av < 1.75f) idx = 3;
    else if (av < 2.5f)  idx = 4; else if (av < 3.5f)  idx = 5;
    else if (av < 5.0f)  idx = 6; else idx = 7;
    return (uint8_t)((sign << 3) | idx);
}

__device__ __forceinline__ uint32_t swizzle_343(uint32_t off) {
    return off ^ ((off >> 3) & 0x70u);
}

// ============================================================================
// MMA: scale_vec::4X with native E4M3FN (ue4m3)
// ============================================================================
__device__ __forceinline__ void mma_nvf4_e4m3_m16n8k64(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
    const float (&c)[4], uint32_t sfa, uint32_t sfb)
{
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
        ".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},"
        "{%10,%11,%12,%13},{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]),
          "r"(sfa), "h"((uint16_t)0), "h"((uint16_t)0),
          "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
}

// ============================================================================
// Atomic Grid Barrier (CUDA-graph safe)
// ============================================================================
__device__ __forceinline__ void grid_barrier_atomic(
    volatile int* counter, int total_ctas, int gen)
{
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        int target = total_ctas * (gen + 1);
        atomicAdd((int*)counter, 1);
        while (atomicAdd((int*)counter, 0) < target) {}
    }
    __syncthreads();
}

__device__ __forceinline__ uint32_t pack_sf4(const uint8_t* sf) {
    return (uint32_t)sf[0] | ((uint32_t)sf[1]<<8)
         | ((uint32_t)sf[2]<<16) | ((uint32_t)sf[3]<<24);
}

// ============================================================================
// Independent Per-Token Fused Cooperative Kernel
//
// Grid: num_pairs × num_tiles CTAs
// Each CTA: ONE (token, expert) pair.
// BF16 in → FP4 quantize → GEMM1 → SwiGLU → requant → GEMM2 → BF16 out
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
verdict_fused_independent_mt(
    const __nv_bfloat16* __restrict__ input_bf16,   // [M, K]
    const uint8_t* __restrict__ all_w1_fp4,         // [E, 2*n_half, K/2]
    const uint8_t* __restrict__ all_w1_sf,          // [E, 2*n_half, K/16]
    const uint8_t* __restrict__ all_w2_fp4,         // [E, K, n_half/2]
    const uint8_t* __restrict__ all_w2_sf,          // [E, K, n_half/16]
    const int*     __restrict__ expert_ids,         // [num_pairs]
    const int*     __restrict__ token_ids,          // [num_pairs]
    const float*   __restrict__ w1_alpha,           // [num_pairs]
    const float*   __restrict__ w2_alpha,           // [num_pairs]
    const float*   __restrict__ expert_wts,         // [num_pairs]
    __nv_bfloat16* __restrict__ output_bf16,        // [M, K]
    float*         __restrict__ output_f32,         // [M, K]
    float*         __restrict__ pair_output_f32,    // [num_pairs, K] per-pair GEMM2 output
    uint8_t*       __restrict__ input_fp4,          // [M, K/2]
    uint8_t*       __restrict__ input_sf,           // [M, K/16]
    float*         __restrict__ partials,           // [num_pairs * num_tiles * PARTIALS_PER_CTA]
    uint8_t*       __restrict__ gmem_inter_fp4,     // [num_pairs, n_half/2]
    uint8_t*       __restrict__ gmem_inter_sf,      // [num_pairs, n_half/SF_BLOCK]
    volatile int*  __restrict__ barrier_counter,
    int num_pairs,
    int M,
    int n_half,
    int k_groups)
{
    // Derive runtime constants
    const int tiles_n       = n_half / BN;
    const int num_tiles     = tiles_n * k_groups;
    const int k_tiles_per_g = (HIDDEN / BK) / k_groups;
    const int k_per_group   = k_tiles_per_g * BK;
    const int n_half_packed = n_half / 2;
    const int sf_cols_w2    = n_half / SF_BLOCK;
    const int n2            = 2 * n_half;

    const int pair_idx = blockIdx.x / num_tiles;
    const int tile     = blockIdx.x % num_tiles;
    const int n_chunk  = tile / k_groups;
    const int k_group  = tile % k_groups;
    const int tid      = threadIdx.x;
    const int warp_id  = tid / WARP_SIZE;
    const int lane_id  = tid % WARP_SIZE;
    const int total_ctas = num_pairs * num_tiles;
    const int eid      = expert_ids[pair_idx];
    const int token_id = token_ids[pair_idx];
    const float alpha1 = w1_alpha[pair_idx];
    const float alpha2 = w2_alpha[pair_idx];
    const float wt     = expert_wts[pair_idx];
    const int n_start  = n_chunk * BN;
    const int k_base   = k_group * k_per_group;

    // ================================================================
    // PROLOGUE: BF16→FP4 quantization + zero output_f32 (ALL M tokens)
    // All CTAs cooperate on this — grid-stride loop.
    // ================================================================
    {
        const int global_tid = blockIdx.x * BLOCK_SIZE + tid;
        const int total_threads = gridDim.x * BLOCK_SIZE;

        // Zero pair_output_f32 (per-pair GEMM2 outputs, summed in epilogue)
        for (int i = global_tid; i < num_pairs * HIDDEN; i += total_threads)
            pair_output_f32[i] = 0.0f;

        // BF16→NVFP4 for all M tokens
        constexpr int num_sf_groups = HIDDEN / SF_BLOCK;  // 256 per token
        const int total_sf_groups = M * num_sf_groups;
        const int half_warp_id = global_tid / 16;
        const int hw_lane = tid % 16;

        if (half_warp_id < total_sf_groups) {
            const int m = half_warp_id / num_sf_groups;
            const int g = half_warp_id % num_sf_groups;
            const int kb = g * SF_BLOCK;

            float val = __bfloat162float(input_bf16[m * HIDDEN + kb + hw_lane]);
            float aval = fabsf(val);

            float wmax = aval;
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1)
                wmax = fmaxf(wmax, __shfl_xor_sync(0xFFFFFFFF, wmax, off));

            float sf_target = fmaxf(wmax / 6.0f, 1e-30f);
            uint8_t sf_byte = d_e4m3fn_encode(sf_target);
            float actual_scale = d_e4m3fn_decode(sf_byte);
            if (actual_scale < 1e-30f) actual_scale = 1e-30f;

            uint8_t nib = d_quantize_e2m1(val / actual_scale);
            uint8_t partner_nib = (uint8_t)__shfl_xor_sync(0xFFFFFFFF, (int)nib, 1);
            if ((hw_lane & 1) == 0)
                input_fp4[m * K_PACKED + kb / 2 + hw_lane / 2] = nib | (partner_nib << 4);
            if (hw_lane == 0)
                input_sf[m * SF_COLS_W1 + g] = sf_byte;
        }
    }

    grid_barrier_atomic(barrier_counter, total_ctas, 0);

    // ================================================================
    // SMEM layout (single token per CTA)
    // ================================================================
    extern __shared__ char smem_raw[];
    uint8_t* s_A        = (uint8_t*)smem_raw;
    uint8_t* s_B_gate   = s_A + 32;
    uint8_t* s_B_up     = s_B_gate + SMEM_B;
    uint8_t* s_SFA      = s_B_up + SMEM_B;
    uint8_t* s_SFB_gate = s_SFA + ((SF_PER_K + 3) & ~3);
    uint8_t* s_SFB_up   = s_SFB_gate + SMEM_SFB;

    const int g = lane_id / 4;
    const int Nl = 4 * (g & 1) + (g >> 1);
    const int sn = warp_id * 8 + Nl;
    const int t0 = lane_id % 4;
    const int rbo = sn * (BK / 2);

    const uint8_t* w1_fp4 = all_w1_fp4 + (long long)eid * n2 * K_PACKED;
    const uint8_t* w1_sf  = all_w1_sf  + (long long)eid * n2 * SF_COLS_W1;

    // ================================================================
    // PHASE 1a: GEMM1 — single token, hybrid K×N
    // ================================================================
    float gate_acc[4] = {0, 0, 0, 0};
    float up_acc[4]   = {0, 0, 0, 0};

    for (int kt = 0; kt < k_tiles_per_g; kt++) {
        const int k_off = k_base + kt * BK;
        const int k_pk  = k_off / 2;
        const int k_sf  = k_off / SF_BLOCK;

        // Load this token's A data
        for (int i = tid; i < 8; i += BLOCK_SIZE) {
            *(uint32_t*)(s_A + i * 4) =
                *(const uint32_t*)&input_fp4[token_id * K_PACKED + k_pk + i * 4];
        }

        // Load gate B
        for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
            int boff = i * 4;
            int row = boff / (BK / 2), col = boff % (BK / 2);
            *(uint32_t*)&s_B_gate[swizzle_343(boff)] =
                *(const uint32_t*)&w1_fp4[(long long)(n_start + row) * K_PACKED + k_pk + col];
        }

        // Load up B
        for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
            int boff = i * 4;
            int row = boff / (BK / 2), col = boff % (BK / 2);
            *(uint32_t*)&s_B_up[swizzle_343(boff)] =
                *(const uint32_t*)&w1_fp4[(long long)(n_half + n_start + row) * K_PACKED + k_pk + col];
        }

        // Load SFA
        if (tid < SF_PER_K) {
            s_SFA[tid] = input_sf[token_id * SF_COLS_W1 + k_sf + tid];
        }

        // Load SFB
        for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
            int row = i / SF_PER_K, col = i % SF_PER_K;
            s_SFB_gate[i] = w1_sf[(long long)(n_start + row) * SF_COLS_W1 + k_sf + col];
        }
        for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
            int row = i / SF_PER_K, col = i % SF_PER_K;
            s_SFB_up[i] = w1_sf[(long long)(n_half + n_start + row) * SF_COLS_W1 + k_sf + col];
        }

        __syncthreads();

        uint32_t bg[2], bu[2];
        bg[0] = *(uint32_t*)&s_B_gate[swizzle_343(rbo + t0 * 4)];
        bg[1] = *(uint32_t*)&s_B_gate[swizzle_343(rbo + 16 + t0 * 4)];
        uint32_t sfbg = pack_sf4(&s_SFB_gate[sn * SF_PER_K]);
        bu[0] = *(uint32_t*)&s_B_up[swizzle_343(rbo + t0 * 4)];
        bu[1] = *(uint32_t*)&s_B_up[swizzle_343(rbo + 16 + t0 * 4)];
        uint32_t sfbu = pack_sf4(&s_SFB_up[sn * SF_PER_K]);

        uint32_t a[4] = {0, 0, 0, 0};
        if (lane_id / 4 == 0) {
            a[0] = *(uint32_t*)(s_A + t0 * 4);
            a[2] = *(uint32_t*)(s_A + 16 + t0 * 4);
        }
        uint32_t sfa_pk = pack_sf4(s_SFA);

        mma_nvf4_e4m3_m16n8k64(gate_acc, a, bg, gate_acc, sfa_pk, sfbg);
        mma_nvf4_e4m3_m16n8k64(up_acc, a, bu, up_acc, sfa_pk, sfbu);

        __syncthreads();
    }

    // Write partials
    if (lane_id < 4) {
        long long pb = (long long)(pair_idx * num_tiles + tile) * PARTIALS_PER_CTA;
        int c0 = warp_id * 8 + lane_id;
        int c1 = c0 + 4;
        partials[pb + c0]      = gate_acc[0];
        partials[pb + c1]      = gate_acc[1];
        partials[pb + BN + c0] = up_acc[0];
        partials[pb + BN + c1] = up_acc[1];
    }

    grid_barrier_atomic(barrier_counter, total_ctas, 1);

    // ================================================================
    // PHASE 1b: Reduce + alpha1×SwiGLU + FP4 requant
    // ================================================================
    if (k_group == 0 && tid < BN) {
        int col = tid;

        float gs = 0, us = 0;
        float gs_c = 0, us_c = 0;

        for (int kg = 0; kg < k_groups; kg++) {
            int partner_tile = n_chunk * k_groups + kg;
            long long base = (long long)(pair_idx * num_tiles + partner_tile) * PARTIALS_PER_CTA;
            float g_y = partials[base + col] - gs_c;
            float g_t = gs + g_y;
            gs_c = (g_t - gs) - g_y;
            gs = g_t;
            float u_y = partials[base + BN + col] - us_c;
            float u_t = us + u_y;
            us_c = (u_t - us) - u_y;
            us = u_t;
        }

        gs *= alpha1;
        us *= alpha1;
        float sw_val = us * d_silu(gs);

        float abs_sw = fabsf(sw_val);
        float gm = abs_sw;
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 1));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 2));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 4));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 8));

        float st = fmaxf(gm / 6.0f, 1e-30f);
        uint8_t sf_enc = d_e4m3fn_encode(st);
        float as = d_e4m3fn_decode(sf_enc);
        if (as < 1e-30f) as = 1e-30f;

        uint8_t nib = d_quantize_e2m1(sw_val / as);

        uint32_t nib32 = (uint32_t)nib;
        uint32_t neighbor32 = __shfl_down_sync(0xFFFFFFFF, nib32, 1);
        if (col % 2 == 0) {
            gmem_inter_fp4[pair_idx * n_half_packed + (n_start + col) / 2] =
                (uint8_t)(nib32 | (neighbor32 << 4));
        }
        if (col % SF_BLOCK == 0) {
            gmem_inter_sf[pair_idx * sf_cols_w2 + (n_start + col) / SF_BLOCK] = sf_enc;
        }
    }

    grid_barrier_atomic(barrier_counter, total_ctas, 2);

    // ================================================================
    // PHASE 2: GEMM2 — single token, output scatter by token_id
    // ================================================================
    {
        const int p2_out_tiles = HIDDEN / BN;  // 64
        const uint8_t* w2_fp4 = all_w2_fp4 + (long long)eid * HIDDEN * n_half_packed;
        const uint8_t* w2_sf  = all_w2_sf  + (long long)eid * HIDDEN * sf_cols_w2;
        uint8_t* s_B2   = s_B_gate;
        uint8_t* s_SFB2 = s_SFB_gate;
        int p2_k_passes = n_half / BK;

        for (int j_tile = tile; j_tile < p2_out_tiles; j_tile += num_tiles) {
            const int j_start = j_tile * BN;

            float p2_acc[4] = {0, 0, 0, 0};

            for (int kp = 0; kp < p2_k_passes; kp++) {
                int ko = kp * BK, kpk = ko / 2, ksf = ko / SF_BLOCK;

                for (int i = tid; i < 8; i += BLOCK_SIZE) {
                    *(uint32_t*)(s_A + i * 4) =
                        *(const uint32_t*)&gmem_inter_fp4[pair_idx * n_half_packed + kpk + i * 4];
                }

                for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
                    int boff = i * 4;
                    int row = boff / (BK / 2), col = boff % (BK / 2);
                    int oc = j_start + row;
                    *(uint32_t*)&s_B2[swizzle_343(boff)] =
                        (oc < HIDDEN)
                        ? *(const uint32_t*)&w2_fp4[(long long)oc * n_half_packed + kpk + col]
                        : 0u;
                }

                if (tid < SF_PER_K) {
                    s_SFA[tid] = gmem_inter_sf[pair_idx * sf_cols_w2 + ksf + tid];
                }

                for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
                    int row = i / SF_PER_K, col = i % SF_PER_K;
                    int oc = j_start + row;
                    s_SFB2[i] = (oc < HIDDEN) ? w2_sf[(long long)oc * sf_cols_w2 + ksf + col] : 0;
                }

                __syncthreads();

                uint32_t br[2];
                br[0] = *(uint32_t*)&s_B2[swizzle_343(rbo + t0 * 4)];
                br[1] = *(uint32_t*)&s_B2[swizzle_343(rbo + 16 + t0 * 4)];
                uint32_t sfbp = pack_sf4(&s_SFB2[sn * SF_PER_K]);

                uint32_t ar[4] = {0, 0, 0, 0};
                if (lane_id / 4 == 0) {
                    ar[0] = *(uint32_t*)(s_A + t0 * 4);
                    ar[2] = *(uint32_t*)(s_A + 16 + t0 * 4);
                }
                uint32_t sfap = pack_sf4(s_SFA);

                mma_nvf4_e4m3_m16n8k64(p2_acc, ar, br, p2_acc, sfap, sfbp);

                __syncthreads();
            }

            // Write per-pair GEMM2 output (deterministic, no atomicAdd)
            if (lane_id < 4) {
                float scale = wt * alpha2;
                int j0 = j_start + warp_id * 8 + lane_id;
                int j1 = j0 + 4;
                if (j0 < HIDDEN) pair_output_f32[pair_idx * HIDDEN + j0] = scale * p2_acc[0];
                if (j1 < HIDDEN) pair_output_f32[pair_idx * HIDDEN + j1] = scale * p2_acc[1];
            }
        }
    }

    grid_barrier_atomic(barrier_counter, total_ctas, 3);

    // ================================================================
    // EPILOGUE: Deterministic reduction + F32 → BF16 for ALL M tokens
    //
    // Sum per-pair outputs in fixed order (pair 0, 1, ..., topk-1) for
    // each token. Eliminates non-deterministic atomicAdd rounding that
    // caused ~1-2% MTP acceptance loss under strict rejection sampling.
    // ================================================================
    {
        const int global_tid = blockIdx.x * BLOCK_SIZE + tid;
        const int total_threads = gridDim.x * BLOCK_SIZE;
        const int topk = num_pairs / M;
        for (int i = global_tid; i < M * HIDDEN; i += total_threads) {
            int token = i / HIDDEN;
            int col   = i % HIDDEN;
            int pair_start = token * topk;
            float sum = 0.0f;
            for (int p = 0; p < topk; p++) {
                sum += pair_output_f32[(pair_start + p) * HIDDEN + col];
            }
            output_bf16[i] = __float2bfloat16(sum);
        }
    }
}

// ============================================================================
// Host Forward — Independent Per-Token Routing (Sprint 9)
//
// Each CTA processes one (token, expert) pair from the flat routing table.
// No union routing, no token masks. CUDA-graph safe: fixed grid per M value.
// ============================================================================
void verdict_cooperative_forward(
    torch::Tensor input,          // [M, K] BF16
    torch::Tensor w1_fp4,         // [E, 2*N, K//2] uint8
    torch::Tensor w1_sf,          // [E, 2*N, K//16] uint8
    torch::Tensor w1_alpha,       // [num_pairs] float32
    torch::Tensor w2_fp4,         // [E, K, N//2] uint8
    torch::Tensor w2_sf,          // [E, K, N//16] uint8
    torch::Tensor w2_alpha,       // [num_pairs] float32
    torch::Tensor output,         // [M, K] BF16
    torch::Tensor expert_ids,     // [num_pairs] int32
    torch::Tensor token_ids,      // [num_pairs] int32
    torch::Tensor expert_weights, // [num_pairs] float32
    torch::Tensor output_f32,     // [M * K] float32 (unused, kept for compat)
    torch::Tensor pair_output_f32,// [num_pairs * K] float32 — per-pair GEMM2 output
    torch::Tensor input_fp4_buf,  // [M * K/2] uint8
    torch::Tensor input_sf_buf,   // [M * K/16] uint8
    torch::Tensor partials_buf,   // [num_pairs * num_tiles * 128] float32
    torch::Tensor inter_fp4_buf,  // [num_pairs * N_HALF/2] uint8
    torch::Tensor inter_sf_buf,   // [num_pairs * N_HALF/16] uint8
    torch::Tensor barrier_buf,    // [1] int32
    int K, int N_half, int num_pairs)
{
    auto stream = c10::cuda::getCurrentCUDAStream();
    int M = input.size(0);

    TORCH_CHECK(K == HIDDEN, "K must be ", HIDDEN, " got ", K);
    TORCH_CHECK(M >= 1, "M must be >= 1, got ", M);

    // Compute runtime grid parameters
    int tiles_n = N_half / BN;
    int total_k_tiles = HIDDEN / BK;  // 64
    int k_groups = std::max(1, 640 / (num_pairs * tiles_n));
    while (total_k_tiles % k_groups != 0 && k_groups > 1) k_groups--;
    int num_tiles = tiles_n * k_groups;
    int grid = num_pairs * num_tiles;

    TORCH_CHECK(grid <= 752,
                "Grid size ", grid, " exceeds 752 max concurrent CTAs on SM120");

    // Single token per CTA — smaller SMEM
    int smem_size = 32 + 2 * SMEM_B
                  + ((SF_PER_K + 3) & ~3) + 2 * SMEM_SFB + 128;

    cudaFuncSetAttribute(verdict_fused_independent_mt,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    // Zero barrier counter
    cudaMemsetAsync(barrier_buf.data_ptr(), 0, sizeof(int), stream);

    // ONE kernel launch — independent per-token routing
    verdict_fused_independent_mt<<<grid, BLOCK_SIZE, smem_size, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const uint8_t*>(w1_fp4.data_ptr()),
        reinterpret_cast<const uint8_t*>(w1_sf.data_ptr()),
        reinterpret_cast<const uint8_t*>(w2_fp4.data_ptr()),
        reinterpret_cast<const uint8_t*>(w2_sf.data_ptr()),
        reinterpret_cast<const int*>(expert_ids.data_ptr()),
        reinterpret_cast<const int*>(token_ids.data_ptr()),
        reinterpret_cast<const float*>(w1_alpha.data_ptr()),
        reinterpret_cast<const float*>(w2_alpha.data_ptr()),
        reinterpret_cast<const float*>(expert_weights.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        reinterpret_cast<float*>(output_f32.data_ptr()),
        reinterpret_cast<float*>(pair_output_f32.data_ptr()),
        reinterpret_cast<uint8_t*>(input_fp4_buf.data_ptr()),
        reinterpret_cast<uint8_t*>(input_sf_buf.data_ptr()),
        reinterpret_cast<float*>(partials_buf.data_ptr()),
        reinterpret_cast<uint8_t*>(inter_fp4_buf.data_ptr()),
        reinterpret_cast<uint8_t*>(inter_sf_buf.data_ptr()),
        reinterpret_cast<volatile int*>(barrier_buf.data_ptr()),
        num_pairs,
        M,
        N_half,
        k_groups);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &verdict_cooperative_forward,
          "VerdictMoE fused independent per-token routing forward (SM120)");
}
