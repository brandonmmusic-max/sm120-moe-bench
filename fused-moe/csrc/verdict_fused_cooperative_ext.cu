/**
 * VerdictMoE Fused Cooperative Extension for vLLM — SM120 Blackwell
 * Sprint 7: Multi-Token + TP=4 Support
 *
 * ONE kernel launch per forward: BF16→FP4 → GEMM1 → SwiGLU → E4M3 requant → GEMM2 → BF16
 *
 * Key changes from Sprint 6:
 *   - Runtime N_HALF, K_GROUPS (supports both EP=4 and TP=4)
 *   - Multi-token support (M>1 inside one launch, weight reuse)
 *   - Compact A storage (M × 32 bytes vs M × 512)
 *   - Kahan summation for K_GROUPS > 4
 *
 * Grid: topk × num_tiles = topk × (tiles_n × k_groups) CTAs
 *   EP=4: topk=10, N_HALF=1024, tiles_n=16, k_groups=4  → 10×64 = 640
 *   TP=4: topk=10, N_HALF=256,  tiles_n=4,  k_groups=16 → 10×64 = 640
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
// Compile-time Constants (model-independent)
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
constexpr int MAX_M       = 4;

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
// Multi-Token Fused Cooperative Kernel
// BF16 in → FP4 quantize → GEMM1 → SwiGLU → requant → GEMM2 → BF16 out
// ALL M tokens processed inside one launch.
// Runtime N_HALF, K_GROUPS for EP=4 and TP=4 support.
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
verdict_fused_cooperative_mt(
    const __nv_bfloat16* __restrict__ input_bf16,   // [M, K]
    const uint8_t* __restrict__ all_w1_fp4,         // [E, 2*n_half, K/2]
    const uint8_t* __restrict__ all_w1_sf,          // [E, 2*n_half, K/16]
    const uint8_t* __restrict__ all_w2_fp4,         // [E, K, n_half/2]
    const uint8_t* __restrict__ all_w2_sf,          // [E, K, n_half/16]
    const int*     __restrict__ expert_ids,         // [num_active]
    const float*   __restrict__ expert_wts,         // [num_active]
    const float*   __restrict__ w1_alpha,           // [num_active]
    const float*   __restrict__ w2_alpha,           // [num_active]
    __nv_bfloat16* __restrict__ output_bf16,        // [M, K]
    float*         __restrict__ output_f32,         // [M, K]
    uint8_t*       __restrict__ input_fp4,          // [M, K/2]
    uint8_t*       __restrict__ input_sf,           // [M, K/16]
    float*         __restrict__ partials,           // [num_active * num_tiles * M * 128]
    uint8_t*       __restrict__ gmem_inter_fp4,     // [num_active * M, n_half/2]
    uint8_t*       __restrict__ gmem_inter_sf,      // [num_active * M, n_half/16]
    volatile int*  __restrict__ barrier_counter,
    int num_active,
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

    const int eidx    = blockIdx.x / num_tiles;
    const int tile    = blockIdx.x % num_tiles;
    const int n_chunk = tile / k_groups;
    const int k_group = tile % k_groups;
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int total_ctas = num_active * num_tiles;
    if (eidx >= num_active) return;

    const int eid     = expert_ids[eidx];
    const float wt    = expert_wts[eidx];
    const float alpha1 = w1_alpha[eidx];
    const float alpha2 = w2_alpha[eidx];
    const int n_start = n_chunk * BN;
    const int k_base  = k_group * k_per_group;

    // ================================================================
    // PROLOGUE: BF16→FP4 quantization + zero output_f32 (ALL M tokens)
    // ================================================================
    {
        const int global_tid = blockIdx.x * BLOCK_SIZE + tid;
        const int total_threads = gridDim.x * BLOCK_SIZE;

        // Zero output_f32 for all M tokens
        for (int i = global_tid; i < M * HIDDEN; i += total_threads)
            output_f32[i] = 0.0f;

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
    // SMEM layout (dynamic, depends on M)
    // ================================================================
    extern __shared__ char smem_raw[];
    uint8_t* s_A_compact = (uint8_t*)smem_raw;
    uint8_t* s_B_gate    = s_A_compact + M * 32;
    uint8_t* s_B_up      = s_B_gate + SMEM_B;
    uint8_t* s_SFA_mt    = s_B_up + SMEM_B;
    uint8_t* s_SFB_gate  = s_SFA_mt + ((M * SF_PER_K + 3) & ~3);
    uint8_t* s_SFB_up    = s_SFB_gate + SMEM_SFB;

    const int g = lane_id / 4;
    const int Nl = 4 * (g & 1) + (g >> 1);
    const int sn = warp_id * 8 + Nl;
    const int t0 = lane_id % 4;
    const int rbo = sn * (BK / 2);

    const uint8_t* w1_fp4 = all_w1_fp4 + (long long)eid * n2 * K_PACKED;
    const uint8_t* w1_sf  = all_w1_sf  + (long long)eid * n2 * SF_COLS_W1;

    // ================================================================
    // PHASE 1a: GEMM1 — Multi-token, hybrid K×N
    // ================================================================
    float gate_acc[MAX_M][4];
    float up_acc[MAX_M][4];
    #pragma unroll
    for (int m = 0; m < MAX_M; m++) {
        gate_acc[m][0] = gate_acc[m][1] = gate_acc[m][2] = gate_acc[m][3] = 0;
        up_acc[m][0]   = up_acc[m][1]   = up_acc[m][2]   = up_acc[m][3]   = 0;
    }

    for (int kt = 0; kt < k_tiles_per_g; kt++) {
        const int k_off = k_base + kt * BK;
        const int k_pk  = k_off / 2;
        const int k_sf  = k_off / SF_BLOCK;

        // Load M tokens' A data (compact)
        for (int i = tid; i < M * 8; i += BLOCK_SIZE) {
            int m = i / 8, w = i % 8;
            *(uint32_t*)(s_A_compact + m * 32 + w * 4) =
                *(const uint32_t*)&input_fp4[m * K_PACKED + k_pk + w * 4];
        }

        // Load gate B (shared)
        for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
            int boff = i * 4;
            int row = boff / (BK / 2), col = boff % (BK / 2);
            *(uint32_t*)&s_B_gate[swizzle_343(boff)] =
                *(const uint32_t*)&w1_fp4[(long long)(n_start + row) * K_PACKED + k_pk + col];
        }

        // Load up B (shared)
        for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
            int boff = i * 4;
            int row = boff / (BK / 2), col = boff % (BK / 2);
            *(uint32_t*)&s_B_up[swizzle_343(boff)] =
                *(const uint32_t*)&w1_fp4[(long long)(n_half + n_start + row) * K_PACKED + k_pk + col];
        }

        // Load M tokens' SFA
        if (tid < M * SF_PER_K) {
            int m = tid / SF_PER_K, si = tid % SF_PER_K;
            s_SFA_mt[m * SF_PER_K + si] = input_sf[m * SF_COLS_W1 + k_sf + si];
        }

        // Load SFB (shared)
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

        for (int m = 0; m < M && m < MAX_M; m++) {
            uint32_t a[4] = {0, 0, 0, 0};
            if (lane_id / 4 == 0) {
                a[0] = *(uint32_t*)(s_A_compact + m * 32 + t0 * 4);
                a[2] = *(uint32_t*)(s_A_compact + m * 32 + 16 + t0 * 4);
            }
            uint32_t sfa_pk = pack_sf4(s_SFA_mt + m * SF_PER_K);
            mma_nvf4_e4m3_m16n8k64(gate_acc[m], a, bg, gate_acc[m], sfa_pk, sfbg);
            mma_nvf4_e4m3_m16n8k64(up_acc[m], a, bu, up_acc[m], sfa_pk, sfbu);
        }

        __syncthreads();
    }

    // Write partials for ALL M tokens
    if (lane_id < 4) {
        for (int m = 0; m < M && m < MAX_M; m++) {
            long long pb = (long long)(eidx * num_tiles + tile) * M * PARTIALS_PER_CTA
                         + (long long)m * PARTIALS_PER_CTA;
            int c0 = warp_id * 8 + lane_id;
            int c1 = c0 + 4;
            partials[pb + c0]      = gate_acc[m][0];
            partials[pb + c1]      = gate_acc[m][1];
            partials[pb + BN + c0] = up_acc[m][0];
            partials[pb + BN + c1] = up_acc[m][1];
        }
    }

    grid_barrier_atomic(barrier_counter, total_ctas, 1);

    // ================================================================
    // PHASE 1b: Reduce + alpha1×SwiGLU + FP4 requant (ALL M tokens)
    // ================================================================
    if (k_group == 0 && tid < BN) {
        int col = tid;

        for (int m = 0; m < M && m < MAX_M; m++) {
            float gs = 0, us = 0;
            float gs_c = 0, us_c = 0;

            for (int kg = 0; kg < k_groups; kg++) {
                int partner_tile = n_chunk * k_groups + kg;
                long long base = (long long)(eidx * num_tiles + partner_tile) * M * PARTIALS_PER_CTA
                               + (long long)m * PARTIALS_PER_CTA;
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
                gmem_inter_fp4[(eidx * M + m) * n_half_packed + (n_start + col) / 2] =
                    (uint8_t)(nib32 | (neighbor32 << 4));
            }
            if (col % SF_BLOCK == 0) {
                gmem_inter_sf[(eidx * M + m) * sf_cols_w2 + (n_start + col) / SF_BLOCK] = sf_enc;
            }
        }
    }

    grid_barrier_atomic(barrier_counter, total_ctas, 2);

    // ================================================================
    // PHASE 2: GEMM2 — Multi-token, alpha2 × wt scatter
    // ================================================================
    {
        const int j_start = tile * BN;
        const uint8_t* w2_fp4 = all_w2_fp4 + (long long)eid * HIDDEN * n_half_packed;
        const uint8_t* w2_sf  = all_w2_sf  + (long long)eid * HIDDEN * sf_cols_w2;
        uint8_t* s_B2   = s_B_gate;
        uint8_t* s_SFB2 = s_SFB_gate;

        float p2_acc[MAX_M][4];
        #pragma unroll
        for (int m = 0; m < MAX_M; m++)
            p2_acc[m][0] = p2_acc[m][1] = p2_acc[m][2] = p2_acc[m][3] = 0;

        int p2_k_passes = n_half / BK;
        for (int kp = 0; kp < p2_k_passes; kp++) {
            int ko = kp * BK, kpk = ko / 2, ksf = ko / SF_BLOCK;

            for (int i = tid; i < M * 8; i += BLOCK_SIZE) {
                int m = i / 8, w = i % 8;
                *(uint32_t*)(s_A_compact + m * 32 + w * 4) =
                    *(const uint32_t*)&gmem_inter_fp4[(eidx * M + m) * n_half_packed + kpk + w * 4];
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

            if (tid < M * SF_PER_K) {
                int m = tid / SF_PER_K, si = tid % SF_PER_K;
                s_SFA_mt[m * SF_PER_K + si] =
                    gmem_inter_sf[(eidx * M + m) * sf_cols_w2 + ksf + si];
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

            for (int m = 0; m < M && m < MAX_M; m++) {
                uint32_t ar[4] = {0, 0, 0, 0};
                if (lane_id / 4 == 0) {
                    ar[0] = *(uint32_t*)(s_A_compact + m * 32 + t0 * 4);
                    ar[2] = *(uint32_t*)(s_A_compact + m * 32 + 16 + t0 * 4);
                }
                uint32_t sfap = pack_sf4(s_SFA_mt + m * SF_PER_K);
                mma_nvf4_e4m3_m16n8k64(p2_acc[m], ar, br, p2_acc[m], sfap, sfbp);
            }

            __syncthreads();
        }

        // Scatter with alpha2 × routing weight
        float scale = wt * alpha2;
        if (lane_id < 4) {
            for (int m = 0; m < M && m < MAX_M; m++) {
                int j0 = j_start + warp_id * 8 + lane_id;
                int j1 = j0 + 4;
                if (j0 < HIDDEN) atomicAdd(&output_f32[m * HIDDEN + j0], scale * p2_acc[m][0]);
                if (j1 < HIDDEN) atomicAdd(&output_f32[m * HIDDEN + j1], scale * p2_acc[m][1]);
            }
        }
    }

    grid_barrier_atomic(barrier_counter, total_ctas, 3);

    // ================================================================
    // EPILOGUE: F32 → BF16 for ALL M tokens
    // ================================================================
    {
        const int global_tid = blockIdx.x * BLOCK_SIZE + tid;
        const int total_threads = gridDim.x * BLOCK_SIZE;
        for (int i = global_tid; i < M * HIDDEN; i += total_threads)
            output_bf16[i] = __float2bfloat16(output_f32[i]);
    }
}

// ============================================================================
// Host Forward — ONE kernel launch for ALL M tokens (Sprint 7)
//
// Shared routing: all M tokens processed through the same topk experts.
// MTP speculative tokens share similar routing — validated at temp=0.
// Weight data loaded ONCE, reused across M tokens → 2.94x vs serial launches.
// ============================================================================
void verdict_cooperative_forward(
    torch::Tensor input,         // [M, K] BF16
    torch::Tensor w1_fp4,        // [E, 2*N, K//2] uint8
    torch::Tensor w1_sf,         // [E, 2*N, K//16] uint8
    torch::Tensor w1_alpha,      // [topk] float32 (shared routing)
    torch::Tensor w2_fp4,        // [E, K, N//2] uint8
    torch::Tensor w2_sf,         // [E, K, N//16] uint8
    torch::Tensor w2_alpha,      // [topk] float32 (shared routing)
    torch::Tensor output,        // [M, K] BF16
    torch::Tensor expert_ids,    // [topk] int32 (shared routing)
    torch::Tensor expert_wts,    // [topk] float32 (shared routing)
    torch::Tensor output_f32,    // [M * K] float32
    torch::Tensor input_fp4_buf, // [M * K/2] uint8
    torch::Tensor input_sf_buf,  // [M * K/16] uint8
    torch::Tensor partials_buf,  // [topk * num_tiles * M * 128] float32
    torch::Tensor inter_fp4_buf, // [topk * M * N_HALF/2] uint8
    torch::Tensor inter_sf_buf,  // [topk * M * N_HALF/16] uint8
    torch::Tensor barrier_buf,   // [1] int32
    int K, int N_half, int num_active, int topk)
{
    auto stream = c10::cuda::getCurrentCUDAStream();
    int M = input.size(0);

    TORCH_CHECK(K == HIDDEN, "K must be ", HIDDEN, " got ", K);
    TORCH_CHECK(M >= 1 && M <= MAX_M,
                "M=", M, " out of range [1, MAX_M=", MAX_M, "]");

    // Compute runtime grid parameters
    int tiles_n = N_half / BN;
    int k_groups = std::max(4, 640 / (topk * tiles_n));  // ensure >= 640 CTAs
    int num_tiles = tiles_n * k_groups;
    int grid = topk * num_tiles;

    TORCH_CHECK(grid <= 752,
                "Grid size ", grid, " exceeds 752 max concurrent CTAs on SM120");

    // Dynamic SMEM for actual M tokens
    int smem_size = M * 32 + 2 * SMEM_B
                  + ((M * SF_PER_K + 3) & ~3) + 2 * SMEM_SFB + 128;

    cudaFuncSetAttribute(verdict_fused_cooperative_mt,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    // Zero barrier counter (ONE reset for ONE launch)
    cudaMemsetAsync(barrier_buf.data_ptr(), 0, sizeof(int), stream);

    // ONE kernel launch — ALL M tokens, shared routing, weight reuse
    verdict_fused_cooperative_mt<<<grid, BLOCK_SIZE, smem_size, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const uint8_t*>(w1_fp4.data_ptr()),
        reinterpret_cast<const uint8_t*>(w1_sf.data_ptr()),
        reinterpret_cast<const uint8_t*>(w2_fp4.data_ptr()),
        reinterpret_cast<const uint8_t*>(w2_sf.data_ptr()),
        reinterpret_cast<const int*>(expert_ids.data_ptr()),
        reinterpret_cast<const float*>(expert_wts.data_ptr()),
        reinterpret_cast<const float*>(w1_alpha.data_ptr()),
        reinterpret_cast<const float*>(w2_alpha.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        reinterpret_cast<float*>(output_f32.data_ptr()),
        reinterpret_cast<uint8_t*>(input_fp4_buf.data_ptr()),
        reinterpret_cast<uint8_t*>(input_sf_buf.data_ptr()),
        reinterpret_cast<float*>(partials_buf.data_ptr()),
        reinterpret_cast<uint8_t*>(inter_fp4_buf.data_ptr()),
        reinterpret_cast<uint8_t*>(inter_sf_buf.data_ptr()),
        reinterpret_cast<volatile int*>(barrier_buf.data_ptr()),
        topk,       // num_active = topk (shared routing)
        M,          // actual token count (1-4 for MTP=3)
        N_half,     // runtime N_HALF (1024 EP=4, 256 TP=4)
        k_groups);  // runtime K_GROUPS (4 EP=4, 16 TP=4)
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &verdict_cooperative_forward,
          "VerdictMoE fused cooperative forward — multi-token + TP=4 (SM120)");
}
