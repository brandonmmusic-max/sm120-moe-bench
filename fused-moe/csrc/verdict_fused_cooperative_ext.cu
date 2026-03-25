/**
 * VerdictMoE Single Fused Cooperative Extension for vLLM — SM120 Blackwell
 *
 * ONE kernel launch per token: BF16→FP4 → GEMM1 → SwiGLU → E4M3 requant → GEMM2 → BF16
 *
 * Architecture (per-token, topk=10 experts):
 *   Prologue: BF16→FP4 quantize input + zero output_f32 (all CTAs)
 *   Barrier 0
 *   Phase 1a: GEMM1 — Hybrid K×N-distributed (640 CTAs, gate+up simultaneous)
 *   Barrier 1
 *   Phase 1b: 4-way reduce + alpha1*SwiGLU + FP4 requant (160 leaders)
 *   Barrier 2
 *   Phase 2: GEMM2 + alpha2*wt scatter via atomicAdd (640 CTAs)
 *   Barrier 3
 *   Epilogue: F32→BF16 output (all CTAs)
 *
 * Grid: topk (10) × NUM_TILES (64) = 640 CTAs ≤ 752 max concurrent
 * Atomic barriers (no cooperative_groups, no -rdc=true). CUDA-graph safe.
 * Consecutive-K packing, scale_vec::4X with native E4M3FN, vectorized uint32 loads.
 *
 * For M>1 (MTP): C++ forward() loops over tokens, launching once per token.
 * Each launch reuses scratch buffers (partials, intermediate, input FP4).
 *
 * Build: torch JIT with -gencode=arch=compute_120a,code=sm_120a -O2
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// ============================================================================
// Model Constants (Qwen3.5-397B, EP=4)
// ============================================================================
constexpr int HIDDEN      = 4096;
constexpr int N_HALF      = 1024;
constexpr int N2          = 2 * N_HALF;

constexpr int BM = 16, BN = 64, BK = 64;
constexpr int SF_BLOCK   = 16;
constexpr int SF_PER_K   = BK / SF_BLOCK;       // 4

constexpr int NUM_WARPS  = 8;
constexpr int WARP_SIZE  = 32;
constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;  // 256

// Hybrid K×N distribution for Phase 1
constexpr int TILES_N       = N_HALF / BN;              // 16
constexpr int K_GROUPS      = 4;
constexpr int NUM_TILES     = TILES_N * K_GROUPS;       // 64
constexpr int K_TILES_PER_G = (HIDDEN / BK) / K_GROUPS; // 16
constexpr int K_PER_GROUP   = K_TILES_PER_G * BK;       // 1024

// Derived packing constants
constexpr int K_PACKED      = HIDDEN / 2;         // 2048
constexpr int N_HALF_PACKED = N_HALF / 2;         // 512
constexpr int SF_COLS_W1    = HIDDEN / SF_BLOCK;   // 256
constexpr int SF_COLS_W2    = N_HALF / SF_BLOCK;   // 64

// SMEM: A + gate_B + up_B + SFA + SFB_gate + SFB_up + pad
constexpr int SMEM_A       = BM * (BK / 2);        // 512
constexpr int SMEM_B       = BN * (BK / 2);        // 2048
constexpr int SMEM_SFA_PAD = 16;
constexpr int SMEM_SFB     = BN * SF_PER_K;        // 256
constexpr int SMEM_TOTAL   = SMEM_A + 2*SMEM_B + SMEM_SFA_PAD + 2*SMEM_SFB + 128;

constexpr int PARTIALS_PER_CTA = 2 * BN;           // 128 floats (gate + up)

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
// Atomic Grid Barrier (CUDA-graph safe, no cooperative_groups)
// ============================================================================
__device__ __forceinline__ void grid_barrier_atomic(
    volatile int* counter, int total_ctas, int gen)
{
    __syncthreads();
    __threadfence();  // CRITICAL: flush all prior GMEM writes device-wide
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
// Single Fused Cooperative Kernel
// BF16 in → FP4 quantize → GEMM1 → SwiGLU → requant → GEMM2 → BF16 out
// ONE launch per token. ALL phases inside.
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
verdict_fused_cooperative_single(
    const __nv_bfloat16* __restrict__ input_bf16,   // [K] one token
    const uint8_t* __restrict__ all_w1_fp4,         // [E, 2*N_HALF, K/2]
    const uint8_t* __restrict__ all_w1_sf,          // [E, 2*N_HALF, K/16]
    const uint8_t* __restrict__ all_w2_fp4,         // [E, K, N_HALF/2]
    const uint8_t* __restrict__ all_w2_sf,          // [E, K, N_HALF/16]
    const int*     __restrict__ expert_ids,         // [num_active]
    const float*   __restrict__ expert_wts,         // [num_active]
    const float*   __restrict__ w1_alpha,           // [num_active]
    const float*   __restrict__ w2_alpha,           // [num_active]
    __nv_bfloat16* __restrict__ output_bf16,        // [K] output
    float*         __restrict__ output_f32,         // [K] atomicAdd accumulator
    uint8_t*       __restrict__ input_fp4,          // [K/2] FP4 scratch
    uint8_t*       __restrict__ input_sf,           // [K/16] scale scratch
    float*         __restrict__ partials,           // [num_active * NUM_TILES * 128]
    uint8_t*       __restrict__ gmem_inter_fp4,     // [num_active * N_HALF/2]
    uint8_t*       __restrict__ gmem_inter_sf,      // [num_active * N_HALF/16]
    volatile int*  __restrict__ barrier_counter,
    int num_active)
{
    const int eidx = blockIdx.x / NUM_TILES;
    const int tile = blockIdx.x % NUM_TILES;
    const int n_chunk = tile / K_GROUPS;    // 0..15
    const int k_group = tile % K_GROUPS;    // 0..3
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int total_ctas = num_active * NUM_TILES;
    if (eidx >= num_active) return;

    const int eid = expert_ids[eidx];
    const float wt = expert_wts[eidx];
    const float alpha1 = w1_alpha[eidx];
    const float alpha2 = w2_alpha[eidx];
    const int n_start = n_chunk * BN;
    const int k_base = k_group * K_PER_GROUP;

    // ================================================================
    // PROLOGUE: BF16→FP4 quantization + zero output_f32
    // All 640 CTAs cooperate. Each half-warp (16 threads) handles one
    // SF group of 16 BF16 elements. 256 groups total for K=4096.
    // ================================================================
    {
        const int global_tid = blockIdx.x * BLOCK_SIZE + tid;
        const int total_threads = gridDim.x * BLOCK_SIZE;

        // Zero output_f32 (K=4096 floats, trivial with 163K threads)
        for (int i = global_tid; i < HIDDEN; i += total_threads)
            output_f32[i] = 0.0f;

        // BF16→NVFP4: one half-warp per SF group
        const int half_warp_id = global_tid / 16;
        const int hw_lane = tid % 16;
        constexpr int num_sf_groups = HIDDEN / SF_BLOCK;  // 256

        if (half_warp_id < num_sf_groups) {
            const int g = half_warp_id;
            const int kb = g * SF_BLOCK;

            float val = __bfloat162float(input_bf16[kb + hw_lane]);
            float aval = fabsf(val);

            // Warp-reduce max across 16 elements in the SF group
            float wmax = aval;
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1)
                wmax = fmaxf(wmax, __shfl_xor_sync(0xFFFFFFFF, wmax, off));

            // E4M3FN scale encode/decode
            float sf_target = fmaxf(wmax / 6.0f, 1e-30f);
            uint8_t sf_byte = d_e4m3fn_encode(sf_target);
            float actual_scale = d_e4m3fn_decode(sf_byte);
            if (actual_scale < 1e-30f) actual_scale = 1e-30f;

            // Quantize to E2M1 nibble
            uint8_t nib = d_quantize_e2m1(val / actual_scale);

            // Pack nibble pairs: even lanes write (low=self, high=partner)
            uint8_t partner_nib = (uint8_t)__shfl_xor_sync(0xFFFFFFFF, (int)nib, 1);
            if ((hw_lane & 1) == 0)
                input_fp4[kb / 2 + hw_lane / 2] = nib | (partner_nib << 4);

            // Write scale (one per 16-element group)
            if (hw_lane == 0)
                input_sf[g] = sf_byte;
        }
    }

    grid_barrier_atomic(barrier_counter, total_ctas, 0);

    // ================================================================
    // PHASE 1a: GEMM1 — Hybrid K×N-distributed
    // Each CTA covers BN=64 N-columns × 16 K-tiles (1024 K-elements).
    // Gate and Up B tiles loaded simultaneously. Vectorized uint32 loads.
    // ================================================================
    extern __shared__ char smem_raw[];
    uint8_t* s_A        = (uint8_t*)smem_raw;
    uint8_t* s_B_gate   = s_A + SMEM_A;
    uint8_t* s_B_up     = s_B_gate + SMEM_B;
    uint8_t* s_SFA      = s_B_up + SMEM_B;
    uint8_t* s_SFB_gate = s_SFA + SMEM_SFA_PAD;
    uint8_t* s_SFB_up   = s_SFB_gate + SMEM_SFB;

    // Per-warp MMA column mapping (CLayout for M=1, scale_vec::4X)
    const int g = lane_id / 4;
    const int Nl = 4 * (g & 1) + (g >> 1);
    const int sn = warp_id * 8 + Nl;
    const int t0 = lane_id % 4;
    const int rbo = sn * (BK / 2);

    const uint8_t* w1_fp4 = all_w1_fp4 + (long long)eid * N2 * K_PACKED;
    const uint8_t* w1_sf  = all_w1_sf  + (long long)eid * N2 * SF_COLS_W1;

    float gate_acc[4] = {0, 0, 0, 0};
    float up_acc[4]   = {0, 0, 0, 0};

    for (int kt = 0; kt < K_TILES_PER_G; kt++) {
        const int k_off = k_base + kt * BK;
        const int k_pk = k_off / 2;
        const int k_sf = k_off / SF_BLOCK;

        // --- Cooperative load: A + gate_B + up_B + scales (vectorized) ---
        for (int i = tid; i < SMEM_A / 4; i += BLOCK_SIZE) {
            int boff = i * 4;
            int row = boff / (BK / 2), col = boff % (BK / 2);
            *(uint32_t*)&s_A[swizzle_343(boff)] =
                (row == 0) ? *(const uint32_t*)&input_fp4[k_pk + col] : 0u;
        }
        for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
            int boff = i * 4;
            int row = boff / (BK / 2), col = boff % (BK / 2);
            *(uint32_t*)&s_B_gate[swizzle_343(boff)] =
                *(const uint32_t*)&w1_fp4[(long long)(n_start + row) * K_PACKED + k_pk + col];
        }
        for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
            int boff = i * 4;
            int row = boff / (BK / 2), col = boff % (BK / 2);
            *(uint32_t*)&s_B_up[swizzle_343(boff)] =
                *(const uint32_t*)&w1_fp4[(long long)(N_HALF + n_start + row) * K_PACKED + k_pk + col];
        }
        if (tid < SF_PER_K) s_SFA[tid] = input_sf[k_sf + tid];
        for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
            int row = i / SF_PER_K, col = i % SF_PER_K;
            s_SFB_gate[i] = w1_sf[(long long)(n_start + row) * SF_COLS_W1 + k_sf + col];
        }
        for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
            int row = i / SF_PER_K, col = i % SF_PER_K;
            s_SFB_up[i] = w1_sf[(long long)(N_HALF + n_start + row) * SF_COLS_W1 + k_sf + col];
        }

        __syncthreads();

        // --- Vectorized pack + MMA ---
        uint32_t a[4] = {0, 0, 0, 0};
        if (lane_id / 4 == 0) {
            a[0] = *(uint32_t*)&s_A[swizzle_343(t0 * 4)];
            a[2] = *(uint32_t*)&s_A[swizzle_343(16 + t0 * 4)];
        }
        uint32_t sfa_pk = pack_sf4(s_SFA);

        // Gate MMA
        uint32_t bg[2];
        bg[0] = *(uint32_t*)&s_B_gate[swizzle_343(rbo + t0 * 4)];
        bg[1] = *(uint32_t*)&s_B_gate[swizzle_343(rbo + 16 + t0 * 4)];
        uint32_t sfbg = pack_sf4(&s_SFB_gate[sn * SF_PER_K]);
        mma_nvf4_e4m3_m16n8k64(gate_acc, a, bg, gate_acc, sfa_pk, sfbg);

        // Up MMA (reuse a and sfa_pk)
        uint32_t bu[2];
        bu[0] = *(uint32_t*)&s_B_up[swizzle_343(rbo + t0 * 4)];
        bu[1] = *(uint32_t*)&s_B_up[swizzle_343(rbo + 16 + t0 * 4)];
        uint32_t sfbu = pack_sf4(&s_SFB_up[sn * SF_PER_K]);
        mma_nvf4_e4m3_m16n8k64(up_acc, a, bu, up_acc, sfa_pk, sfbu);

        __syncthreads();
    }

    // Write partial gate[64] + up[64] to partials buffer
    if (lane_id < 4) {
        long long pb = (long long)eidx * NUM_TILES * PARTIALS_PER_CTA
                     + (long long)tile * PARTIALS_PER_CTA;
        int c0 = warp_id * 8 + lane_id;
        int c1 = c0 + 4;
        partials[pb + c0]      = gate_acc[0];
        partials[pb + c1]      = gate_acc[1];
        partials[pb + BN + c0] = up_acc[0];
        partials[pb + BN + c1] = up_acc[1];
    }

    grid_barrier_atomic(barrier_counter, total_ctas, 1);

    // ================================================================
    // PHASE 1b: Reduce 4 K-group partials + alpha1×SwiGLU + FP4 requant
    // 160 leader CTAs (k_group==0), 64 threads each.
    // ================================================================
    if (k_group == 0 && tid < BN) {
        int col = tid;
        float gs = 0, us = 0;

        #pragma unroll
        for (int kg = 0; kg < K_GROUPS; kg++) {
            int partner_tile = n_chunk * K_GROUPS + kg;
            long long base = (long long)eidx * NUM_TILES * PARTIALS_PER_CTA
                           + (long long)partner_tile * PARTIALS_PER_CTA;
            gs += partials[base + col];
            us += partials[base + BN + col];
        }

        // Apply alpha1 weight scale before SwiGLU
        gs *= alpha1;
        us *= alpha1;

        float sw_val = us * d_silu(gs);

        // Group max across SF_BLOCK=16 columns via warp shuffle
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

        // Pack nibble pairs
        uint32_t nib32 = (uint32_t)nib;
        uint32_t neighbor32 = __shfl_down_sync(0xFFFFFFFF, nib32, 1);
        if (col % 2 == 0) {
            gmem_inter_fp4[eidx * N_HALF_PACKED + (n_start + col) / 2] =
                (uint8_t)(nib32 | (neighbor32 << 4));
        }

        // Write scale factor
        if (col % SF_BLOCK == 0) {
            gmem_inter_sf[eidx * SF_COLS_W2 + (n_start + col) / SF_BLOCK] = sf_enc;
        }
    }

    grid_barrier_atomic(barrier_counter, total_ctas, 2);

    // ================================================================
    // PHASE 2: GEMM2 — N-distributed + alpha2 × weighted atomicAdd scatter
    // All 640 CTAs. Each handles 1 output tile of BN=64 columns.
    // ================================================================
    {
        const int j_start = tile * BN;
        const uint8_t* w2_fp4 = all_w2_fp4 + (long long)eid * HIDDEN * N_HALF_PACKED;
        const uint8_t* w2_sf  = all_w2_sf  + (long long)eid * HIDDEN * SF_COLS_W2;

        // Reuse SMEM
        uint8_t* s_B2   = s_B_gate;
        uint8_t* s_SFB2 = s_SFB_gate;

        float acc[4] = {0, 0, 0, 0};

        for (int kp = 0; kp < N_HALF / BK; kp++) {
            int ko = kp * BK, kpk = ko / 2, ksf = ko / SF_BLOCK;

            // Load intermediate A (vectorized)
            for (int i = tid; i < SMEM_A / 4; i += BLOCK_SIZE) {
                int boff = i * 4;
                int row = boff / (BK / 2), col = boff % (BK / 2);
                *(uint32_t*)&s_A[swizzle_343(boff)] =
                    (row == 0) ? *(const uint32_t*)&gmem_inter_fp4[eidx * N_HALF_PACKED + kpk + col] : 0u;
            }

            // Load W2 B (vectorized)
            for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
                int boff = i * 4;
                int row = boff / (BK / 2), col = boff % (BK / 2);
                int oc = j_start + row;
                *(uint32_t*)&s_B2[swizzle_343(boff)] =
                    (oc < HIDDEN)
                    ? *(const uint32_t*)&w2_fp4[(long long)oc * N_HALF_PACKED + kpk + col]
                    : 0u;
            }

            // Load scales
            if (tid < SF_PER_K)
                s_SFA[tid] = gmem_inter_sf[eidx * SF_COLS_W2 + ksf + tid];
            for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
                int row = i / SF_PER_K, col = i % SF_PER_K;
                int oc = j_start + row;
                s_SFB2[i] = (oc < HIDDEN) ? w2_sf[(long long)oc * SF_COLS_W2 + ksf + col] : 0;
            }

            __syncthreads();

            // Vectorized pack + MMA
            uint32_t ar[4] = {0, 0, 0, 0};
            if (lane_id / 4 == 0) {
                ar[0] = *(uint32_t*)&s_A[swizzle_343(t0 * 4)];
                ar[2] = *(uint32_t*)&s_A[swizzle_343(16 + t0 * 4)];
            }
            uint32_t sfap = pack_sf4(s_SFA);

            uint32_t br[2];
            br[0] = *(uint32_t*)&s_B2[swizzle_343(rbo + t0 * 4)];
            br[1] = *(uint32_t*)&s_B2[swizzle_343(rbo + 16 + t0 * 4)];
            uint32_t sfbp = pack_sf4(&s_SFB2[sn * SF_PER_K]);

            mma_nvf4_e4m3_m16n8k64(acc, ar, br, acc, sfap, sfbp);

            __syncthreads();
        }

        // Scatter with alpha2 × routing weight
        if (lane_id < 4) {
            float scale = wt * alpha2;
            int j0 = j_start + warp_id * 8 + lane_id;
            int j1 = j0 + 4;
            if (j0 < HIDDEN) atomicAdd(&output_f32[j0], scale * acc[0]);
            if (j1 < HIDDEN) atomicAdd(&output_f32[j1], scale * acc[1]);
        }
    }

    grid_barrier_atomic(barrier_counter, total_ctas, 3);

    // ================================================================
    // EPILOGUE: F32 → BF16 output conversion
    // All CTAs cooperate. K=4096 elements trivial with 163K threads.
    // ================================================================
    {
        const int global_tid = blockIdx.x * BLOCK_SIZE + tid;
        const int total_threads = gridDim.x * BLOCK_SIZE;
        for (int i = global_tid; i < HIDDEN; i += total_threads)
            output_bf16[i] = __float2bfloat16(output_f32[i]);
    }
}

// ============================================================================
// Host Orchestrator — loops over M tokens, one kernel launch each
// ============================================================================
void verdict_cooperative_forward(
    torch::Tensor input,         // [M, K] BF16
    torch::Tensor w1_fp4,        // [E, 2*N, K//2] uint8
    torch::Tensor w1_sf,         // [E, 2*N, K//16] uint8
    torch::Tensor w1_alpha,      // [num_active] float32
    torch::Tensor w2_fp4,        // [E, K, N//2] uint8
    torch::Tensor w2_sf,         // [E, K, N//16] uint8
    torch::Tensor w2_alpha,      // [num_active] float32
    torch::Tensor output,        // [M, K] BF16
    torch::Tensor expert_ids,    // [num_active] int32
    torch::Tensor expert_wts,    // [num_active] float32
    torch::Tensor output_f32,    // [M * K] float32
    torch::Tensor input_fp4_buf, // [K/2] uint8  (reused per token)
    torch::Tensor input_sf_buf,  // [K/16] uint8 (reused per token)
    torch::Tensor partials_buf,  // [topk * NUM_TILES * 128] float32
    torch::Tensor inter_fp4_buf, // [topk * N_HALF/2] uint8 (reused per token)
    torch::Tensor inter_sf_buf,  // [topk * N_HALF/16] uint8 (reused per token)
    torch::Tensor barrier_buf,   // [1] int32
    int K, int N_half, int num_active, int topk)
{
    auto stream = c10::cuda::getCurrentCUDAStream();
    int M = input.size(0);

    TORCH_CHECK(K == HIDDEN, "K must be ", HIDDEN, " got ", K);
    TORCH_CHECK(N_half == N_HALF, "N_half must be ", N_HALF, " got ", N_half);
    TORCH_CHECK(num_active == M * topk,
                "num_active (", num_active, ") must equal M*topk (", M, "*", topk, ")");

    int grid = topk * NUM_TILES;  // 10 * 64 = 640
    TORCH_CHECK(grid <= 752,
                "Grid size ", grid, " exceeds 752 max concurrent CTAs on SM120");

    // Set max dynamic SMEM (idempotent, cheap host call)
    cudaFuncSetAttribute(verdict_fused_cooperative_single,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);

    // Process one token at a time (cooperative kernel needs all CTAs to fit)
    for (int tok = 0; tok < M; tok++) {
        // Reset barrier counter (CUDA-graph safe)
        cudaMemsetAsync(barrier_buf.data_ptr(), 0, sizeof(int), stream);

        verdict_fused_cooperative_single<<<grid, BLOCK_SIZE, SMEM_TOTAL, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()) + tok * K,
            reinterpret_cast<const uint8_t*>(w1_fp4.data_ptr()),
            reinterpret_cast<const uint8_t*>(w1_sf.data_ptr()),
            reinterpret_cast<const uint8_t*>(w2_fp4.data_ptr()),
            reinterpret_cast<const uint8_t*>(w2_sf.data_ptr()),
            reinterpret_cast<const int*>(expert_ids.data_ptr()) + tok * topk,
            reinterpret_cast<const float*>(expert_wts.data_ptr()) + tok * topk,
            reinterpret_cast<const float*>(w1_alpha.data_ptr()) + tok * topk,
            reinterpret_cast<const float*>(w2_alpha.data_ptr()) + tok * topk,
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr()) + tok * K,
            reinterpret_cast<float*>(output_f32.data_ptr()) + tok * K,
            reinterpret_cast<uint8_t*>(input_fp4_buf.data_ptr()),   // reused
            reinterpret_cast<uint8_t*>(input_sf_buf.data_ptr()),    // reused
            reinterpret_cast<float*>(partials_buf.data_ptr()),      // reused
            reinterpret_cast<uint8_t*>(inter_fp4_buf.data_ptr()),   // reused
            reinterpret_cast<uint8_t*>(inter_sf_buf.data_ptr()),    // reused
            reinterpret_cast<volatile int*>(barrier_buf.data_ptr()),
            topk);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &verdict_cooperative_forward,
          "VerdictMoE single fused cooperative forward (SM120)");
}
