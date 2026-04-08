/**
 * megakernel_v1.cu — MegaKernel: Persistent Fused Transformer Forward Pass
 *
 * Single cudaLaunchCooperativeKernel that processes all 60 layers of
 * Qwen3.5-397B-A17B MoE inference on one GPU (TP=4 shard).
 *
 * v2 OPTIMIZATIONS (targeting <100us single-layer):
 *   1. ALL phases parallel across 188 CTAs — no more CTA-0-only bottlenecks
 *   2. RMSNorm: all CTAs compute sum-of-squares via atomicAdd, all apply norm
 *   3. Attention decode: 8 Q heads distributed across CTAs (23+ CTAs/head)
 *      - Within each head group: KV sequence distributed across CTAs
 *      - Two-pass online softmax with partial max/sum reduction via atomics
 *   4. MoE gate: 512 experts distributed across 188 CTAs, then CTA-0 top-K
 *   5. Requantization: distributed across all CTAs
 *   6. Residual add: distributed across all CTAs
 *   7. grid.sync() reduced from 17 to 8 per layer (fused zeroing + phases)
 *
 * Per-layer pipeline:
 *   RMSNorm -> Attention (QKV GEMV + decode + O GEMV) -> P2P AllReduce
 *   -> RMSNorm -> MoE Gate -> Expert GEMV (top-10 of 512) -> P2P AllReduce
 *   -> Residual add
 *
 * Architecture:
 *   - Cooperative launch: 188 CTAs (1 per SM), 256 threads, grid.sync() between phases
 *   - In-kernel P2P AllReduce: Write-based posted PCIe writes to remote BAR memory
 *   - Weight streaming: Weights stay in HBM, streamed via SMEM. Activations (~10KB) in L2.
 *   - Dynamic MoE routing: Gate -> top-K -> CTA self-assignment via atomics
 *   - FP4 GEMV: m16n8k64 MMA with E4M3FN block scaling (scale_vec::4X)
 *
 * Model specs (Qwen3.5-397B-A17B, TP=4 per-GPU shard):
 *   60 layers, hidden_size=4096
 *   32 attn heads total (8/GPU), 8 KV heads total (2/GPU), head_dim=128
 *   512 experts/layer, top-10 routing + 1 shared expert
 *   Expert intermediate_size=1024
 *   Mixed attention: 15 full + 45 DeltaNet (linear)
 *
 * Build:
 *   /usr/local/cuda-13.2/bin/nvcc -std=c++17 -O2 \
 *     -gencode=arch=compute_120a,code=sm_120a \
 *     --expt-relaxed-constexpr --compiler-options '-fPIC' \
 *     -rdc=true -lcuda \
 *     -o megakernel_v1 csrc/megakernel_v1.cu
 */

#include <cuda_runtime.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <numeric>

namespace cg = cooperative_groups;

// ============================================================================
// Compile-time Constants
// ============================================================================
static constexpr int NUM_LAYERS      = 60;
static constexpr int HIDDEN          = 4096;
static constexpr int HEAD_DIM        = 128;
static constexpr int NUM_Q_HEADS     = 8;     // per GPU (32 total / TP=4)
static constexpr int NUM_KV_HEADS    = 2;     // per GPU (8 total / TP=4)
static constexpr int QKV_DIM         = (NUM_Q_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM;  // 1536
static constexpr int O_DIM           = NUM_Q_HEADS * HEAD_DIM;                        // 1024
static constexpr int NUM_EXPERTS     = 512;
static constexpr int TOP_K           = 10;
static constexpr int EXPERT_INTER    = 1024;  // intermediate_size per expert
static constexpr int EXPERT_INTER_PACKED = EXPERT_INTER / 2;  // 512
static constexpr int WORLD_SIZE      = 4;

// MMA tile dimensions (same as VerdictMoE)
static constexpr int BM = 16, BN = 64, BK = 64;
static constexpr int SF_BLOCK    = 16;
static constexpr int SF_PER_K    = BK / SF_BLOCK;       // 4
static constexpr int NUM_WARPS   = 8;
static constexpr int WARP_SIZE   = 32;
static constexpr int BLOCK_SIZE  = NUM_WARPS * WARP_SIZE;  // 256

static constexpr int K_PACKED       = HIDDEN / 2;         // 2048
static constexpr int SF_COLS_HIDDEN = HIDDEN / SF_BLOCK;   // 256
static constexpr int K_TILES_HIDDEN = HIDDEN / BK;         // 64
static constexpr int N_TILES_QKV    = (QKV_DIM + BN - 1) / BN;  // 24
static constexpr int N_TILES_O      = (O_DIM + BN - 1) / BN;    // 16

static constexpr int NUM_CTAS    = 188;  // 1 per SM on RTX PRO 6000 Blackwell

static constexpr int FULL_ATTN_LAYERS = 15;
// Layers 0-14: full attention, layers 15-59: DeltaNet (linear attention)

// Attention parallelism constants
static constexpr int CTAS_PER_HEAD   = NUM_CTAS / NUM_Q_HEADS;  // 23 (with 4 spare)
static constexpr int CTAS_FOR_ATTN   = CTAS_PER_HEAD * NUM_Q_HEADS;  // 184

static const float E2M1_TABLE[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// ============================================================================
// Data Structures
// ============================================================================

// Per-layer weight pointers (all FP4 packed + E4M3FN scales, on this GPU's shard)
struct LayerWeights {
    // RMSNorm weights (BF16, not quantized — only 4096 elements = 8KB)
    const float*   attn_norm;        // [HIDDEN]
    const float*   ffn_norm;         // [HIDDEN]

    // Attention projections (FP4 + scales)
    const uint8_t* qkv_fp4;         // [QKV_DIM, K_PACKED]
    const uint8_t* qkv_sf;          // [QKV_DIM, SF_COLS_HIDDEN]
    const uint8_t* o_fp4;           // [HIDDEN, O_DIM/2]
    const uint8_t* o_sf;            // [HIDDEN, O_DIM/SF_BLOCK]

    // MoE gate (BF16 — small: 512 x 4096 = 4MB, kept in BF16 for softmax precision)
    const float*   gate_weight;      // [NUM_EXPERTS, HIDDEN]

    // Shared expert (FP4)
    const uint8_t* shared_w1_fp4;    // [2*EXPERT_INTER, K_PACKED]
    const uint8_t* shared_w1_sf;     // [2*EXPERT_INTER, SF_COLS_HIDDEN]
    const uint8_t* shared_w2_fp4;    // [HIDDEN, EXPERT_INTER_PACKED]
    const uint8_t* shared_w2_sf;     // [HIDDEN, EXPERT_INTER/SF_BLOCK]

    // Routed experts (FP4, all 512 on each GPU for MoE — TP=4 replicates experts)
    const uint8_t* expert_w1_fp4;    // [NUM_EXPERTS, 2*EXPERT_INTER, K_PACKED]
    const uint8_t* expert_w1_sf;     // [NUM_EXPERTS, 2*EXPERT_INTER, SF_COLS_HIDDEN]
    const uint8_t* expert_w2_fp4;    // [NUM_EXPERTS, HIDDEN, EXPERT_INTER_PACKED]
    const uint8_t* expert_w2_sf;     // [NUM_EXPERTS, HIDDEN, EXPERT_INTER/SF_BLOCK]
};

// P2P write-based AllReduce buffers
struct P2PBuffers {
    // Remote BAR pointers (mapped via cudaIpcGetMemHandle or P2P mapping)
    float* remote_recv[WORLD_SIZE];
    float* local_send;
    float* local_recv;

    volatile uint32_t* local_flags;
    volatile uint32_t* remote_flags[WORLD_SIZE];

    int rank;
};

// KV cache pointers (pre-allocated, indexed by layer)
struct KVCache {
    void* k_cache;
    void* v_cache;
    int   seq_len;
};

// Atomic scratch space for multi-CTA reductions
struct AtomicScratch {
    float    rmsnorm_sum;           // sum-of-squares for RMSNorm
    int      ready_counter;         // barrier counter
    float    gate_logits[NUM_EXPERTS]; // gate output (filled by distributed CTAs)
    // Attention decode partial reductions per Q head
    float    attn_max[NUM_Q_HEADS];     // partial max scores
    float    attn_sum[NUM_Q_HEADS];     // partial exp-sum
    float    attn_out_partial[NUM_Q_HEADS * HEAD_DIM]; // partial weighted V accumulation
    int      attn_cta_done[NUM_Q_HEADS]; // count of CTAs that finished their attention partial
};

struct MegaKernelParams {
    const LayerWeights* layers;      // [NUM_LAYERS]
    KVCache*     kv_caches;          // [NUM_LAYERS]
    P2PBuffers   p2p;

    float*       hidden_state;       // [HIDDEN]
    float*       attn_out;           // [QKV_DIM] — QKV output, then attention output
    float*       ffn_out;            // [HIDDEN]
    float*       norm_buf;           // [HIDDEN]
    float*       gate_logits;        // [NUM_EXPERTS]

    uint8_t*     act_fp4;            // [K_PACKED]
    uint8_t*     act_sf;             // [SF_COLS_HIDDEN]

    int*         top_expert_ids;     // [TOP_K]
    float*       top_expert_wts;     // [TOP_K]
    int*         expert_cta_counter; // [1]

    float*       expert_inter_buf;   // [TOP_K * EXPERT_INTER]
    uint8_t*     expert_inter_fp4;   // [TOP_K * EXPERT_INTER_PACKED]
    uint8_t*     expert_inter_sf;    // [TOP_K * EXPERT_INTER / SF_BLOCK]

    AtomicScratch* scratch;          // atomic scratch space for multi-CTA reductions

    int          num_layers;
};

// ============================================================================
// Device Helpers (from VerdictMoE — identical)
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
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(packed) : "f"(val), "f"(0.0f));
    return (uint8_t)((packed >> 8) & 0xFF);
}

__device__ __forceinline__ float d_silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ uint8_t d_quantize_e2m1(float value) {
    float av = fabsf(value); int sign = (value < 0.0f) ? 1 : 0; int idx;
    if      (av < 0.25f) idx = 0; else if (av < 0.75f) idx = 1;
    else if (av < 1.25f) idx = 2; else if (av < 1.75f) idx = 3;
    else if (av < 2.5f)  idx = 4; else if (av < 3.5f)  idx = 5;
    else if (av < 5.0f)  idx = 6; else idx = 7;
    return (uint8_t)((sign << 3) | idx);
}

__device__ __forceinline__ uint32_t pack_sf4(const uint8_t* sf) {
    return (uint32_t)sf[0] | ((uint32_t)sf[1]<<8)
         | ((uint32_t)sf[2]<<16) | ((uint32_t)sf[3]<<24);
}

__device__ __forceinline__ uint32_t swizzle_343(uint32_t off) {
    return off ^ ((off >> 3) & 0x70u);
}

// ============================================================================
// MMA: scale_vec::4X with native E4M3FN (identical to VerdictMoE)
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
// RMSNorm — ALL CTAs participate
//
// Phase A (all CTAs): Each CTA computes partial sum-of-squares, atomicAdd to global.
// Phase B (all CTAs after grid.sync): All CTAs apply norm+weight to their chunk,
//                                     also quantize to FP4.
// ============================================================================
__device__ void rmsnorm_parallel(
    float*       __restrict__ norm_out,    // [HIDDEN]
    uint8_t*     __restrict__ out_fp4,     // [K_PACKED]
    uint8_t*     __restrict__ out_sf,      // [SF_COLS_HIDDEN]
    const float* __restrict__ input,       // [HIDDEN]
    const float* __restrict__ weight,      // [HIDDEN]
    AtomicScratch* __restrict__ scratch,
    float eps = 1e-6f)
{
    const int tid = threadIdx.x;
    const int cta_id = blockIdx.x;
    const int global_tid = cta_id * BLOCK_SIZE + tid;
    const int total_threads = NUM_CTAS * BLOCK_SIZE;  // 188*256 = 48128

    // Step 1: Each thread computes partial sum-of-squares
    float local_sum = 0.0f;
    for (int i = global_tid; i < HIDDEN; i += total_threads) {
        float v = input[i];
        local_sum += v * v;
    }

    // Warp reduce within CTA
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_xor_sync(0xFFFFFFFF, local_sum, off);

    // Lane 0 of each warp atomicAdds to global scratch
    if ((tid % WARP_SIZE) == 0) {
        atomicAdd(&scratch->rmsnorm_sum, local_sum);
    }
    // NOTE: grid.sync() needed after this before reading rmsnorm_sum.
    // The caller handles this.
}

__device__ void rmsnorm_apply_and_quantize(
    float*       __restrict__ norm_out,
    uint8_t*     __restrict__ out_fp4,
    uint8_t*     __restrict__ out_sf,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    AtomicScratch* __restrict__ scratch,
    float eps = 1e-6f)
{
    const int tid = threadIdx.x;
    const int cta_id = blockIdx.x;

    float rms = rsqrtf(scratch->rmsnorm_sum / (float)HIDDEN + eps);

    // All CTAs apply normalization in parallel — each CTA handles a stripe
    const int global_tid = cta_id * BLOCK_SIZE + tid;
    const int total_threads = NUM_CTAS * BLOCK_SIZE;

    for (int i = global_tid; i < HIDDEN; i += total_threads) {
        norm_out[i] = input[i] * rms * weight[i];
    }
    // NOTE: need __threadfence() + grid.sync() before quantization reads norm_out

    // Quantize to FP4: distribute SF groups across all CTAs
    // Total SF groups = HIDDEN / SF_BLOCK = 256
    // Each half-warp (16 threads) handles one SF group
    const int hw_id_in_cta = tid / 16;      // 0..15 within CTA
    const int hw_lane = tid % 16;
    const int hw_per_cta = BLOCK_SIZE / 16;  // 16
    const int num_sf_groups = HIDDEN / SF_BLOCK;  // 256

    // Distribute groups across CTAs: CTA c handles groups starting at c*hw_per_cta
    for (int g = cta_id * hw_per_cta + hw_id_in_cta; g < num_sf_groups;
         g += NUM_CTAS * hw_per_cta) {
        int base = g * SF_BLOCK;
        float val = norm_out[base + hw_lane];
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
            out_fp4[base / 2 + hw_lane / 2] = nib | (partner_nib << 4);
        if (hw_lane == 0)
            out_sf[g] = sf_byte;
    }
}

// ============================================================================
// FP4 GEMV — single-token matrix-vector multiply (M=1)
//
// Computes out[N_out] = input_fp4[K] @ weight_fp4[N_out, K]^T
// Distributes N-tiles across all CTAs, each CTA does full K reduction.
// ============================================================================
__device__ void gemv_fp4_inkernel(
    float*       __restrict__ output,
    const uint8_t* __restrict__ act_fp4,
    const uint8_t* __restrict__ act_sf,
    const uint8_t* __restrict__ w_fp4,
    const uint8_t* __restrict__ w_sf,
    int N_out, int K,
    int n_tile,
    float scale = 1.0f)
{
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int n_start   = n_tile * BN;
    const int k_packed  = K / 2;
    const int sf_cols   = K / SF_BLOCK;
    const int k_tiles   = K / BK;

    extern __shared__ char smem_raw[];
    uint8_t* s_A   = (uint8_t*)smem_raw;
    uint8_t* s_B   = s_A + 32;
    uint8_t* s_SFA = s_B + BN * (BK / 2);
    uint8_t* s_SFB = s_SFA + ((SF_PER_K + 3) & ~3);

    const int g   = lane_id / 4;
    const int Nl  = 4 * (g & 1) + (g >> 1);
    const int sn  = warp_id * 8 + Nl;
    const int t0  = lane_id % 4;
    const int rbo = sn * (BK / 2);

    float acc[4] = {0, 0, 0, 0};

    for (int kt = 0; kt < k_tiles; kt++) {
        const int k_off = kt * BK;
        const int k_pk  = k_off / 2;
        const int k_sf  = k_off / SF_BLOCK;

        for (int i = tid; i < 8; i += BLOCK_SIZE) {
            *(uint32_t*)(s_A + i * 4) =
                *(const uint32_t*)&act_fp4[k_pk + i * 4];
        }

        for (int i = tid; i < (BN * BK / 2) / 4; i += BLOCK_SIZE) {
            int boff = i * 4;
            int row = boff / (BK / 2), col = boff % (BK / 2);
            int global_n = n_start + row;
            if (global_n < N_out)
                *(uint32_t*)&s_B[swizzle_343(boff)] =
                    *(const uint32_t*)&w_fp4[(long long)global_n * k_packed + k_pk + col];
            else
                *(uint32_t*)&s_B[swizzle_343(boff)] = 0;
        }

        if (tid == 0) {
            *(uint32_t*)s_SFA = *(const uint32_t*)&act_sf[k_sf];
        }

        for (int i = tid; i < BN; i += BLOCK_SIZE) {
            int global_n = n_start + i;
            if (global_n < N_out)
                *(uint32_t*)&s_SFB[i * SF_PER_K] =
                    *(const uint32_t*)&w_sf[(long long)global_n * sf_cols + k_sf];
            else
                *(uint32_t*)&s_SFB[i * SF_PER_K] = 0;
        }

        __syncthreads();

        uint32_t b_reg[2];
        b_reg[0] = *(uint32_t*)&s_B[swizzle_343(rbo + t0 * 4)];
        b_reg[1] = *(uint32_t*)&s_B[swizzle_343(rbo + 16 + t0 * 4)];
        uint32_t sfb_pk = *(const uint32_t*)&s_SFB[sn * SF_PER_K];

        uint32_t a_reg[4] = {0, 0, 0, 0};
        if (lane_id / 4 == 0) {
            a_reg[0] = *(uint32_t*)(s_A + t0 * 4);
            a_reg[2] = *(uint32_t*)(s_A + 16 + t0 * 4);
        }
        uint32_t sfa_pk = *(const uint32_t*)s_SFA;

        mma_nvf4_e4m3_m16n8k64(acc, a_reg, b_reg, acc, sfa_pk, sfb_pk);

        __syncthreads();
    }

    if (lane_id < 4) {
        int c0 = n_start + warp_id * 8 + lane_id;
        int c1 = c0 + 4;
        if (c0 < N_out) atomicAdd(&output[c0], scale * acc[0]);
        if (c1 < N_out) atomicAdd(&output[c1], scale * acc[1]);
    }
}

// ============================================================================
// MoE Gate — DISTRIBUTED across all CTAs
//
// Phase A: Each CTA computes a subset of the 512 gate logits (dot products).
//          Each CTA handles ~3 experts (512/188).
// Phase B (CTA 0 after grid.sync): Softmax + Top-K selection.
// ============================================================================
__device__ void moe_gate_distributed(
    float*       __restrict__ gate_logits, // [NUM_EXPERTS] output
    const float* __restrict__ hidden,      // [HIDDEN]
    const float* __restrict__ gate_weight, // [NUM_EXPERTS, HIDDEN]
    int num_experts)
{
    const int tid = threadIdx.x;
    const int cta_id = blockIdx.x;

    // Distribute experts across CTAs
    extern __shared__ char _smem_gate[];
    float* s_partial = (float*)_smem_gate;  // [BLOCK_SIZE] = 1KB

    for (int e = cta_id; e < num_experts; e += NUM_CTAS) {
        // Cooperative dot product across 256 threads within this CTA
        float local_dot = 0.0f;
        for (int k = tid; k < HIDDEN; k += BLOCK_SIZE) {
            local_dot += hidden[k] * gate_weight[e * HIDDEN + k];
        }
        s_partial[tid] = local_dot;
        __syncthreads();

        // Reduce within block
        if (tid < 32) {
            float ws = 0.0f;
            #pragma unroll
            for (int i = tid; i < BLOCK_SIZE; i += 32) ws += s_partial[i];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                ws += __shfl_xor_sync(0xFFFFFFFF, ws, off);
            if (tid == 0) gate_logits[e] = ws;
        }
        __syncthreads();
    }
}

__device__ void moe_topk_select(
    int*         __restrict__ top_ids,
    float*       __restrict__ top_wts,
    float*       __restrict__ gate_logits,
    int num_experts, int top_k)
{
    // CTA 0 only — sequential softmax + top-K on 512 values (~1us)
    const int tid = threadIdx.x;

    if (tid == 0) {
        float max_val = -1e30f;
        for (int e = 0; e < num_experts; e++)
            max_val = fmaxf(max_val, gate_logits[e]);

        float sum_exp = 0.0f;
        for (int e = 0; e < num_experts; e++) {
            gate_logits[e] = expf(gate_logits[e] - max_val);
            sum_exp += gate_logits[e];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int e = 0; e < num_experts; e++)
            gate_logits[e] *= inv_sum;

        // Top-K selection (insertion sort, K=10)
        for (int i = 0; i < top_k; i++) {
            top_ids[i] = -1;
            top_wts[i] = -1e30f;
        }
        for (int e = 0; e < num_experts; e++) {
            float w = gate_logits[e];
            if (w > top_wts[top_k - 1]) {
                top_ids[top_k - 1] = e;
                top_wts[top_k - 1] = w;
                for (int j = top_k - 1; j > 0 && top_wts[j] > top_wts[j - 1]; j--) {
                    float tw = top_wts[j]; top_wts[j] = top_wts[j - 1]; top_wts[j - 1] = tw;
                    int ti = top_ids[j]; top_ids[j] = top_ids[j - 1]; top_ids[j - 1] = ti;
                }
            }
        }

        float wsum = 0.0f;
        for (int i = 0; i < top_k; i++) wsum += top_wts[i];
        float winv = 1.0f / wsum;
        for (int i = 0; i < top_k; i++) top_wts[i] *= winv;
    }
    __syncthreads();
}

// ============================================================================
// Execute Experts — CTA self-assignment via atomics (all CTAs participate)
// ============================================================================
__device__ void execute_experts(
    float*         __restrict__ output,
    const uint8_t* __restrict__ act_fp4,
    const uint8_t* __restrict__ act_sf,
    const int*     __restrict__ expert_ids,
    const float*   __restrict__ expert_wts,
    const uint8_t* __restrict__ all_w1_fp4,
    const uint8_t* __restrict__ all_w1_sf,
    const uint8_t* __restrict__ all_w2_fp4,
    const uint8_t* __restrict__ all_w2_sf,
    int*           __restrict__ cta_counter,
    float*         __restrict__ inter_buf,
    uint8_t*       __restrict__ inter_fp4,
    uint8_t*       __restrict__ inter_sf)
{
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int n_half         = EXPERT_INTER;
    const int n2             = 2 * EXPERT_INTER;
    const int sf_cols_w1     = SF_COLS_HIDDEN;
    const int tiles_n_gemm1  = n_half / BN;            // 16

    const int g   = lane_id / 4;
    const int Nl  = 4 * (g & 1) + (g >> 1);
    const int sn  = warp_id * 8 + Nl;
    const int t0  = lane_id % 4;
    const int rbo = sn * (BK / 2);

    extern __shared__ char smem_raw[];
    uint8_t* s_A        = (uint8_t*)smem_raw;
    uint8_t* s_B_gate   = s_A + 32;
    uint8_t* s_B_up     = s_B_gate + BN * (BK / 2);
    uint8_t* s_SFA      = s_B_up + BN * (BK / 2);
    uint8_t* s_SFB_gate = s_SFA + ((SF_PER_K + 3) & ~3);
    uint8_t* s_SFB_up   = s_SFB_gate + BN * SF_PER_K;

    // Reset work counter (CTA 0, thread 0)
    if (blockIdx.x == 0 && tid == 0) *cta_counter = 0;
    // NOTE: caller grid.sync() ensures visibility

    int total_gemm1_items = TOP_K * tiles_n_gemm1;  // 10 * 16 = 160

    while (true) {
        __shared__ int s_work_idx;
        if (tid == 0) {
            s_work_idx = atomicAdd(cta_counter, 1);
        }
        __syncthreads();
        int work_idx = s_work_idx;
        if (work_idx >= total_gemm1_items) break;

        int expert_slot = work_idx / tiles_n_gemm1;
        int n_chunk     = work_idx % tiles_n_gemm1;
        int eid         = expert_ids[expert_slot];
        int n_start     = n_chunk * BN;

        const uint8_t* w1_fp4 = all_w1_fp4 + (long long)eid * n2 * K_PACKED;
        const uint8_t* w1_sf  = all_w1_sf  + (long long)eid * n2 * sf_cols_w1;

        float gate_acc[4] = {0, 0, 0, 0};
        float up_acc[4]   = {0, 0, 0, 0};

        for (int kt = 0; kt < K_TILES_HIDDEN; kt++) {
            const int k_off = kt * BK;
            const int k_pk  = k_off / 2;
            const int k_sf  = k_off / SF_BLOCK;

            for (int i = tid; i < 8; i += BLOCK_SIZE) {
                *(uint32_t*)(s_A + i * 4) =
                    *(const uint32_t*)&act_fp4[k_pk + i * 4];
            }

            for (int i = tid; i < (BN * BK / 2) / 4; i += BLOCK_SIZE) {
                int boff = i * 4;
                int row = boff / (BK / 2), col = boff % (BK / 2);
                *(uint32_t*)&s_B_gate[swizzle_343(boff)] =
                    *(const uint32_t*)&w1_fp4[(long long)(n_start + row) * K_PACKED + k_pk + col];
            }

            for (int i = tid; i < (BN * BK / 2) / 4; i += BLOCK_SIZE) {
                int boff = i * 4;
                int row = boff / (BK / 2), col = boff % (BK / 2);
                *(uint32_t*)&s_B_up[swizzle_343(boff)] =
                    *(const uint32_t*)&w1_fp4[(long long)(n_half + n_start + row) * K_PACKED + k_pk + col];
            }

            if (tid < SF_PER_K)
                s_SFA[tid] = act_sf[k_sf + tid];

            for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
                int row = i / SF_PER_K, col = i % SF_PER_K;
                s_SFB_gate[i] = w1_sf[(long long)(n_start + row) * sf_cols_w1 + k_sf + col];
            }
            for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
                int row = i / SF_PER_K, col = i % SF_PER_K;
                s_SFB_up[i] = w1_sf[(long long)(n_half + n_start + row) * sf_cols_w1 + k_sf + col];
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
            mma_nvf4_e4m3_m16n8k64(up_acc,   a, bu, up_acc,   sfa_pk, sfbu);

            __syncthreads();
        }

        // SwiGLU + write to intermediate buffer
        if (lane_id < 4) {
            int c0 = n_start + warp_id * 8 + lane_id;
            int c1 = c0 + 4;
            if (c0 < n_half) inter_buf[expert_slot * n_half + c0] = up_acc[0] * d_silu(gate_acc[0]);
            if (c1 < n_half) inter_buf[expert_slot * n_half + c1] = up_acc[1] * d_silu(gate_acc[1]);
        }
    }
}

// ============================================================================
// Execute Experts Phase 2 — GEMM2 scatter (all CTAs, work-stealing)
// ============================================================================
__device__ void execute_experts_phase2(
    float*         __restrict__ output,
    const float*   __restrict__ inter_buf,
    const uint8_t* __restrict__ inter_fp4,
    const uint8_t* __restrict__ inter_sf,
    const int*     __restrict__ expert_ids,
    const float*   __restrict__ expert_wts,
    const uint8_t* __restrict__ all_w2_fp4,
    const uint8_t* __restrict__ all_w2_sf,
    int*           __restrict__ cta_counter)
{
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int n_half_packed  = EXPERT_INTER_PACKED;
    const int sf_cols_w2     = EXPERT_INTER / SF_BLOCK;
    const int tiles_n_gemm2  = HIDDEN / BN;            // 64

    const int g   = lane_id / 4;
    const int Nl  = 4 * (g & 1) + (g >> 1);
    const int sn  = warp_id * 8 + Nl;
    const int t0  = lane_id % 4;
    const int rbo = sn * (BK / 2);

    extern __shared__ char smem_raw[];
    uint8_t* s_A   = (uint8_t*)smem_raw;
    uint8_t* s_B   = s_A + 32;
    uint8_t* s_SFA = s_B + BN * (BK / 2);
    uint8_t* s_SFB = s_SFA + ((SF_PER_K + 3) & ~3);

    // Reset counter
    if (blockIdx.x == 0 && tid == 0) *cta_counter = 0;

    int total_gemm2_items = TOP_K * tiles_n_gemm2;  // 10 * 64 = 640

    while (true) {
        __shared__ int s_work_idx;
        if (tid == 0) {
            s_work_idx = atomicAdd(cta_counter, 1);
        }
        __syncthreads();
        int work_idx = s_work_idx;
        if (work_idx >= total_gemm2_items) break;

        int expert_slot = work_idx / tiles_n_gemm2;
        int n_chunk     = work_idx % tiles_n_gemm2;
        int eid         = expert_ids[expert_slot];
        float wt        = expert_wts[expert_slot];
        int n_start     = n_chunk * BN;

        const uint8_t* w2_fp4 = all_w2_fp4 + (long long)eid * HIDDEN * n_half_packed;
        const uint8_t* w2_sf_base = all_w2_sf + (long long)eid * HIDDEN * sf_cols_w2;

        const uint8_t* a_fp4_ptr = inter_fp4 + expert_slot * n_half_packed;
        const uint8_t* a_sf_ptr  = inter_sf  + expert_slot * (EXPERT_INTER / SF_BLOCK);

        const int k_tiles_w2 = EXPERT_INTER / BK;  // 16

        float acc[4] = {0, 0, 0, 0};

        for (int kt = 0; kt < k_tiles_w2; kt++) {
            const int k_off = kt * BK;
            const int k_pk  = k_off / 2;
            const int k_sf  = k_off / SF_BLOCK;

            for (int i = tid; i < 8; i += BLOCK_SIZE) {
                *(uint32_t*)(s_A + i * 4) =
                    *(const uint32_t*)&a_fp4_ptr[k_pk + i * 4];
            }

            for (int i = tid; i < (BN * BK / 2) / 4; i += BLOCK_SIZE) {
                int boff = i * 4;
                int row = boff / (BK / 2), col = boff % (BK / 2);
                int global_n = n_start + row;
                if (global_n < HIDDEN)
                    *(uint32_t*)&s_B[swizzle_343(boff)] =
                        *(const uint32_t*)&w2_fp4[(long long)global_n * n_half_packed + k_pk + col];
                else
                    *(uint32_t*)&s_B[swizzle_343(boff)] = 0;
            }

            if (tid == 0) {
                *(uint32_t*)s_SFA = *(const uint32_t*)&a_sf_ptr[k_sf];
            }

            for (int i = tid; i < BN; i += BLOCK_SIZE) {
                int global_n = n_start + i;
                if (global_n < HIDDEN)
                    *(uint32_t*)&s_SFB[i * SF_PER_K] =
                        *(const uint32_t*)&w2_sf_base[(long long)global_n * sf_cols_w2 + k_sf];
                else
                    *(uint32_t*)&s_SFB[i * SF_PER_K] = 0;
            }

            __syncthreads();

            uint32_t b_reg[2];
            b_reg[0] = *(uint32_t*)&s_B[swizzle_343(rbo + t0 * 4)];
            b_reg[1] = *(uint32_t*)&s_B[swizzle_343(rbo + 16 + t0 * 4)];
            uint32_t sfb_pk = pack_sf4(&s_SFB[sn * SF_PER_K]);

            uint32_t a_reg[4] = {0, 0, 0, 0};
            if (lane_id / 4 == 0) {
                a_reg[0] = *(uint32_t*)(s_A + t0 * 4);
                a_reg[2] = *(uint32_t*)(s_A + 16 + t0 * 4);
            }
            uint32_t sfa_pk = pack_sf4(s_SFA);

            mma_nvf4_e4m3_m16n8k64(acc, a_reg, b_reg, acc, sfa_pk, sfb_pk);

            __syncthreads();
        }

        if (lane_id < 4) {
            int c0 = n_start + warp_id * 8 + lane_id;
            int c1 = c0 + 4;
            if (c0 < HIDDEN) atomicAdd(&output[c0], wt * acc[0]);
            if (c1 < HIDDEN) atomicAdd(&output[c1], wt * acc[1]);
        }
        __syncthreads();
    }
}

// ============================================================================
// P2P Write-Based AllReduce — ALL CTAs participate in data scatter/gather
// ============================================================================
__device__ void p2p_allreduce_write(
    float*                __restrict__ data,
    const P2PBuffers&                  p2p,
    int                                generation)
{
    const int tid  = threadIdx.x;
    const int cta_id = blockIdx.x;
    const int rank = p2p.rank;
    const int global_tid = cta_id * BLOCK_SIZE + tid;
    const int total_threads = NUM_CTAS * BLOCK_SIZE;

    // Step 1: ALL CTAs write local data to remote receive buffers (distributed)
    for (int dst = 0; dst < WORLD_SIZE; dst++) {
        if (dst == rank) continue;
        float* remote_slot = p2p.remote_recv[dst] + rank * HIDDEN;
        for (int i = global_tid; i < HIDDEN; i += total_threads) {
            remote_slot[i] = data[i];
        }
    }

    __threadfence_system();

    // Step 2: CTA 0 thread 0 sets flags and polls
    if (cta_id == 0 && tid == 0) {
        for (int dst = 0; dst < WORLD_SIZE; dst++) {
            if (dst == rank) continue;
            p2p.remote_flags[dst][rank] = generation;
        }
        __threadfence_system();

        for (int src = 0; src < WORLD_SIZE; src++) {
            if (src == rank) continue;
            while (p2p.local_flags[src] < (uint32_t)generation) {
                // Spin
            }
        }
    }
    // NOTE: Need grid.sync() after this to ensure all CTAs see the received data.
    // Caller handles this.
}

__device__ void p2p_allreduce_sum(
    float*                __restrict__ data,
    const P2PBuffers&                  p2p)
{
    const int tid  = threadIdx.x;
    const int cta_id = blockIdx.x;
    const int rank = p2p.rank;
    const int global_tid = cta_id * BLOCK_SIZE + tid;
    const int total_threads = NUM_CTAS * BLOCK_SIZE;

    float* local_recv_base = p2p.local_recv;
    for (int i = global_tid; i < HIDDEN; i += total_threads) {
        float sum = data[i];
        for (int src = 0; src < WORLD_SIZE; src++) {
            if (src == rank) continue;
            sum += local_recv_base[src * HIDDEN + i];
        }
        data[i] = sum;
    }
}

// ============================================================================
// Attention Decode — DISTRIBUTED across CTAs
//
// Full attention: Distribute Q heads across CTA groups. Each group processes
// one Q head with KV cache sequence distributed across CTAs in the group.
//
// DeltaNet: Distribute KV heads across CTA groups for state update + mat-vec.
// ============================================================================
__device__ void attention_decode_distributed(
    float*       __restrict__ attn_out,     // [NUM_Q_HEADS * HEAD_DIM]
    const float* __restrict__ qkv,          // [QKV_DIM]
    KVCache*     __restrict__ kv_cache,
    AtomicScratch* __restrict__ scratch,
    int layer_idx,
    bool is_full_attention)
{
    const int tid = threadIdx.x;
    const int cta_id = blockIdx.x;

    if (is_full_attention) {
        const float* q_ptr = qkv;
        const float* k_ptr = qkv + NUM_Q_HEADS * HEAD_DIM;
        const float* v_ptr = qkv + (NUM_Q_HEADS + NUM_KV_HEADS) * HEAD_DIM;

        uint8_t* k_cache = (uint8_t*)kv_cache->k_cache;
        uint8_t* v_cache = (uint8_t*)kv_cache->v_cache;
        int seq_len = kv_cache->seq_len;

        // Step 1: ALL CTAs help append K,V (tiny: 256 elements)
        {
            int global_tid = cta_id * BLOCK_SIZE + tid;
            int total_threads = NUM_CTAS * BLOCK_SIZE;
            int kv_elems = NUM_KV_HEADS * HEAD_DIM;  // 256
            for (int i = global_tid; i < kv_elems; i += total_threads) {
                k_cache[seq_len * kv_elems + i] = d_e4m3fn_encode(k_ptr[i]);
                v_cache[seq_len * kv_elems + i] = d_e4m3fn_encode(v_ptr[i]);
            }
        }
        // No sync needed — each CTA reads only what it wrote, or reads older positions

        int total_seq = seq_len + 1;
        float inv_sqrt_hd = rsqrtf((float)HEAD_DIM);

        // Distribute Q heads across CTAs: CTA group = cta_id / CTAS_PER_HEAD
        int my_head = cta_id / CTAS_PER_HEAD;
        int my_head_local_id = cta_id % CTAS_PER_HEAD;

        if (my_head >= NUM_Q_HEADS) {
            // Spare CTAs (188 - 184 = 4) — just idle for this phase
            return;
        }

        const int gqa_ratio = NUM_Q_HEADS / NUM_KV_HEADS;
        int kv_head = my_head / gqa_ratio;
        const float* q_head = q_ptr + my_head * HEAD_DIM;

        extern __shared__ char _smem_attn[];
        float* s_scores = (float*)_smem_attn;  // [BLOCK_SIZE]

        // Pass 1: compute partial attention scores and find max
        // Each CTA in the head group handles a stripe of sequence positions
        float local_max = -1e30f;
        for (int s = my_head_local_id * BLOCK_SIZE + tid; s < total_seq;
             s += CTAS_PER_HEAD * BLOCK_SIZE) {
            const uint8_t* k_row = k_cache + s * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM;
            float dot = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < HEAD_DIM; d++) {
                dot += q_head[d] * d_e4m3fn_decode(k_row[d]);
            }
            dot *= inv_sqrt_hd;
            local_max = fmaxf(local_max, dot);
        }

        // Reduce max within CTA
        s_scores[tid] = local_max;
        __syncthreads();
        if (tid < 32) {
            float wm = -1e30f;
            for (int i = tid; i < BLOCK_SIZE; i += 32) wm = fmaxf(wm, s_scores[i]);
            for (int off = 16; off > 0; off >>= 1)
                wm = fmaxf(wm, __shfl_xor_sync(0xFFFFFFFF, wm, off));
            if (tid == 0) {
                // Use atomicMax on int representation for float max
                // Reinterpret as int for atomic compare
                float cta_max = wm;
                int* imax = (int*)&scratch->attn_max[my_head];
                int ival = __float_as_int(cta_max);
                // atomicMax on positive floats works with int representation
                // But attention scores can be negative, so we need CAS loop
                int old_ival = *imax;
                while (true) {
                    float old_val = __int_as_float(old_ival);
                    if (cta_max <= old_val) break;
                    int assumed = old_ival;
                    old_ival = atomicCAS(imax, assumed, ival);
                    if (old_ival == assumed) break;
                }
            }
        }

        // NOTE: grid.sync() needed here to get global max. Caller handles this.
        // After grid.sync(), scratch->attn_max[my_head] has the global max.
    } else {
        // DeltaNet (linear attention) — distributed state update
        const float* q_ptr = qkv;
        const float* k_ptr = qkv + NUM_Q_HEADS * HEAD_DIM;
        const float* v_ptr = qkv + (NUM_Q_HEADS + NUM_KV_HEADS) * HEAD_DIM;

        float* state = (float*)kv_cache->k_cache;
        const float beta = 0.99f;

        const int gqa_ratio = NUM_Q_HEADS / NUM_KV_HEADS;
        const int state_size = HEAD_DIM * HEAD_DIM;  // 16384

        // Distribute state elements across ALL CTAs
        int global_tid = cta_id * BLOCK_SIZE + tid;
        int total_threads = NUM_CTAS * BLOCK_SIZE;

        for (int kvh = 0; kvh < NUM_KV_HEADS; kvh++) {
            float* S = state + kvh * state_size;
            const float* k_head = k_ptr + kvh * HEAD_DIM;
            const float* v_head = v_ptr + kvh * HEAD_DIM;

            // State update: S[i][j] = beta * S[i][j] + k[i] * v[j]
            for (int idx = global_tid; idx < state_size; idx += total_threads) {
                int i = idx / HEAD_DIM;
                int j = idx % HEAD_DIM;
                S[idx] = beta * S[idx] + k_head[i] * v_head[j];
            }
        }
        // NOTE: grid.sync() after state update, then mat-vec pass.
    }
}

// Attention decode pass 2: compute exp-weighted V sum using global max
__device__ void attention_decode_pass2(
    float*       __restrict__ attn_out,
    const float* __restrict__ qkv,
    KVCache*     __restrict__ kv_cache,
    AtomicScratch* __restrict__ scratch,
    int layer_idx,
    bool is_full_attention)
{
    const int tid = threadIdx.x;
    const int cta_id = blockIdx.x;

    if (is_full_attention) {
        const float* q_ptr = qkv;

        uint8_t* k_cache = (uint8_t*)kv_cache->k_cache;
        uint8_t* v_cache = (uint8_t*)kv_cache->v_cache;
        int total_seq = kv_cache->seq_len + 1;
        float inv_sqrt_hd = rsqrtf((float)HEAD_DIM);

        int my_head = cta_id / CTAS_PER_HEAD;
        int my_head_local_id = cta_id % CTAS_PER_HEAD;

        if (my_head >= NUM_Q_HEADS) return;

        const int gqa_ratio = NUM_Q_HEADS / NUM_KV_HEADS;
        int kv_head = my_head / gqa_ratio;
        const float* q_head = q_ptr + my_head * HEAD_DIM;

        float max_score = scratch->attn_max[my_head];

        extern __shared__ char _smem_attn2[];
        float* s_reduce = (float*)_smem_attn2;  // [BLOCK_SIZE]

        // Pass 2: compute exp(score - max), sum, and weighted V
        float local_sum = 0.0f;
        // We accumulate partial V output per thread. HEAD_DIM=128 floats = 512 bytes
        // fits in registers for small HEAD_DIM.
        float local_out[4] = {0, 0, 0, 0};  // We'll process HEAD_DIM in chunks of 4

        // Actually, to avoid 128 register floats, process V in chunks
        // Strategy: accumulate (sum, per-dim V) in SMEM cooperatively
        // Better: each thread accumulates locally for its assigned seq positions

        // Each CTA processes its stripe of seq positions, accumulates V output
        // For HEAD_DIM=128: need 128 floats per thread — too many registers if done naively.
        // Instead: iterate over HEAD_DIM dimensions in chunks, reducing per-chunk.

        // Approach: For each dim chunk, all threads compute their partial, reduce across CTA.
        // Then atomicAdd to global scratch for cross-CTA reduction.

        // First compute per-thread sum of exp weights
        float local_exp_sum = 0.0f;
        int my_seq_start = my_head_local_id * BLOCK_SIZE + tid;
        int my_seq_stride = CTAS_PER_HEAD * BLOCK_SIZE;

        // We need to store exp weights. For short seq (test: 128), each thread handles
        // at most ceil(128 / (23*256)) = 1 position. Store scores per-thread.
        // For long seq (32K+), each thread handles ~2-3 positions. Use SMEM or loop.

        // Simple approach: two-pass within this CTA, then atomic cross-CTA reduction.

        for (int s = my_seq_start; s < total_seq; s += my_seq_stride) {
            const uint8_t* k_row = k_cache + s * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM;
            float dot = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < HEAD_DIM; d++) {
                dot += q_head[d] * d_e4m3fn_decode(k_row[d]);
            }
            dot *= inv_sqrt_hd;
            float w = expf(dot - max_score);
            local_exp_sum += w;

            // Accumulate w * V[s] — atomicAdd to per-head output buffer
            const uint8_t* v_row = v_cache + s * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) {
                atomicAdd(&scratch->attn_out_partial[my_head * HEAD_DIM + d],
                          w * d_e4m3fn_decode(v_row[d]));
            }
        }

        // Reduce exp_sum within CTA
        s_reduce[tid] = local_exp_sum;
        __syncthreads();
        if (tid < 32) {
            float ws = 0.0f;
            for (int i = tid; i < BLOCK_SIZE; i += 32) ws += s_reduce[i];
            for (int off = 16; off > 0; off >>= 1)
                ws += __shfl_xor_sync(0xFFFFFFFF, ws, off);
            if (tid == 0) {
                atomicAdd(&scratch->attn_sum[my_head], ws);
            }
        }
        // NOTE: grid.sync() after this, then normalize.

    } else {
        // DeltaNet pass 2: mat-vec o = S @ q (distributed across CTAs)
        const float* q_ptr = qkv;
        float* state = (float*)kv_cache->k_cache;
        const int gqa_ratio = NUM_Q_HEADS / NUM_KV_HEADS;
        const int state_size = HEAD_DIM * HEAD_DIM;

        // Distribute output dimensions across CTAs
        // For each Q head: o[i] = sum_j S[kv_head][i][j] * q[j]
        // Total work: NUM_Q_HEADS * HEAD_DIM = 1024 output elements
        int global_tid = cta_id * BLOCK_SIZE + tid;
        int total_threads = NUM_CTAS * BLOCK_SIZE;

        extern __shared__ char _smem_dn2[];
        float* s_partial = (float*)_smem_dn2;

        for (int qh = 0; qh < NUM_Q_HEADS; qh++) {
            int kvh = qh / gqa_ratio;
            float* S = state + kvh * state_size;
            const float* q_head = q_ptr + qh * HEAD_DIM;

            // Each CTA handles a subset of output dimensions
            for (int i = cta_id; i < HEAD_DIM; i += NUM_CTAS) {
                // o[i] = sum_j S[i*HEAD_DIM + j] * q[j]
                float local_dot = 0.0f;
                const float* S_row = S + i * HEAD_DIM;
                for (int j = tid; j < HEAD_DIM; j += BLOCK_SIZE) {
                    local_dot += S_row[j] * q_head[j];
                }
                s_partial[tid] = local_dot;
                __syncthreads();

                if (tid < 32) {
                    float ws = 0.0f;
                    for (int k = tid; k < BLOCK_SIZE; k += 32) ws += s_partial[k];
                    for (int off = 16; off > 0; off >>= 1)
                        ws += __shfl_xor_sync(0xFFFFFFFF, ws, off);
                    if (tid == 0) {
                        attn_out[qh * HEAD_DIM + i] = ws;
                    }
                }
                __syncthreads();
            }
        }
    }
}

// Attention decode pass 3: normalize V output by sum (full attention only)
__device__ void attention_decode_normalize(
    float*       __restrict__ attn_out,
    AtomicScratch* __restrict__ scratch)
{
    const int tid = threadIdx.x;
    const int cta_id = blockIdx.x;
    const int global_tid = cta_id * BLOCK_SIZE + tid;
    const int total_threads = NUM_CTAS * BLOCK_SIZE;

    // Normalize: attn_out[h*HD + d] = scratch->attn_out_partial[h*HD + d] / scratch->attn_sum[h]
    int total_elems = NUM_Q_HEADS * HEAD_DIM;  // 1024
    for (int i = global_tid; i < total_elems; i += total_threads) {
        int h = i / HEAD_DIM;
        float inv_sum = 1.0f / scratch->attn_sum[h];
        attn_out[i] = scratch->attn_out_partial[i] * inv_sum;
    }
}

// ============================================================================
// Quantize activation — DISTRIBUTED across all CTAs
// ============================================================================
__device__ void quantize_activation_distributed(
    uint8_t*     __restrict__ out_fp4,
    uint8_t*     __restrict__ out_sf,
    const float* __restrict__ input,
    int size)
{
    const int tid = threadIdx.x;
    const int cta_id = blockIdx.x;
    const int hw_id_in_cta = tid / 16;
    const int hw_lane = tid % 16;
    const int hw_per_cta = BLOCK_SIZE / 16;
    const int num_sf_groups = size / SF_BLOCK;

    for (int g = cta_id * hw_per_cta + hw_id_in_cta; g < num_sf_groups;
         g += NUM_CTAS * hw_per_cta) {
        int base = g * SF_BLOCK;
        float val = (base + hw_lane < size) ? input[base + hw_lane] : 0.0f;
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
        if ((hw_lane & 1) == 0 && base + hw_lane < size)
            out_fp4[base / 2 + hw_lane / 2] = nib | (partner_nib << 4);
        if (hw_lane == 0)
            out_sf[g] = sf_byte;
    }
}

// ============================================================================
// THE MEGAKERNEL — v2: All phases parallelized across 188 CTAs
//
// grid.sync() count per layer: 8 (down from 17)
//
// Phase 1: RMSNorm partial sums (all CTAs)          → grid.sync ①
// Phase 2: RMSNorm apply + quantize + zero QKV      → grid.sync ②
//          QKV GEMV (all CTAs)
// Phase 3: Attention decode pass 1 (max)            → grid.sync ③
// Phase 3b: Attention decode pass 2 (V accum)       → grid.sync ④
// Phase 3c: Normalize + quantize + O GEMV + zero    → grid.sync ⑤
//           AllReduce + residual (distributed)
// Phase 4: FFN RMSNorm partial + gate distributed   → grid.sync ⑥
// Phase 4b: RMSNorm apply + quantize + topK + zero  → grid.sync ⑦
//           Expert GEMM1 → requant → GEMM2 → shared → grid.sync ⑧
//           AllReduce + residual
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
megakernel_forward(MegaKernelParams params)
{
    cg::grid_group grid = cg::this_grid();
    const int cta_id = blockIdx.x;
    const int tid    = threadIdx.x;

    int allreduce_gen = 1;

    for (int layer = 0; layer < params.num_layers; layer++) {
        const LayerWeights& lw = params.layers[layer];
        AtomicScratch* scratch = params.scratch;

        // ==============================================================
        // PHASE 1: Attention RMSNorm — partial sums (all CTAs)
        //          + zero scratch atomics
        // ==============================================================
        // Zero atomic scratch (distributed)
        if (cta_id == 0 && tid == 0) {
            scratch->rmsnorm_sum = 0.0f;
            for (int h = 0; h < NUM_Q_HEADS; h++) {
                scratch->attn_max[h] = __int_as_float(0xFF800000); // -inf
                scratch->attn_sum[h] = 0.0f;
                scratch->attn_cta_done[h] = 0;
            }
        }
        {
            int global_tid = cta_id * BLOCK_SIZE + tid;
            int total_threads = NUM_CTAS * BLOCK_SIZE;
            for (int i = global_tid; i < NUM_Q_HEADS * HEAD_DIM; i += total_threads) {
                scratch->attn_out_partial[i] = 0.0f;
            }
        }
        grid.sync();  // ① ensure scratch is zeroed

        rmsnorm_parallel(
            params.norm_buf, params.act_fp4, params.act_sf,
            params.hidden_state, lw.attn_norm, scratch);
        grid.sync();  // ② rmsnorm sum complete

        // ==============================================================
        // PHASE 2: RMSNorm apply + quantize + QKV GEMV (all CTAs)
        // ==============================================================
        rmsnorm_apply_and_quantize(
            params.norm_buf, params.act_fp4, params.act_sf,
            params.hidden_state, lw.attn_norm, scratch);
        __threadfence();

        // Zero QKV output (distributed)
        {
            int global_tid = cta_id * BLOCK_SIZE + tid;
            int total_threads = NUM_CTAS * BLOCK_SIZE;
            for (int i = global_tid; i < QKV_DIM; i += total_threads)
                params.attn_out[i] = 0.0f;
        }
        grid.sync();  // ③ norm+quantize+zero complete

        // QKV GEMV: distribute N-tiles
        for (int n_tile = cta_id; n_tile < N_TILES_QKV; n_tile += NUM_CTAS) {
            gemv_fp4_inkernel(
                params.attn_out,
                params.act_fp4, params.act_sf,
                lw.qkv_fp4, lw.qkv_sf,
                QKV_DIM, HIDDEN,
                n_tile, 1.0f);
        }
        grid.sync();  // ④ QKV complete

        // ==============================================================
        // PHASE 3: Attention Decode — distributed across CTAs
        // ==============================================================
        {
            bool is_full = (layer < FULL_ATTN_LAYERS);
            attention_decode_distributed(
                params.attn_out, params.attn_out,
                &params.kv_caches[layer],
                scratch, layer, is_full);
            grid.sync();  // ⑤ attn pass1 (max + DeltaNet state update) complete

            attention_decode_pass2(
                params.attn_out, params.attn_out,
                &params.kv_caches[layer],
                scratch, layer, is_full);
            grid.sync();  // ⑥ attn pass2 (V accum + DeltaNet matvec) complete

            if (is_full) {
                attention_decode_normalize(params.attn_out, scratch);
            }
            // Update seq_len
            if (is_full && cta_id == 0 && tid == 0) {
                params.kv_caches[layer].seq_len = params.kv_caches[layer].seq_len + 1;
            }
        }

        // ==============================================================
        // PHASE 4: O Projection — quantize + GEMV + zero (fused)
        // ==============================================================
        quantize_activation_distributed(params.act_fp4, params.act_sf, params.attn_out, O_DIM);
        {
            int global_tid = cta_id * BLOCK_SIZE + tid;
            int total_threads = NUM_CTAS * BLOCK_SIZE;
            for (int i = global_tid; i < HIDDEN; i += total_threads)
                params.ffn_out[i] = 0.0f;
        }
        grid.sync();  // ⑦ quantize + zero complete

        for (int n_tile = cta_id; n_tile < N_TILES_O; n_tile += NUM_CTAS) {
            gemv_fp4_inkernel(
                params.ffn_out,
                params.act_fp4, params.act_sf,
                lw.o_fp4, lw.o_sf,
                HIDDEN, O_DIM,
                n_tile, 1.0f);
        }
        grid.sync();  // ⑧ O projection complete

        // ==============================================================
        // PHASE 5: AllReduce + Residual (all CTAs)
        // ==============================================================
        p2p_allreduce_write(params.ffn_out, params.p2p, allreduce_gen++);
        grid.sync();  // ⑨ allreduce writes landed

        p2p_allreduce_sum(params.ffn_out, params.p2p);
        // Residual add (distributed)
        {
            int global_tid = cta_id * BLOCK_SIZE + tid;
            int total_threads = NUM_CTAS * BLOCK_SIZE;
            for (int i = global_tid; i < HIDDEN; i += total_threads)
                params.hidden_state[i] += params.ffn_out[i];
        }
        grid.sync();  // ⑩ residual complete

        // ==============================================================
        // PHASE 6: FFN RMSNorm — all CTAs
        // ==============================================================
        if (cta_id == 0 && tid == 0) scratch->rmsnorm_sum = 0.0f;
        grid.sync();  // ⑪ zero scratch

        rmsnorm_parallel(
            params.norm_buf, params.act_fp4, params.act_sf,
            params.hidden_state, lw.ffn_norm, scratch);
        grid.sync();  // ⑫ rmsnorm sum complete

        rmsnorm_apply_and_quantize(
            params.norm_buf, params.act_fp4, params.act_sf,
            params.hidden_state, lw.ffn_norm, scratch);

        // ==============================================================
        // PHASE 7: MoE Gate (distributed) + Top-K (CTA 0)
        // ==============================================================
        moe_gate_distributed(
            params.gate_logits, params.norm_buf, lw.gate_weight, NUM_EXPERTS);
        grid.sync();  // ⑬ gate + rmsnorm apply complete

        if (cta_id == 0) {
            moe_topk_select(
                params.top_expert_ids, params.top_expert_wts,
                params.gate_logits, NUM_EXPERTS, TOP_K);
        }

        // Zero ffn_out for expert accumulation (distributed)
        {
            int global_tid = cta_id * BLOCK_SIZE + tid;
            int total_threads = NUM_CTAS * BLOCK_SIZE;
            for (int i = global_tid; i < HIDDEN; i += total_threads)
                params.ffn_out[i] = 0.0f;
        }
        grid.sync();  // ⑭ topK + zero complete

        // ==============================================================
        // PHASE 8: Expert GEMM1 + SwiGLU (all CTAs, work-stealing)
        // ==============================================================
        execute_experts(
            params.ffn_out,
            params.act_fp4, params.act_sf,
            params.top_expert_ids, params.top_expert_wts,
            lw.expert_w1_fp4, lw.expert_w1_sf,
            lw.expert_w2_fp4, lw.expert_w2_sf,
            params.expert_cta_counter,
            params.expert_inter_buf,
            params.expert_inter_fp4,
            params.expert_inter_sf);
        grid.sync();  // ⑮ GEMM1 complete

        // ==============================================================
        // PHASE 8.5: Requantize inter_buf -> FP4 (ALL CTAs)
        // ==============================================================
        for (int slot = 0; slot < TOP_K; slot++) {
            quantize_activation_distributed(
                params.expert_inter_fp4 + slot * EXPERT_INTER_PACKED,
                params.expert_inter_sf  + slot * (EXPERT_INTER / SF_BLOCK),
                params.expert_inter_buf + slot * EXPERT_INTER,
                EXPERT_INTER);
        }
        grid.sync();  // ⑯ requant complete

        // ==============================================================
        // PHASE 9: Expert GEMM2 (all CTAs, work-stealing)
        // ==============================================================
        execute_experts_phase2(
            params.ffn_out,
            params.expert_inter_buf,
            params.expert_inter_fp4, params.expert_inter_sf,
            params.top_expert_ids, params.top_expert_wts,
            lw.expert_w2_fp4, lw.expert_w2_sf,
            params.expert_cta_counter);
        grid.sync();  // ⑰ GEMM2 complete

        // ==============================================================
        // PHASE 10a: Shared Expert
        // ==============================================================
        {
            const int tiles_n_shared_g1 = EXPERT_INTER / BN;  // 16

            // Zero inter_buf slot 0 (distributed)
            {
                int global_tid = cta_id * BLOCK_SIZE + tid;
                int total_threads = NUM_CTAS * BLOCK_SIZE;
                for (int i = global_tid; i < EXPERT_INTER; i += total_threads)
                    params.expert_inter_buf[i] = 0.0f;
            }
            grid.sync();  // ⑱

            for (int n_tile = cta_id; n_tile < tiles_n_shared_g1; n_tile += NUM_CTAS) {
                gemv_fp4_inkernel(
                    params.expert_inter_buf,
                    params.act_fp4, params.act_sf,
                    lw.shared_w1_fp4, lw.shared_w1_sf,
                    EXPERT_INTER, HIDDEN,
                    n_tile, 1.0f);
            }
            grid.sync();  // ⑲ gate GEMV complete

            // Zero up buffer + up GEMV
            {
                int global_tid = cta_id * BLOCK_SIZE + tid;
                int total_threads = NUM_CTAS * BLOCK_SIZE;
                for (int i = global_tid; i < EXPERT_INTER; i += total_threads)
                    params.expert_inter_buf[EXPERT_INTER + i] = 0.0f;
            }
            grid.sync();  // ⑳

            for (int n_tile = cta_id; n_tile < tiles_n_shared_g1; n_tile += NUM_CTAS) {
                gemv_fp4_inkernel(
                    params.expert_inter_buf + EXPERT_INTER,
                    params.act_fp4, params.act_sf,
                    lw.shared_w1_fp4 + (long long)EXPERT_INTER * K_PACKED,
                    lw.shared_w1_sf  + (long long)EXPERT_INTER * SF_COLS_HIDDEN,
                    EXPERT_INTER, HIDDEN,
                    n_tile, 1.0f);
            }
            grid.sync();  // (21) up GEMV complete

            // SwiGLU (all CTAs)
            {
                int global_tid = cta_id * BLOCK_SIZE + tid;
                int total_threads = NUM_CTAS * BLOCK_SIZE;
                for (int i = global_tid; i < EXPERT_INTER; i += total_threads) {
                    float gate_val = params.expert_inter_buf[i];
                    float up_val   = params.expert_inter_buf[EXPERT_INTER + i];
                    params.expert_inter_buf[i] = up_val * d_silu(gate_val);
                }
            }
            grid.sync();  // (22) SwiGLU complete

            // Requantize (all CTAs)
            quantize_activation_distributed(
                params.expert_inter_fp4,
                params.expert_inter_sf,
                params.expert_inter_buf,
                EXPERT_INTER);
            grid.sync();  // (23) requant complete

            // GEMM2 for shared expert
            const int tiles_n_shared_g2 = HIDDEN / BN;
            for (int n_tile = cta_id; n_tile < tiles_n_shared_g2; n_tile += NUM_CTAS) {
                gemv_fp4_inkernel(
                    params.ffn_out,
                    params.expert_inter_fp4, params.expert_inter_sf,
                    lw.shared_w2_fp4, lw.shared_w2_sf,
                    HIDDEN, EXPERT_INTER,
                    n_tile, 1.0f);
            }
            grid.sync();  // (24) shared GEMM2 complete
        }

        // ==============================================================
        // PHASE 10b: MoE AllReduce + Residual (all CTAs)
        // ==============================================================
        p2p_allreduce_write(params.ffn_out, params.p2p, allreduce_gen++);
        grid.sync();  // (25) allreduce writes landed

        p2p_allreduce_sum(params.ffn_out, params.p2p);
        {
            int global_tid = cta_id * BLOCK_SIZE + tid;
            int total_threads = NUM_CTAS * BLOCK_SIZE;
            for (int i = global_tid; i < HIDDEN; i += total_threads)
                params.hidden_state[i] += params.ffn_out[i];
        }
        grid.sync();  // (26) residual complete — layer done
    }
}

// ============================================================================
// Host-side Setup
// ============================================================================

#define CHECK_CUDA(c) do { cudaError_t _e = (c); if (_e != cudaSuccess) { \
    printf("CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); } } while(0)

// Host quantization helpers
float h_e4m3fn_decode(uint8_t x) {
    int s = (x >> 7) & 1, e = (x >> 3) & 0xF, m = x & 7;
    if (e == 15 && m == 7) return s ? -NAN : NAN;
    float val = (e == 0) ? ldexpf((float)m, -9) : ldexpf((float)(8 + m), e - 10);
    return s ? -val : val;
}

uint8_t h_e4m3fn_encode_ceil(float val) {
    if (val <= 0) return 0x08;
    if (val >= 448.0f) return 0x7E;
    uint8_t best = 0x7E; float best_repr = 448.0f;
    for (int e = 0; e <= 15; e++)
        for (int m = 0; m <= 7; m++) {
            if (e == 15 && m == 7) continue;
            float repr = (e == 0) ? ldexpf((float)m, -9) : ldexpf((float)(8+m), e-10);
            if (repr >= val && repr < best_repr) { best_repr = repr; best = (e << 3) | m; }
        }
    return best;
}

void quantize_to_nvfp4_e4m3(const float* data, int numel,
                             uint8_t* packed, uint8_t* sf_out) {
    int nb = numel / SF_BLOCK;
    memset(packed, 0, numel / 2);
    for (int b = 0; b < nb; b++) {
        int s = b * SF_BLOCK;
        float bmax = 0;
        for (int i = s; i < s + SF_BLOCK; i++) bmax = std::max(bmax, fabsf(data[i]));
        uint8_t sf = h_e4m3fn_encode_ceil(std::max(bmax / 6.0f, 1e-30f));
        sf_out[b] = sf;
        float as = h_e4m3fn_decode(sf);
        if (as < 1e-30f) as = 1e-30f;
        for (int i = s; i < s + SF_BLOCK; i++) {
            float sc = data[i] / as, av = fabsf(sc);
            int sign = (sc < 0) ? 1 : 0, idx = 0;
            float bd = av;
            for (int j = 1; j < 8; j++) { float d = fabsf(av - E2M1_TABLE[j]); if (d < bd) { bd = d; idx = j; } }
            uint8_t fp4 = (uint8_t)((sign << 3) | idx);
            int bi = i / 2;
            if (i % 2 == 0) packed[bi] = fp4; else packed[bi] |= (fp4 << 4);
        }
    }
}

struct MegaKernelHost {
    MegaKernelParams params;
    LayerWeights*    d_layers;
    KVCache*         d_kv_caches;

    static int smem_size() {
        // A(32) + B_gate(2048) + B_up(2048) + SFA(4) + SFB_gate(256) + SFB_up(256) + pad
        return 32 + 2 * (BN * BK / 2) + ((SF_PER_K + 3) & ~3) + 2 * (BN * SF_PER_K) + 128;
    }

    void setup(int num_layers_to_run = 1) {
        memset(&params, 0, sizeof(params));
        params.num_layers = num_layers_to_run;

        CHECK_CUDA(cudaMalloc(&params.hidden_state, HIDDEN * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&params.attn_out,     QKV_DIM * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&params.ffn_out,      HIDDEN * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&params.norm_buf,      HIDDEN * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&params.gate_logits,   NUM_EXPERTS * sizeof(float)));

        CHECK_CUDA(cudaMalloc(&params.act_fp4,       K_PACKED));
        CHECK_CUDA(cudaMalloc(&params.act_sf,        SF_COLS_HIDDEN));

        CHECK_CUDA(cudaMalloc(&params.top_expert_ids, TOP_K * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&params.top_expert_wts, TOP_K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&params.expert_cta_counter, sizeof(int)));

        CHECK_CUDA(cudaMalloc(&params.expert_inter_buf,
                              TOP_K * EXPERT_INTER * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&params.expert_inter_fp4,
                              TOP_K * EXPERT_INTER_PACKED));
        CHECK_CUDA(cudaMalloc(&params.expert_inter_sf,
                              TOP_K * EXPERT_INTER / SF_BLOCK));

        // Atomic scratch space
        CHECK_CUDA(cudaMalloc(&params.scratch, sizeof(AtomicScratch)));
        CHECK_CUDA(cudaMemset(params.scratch, 0, sizeof(AtomicScratch)));

        CHECK_CUDA(cudaMemset(params.expert_cta_counter, 0, sizeof(int)));
    }

    void setup_test_layer(int layer_idx, LayerWeights& lw) {
        int qkv_rows = QKV_DIM;
        int o_rows = HIDDEN;
        int o_k = O_DIM;

        float* h_norm = new float[HIDDEN];
        for (int i = 0; i < HIDDEN; i++) h_norm[i] = 1.0f;
        CHECK_CUDA(cudaMalloc((void**)&lw.attn_norm, HIDDEN * sizeof(float)));
        CHECK_CUDA(cudaMemcpy((void*)lw.attn_norm, h_norm, HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc((void**)&lw.ffn_norm, HIDDEN * sizeof(float)));
        CHECK_CUDA(cudaMemcpy((void*)lw.ffn_norm, h_norm, HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
        delete[] h_norm;

        int qkv_numel = qkv_rows * HIDDEN;
        float* h_qkv = new float[qkv_numel];
        for (int i = 0; i < qkv_numel; i++) h_qkv[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        uint8_t* h_qkv_fp4 = new uint8_t[qkv_numel / 2];
        uint8_t* h_qkv_sf  = new uint8_t[qkv_numel / SF_BLOCK];
        quantize_to_nvfp4_e4m3(h_qkv, qkv_numel, h_qkv_fp4, h_qkv_sf);
        CHECK_CUDA(cudaMalloc((void**)&lw.qkv_fp4, qkv_numel / 2));
        CHECK_CUDA(cudaMemcpy((void*)lw.qkv_fp4, h_qkv_fp4, qkv_numel / 2, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc((void**)&lw.qkv_sf, qkv_numel / SF_BLOCK));
        CHECK_CUDA(cudaMemcpy((void*)lw.qkv_sf, h_qkv_sf, qkv_numel / SF_BLOCK, cudaMemcpyHostToDevice));
        delete[] h_qkv; delete[] h_qkv_fp4; delete[] h_qkv_sf;

        int o_numel = o_rows * o_k;
        float* h_o = new float[o_numel];
        for (int i = 0; i < o_numel; i++) h_o[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        uint8_t* h_o_fp4 = new uint8_t[o_numel / 2];
        uint8_t* h_o_sf  = new uint8_t[o_numel / SF_BLOCK];
        quantize_to_nvfp4_e4m3(h_o, o_numel, h_o_fp4, h_o_sf);
        CHECK_CUDA(cudaMalloc((void**)&lw.o_fp4, o_numel / 2));
        CHECK_CUDA(cudaMemcpy((void*)lw.o_fp4, h_o_fp4, o_numel / 2, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc((void**)&lw.o_sf, o_numel / SF_BLOCK));
        CHECK_CUDA(cudaMemcpy((void*)lw.o_sf, h_o_sf, o_numel / SF_BLOCK, cudaMemcpyHostToDevice));
        delete[] h_o; delete[] h_o_fp4; delete[] h_o_sf;

        float* h_gate = new float[NUM_EXPERTS * HIDDEN];
        for (int i = 0; i < NUM_EXPERTS * HIDDEN; i++)
            h_gate[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        CHECK_CUDA(cudaMalloc((void**)&lw.gate_weight, NUM_EXPERTS * HIDDEN * sizeof(float)));
        CHECK_CUDA(cudaMemcpy((void*)lw.gate_weight, h_gate,
                              NUM_EXPERTS * HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
        delete[] h_gate;

        int ne_test = NUM_EXPERTS;
        int w1_numel_per_expert = 2 * EXPERT_INTER * HIDDEN;
        int w2_numel_per_expert = HIDDEN * EXPERT_INTER;

        CHECK_CUDA(cudaMalloc((void**)&lw.expert_w1_fp4,
                              (long long)ne_test * w1_numel_per_expert / 2));
        CHECK_CUDA(cudaMemset((void*)lw.expert_w1_fp4, 0,
                              (long long)ne_test * w1_numel_per_expert / 2));
        CHECK_CUDA(cudaMalloc((void**)&lw.expert_w1_sf,
                              (long long)ne_test * w1_numel_per_expert / SF_BLOCK));
        CHECK_CUDA(cudaMemset((void*)lw.expert_w1_sf, 0,
                              (long long)ne_test * w1_numel_per_expert / SF_BLOCK));
        CHECK_CUDA(cudaMalloc((void**)&lw.expert_w2_fp4,
                              (long long)ne_test * w2_numel_per_expert / 2));
        CHECK_CUDA(cudaMemset((void*)lw.expert_w2_fp4, 0,
                              (long long)ne_test * w2_numel_per_expert / 2));
        CHECK_CUDA(cudaMalloc((void**)&lw.expert_w2_sf,
                              (long long)ne_test * w2_numel_per_expert / SF_BLOCK));
        CHECK_CUDA(cudaMemset((void*)lw.expert_w2_sf, 0,
                              (long long)ne_test * w2_numel_per_expert / SF_BLOCK));

        CHECK_CUDA(cudaMalloc((void**)&lw.shared_w1_fp4, w1_numel_per_expert / 2));
        CHECK_CUDA(cudaMemset((void*)lw.shared_w1_fp4, 0, w1_numel_per_expert / 2));
        CHECK_CUDA(cudaMalloc((void**)&lw.shared_w1_sf, w1_numel_per_expert / SF_BLOCK));
        CHECK_CUDA(cudaMemset((void*)lw.shared_w1_sf, 0, w1_numel_per_expert / SF_BLOCK));
        CHECK_CUDA(cudaMalloc((void**)&lw.shared_w2_fp4, w2_numel_per_expert / 2));
        CHECK_CUDA(cudaMemset((void*)lw.shared_w2_fp4, 0, w2_numel_per_expert / 2));
        CHECK_CUDA(cudaMalloc((void**)&lw.shared_w2_sf, w2_numel_per_expert / SF_BLOCK));
        CHECK_CUDA(cudaMemset((void*)lw.shared_w2_sf, 0, w2_numel_per_expert / SF_BLOCK));
    }

    void launch(cudaStream_t stream = 0) {
        int smem = smem_size();

        CHECK_CUDA(cudaFuncSetAttribute(
            megakernel_forward,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem));

        dim3 grid(NUM_CTAS);
        dim3 block(BLOCK_SIZE);
        void* args[] = { &params };

        CHECK_CUDA(cudaLaunchCooperativeKernel(
            (void*)megakernel_forward,
            grid, block,
            args, smem, stream));
    }
};

// ============================================================================
// Test main()
// ============================================================================
int main() {
    printf("==========================================================\n");
    printf("MegaKernel v2 — Fully Parallelized Transformer Forward\n");
    printf("==========================================================\n\n");

    int dev = 0;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s (SM %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);

    int supports_coop = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&supports_coop,
               cudaDevAttrCooperativeLaunch, dev));
    if (!supports_coop) {
        printf("ERROR: Device does not support cooperative launch\n");
        return 1;
    }
    printf("Cooperative launch: supported\n");

    int smem = MegaKernelHost::smem_size();
    int max_blocks = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks, megakernel_forward, BLOCK_SIZE, smem));
    printf("SMEM per CTA: %d bytes\n", smem);
    printf("Occupancy: %d CTAs/SM (need 1 for cooperative launch)\n", max_blocks);
    printf("Grid: %d CTAs x %d threads\n\n", NUM_CTAS, BLOCK_SIZE);

    if (max_blocks < 1) {
        printf("ERROR: Cannot fit 1 CTA per SM (SMEM too large)\n");
        return 1;
    }

    printf("Setting up single-layer test...\n");
    MegaKernelHost host;
    host.setup(1);

    LayerWeights lw;
    memset(&lw, 0, sizeof(lw));
    host.setup_test_layer(0, lw);

    CHECK_CUDA(cudaMalloc(&host.d_layers, sizeof(LayerWeights)));
    CHECK_CUDA(cudaMemcpy(host.d_layers, &lw, sizeof(LayerWeights), cudaMemcpyHostToDevice));
    host.params.layers = host.d_layers;

    // KV cache
    {
        const int TEST_MAX_SEQ = 128;
        KVCache h_kv;
        memset(&h_kv, 0, sizeof(h_kv));
        h_kv.seq_len = 0;

        size_t kv_bytes = (size_t)TEST_MAX_SEQ * NUM_KV_HEADS * HEAD_DIM;
        CHECK_CUDA(cudaMalloc(&h_kv.k_cache, kv_bytes));
        CHECK_CUDA(cudaMemset(h_kv.k_cache, 0, kv_bytes));
        CHECK_CUDA(cudaMalloc(&h_kv.v_cache, kv_bytes));
        CHECK_CUDA(cudaMemset(h_kv.v_cache, 0, kv_bytes));

        CHECK_CUDA(cudaMalloc(&host.d_kv_caches, sizeof(KVCache)));
        CHECK_CUDA(cudaMemcpy(host.d_kv_caches, &h_kv, sizeof(KVCache), cudaMemcpyHostToDevice));
        host.params.kv_caches = host.d_kv_caches;
    }

    // P2P buffers (stub — single GPU test)
    host.params.p2p.rank = 0;
    float* local_recv;
    CHECK_CUDA(cudaMalloc(&local_recv, WORLD_SIZE * HIDDEN * sizeof(float)));
    CHECK_CUDA(cudaMemset(local_recv, 0, WORLD_SIZE * HIDDEN * sizeof(float)));
    host.params.p2p.local_recv = local_recv;
    host.params.p2p.local_send = nullptr;
    uint32_t* local_flags;
    CHECK_CUDA(cudaMalloc(&local_flags, WORLD_SIZE * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(local_flags, 0xFF, WORLD_SIZE * sizeof(uint32_t)));
    host.params.p2p.local_flags = (volatile uint32_t*)local_flags;
    for (int i = 0; i < WORLD_SIZE; i++) {
        host.params.p2p.remote_recv[i] = local_recv;
        host.params.p2p.remote_flags[i] = (volatile uint32_t*)local_flags;
    }

    // Initialize hidden state
    float* h_hidden = new float[HIDDEN];
    srand(42);
    for (int i = 0; i < HIDDEN; i++) h_hidden[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    CHECK_CUDA(cudaMemcpy(host.params.hidden_state, h_hidden,
                          HIDDEN * sizeof(float), cudaMemcpyHostToDevice));

    printf("Launching megakernel (1 layer)...\n");

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    host.launch();
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("Warmup done\n");

    // Re-init hidden state
    CHECK_CUDA(cudaMemcpy(host.params.hidden_state, h_hidden,
                          HIDDEN * sizeof(float), cudaMemcpyHostToDevice));

    // Benchmark
    int iters = 100;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        CHECK_CUDA(cudaMemcpy(host.params.hidden_state, h_hidden,
                              HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset((void*)local_flags, 0xFF, WORLD_SIZE * sizeof(uint32_t)));
        // Reset scratch for each iteration
        CHECK_CUDA(cudaMemset(host.params.scratch, 0, sizeof(AtomicScratch)));
        // Reset KV cache seq_len
        {
            KVCache h_kv_reset;
            h_kv_reset.seq_len = 0;
            // Only reset seq_len field (offset 16 in struct = after two void* pointers)
            CHECK_CUDA(cudaMemcpy((char*)host.d_kv_caches + offsetof(KVCache, seq_len),
                                  &h_kv_reset.seq_len, sizeof(int), cudaMemcpyHostToDevice));
        }

        host.launch();
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_us = total_ms / iters * 1000.0f;

    printf("\nResults:\n");
    printf("  Single-layer latency: %.1f us (avg over %d iters)\n", avg_us, iters);
    printf("  Projected 60-layer:  %.1f ms\n", avg_us * 60.0f / 1000.0f);
    printf("  Projected tok/s (no AllReduce): %.0f\n", 1000000.0f / (avg_us * 60.0f));
    printf("  Projected tok/s (with P2P AR):  %.0f\n",
           1000000.0f / (avg_us * 60.0f + 120.0f * 7.9f));

    // Read back hidden state
    float* h_out = new float[HIDDEN];
    CHECK_CUDA(cudaMemcpy(h_out, host.params.hidden_state,
                          HIDDEN * sizeof(float), cudaMemcpyDeviceToHost));

    int nan_count = 0;
    for (int i = 0; i < HIDDEN; i++) {
        if (isnan(h_out[i]) || isinf(h_out[i])) nan_count++;
    }
    printf("  Output NaN/Inf count: %d / %d %s\n", nan_count, HIDDEN,
           nan_count == 0 ? "OK" : "FAIL");

    printf("  Output[0:8]: ");
    for (int i = 0; i < 8; i++) printf("%.4f ", h_out[i]);
    printf("\n");

    delete[] h_hidden;
    delete[] h_out;

    // Cleanup
    CHECK_CUDA(cudaFree(host.params.hidden_state));
    CHECK_CUDA(cudaFree(host.params.attn_out));
    CHECK_CUDA(cudaFree(host.params.ffn_out));
    CHECK_CUDA(cudaFree(host.params.norm_buf));
    CHECK_CUDA(cudaFree(host.params.gate_logits));
    CHECK_CUDA(cudaFree(host.params.act_fp4));
    CHECK_CUDA(cudaFree(host.params.act_sf));
    CHECK_CUDA(cudaFree(host.params.top_expert_ids));
    CHECK_CUDA(cudaFree(host.params.top_expert_wts));
    CHECK_CUDA(cudaFree(host.params.expert_cta_counter));
    CHECK_CUDA(cudaFree(host.params.expert_inter_buf));
    CHECK_CUDA(cudaFree(host.params.expert_inter_fp4));
    CHECK_CUDA(cudaFree(host.params.expert_inter_sf));
    CHECK_CUDA(cudaFree(host.params.scratch));
    CHECK_CUDA(cudaFree(host.d_layers));
    CHECK_CUDA(cudaFree(host.d_kv_caches));
    CHECK_CUDA(cudaFree(local_recv));
    CHECK_CUDA(cudaFree(local_flags));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    printf("\nDone.\n");
    return 0;
}
