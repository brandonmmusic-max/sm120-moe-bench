/**
 * megakernel_v1.cu — MegaKernel: Persistent Fused Transformer Forward Pass
 *
 * Single cudaLaunchCooperativeKernel that processes all 60 layers of
 * Qwen3.5-397B-A17B MoE inference on one GPU (TP=4 shard).
 *
 * Per-layer pipeline:
 *   RMSNorm -> Attention (QKV GEMV + decode + O GEMV) -> P2P AllReduce
 *   -> RMSNorm -> MoE Gate -> Expert GEMV (top-10 of 512) -> P2P AllReduce
 *   -> Residual add
 *   grid.sync()
 *
 * Architecture:
 *   - Cooperative launch: 99 CTAs (1 per SM), 256 threads, grid.sync() between layers
 *   - In-kernel P2P AllReduce: Write-based posted PCIe writes to remote BAR memory
 *   - Weight streaming: Weights stay in HBM, streamed via TMA. Activations (~10KB) in L2.
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

static constexpr int NUM_CTAS    = 99;  // 1 per SM on RTX PRO 6000 Blackwell

static constexpr int FULL_ATTN_LAYERS = 15;
// Layers 0-14: full attention, layers 15-59: DeltaNet (linear attention)

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
    // Each GPU writes its partial to remote GPUs' receive slots
    float* remote_recv[WORLD_SIZE];  // remote_recv[i] = pointer to GPU i's receive buffer
    float* local_send;               // local send/accumulation buffer
    float* local_recv;               // local receive buffer (others write here)

    // Barrier flags for write-based protocol
    volatile uint32_t* local_flags;           // [WORLD_SIZE] — set by remote writers
    volatile uint32_t* remote_flags[WORLD_SIZE]; // remote flag arrays

    int rank;
};

// KV cache pointers (pre-allocated, indexed by layer)
struct KVCache {
    // For full attention layers (0-14): standard paged KV cache
    // For DeltaNet layers (15-59): linear attention state
    void* k_cache;       // [max_seq_len, NUM_KV_HEADS, HEAD_DIM] per layer
    void* v_cache;       // [max_seq_len, NUM_KV_HEADS, HEAD_DIM] per layer
    int   seq_len;       // current sequence length
};

struct MegaKernelParams {
    // Layer weights (device array of structs)
    const LayerWeights* layers;      // [NUM_LAYERS]

    // KV cache (device array)
    KVCache*     kv_caches;          // [NUM_LAYERS]

    // P2P AllReduce
    P2PBuffers   p2p;

    // Activation buffers (L2-resident, ~10KB for M=1)
    float*       hidden_state;       // [HIDDEN] — main residual stream
    float*       attn_out;           // [HIDDEN] — attention output (pre-allreduce)
    float*       ffn_out;            // [HIDDEN] — MoE output (pre-allreduce)
    float*       norm_buf;           // [HIDDEN] — RMSNorm output
    float*       gate_logits;        // [NUM_EXPERTS] — gate scores

    // Quantized activation workspace (for FP4 GEMV input)
    uint8_t*     act_fp4;            // [K_PACKED] — quantized hidden state
    uint8_t*     act_sf;             // [SF_COLS_HIDDEN] — scale factors

    // MoE routing workspace
    int*         top_expert_ids;     // [TOP_K]
    float*       top_expert_wts;     // [TOP_K]

    // Atomic counters for MoE CTA self-assignment
    int*         expert_cta_counter; // [NUM_EXPERTS] — atomicAdd for work stealing

    // Scratch for intermediate MoE activations
    float*       expert_inter_buf;   // [TOP_K * EXPERT_INTER] — SwiGLU intermediates
    uint8_t*     expert_inter_fp4;   // [TOP_K * EXPERT_INTER_PACKED]
    uint8_t*     expert_inter_sf;    // [TOP_K * EXPERT_INTER / SF_BLOCK]

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
// RMSNorm — cooperative across all threads in CTA 0
//
// Only CTA 0 computes RMSNorm (4096 elements, single token).
// Other CTAs skip. grid.sync() ensures all see the result.
// Output: norm_buf[HIDDEN] and quantized act_fp4/act_sf for GEMV.
// ============================================================================
__device__ void rmsnorm_inkernel(
    float*       __restrict__ norm_out,    // [HIDDEN]
    uint8_t*     __restrict__ out_fp4,     // [K_PACKED]
    uint8_t*     __restrict__ out_sf,      // [SF_COLS_HIDDEN]
    const float* __restrict__ input,       // [HIDDEN]
    const float* __restrict__ weight,      // [HIDDEN]
    float eps = 1e-6f)
{
    // Only CTA 0 runs this
    const int tid = threadIdx.x;

    // Step 1: Compute sum of squares (cooperative across 256 threads)
    // Use dynamic SMEM (same base as other functions — safe since phases don't overlap)
    extern __shared__ char _smem_norm[];
    float* s_sum = (float*)_smem_norm;  // [BLOCK_SIZE] = 1KB, fits in dynamic SMEM
    float local_sum = 0.0f;
    for (int i = tid; i < HIDDEN; i += BLOCK_SIZE) {
        float v = input[i];
        local_sum += v * v;
    }
    s_sum[tid] = local_sum;
    __syncthreads();

    // Warp reduce
    if (tid < 32) {
        float ws = 0.0f;
        #pragma unroll
        for (int i = tid; i < BLOCK_SIZE; i += 32) ws += s_sum[i];
        // Warp-level reduce
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            ws += __shfl_xor_sync(0xFFFFFFFF, ws, off);
        if (tid == 0) s_sum[0] = ws;
    }
    __syncthreads();

    float rms = rsqrtf(s_sum[0] / (float)HIDDEN + eps);

    // Step 2: Apply normalization + weight, write to gmem output + quantize to FP4
    // Write normalized values to gmem, then quantize reading back from gmem.
    // Avoids large static SMEM (HIDDEN=4096 floats = 16KB).
    for (int i = tid; i < HIDDEN; i += BLOCK_SIZE) {
        norm_out[i] = input[i] * rms * weight[i];
    }
    __syncthreads();

    // Quantize to FP4: each half-warp (16 threads) handles one SF group
    const int half_warp_id = tid / 16;
    const int hw_lane = tid % 16;
    const int num_sf_groups = HIDDEN / SF_BLOCK;  // 256

    for (int g = half_warp_id; g < num_sf_groups; g += BLOCK_SIZE / 16) {
        int base = g * SF_BLOCK;
        float val = norm_out[base + hw_lane];  // read from gmem (L2-resident)
        float aval = fabsf(val);

        // Warp-shuffle max across 16 lanes
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
// For M=1 the K-loop is sequential within each CTA (bandwidth-bound GEMV).
//
// n_tile: which N-tile this CTA is responsible for
// ============================================================================
__device__ void gemv_fp4_inkernel(
    float*       __restrict__ output,      // [N_out] — output vector (atomicAdd for multi-CTA)
    const uint8_t* __restrict__ act_fp4,   // [K/2] — quantized input
    const uint8_t* __restrict__ act_sf,    // [K/SF_BLOCK] — input scales
    const uint8_t* __restrict__ w_fp4,     // [N_out, K/2] — weight matrix FP4
    const uint8_t* __restrict__ w_sf,      // [N_out, K/SF_BLOCK] — weight scales
    int N_out, int K,
    int n_tile,                            // which BN-sized N-tile
    float scale = 1.0f)                    // output scaling
{
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int n_start   = n_tile * BN;
    const int k_packed  = K / 2;
    const int sf_cols   = K / SF_BLOCK;
    const int k_tiles   = K / BK;

    // SMEM layout (same as VerdictDense)
    extern __shared__ char smem_raw[];
    uint8_t* s_A   = (uint8_t*)smem_raw;
    uint8_t* s_B   = s_A + 32;
    uint8_t* s_SFA = s_B + BN * (BK / 2);
    uint8_t* s_SFB = s_SFA + ((SF_PER_K + 3) & ~3);

    // MMA column mapping (identical to VerdictMoE)
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

        // Load A (input, 32 bytes)
        for (int i = tid; i < 8; i += BLOCK_SIZE) {
            *(uint32_t*)(s_A + i * 4) =
                *(const uint32_t*)&act_fp4[k_pk + i * 4];
        }

        // Load B (weight tile, BN x BK/2 bytes) with swizzle
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

        // Load SFA
        if (tid == 0) {
            *(uint32_t*)s_SFA = *(const uint32_t*)&act_sf[k_sf];
        }

        // Load SFB
        for (int i = tid; i < BN; i += BLOCK_SIZE) {
            int global_n = n_start + i;
            if (global_n < N_out)
                *(uint32_t*)&s_SFB[i * SF_PER_K] =
                    *(const uint32_t*)&w_sf[(long long)global_n * sf_cols + k_sf];
            else
                *(uint32_t*)&s_SFB[i * SF_PER_K] = 0;
        }

        __syncthreads();

        // MMA operands
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

    // Write output (M=1: only lane_id < 4 hold valid data)
    if (lane_id < 4) {
        int c0 = n_start + warp_id * 8 + lane_id;
        int c1 = c0 + 4;
        if (c0 < N_out) atomicAdd(&output[c0], scale * acc[0]);
        if (c1 < N_out) atomicAdd(&output[c1], scale * acc[1]);
    }
}

// ============================================================================
// MoE Gate + Top-K Selection
//
// Gate: softmax(hidden @ gate_weight^T) -> top-K expert selection.
// Only CTA 0 computes this (small: 512 dot products of dim 4096).
// ============================================================================
__device__ void moe_gate_topk(
    int*         __restrict__ top_ids,     // [TOP_K] output
    float*       __restrict__ top_wts,     // [TOP_K] output
    float*       __restrict__ gate_logits, // [NUM_EXPERTS] scratch
    const float* __restrict__ hidden,      // [HIDDEN] — normed hidden state
    const float* __restrict__ gate_weight, // [NUM_EXPERTS, HIDDEN]
    int num_experts, int top_k)
{
    const int tid = threadIdx.x;

    // Step 1: Compute gate logits — each thread handles multiple experts
    // gate_logits[e] = dot(hidden, gate_weight[e])
    extern __shared__ char _smem_gate[];
    float* s_partial = (float*)_smem_gate;  // [BLOCK_SIZE] = 1KB

    for (int e = 0; e < num_experts; e++) {
        // Cooperative dot product across 256 threads
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

    // Step 2: Softmax (thread 0 — sequential, only 512 elements)
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

        // Step 3: Top-K selection (insertion sort, K=10)
        for (int i = 0; i < top_k; i++) {
            top_ids[i] = -1;
            top_wts[i] = -1e30f;
        }
        for (int e = 0; e < num_experts; e++) {
            float w = gate_logits[e];
            if (w > top_wts[top_k - 1]) {
                top_ids[top_k - 1] = e;
                top_wts[top_k - 1] = w;
                // Bubble up
                for (int j = top_k - 1; j > 0 && top_wts[j] > top_wts[j - 1]; j--) {
                    float tw = top_wts[j]; top_wts[j] = top_wts[j - 1]; top_wts[j - 1] = tw;
                    int ti = top_ids[j]; top_ids[j] = top_ids[j - 1]; top_ids[j - 1] = ti;
                }
            }
        }

        // Renormalize top-K weights
        float wsum = 0.0f;
        for (int i = 0; i < top_k; i++) wsum += top_wts[i];
        float winv = 1.0f / wsum;
        for (int i = 0; i < top_k; i++) top_wts[i] *= winv;
    }
    __syncthreads();
}

// ============================================================================
// Execute Experts — CTA self-assignment via atomics
//
// Each CTA atomically claims (expert, N-tile) work items.
// Full GEMM1 -> SwiGLU -> FP4 requant -> GEMM2 pipeline per expert.
// Same MMA approach as VerdictMoE persistent kernel.
//
// All 99 CTAs participate. Work items = TOP_K experts x N-tiles.
// ============================================================================
__device__ void execute_experts(
    float*         __restrict__ output,           // [HIDDEN] — accumulated output
    const uint8_t* __restrict__ act_fp4,          // [K_PACKED]
    const uint8_t* __restrict__ act_sf,           // [SF_COLS_HIDDEN]
    const int*     __restrict__ expert_ids,       // [TOP_K]
    const float*   __restrict__ expert_wts,       // [TOP_K]
    const uint8_t* __restrict__ all_w1_fp4,       // [NUM_EXPERTS, 2*EXPERT_INTER, K_PACKED]
    const uint8_t* __restrict__ all_w1_sf,        // [NUM_EXPERTS, 2*EXPERT_INTER, SF_COLS_HIDDEN]
    const uint8_t* __restrict__ all_w2_fp4,       // [NUM_EXPERTS, HIDDEN, EXPERT_INTER_PACKED]
    const uint8_t* __restrict__ all_w2_sf,        // [NUM_EXPERTS, HIDDEN, EXPERT_INTER/SF_BLOCK]
    int*           __restrict__ cta_counter,       // [1] — atomic work counter
    float*         __restrict__ inter_buf,         // scratch for SwiGLU intermediates
    uint8_t*       __restrict__ inter_fp4,         // scratch for requantized intermediates
    uint8_t*       __restrict__ inter_sf)          // scratch for intermediate scales
{
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int n_half         = EXPERT_INTER;           // 1024
    const int n_half_packed  = EXPERT_INTER_PACKED;    // 512
    const int n2             = 2 * EXPERT_INTER;       // 2048
    const int sf_cols_w1     = SF_COLS_HIDDEN;         // 256
    const int sf_cols_w2     = EXPERT_INTER / SF_BLOCK; // 64
    const int tiles_n_gemm1  = n_half / BN;            // 16
    const int tiles_n_gemm2  = HIDDEN / BN;            // 64

    // Total work items: TOP_K * tiles_n_gemm1 (GEMM1) then TOP_K * tiles_n_gemm2 (GEMM2)
    // Phase 1: GEMM1 + SwiGLU + requant (one N-tile per work item)
    // Phase 2: GEMM2 scatter (one N-tile per work item)
    // Separated by grid.sync() since GEMM2 reads from requant buffer

    // MMA column mapping
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
    // NOTE: caller must grid.sync() before this function to ensure counter is visible
    // The grid.sync() after rmsnorm+quantize serves this purpose.

    // ---- PHASE 1: GEMM1 + SwiGLU + FP4 requant per (expert, N-tile) ----
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

        // Weight pointers
        const uint8_t* w1_fp4 = all_w1_fp4 + (long long)eid * n2 * K_PACKED;
        const uint8_t* w1_sf  = all_w1_sf  + (long long)eid * n2 * sf_cols_w1;

        // Full K-reduction GEMM1 (gate + up, 64 K-tiles)
        float gate_acc[4] = {0, 0, 0, 0};
        float up_acc[4]   = {0, 0, 0, 0};

        for (int kt = 0; kt < K_TILES_HIDDEN; kt++) {
            const int k_off = kt * BK;
            const int k_pk  = k_off / 2;
            const int k_sf  = k_off / SF_BLOCK;

            // Load A
            for (int i = tid; i < 8; i += BLOCK_SIZE) {
                *(uint32_t*)(s_A + i * 4) =
                    *(const uint32_t*)&act_fp4[k_pk + i * 4];
            }

            // Load gate B tile
            for (int i = tid; i < (BN * BK / 2) / 4; i += BLOCK_SIZE) {
                int boff = i * 4;
                int row = boff / (BK / 2), col = boff % (BK / 2);
                *(uint32_t*)&s_B_gate[swizzle_343(boff)] =
                    *(const uint32_t*)&w1_fp4[(long long)(n_start + row) * K_PACKED + k_pk + col];
            }

            // Load up B tile
            for (int i = tid; i < (BN * BK / 2) / 4; i += BLOCK_SIZE) {
                int boff = i * 4;
                int row = boff / (BK / 2), col = boff % (BK / 2);
                *(uint32_t*)&s_B_up[swizzle_343(boff)] =
                    *(const uint32_t*)&w1_fp4[(long long)(n_half + n_start + row) * K_PACKED + k_pk + col];
            }

            // Load SFA
            if (tid < SF_PER_K)
                s_SFA[tid] = act_sf[k_sf + tid];

            // Load SFB gate + up
            for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
                int row = i / SF_PER_K, col = i % SF_PER_K;
                s_SFB_gate[i] = w1_sf[(long long)(n_start + row) * sf_cols_w1 + k_sf + col];
            }
            for (int i = tid; i < BN * SF_PER_K; i += BLOCK_SIZE) {
                int row = i / SF_PER_K, col = i % SF_PER_K;
                s_SFB_up[i] = w1_sf[(long long)(n_half + n_start + row) * sf_cols_w1 + k_sf + col];
            }

            __syncthreads();

            // MMA operands
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

    // NOTE: Need grid.sync() here before GEMM2 can read inter_buf.
    // The caller handles this via cooperative grid sync.
}

// ============================================================================
// Execute Experts Phase 2 — GEMM2 scatter
//
// Reads requantized SwiGLU intermediates, does GEMM2 = W2 @ intermediate,
// atomicAdd weighted result to output.
// ============================================================================
__device__ void execute_experts_phase2(
    float*         __restrict__ output,
    const float*   __restrict__ inter_buf,       // [TOP_K * EXPERT_INTER] SwiGLU results (unused now, kept for API)
    const uint8_t* __restrict__ inter_fp4,       // [TOP_K * EXPERT_INTER_PACKED] FP4 requantized
    const uint8_t* __restrict__ inter_sf,        // [TOP_K * EXPERT_INTER / SF_BLOCK] scales
    const int*     __restrict__ expert_ids,
    const float*   __restrict__ expert_wts,
    const uint8_t* __restrict__ all_w2_fp4,
    const uint8_t* __restrict__ all_w2_sf,
    int*           __restrict__ cta_counter)
{
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    const int n_half         = EXPERT_INTER;
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
    // Caller grid.sync() ensures visibility

    int total_gemm2_items = TOP_K * tiles_n_gemm2;  // 10 * 64 = 640

    // GEMM2 uses the FP4 requantized intermediate as A operand.
    // Requantization is done by CTA 0 between Phase 1 and Phase 2 (caller handles this).
    // inter_fp4/inter_sf contain the FP4-quantized SwiGLU output per expert slot.

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

        // W2 shape: [HIDDEN, EXPERT_INTER/2] packed FP4, scales [HIDDEN, EXPERT_INTER/SF_BLOCK]
        // A shape: [1, EXPERT_INTER/2] FP4 (requantized SwiGLU output for this expert slot)
        const uint8_t* w2_fp4 = all_w2_fp4 + (long long)eid * HIDDEN * n_half_packed;
        const uint8_t* w2_sf_base = all_w2_sf + (long long)eid * HIDDEN * sf_cols_w2;

        // A operand: requantized intermediate for this expert slot
        const uint8_t* a_fp4_ptr = inter_fp4 + expert_slot * n_half_packed;
        const uint8_t* a_sf_ptr  = inter_sf  + expert_slot * (EXPERT_INTER / SF_BLOCK);

        const int k_tiles_w2 = EXPERT_INTER / BK;  // 1024/64 = 16

        float acc[4] = {0, 0, 0, 0};

        for (int kt = 0; kt < k_tiles_w2; kt++) {
            const int k_off = kt * BK;
            const int k_pk  = k_off / 2;
            const int k_sf  = k_off / SF_BLOCK;

            // Load A (requantized intermediate, 32 bytes per BK/2)
            for (int i = tid; i < 8; i += BLOCK_SIZE) {
                *(uint32_t*)(s_A + i * 4) =
                    *(const uint32_t*)&a_fp4_ptr[k_pk + i * 4];
            }

            // Load B (W2 weight tile, BN x BK/2 bytes) with swizzle
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

            // Load SFA (activation scales)
            if (tid == 0) {
                *(uint32_t*)s_SFA = *(const uint32_t*)&a_sf_ptr[k_sf];
            }

            // Load SFB (W2 weight scales)
            for (int i = tid; i < BN; i += BLOCK_SIZE) {
                int global_n = n_start + i;
                if (global_n < HIDDEN)
                    *(uint32_t*)&s_SFB[i * SF_PER_K] =
                        *(const uint32_t*)&w2_sf_base[(long long)global_n * sf_cols_w2 + k_sf];
                else
                    *(uint32_t*)&s_SFB[i * SF_PER_K] = 0;
            }

            __syncthreads();

            // MMA operands (vectorized uint32 SMEM loads)
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

        // Write weighted output (M=1: only lane_id < 4 hold valid data)
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
// P2P Write-Based AllReduce (in-kernel)
//
// Protocol for 4 GPUs:
//   1. Each GPU writes its partial to ALL other GPUs' receive buffers (posted PCIe writes)
//   2. Each GPU sets a flag on each remote GPU after write completes
//   3. Each GPU polls its local flags until all 3 remotes have written
//   4. Each GPU sums local data + 3 received partials
//
// Write-based is better for PCIe: posted writes are fire-and-forget (no ACK stall).
// Total data movement: 3 * HIDDEN * 4 bytes = 48KB per GPU per AllReduce.
// ============================================================================
__device__ void p2p_allreduce_write(
    float*                __restrict__ data,          // [HIDDEN] — local data (in-place result)
    const P2PBuffers&                  p2p,
    int                                generation)    // monotonic counter for flag disambiguation
{
    const int tid  = threadIdx.x;
    const int rank = p2p.rank;

    // Step 1: Write local data to all remote receive buffers
    // Each GPU has a receive buffer with WORLD_SIZE slots: recv[src_rank * HIDDEN ... ]
    // We write our data to slot [rank] on each remote GPU.
    for (int dst = 0; dst < WORLD_SIZE; dst++) {
        if (dst == rank) continue;
        float* remote_slot = p2p.remote_recv[dst] + rank * HIDDEN;
        for (int i = tid; i < HIDDEN; i += BLOCK_SIZE) {
            remote_slot[i] = data[i];
        }
    }

    // Fence: ensure all posted writes are visible before setting flags
    __threadfence_system();

    // Step 2: Set flag on each remote GPU
    if (tid == 0) {
        for (int dst = 0; dst < WORLD_SIZE; dst++) {
            if (dst == rank) continue;
            // Write our generation to remote GPU's flag slot for our rank
            p2p.remote_flags[dst][rank] = generation;
        }
    }
    __threadfence_system();

    // Step 3: Poll local flags until all remotes have written
    if (tid == 0) {
        for (int src = 0; src < WORLD_SIZE; src++) {
            if (src == rank) continue;
            while (p2p.local_flags[src] < (uint32_t)generation) {
                // Spin — posted writes from remote GPUs will eventually arrive
            }
        }
    }
    __syncthreads();  // all threads wait for tid==0 to confirm

    // Step 4: Sum received data into local buffer
    float* local_recv_base = p2p.local_recv;
    for (int i = tid; i < HIDDEN; i += BLOCK_SIZE) {
        float sum = data[i];
        for (int src = 0; src < WORLD_SIZE; src++) {
            if (src == rank) continue;
            sum += local_recv_base[src * HIDDEN + i];
        }
        data[i] = sum;
    }
    __syncthreads();
}

// ============================================================================
// Attention Decode — Full Attention + DeltaNet (linear attention)
//
// Full attention (layers 0-14): standard softmax(Q @ K^T / sqrt(d)) @ V
//   - FP8 (E4M3) KV cache for memory efficiency
//   - GQA: 8 Q heads, 2 KV heads (ratio 4:1)
//   - Single-token decode: GEMV against growing KV cache
//
// DeltaNet (layers 15-59): linear attention with recurrent state
//   - Fixed-size state S per head: [HEAD_DIM, HEAD_DIM] = 16KB
//   - S_t = beta * S_{t-1} + k_t * v_t^T (rank-1 update)
//   - o_t = S_t @ q_t (mat-vec, O(HEAD_DIM^2) per head)
//   - O(1) memory per token (no growing KV cache)
//
// The SM120 flash attention decode kernel (99 TFLOPS) could replace
// the full attention path for better performance at long sequences.
// ============================================================================
__device__ void attention_decode(
    float*       __restrict__ attn_out,     // [NUM_Q_HEADS * HEAD_DIM] = [1024]
    const float* __restrict__ qkv,          // [QKV_DIM] = [1536] — Q,K,V concatenated
    KVCache*     __restrict__ kv_cache,
    int layer_idx,
    bool is_full_attention)                 // true for layers 0-14, false for DeltaNet
{
    const int tid = threadIdx.x;

    if (is_full_attention) {
        // Full self-attention decode (M=1 single-token)
        // QKV layout: Q=[NUM_Q_HEADS * HEAD_DIM], K=[NUM_KV_HEADS * HEAD_DIM], V=[same]
        const float* q_ptr = qkv;                                        // [0 .. NUM_Q_HEADS*HEAD_DIM)
        const float* k_ptr = qkv + NUM_Q_HEADS * HEAD_DIM;              // next NUM_KV_HEADS*HEAD_DIM
        const float* v_ptr = qkv + (NUM_Q_HEADS + NUM_KV_HEADS) * HEAD_DIM;

        // KV cache: FP8 (E4M3) packed, [max_seq_len, NUM_KV_HEADS, HEAD_DIM]
        uint8_t* k_cache = (uint8_t*)kv_cache->k_cache;
        uint8_t* v_cache = (uint8_t*)kv_cache->v_cache;
        int seq_len = kv_cache->seq_len;

        // Step 1: Append current K,V to KV cache at position seq_len (FP8)
        // Each thread handles multiple elements
        for (int i = tid; i < NUM_KV_HEADS * HEAD_DIM; i += BLOCK_SIZE) {
            k_cache[seq_len * NUM_KV_HEADS * HEAD_DIM + i] = d_e4m3fn_encode(k_ptr[i]);
            v_cache[seq_len * NUM_KV_HEADS * HEAD_DIM + i] = d_e4m3fn_encode(v_ptr[i]);
        }
        __syncthreads();

        int total_seq = seq_len + 1;  // including the just-appended token
        float inv_sqrt_hd = rsqrtf((float)HEAD_DIM);

        // SMEM for attention: reuse dynamic SMEM
        // s_scores: [max threads can handle] — we process per Q-head sequentially
        extern __shared__ char _smem_attn[];
        float* s_scores = (float*)_smem_attn;  // [BLOCK_SIZE] for partial dot products
        float* s_max    = s_scores + BLOCK_SIZE;  // [1]
        float* s_sum    = s_max + 1;               // [1]

        // GQA: NUM_Q_HEADS=8, NUM_KV_HEADS=2, ratio=4 (4 Q heads per KV head)
        const int gqa_ratio = NUM_Q_HEADS / NUM_KV_HEADS;

        // Process each Q head sequentially (8 heads, each does GEMV against KV cache)
        for (int qh = 0; qh < NUM_Q_HEADS; qh++) {
            int kv_head = qh / gqa_ratio;
            const float* q_head = q_ptr + qh * HEAD_DIM;

            // Step 2: Compute attention scores: Q @ K_cache^T / sqrt(HEAD_DIM)
            // Each thread computes dot product for a subset of sequence positions
            // Then we need softmax across all positions

            // Pass 1: compute scores and find max (for numerical stability)
            float local_max = -1e30f;
            for (int s = tid; s < total_seq; s += BLOCK_SIZE) {
                // Dot product: q_head[HEAD_DIM] . k_cache[s, kv_head, HEAD_DIM]
                const uint8_t* k_row = k_cache + s * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM;
                float dot = 0.0f;
                #pragma unroll 8
                for (int d = 0; d < HEAD_DIM; d++) {
                    dot += q_head[d] * d_e4m3fn_decode(k_row[d]);
                }
                dot *= inv_sqrt_hd;
                // Store score temporarily in global scratch (reuse attn_out tail)
                // We have enough space: attn_out is QKV_DIM=1536, we only write O_DIM=1024
                // Use s_scores for block-local reduction
                local_max = fmaxf(local_max, dot);
            }

            // Reduce max across block
            s_scores[tid] = local_max;
            __syncthreads();
            if (tid < 32) {
                float wm = -1e30f;
                for (int i = tid; i < BLOCK_SIZE; i += 32) wm = fmaxf(wm, s_scores[i]);
                for (int off = 16; off > 0; off >>= 1)
                    wm = fmaxf(wm, __shfl_xor_sync(0xFFFFFFFF, wm, off));
                if (tid == 0) s_max[0] = wm;
            }
            __syncthreads();
            float max_score = s_max[0];

            // Pass 2: compute exp(score - max) and sum, also accumulate V weighted
            // For M=1 decode we fuse softmax denominator and V accumulation
            float local_sum = 0.0f;
            float local_out[HEAD_DIM];
            for (int d = 0; d < HEAD_DIM; d++) local_out[d] = 0.0f;

            for (int s = tid; s < total_seq; s += BLOCK_SIZE) {
                const uint8_t* k_row = k_cache + s * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM;
                float dot = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    dot += q_head[d] * d_e4m3fn_decode(k_row[d]);
                }
                dot *= inv_sqrt_hd;
                float w = expf(dot - max_score);
                local_sum += w;

                // Accumulate w * V[s]
                const uint8_t* v_row = v_cache + s * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; d++) {
                    local_out[d] += w * d_e4m3fn_decode(v_row[d]);
                }
            }

            // Reduce sum across block
            s_scores[tid] = local_sum;
            __syncthreads();
            if (tid < 32) {
                float ws = 0.0f;
                for (int i = tid; i < BLOCK_SIZE; i += 32) ws += s_scores[i];
                for (int off = 16; off > 0; off >>= 1)
                    ws += __shfl_xor_sync(0xFFFFFFFF, ws, off);
                if (tid == 0) s_sum[0] = ws;
            }
            __syncthreads();
            float inv_sum = 1.0f / s_sum[0];

            // Reduce V accumulations across block and write output
            // Each dimension needs a block-wide reduction. We do this element-wise.
            // For HEAD_DIM=128, each thread writes its partial to SMEM and we reduce.
            for (int d = 0; d < HEAD_DIM; d++) {
                s_scores[tid] = local_out[d];
                __syncthreads();
                if (tid < 32) {
                    float ws = 0.0f;
                    for (int i = tid; i < BLOCK_SIZE; i += 32) ws += s_scores[i];
                    for (int off = 16; off > 0; off >>= 1)
                        ws += __shfl_xor_sync(0xFFFFFFFF, ws, off);
                    if (tid == 0) {
                        attn_out[qh * HEAD_DIM + d] = ws * inv_sum;
                    }
                }
                __syncthreads();
            }
        }

        // Increment seq_len in KV cache
        if (tid == 0) {
            kv_cache->seq_len = total_seq;
        }
    } else {
        // DeltaNet (linear attention) — state update
        // State matrix S: [NUM_KV_HEADS, HEAD_DIM, HEAD_DIM] stored in kv_cache->k_cache
        // Beta (decay) stored per-head in kv_cache->v_cache as float[NUM_KV_HEADS]
        //
        // Per-head update:
        //   S_t = beta * S_{t-1} + k_t * v_t^T    (rank-1 update, HEAD_DIM x HEAD_DIM)
        //   o_t = S_t @ q_t                         (mat-vec, HEAD_DIM output)
        //
        // QKV layout same as full attention
        const float* q_ptr = qkv;
        const float* k_ptr = qkv + NUM_Q_HEADS * HEAD_DIM;
        const float* v_ptr = qkv + (NUM_Q_HEADS + NUM_KV_HEADS) * HEAD_DIM;

        float* state = (float*)kv_cache->k_cache;  // [NUM_KV_HEADS, HEAD_DIM, HEAD_DIM]
        // Default beta (decay factor) — in production this comes from the model
        const float beta = 0.99f;

        const int gqa_ratio = NUM_Q_HEADS / NUM_KV_HEADS;
        const int state_size = HEAD_DIM * HEAD_DIM;  // 128*128 = 16384 floats per head

        // Process each KV head
        for (int kvh = 0; kvh < NUM_KV_HEADS; kvh++) {
            float* S = state + kvh * state_size;  // [HEAD_DIM, HEAD_DIM]
            const float* k_head = k_ptr + kvh * HEAD_DIM;
            const float* v_head = v_ptr + kvh * HEAD_DIM;

            // Step 1: State update S = beta * S + k * v^T
            // S[i][j] = beta * S[i][j] + k[i] * v[j]
            // 16384 elements, distribute across 256 threads
            for (int idx = tid; idx < state_size; idx += BLOCK_SIZE) {
                int i = idx / HEAD_DIM;
                int j = idx % HEAD_DIM;
                S[idx] = beta * S[idx] + k_head[i] * v_head[j];
            }
            __syncthreads();

            // Step 2: Output o = S @ q for each Q head mapped to this KV head
            // o[i] = sum_j S[i][j] * q[j]
            for (int qoff = 0; qoff < gqa_ratio; qoff++) {
                int qh = kvh * gqa_ratio + qoff;
                const float* q_head = q_ptr + qh * HEAD_DIM;

                // Each thread computes partial dot products for assigned output dims
                extern __shared__ char _smem_dn[];
                float* s_partial = (float*)_smem_dn;  // [BLOCK_SIZE]

                for (int i = 0; i < HEAD_DIM; i++) {
                    // Compute o[i] = sum_j S[i*HEAD_DIM + j] * q[j]
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
    __syncthreads();
}

// ============================================================================
// Quantize activation buffer (cooperative, CTA 0)
// BF16/FP32 -> FP4 + E4M3FN scales, identical to VerdictMoE prologue
// ============================================================================
__device__ void quantize_activation(
    uint8_t*     __restrict__ out_fp4,   // [HIDDEN/2]
    uint8_t*     __restrict__ out_sf,    // [HIDDEN/SF_BLOCK]
    const float* __restrict__ input,     // [HIDDEN]
    int size)
{
    const int tid = threadIdx.x;
    const int half_warp_id = tid / 16;
    const int hw_lane = tid % 16;
    const int num_sf_groups = size / SF_BLOCK;

    for (int g = half_warp_id; g < num_sf_groups; g += BLOCK_SIZE / 16) {
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
// THE MEGAKERNEL
//
// Single cooperative launch: 99 CTAs x 256 threads.
// Processes all NUM_LAYERS layers sequentially with grid.sync() between layers.
//
// CTA role assignment per phase:
//   RMSNorm + Quantize:  CTA 0 only (4096 elements, ~16us)
//   Attention GEMV:      All CTAs distribute N-tiles (QKV: 24 tiles, O: 16 tiles)
//   P2P AllReduce:       CTA 0 only (coordinator, ~10us target)
//   MoE Gate + Top-K:    CTA 0 only (512 experts, sequential)
//   Expert GEMV:         All CTAs work-steal via atomics
//   Residual:            CTA 0 only
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
megakernel_forward(MegaKernelParams params)
{
    cg::grid_group grid = cg::this_grid();
    const int cta_id = blockIdx.x;
    const int tid    = threadIdx.x;

    int allreduce_gen = 1;  // monotonic generation counter for P2P flags

    for (int layer = 0; layer < params.num_layers; layer++) {
        const LayerWeights& lw = params.layers[layer];

        // ==============================================================
        // PHASE 1: Attention RMSNorm + Quantize
        // ==============================================================
        if (cta_id == 0) {
            rmsnorm_inkernel(
                params.norm_buf, params.act_fp4, params.act_sf,
                params.hidden_state, lw.attn_norm);
        }
        grid.sync();

        // ==============================================================
        // PHASE 2: QKV Projection (GEMV, all CTAs)
        // hidden[4096] @ W_qkv[1536, 4096]^T -> qkv[1536]
        // ==============================================================

        // Zero QKV output (CTA 0)
        if (cta_id == 0) {
            for (int i = tid; i < QKV_DIM; i += BLOCK_SIZE)
                params.attn_out[i] = 0.0f;
        }
        grid.sync();

        // Distribute N-tiles across CTAs
        for (int n_tile = cta_id; n_tile < N_TILES_QKV; n_tile += NUM_CTAS) {
            gemv_fp4_inkernel(
                params.attn_out,
                params.act_fp4, params.act_sf,
                lw.qkv_fp4, lw.qkv_sf,
                QKV_DIM, HIDDEN,
                n_tile, 1.0f);
        }
        grid.sync();

        // ==============================================================
        // PHASE 3: Attention Decode
        // ==============================================================
        if (cta_id == 0) {
            bool is_full = (layer < FULL_ATTN_LAYERS);
            // attn_out currently holds QKV projection.
            // After decode, first O_DIM elements hold attention output.
            attention_decode(
                params.attn_out, params.attn_out,
                &params.kv_caches[layer],
                layer, is_full);
        }
        grid.sync();

        // ==============================================================
        // PHASE 4: O Projection (GEMV, all CTAs)
        // attn_decoded[1024] @ W_o[4096, 1024]^T -> attn_proj[4096]
        // ==============================================================

        // Quantize attention output for O projection
        if (cta_id == 0) {
            quantize_activation(params.act_fp4, params.act_sf, params.attn_out, O_DIM);
        }
        grid.sync();

        // Zero ffn_out (reuse as O projection target)
        if (cta_id == 0) {
            for (int i = tid; i < HIDDEN; i += BLOCK_SIZE)
                params.ffn_out[i] = 0.0f;
        }
        grid.sync();

        for (int n_tile = cta_id; n_tile < N_TILES_O; n_tile += NUM_CTAS) {
            gemv_fp4_inkernel(
                params.ffn_out,
                params.act_fp4, params.act_sf,
                lw.o_fp4, lw.o_sf,
                HIDDEN, O_DIM,
                n_tile, 1.0f);
        }
        grid.sync();

        // ==============================================================
        // PHASE 5: Attention AllReduce + Residual
        // ==============================================================
        if (cta_id == 0) {
            p2p_allreduce_write(params.ffn_out, params.p2p, allreduce_gen++);

            // Residual add: hidden_state += attn_output
            for (int i = tid; i < HIDDEN; i += BLOCK_SIZE)
                params.hidden_state[i] += params.ffn_out[i];
        }
        grid.sync();

        // ==============================================================
        // PHASE 6: FFN RMSNorm + Quantize
        // ==============================================================
        if (cta_id == 0) {
            rmsnorm_inkernel(
                params.norm_buf, params.act_fp4, params.act_sf,
                params.hidden_state, lw.ffn_norm);
        }
        grid.sync();

        // ==============================================================
        // PHASE 7: MoE Gate + Top-K
        // ==============================================================
        if (cta_id == 0) {
            moe_gate_topk(
                params.top_expert_ids, params.top_expert_wts,
                params.gate_logits,
                params.norm_buf, lw.gate_weight,
                NUM_EXPERTS, TOP_K);
        }
        grid.sync();

        // ==============================================================
        // PHASE 8: Expert GEMM1 + SwiGLU (all CTAs, work-stealing)
        // ==============================================================

        // Zero output buffer for expert accumulation
        if (cta_id == 0) {
            for (int i = tid; i < HIDDEN; i += BLOCK_SIZE)
                params.ffn_out[i] = 0.0f;
        }
        grid.sync();

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
        grid.sync();

        // ==============================================================
        // PHASE 8.5: Requantize inter_buf -> FP4 (CTA 0)
        // Each expert slot has EXPERT_INTER=1024 float values in inter_buf.
        // Quantize each to FP4 for GEMM2 A operand.
        // ==============================================================
        if (cta_id == 0) {
            for (int slot = 0; slot < TOP_K; slot++) {
                quantize_activation(
                    params.expert_inter_fp4 + slot * EXPERT_INTER_PACKED,
                    params.expert_inter_sf  + slot * (EXPERT_INTER / SF_BLOCK),
                    params.expert_inter_buf + slot * EXPERT_INTER,
                    EXPERT_INTER);
            }
        }
        grid.sync();

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
        grid.sync();

        // ==============================================================
        // PHASE 10a: Shared Expert (same architecture as routed experts)
        // GEMM1: act_fp4[HIDDEN] @ shared_w1[2*EXPERT_INTER, HIDDEN]^T -> gate+up
        // SwiGLU -> requant -> GEMM2: inter_fp4[EXPERT_INTER] @ shared_w2[HIDDEN, EXPERT_INTER]^T
        // Weight = 1.0, added to ffn_out.
        // Reuse expert_inter_buf/fp4/sf slot 0 as scratch.
        // ==============================================================

        // Shared expert GEMM1: distribute N-tiles across CTAs
        {
            const int tiles_n_shared_g1 = EXPERT_INTER / BN;  // 1024/64 = 16
            // Zero inter_buf slot 0
            if (cta_id == 0) {
                for (int i = tid; i < EXPERT_INTER; i += BLOCK_SIZE)
                    params.expert_inter_buf[i] = 0.0f;
            }
            grid.sync();

            // GEMM1 gate+up via gemv_fp4_inkernel for each half
            // gate: shared_w1[0:EXPERT_INTER, :], up: shared_w1[EXPERT_INTER:2*EXPERT_INTER, :]
            // We need both gate and up, then SiLU-gate. Use per-element approach.
            // Distribute N-tiles for gate GEMV
            for (int n_tile = cta_id; n_tile < tiles_n_shared_g1; n_tile += NUM_CTAS) {
                // Gate GEMV: act_fp4 @ shared_w1[0:EXPERT_INTER]^T
                gemv_fp4_inkernel(
                    params.expert_inter_buf,             // first EXPERT_INTER = gate output
                    params.act_fp4, params.act_sf,
                    lw.shared_w1_fp4, lw.shared_w1_sf,
                    EXPERT_INTER, HIDDEN,
                    n_tile, 1.0f);
            }
            grid.sync();

            // Now do up projection into second half of inter_buf
            if (cta_id == 0) {
                for (int i = tid; i < EXPERT_INTER; i += BLOCK_SIZE)
                    params.expert_inter_buf[EXPERT_INTER + i] = 0.0f;
            }
            grid.sync();

            for (int n_tile = cta_id; n_tile < tiles_n_shared_g1; n_tile += NUM_CTAS) {
                // Up GEMV: act_fp4 @ shared_w1[EXPERT_INTER:2*EXPERT_INTER]^T
                gemv_fp4_inkernel(
                    params.expert_inter_buf + EXPERT_INTER,
                    params.act_fp4, params.act_sf,
                    lw.shared_w1_fp4 + (long long)EXPERT_INTER * K_PACKED,
                    lw.shared_w1_sf  + (long long)EXPERT_INTER * SF_COLS_HIDDEN,
                    EXPERT_INTER, HIDDEN,
                    n_tile, 1.0f);
            }
            grid.sync();

            // SwiGLU: inter_buf[i] = up[i] * silu(gate[i])
            if (cta_id == 0) {
                for (int i = tid; i < EXPERT_INTER; i += BLOCK_SIZE) {
                    float gate_val = params.expert_inter_buf[i];
                    float up_val   = params.expert_inter_buf[EXPERT_INTER + i];
                    params.expert_inter_buf[i] = up_val * d_silu(gate_val);
                }
            }
            grid.sync();

            // Requantize to FP4
            if (cta_id == 0) {
                quantize_activation(
                    params.expert_inter_fp4,
                    params.expert_inter_sf,
                    params.expert_inter_buf,
                    EXPERT_INTER);
            }
            grid.sync();

            // GEMM2: inter_fp4[EXPERT_INTER] @ shared_w2[HIDDEN, EXPERT_INTER]^T
            // Output added to ffn_out with weight=1.0
            const int tiles_n_shared_g2 = HIDDEN / BN;  // 4096/64 = 64
            for (int n_tile = cta_id; n_tile < tiles_n_shared_g2; n_tile += NUM_CTAS) {
                gemv_fp4_inkernel(
                    params.ffn_out,
                    params.expert_inter_fp4, params.expert_inter_sf,
                    lw.shared_w2_fp4, lw.shared_w2_sf,
                    HIDDEN, EXPERT_INTER,
                    n_tile, 1.0f);
            }
            grid.sync();
        }

        // ==============================================================
        // PHASE 10b: MoE AllReduce + Residual
        // ==============================================================
        if (cta_id == 0) {
            p2p_allreduce_write(params.ffn_out, params.p2p, allreduce_gen++);

            for (int i = tid; i < HIDDEN; i += BLOCK_SIZE)
                params.hidden_state[i] += params.ffn_out[i];
        }
        grid.sync();
    }
}

// ============================================================================
// Host-side Setup
// ============================================================================

#define CHECK_CUDA(c) do { cudaError_t _e = (c); if (_e != cudaSuccess) { \
    printf("CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(1); } } while(0)

// Host quantization helpers (from VerdictMoE)
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

    // SMEM requirement for the megakernel
    static int smem_size() {
        // A(32) + B_gate(2048) + B_up(2048) + SFA(4) + SFB_gate(256) + SFB_up(256) + pad
        return 32 + 2 * (BN * BK / 2) + ((SF_PER_K + 3) & ~3) + 2 * (BN * SF_PER_K) + 128;
    }

    void setup(int num_layers_to_run = 1) {
        memset(&params, 0, sizeof(params));
        params.num_layers = num_layers_to_run;

        // Allocate activation buffers
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

        // Zero activation buffers
        CHECK_CUDA(cudaMemset(params.expert_cta_counter, 0, sizeof(int)));
    }

    void setup_test_layer(int layer_idx, LayerWeights& lw) {
        // Allocate small test weights for single-layer validation
        int qkv_rows = QKV_DIM;
        int o_rows = HIDDEN;
        int o_k = O_DIM;

        // RMSNorm weights (all ones)
        float* h_norm = new float[HIDDEN];
        for (int i = 0; i < HIDDEN; i++) h_norm[i] = 1.0f;
        CHECK_CUDA(cudaMalloc((void**)&lw.attn_norm, HIDDEN * sizeof(float)));
        CHECK_CUDA(cudaMemcpy((void*)lw.attn_norm, h_norm, HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMalloc((void**)&lw.ffn_norm, HIDDEN * sizeof(float)));
        CHECK_CUDA(cudaMemcpy((void*)lw.ffn_norm, h_norm, HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
        delete[] h_norm;

        // QKV weights (random FP4)
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

        // O projection weights
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

        // Gate weights (FP32, small random)
        float* h_gate = new float[NUM_EXPERTS * HIDDEN];
        for (int i = 0; i < NUM_EXPERTS * HIDDEN; i++)
            h_gate[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        CHECK_CUDA(cudaMalloc((void**)&lw.gate_weight, NUM_EXPERTS * HIDDEN * sizeof(float)));
        CHECK_CUDA(cudaMemcpy((void*)lw.gate_weight, h_gate,
                              NUM_EXPERTS * HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
        delete[] h_gate;

        // Expert weights (allocate small dummy — 2 experts for speed)
        // In real deployment, all 512 experts would be allocated
        int ne_test = NUM_EXPERTS;  // must allocate for all since gate might route anywhere
        int w1_numel_per_expert = 2 * EXPERT_INTER * HIDDEN;
        int w2_numel_per_expert = HIDDEN * EXPERT_INTER;

        // For test: allocate minimal, zero-initialized
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

        // Shared expert (zero-initialized for test)
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

        // Set max dynamic SMEM for the kernel
        CHECK_CUDA(cudaFuncSetAttribute(
            megakernel_forward,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem));

        // Cooperative launch: all CTAs must be resident simultaneously
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
// Test main() — single-layer validation on small model
// ============================================================================
int main() {
    printf("==========================================================\n");
    printf("MegaKernel v1 — Fused Transformer Forward Pass Skeleton\n");
    printf("==========================================================\n\n");

    // Check cooperative launch support
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

    // Check occupancy
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

    // Setup single-layer test
    printf("Setting up single-layer test...\n");
    MegaKernelHost host;
    host.setup(1);  // 1 layer

    // Create layer weights on device
    LayerWeights lw;
    memset(&lw, 0, sizeof(lw));
    host.setup_test_layer(0, lw);

    // Copy LayerWeights array to device
    CHECK_CUDA(cudaMalloc(&host.d_layers, sizeof(LayerWeights)));
    CHECK_CUDA(cudaMemcpy(host.d_layers, &lw, sizeof(LayerWeights), cudaMemcpyHostToDevice));
    host.params.layers = host.d_layers;

    // KV cache — allocate real FP8 buffers for attention decode
    // Layer 0 is full attention: needs [max_seq_len, NUM_KV_HEADS, HEAD_DIM] byte arrays
    // For test: max_seq_len = 128 (small)
    {
        const int TEST_MAX_SEQ = 128;
        KVCache h_kv;
        memset(&h_kv, 0, sizeof(h_kv));
        h_kv.seq_len = 0;

        // FP8 KV cache: 1 byte per element
        size_t kv_bytes = (size_t)TEST_MAX_SEQ * NUM_KV_HEADS * HEAD_DIM;
        CHECK_CUDA(cudaMalloc(&h_kv.k_cache, kv_bytes));
        CHECK_CUDA(cudaMemset(h_kv.k_cache, 0, kv_bytes));
        CHECK_CUDA(cudaMalloc(&h_kv.v_cache, kv_bytes));
        CHECK_CUDA(cudaMemset(h_kv.v_cache, 0, kv_bytes));

        CHECK_CUDA(cudaMalloc(&host.d_kv_caches, sizeof(KVCache)));
        CHECK_CUDA(cudaMemcpy(host.d_kv_caches, &h_kv, sizeof(KVCache), cudaMemcpyHostToDevice));
        host.params.kv_caches = host.d_kv_caches;
    }

    // P2P buffers (stub — single GPU test, no actual P2P)
    // For single-GPU test, AllReduce is a no-op (rank 0, no remote peers)
    host.params.p2p.rank = 0;
    // Allocate local recv buffer (WORLD_SIZE * HIDDEN floats)
    float* local_recv;
    CHECK_CUDA(cudaMalloc(&local_recv, WORLD_SIZE * HIDDEN * sizeof(float)));
    CHECK_CUDA(cudaMemset(local_recv, 0, WORLD_SIZE * HIDDEN * sizeof(float)));
    host.params.p2p.local_recv = local_recv;
    host.params.p2p.local_send = nullptr;
    // For single-GPU test, set all remote pointers to local (self-loop, harmless)
    uint32_t* local_flags;
    CHECK_CUDA(cudaMalloc(&local_flags, WORLD_SIZE * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(local_flags, 0xFF, WORLD_SIZE * sizeof(uint32_t)));  // pre-set high
    host.params.p2p.local_flags = (volatile uint32_t*)local_flags;
    for (int i = 0; i < WORLD_SIZE; i++) {
        host.params.p2p.remote_recv[i] = local_recv;
        host.params.p2p.remote_flags[i] = (volatile uint32_t*)local_flags;
    }

    // Initialize hidden state with small random values
    float* h_hidden = new float[HIDDEN];
    srand(42);
    for (int i = 0; i < HIDDEN; i++) h_hidden[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    CHECK_CUDA(cudaMemcpy(host.params.hidden_state, h_hidden,
                          HIDDEN * sizeof(float), cudaMemcpyHostToDevice));

    // Launch
    printf("Launching megakernel (1 layer)...\n");

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    host.launch();
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("Warmup done\n");

    // Re-init hidden state (warmup consumed it)
    CHECK_CUDA(cudaMemcpy(host.params.hidden_state, h_hidden,
                          HIDDEN * sizeof(float), cudaMemcpyHostToDevice));

    // Benchmark
    int iters = 100;
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        // Re-init each iteration for consistent timing
        CHECK_CUDA(cudaMemcpy(host.params.hidden_state, h_hidden,
                              HIDDEN * sizeof(float), cudaMemcpyHostToDevice));
        // Reset P2P flags
        CHECK_CUDA(cudaMemset((void*)local_flags, 0xFF, WORLD_SIZE * sizeof(uint32_t)));

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

    // Read back hidden state to verify non-NaN
    float* h_out = new float[HIDDEN];
    CHECK_CUDA(cudaMemcpy(h_out, host.params.hidden_state,
                          HIDDEN * sizeof(float), cudaMemcpyDeviceToHost));

    int nan_count = 0;
    for (int i = 0; i < HIDDEN; i++) {
        if (isnan(h_out[i]) || isinf(h_out[i])) nan_count++;
    }
    printf("  Output NaN/Inf count: %d / %d %s\n", nan_count, HIDDEN,
           nan_count == 0 ? "OK" : "FAIL");

    // Print first few values
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
    CHECK_CUDA(cudaFree(host.d_layers));
    CHECK_CUDA(cudaFree(host.d_kv_caches));
    CHECK_CUDA(cudaFree(local_recv));
    CHECK_CUDA(cudaFree(local_flags));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    printf("\nDone.\n");
    return 0;
}
