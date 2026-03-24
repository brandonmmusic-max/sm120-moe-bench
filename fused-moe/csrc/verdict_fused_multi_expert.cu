/**
 * Task 4: Multi-Expert Fused Kernel for SM120
 *
 * Grid: (num_active_experts=10, N_tiles=64, 1) = 640 CTAs across 188 SMs
 * Each CTA: lookup expert from routing table, gather input token,
 *           load expert weights, GEMM1->SwiGLU->E4M3->GEMM2, atomicAdd weighted output.
 *
 * Architecture: Independent CTAs (no cooperative sync needed).
 *   - Each CTA redundantly computes full GEMM1+SwiGLU for its expert (L2 cached)
 *   - Each CTA computes its 64-col GEMM2 tile
 *   - Output scatter via atomicAdd for multi-expert accumulation
 *
 * SM120: compute_120a, 99KB SMEM, no tcgen05/TMEM, no BF16 MMA, 1x1x1 cluster.
 * SMEM per block: 16KB (input cache), 640 blocks / 188 SMs = 3.4 blocks/SM.
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a \
 *     -rdc=true -o verdict_fused_multi_expert verdict_fused_multi_expert.cu
 */

#include <cooperative_groups.h>
#include <cuda_runtime.h>
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
// Constants
// ============================================================================
constexpr int BLOCK_SIZE = 256;
constexpr int TILES_PER_EXPERT = 64;
constexpr int COLS_PER_TILE = 64;  // K=4096 / 64 tiles

// ============================================================================
// E4M3 Lookup Table
// ============================================================================
__constant__ float c_e4m3_lut[256];
static float h_e4m3_lut[256];

void init_e4m3_lut() {
    for (int i = 0; i < 256; i++) {
        int s = (i >> 7) & 1, e = (i >> 3) & 0xF, m = i & 7;
        float val;
        if (e == 15 && m == 7) val = 0.0f;
        else if (e == 0) val = ldexpf((float)m / 8.0f, -6);
        else val = ldexpf(1.0f + (float)m / 8.0f, e - 7);
        h_e4m3_lut[i] = s ? -val : val;
    }
    cudaMemcpyToSymbol(c_e4m3_lut, h_e4m3_lut, sizeof(h_e4m3_lut));
}

float host_e4m3_decode(uint8_t x) { return h_e4m3_lut[x]; }

uint8_t host_e4m3_encode(float v) {
    if (isnan(v)) return 0x7F;
    int s = v < 0 ? 1 : 0;
    float av = fabsf(v);
    if (av > 448.0f) av = 448.0f;
    uint8_t best = 0; float best_err = FLT_MAX;
    for (int e = 0; e <= 15; e++)
        for (int m = 0; m <= 7; m++) {
            if (e == 15 && m == 7) continue;
            float repr = (e == 0) ? ldexpf((float)m / 8.0f, -6)
                                  : ldexpf(1.0f + (float)m / 8.0f, e - 7);
            float err = fabsf(av - repr);
            if (err < best_err) { best_err = err; best = (e << 3) | m; }
        }
    return (s << 7) | best;
}

float host_silu(float x) { return x / (1.0f + expf(-x)); }

// ============================================================================
// Device Helpers
// ============================================================================
__device__ __forceinline__ uint8_t float_to_e4m3(float v) {
    uint16_t packed;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(packed) : "f"(0.0f), "f"(v));
    return (uint8_t)(packed & 0xFF);
}

__device__ __forceinline__ float d_silu(float x) {
    return x / (1.0f + __expf(-x));
}

// ============================================================================
// MULTI-EXPERT FUSED KERNEL (640 CTAs, independent, no cooperative sync)
// ============================================================================
//
// Grid: (num_active * tiles_per_expert, 1, 1) = 640 blocks
// Each block: 256 threads
//
// Phase 1 (GEMM1 + SwiGLU + E4M3): All 256 threads compute full GEMM1 for this
//   expert (redundant across 64 tiles, but L2 cached). tid < N_half maps 1:1 to
//   gate/up columns. SwiGLU + E4M3 requant, store to SMEM.
//
// Phase 2 (GEMM2 + scatter): 64 output cols per tile. 256 threads = 4 threads per
//   col doing K-reduction over N_half=256. Warp shuffle reduce. atomicAdd weighted
//   result to output.
//
__global__ void verdict_fused_multi_expert(
    const float* __restrict__ input,       // [M, K]
    const float* __restrict__ all_w1,      // [E, K, 2*N_half]
    const float* __restrict__ all_w2,      // [E, N_half, K]
    float* __restrict__ output,            // [M, K] (zeroed, atomicAdd scatter)
    const int* __restrict__ expert_ids,    // [num_active]
    const float* __restrict__ expert_wts,  // [num_active]
    const int* __restrict__ token_ids,     // [num_active]
    int K, int N_half, int num_active, int tiles_per_expert)
{
    const int eidx = blockIdx.x / tiles_per_expert;   // expert index 0..9
    const int tile  = blockIdx.x % tiles_per_expert;   // tile 0..63
    const int tid   = threadIdx.x;
    if (eidx >= num_active) return;

    const int eid  = expert_ids[eidx];
    const float wt = expert_wts[eidx];
    const int tok  = token_ids[eidx];
    const int N2   = 2 * N_half;

    const float* inp = input + tok * K;
    const float* w1  = all_w1 + (long long)eid * K * N2;
    const float* w2  = all_w2 + (long long)eid * N_half * K;

    // SMEM: K floats for input, reused as N_half floats for intermediate
    extern __shared__ float smem[];

    // Load input row to SMEM
    for (int i = tid; i < K; i += BLOCK_SIZE)
        smem[i] = inp[i];
    __syncthreads();

    // --- Phase 1: GEMM1 + SwiGLU + E4M3 requant ---
    float swiglu = 0.0f;
    if (tid < N_half) {
        float gate_acc = 0.0f, up_acc = 0.0f;
        for (int k = 0; k < K; k++) {
            float inp_k = smem[k];
            gate_acc += inp_k * w1[k * N2 + tid];
            up_acc   += inp_k * w1[k * N2 + tid + N_half];
        }
        swiglu = up_acc * d_silu(gate_acc);
    }
    uint8_t e4m3 = float_to_e4m3(swiglu);
    float inter_q = c_e4m3_lut[e4m3];

    __syncthreads();
    // Store intermediate to SMEM (reuse, N_half=256 floats = 1KB)
    if (tid < N_half) smem[tid] = inter_q;
    __syncthreads();

    // --- Phase 2: GEMM2 tile (64 cols) + weighted atomicAdd ---
    // 256 threads / 64 cols = 4 threads per col, K-reduction over N_half/4=64
    int col_local = tid >> 2;          // tid / 4, range 0..63
    int k_quarter = tid & 3;           // tid % 4, range 0..3
    int j = tile * COLS_PER_TILE + col_local;

    if (j < K) {
        int k_per = N_half >> 2;       // 64
        int k_start = k_quarter * k_per;
        float acc = 0.0f;
        #pragma unroll 8
        for (int k = k_start; k < k_start + k_per; k++)
            acc += smem[k] * w2[k * K + j];

        // Warp shuffle reduce across 4 consecutive threads
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, 1);
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, 2);

        if (k_quarter == 0)
            atomicAdd(&output[tok * K + j], wt * acc);
    }
}

// ============================================================================
// MULTI-EXPERT COOPERATIVE (640 CTAs, grid.sync, distributed GEMM1)
// ============================================================================
//
// Same 640-CTA grid but GEMM1 K is distributed across 64 tiles per expert.
// Eliminates redundant GEMM1 compute at the cost of 2 grid barriers.
//
// Phase 1a: Each of 64 blocks per expert handles K/64=64 elements of GEMM1.
//           Writes partial gate/up sums to GMEM partials buffer.
// grid.sync()
// Phase 1b: 10 leader blocks (tile==0) reduce 64 partials → SwiGLU → E4M3 → GMEM.
// grid.sync()
// Phase 2:  Each of 64 blocks handles 64 cols of GEMM2, atomicAdd weighted output.
//
__global__ void verdict_fused_multi_expert_coop(
    const float* __restrict__ input,       // [M, K]
    const float* __restrict__ all_w1,      // [E, K, 2*N_half]
    const float* __restrict__ all_w2,      // [E, N_half, K]
    float* __restrict__ output,            // [M, K] (zeroed, atomicAdd scatter)
    const int* __restrict__ expert_ids,    // [num_active]
    const float* __restrict__ expert_wts,  // [num_active]
    const int* __restrict__ token_ids,     // [num_active]
    float* __restrict__ partials,          // [num_active, tiles, 2, N_half]
    float* __restrict__ gmem_inter,        // [num_active, N_half]
    int K, int N_half, int num_active, int tiles_per_expert)
{
    cg::grid_group grid = cg::this_grid();
    const int eidx = blockIdx.x / tiles_per_expert;
    const int tile  = blockIdx.x % tiles_per_expert;
    const int tid   = threadIdx.x;
    if (eidx >= num_active) return;

    const int eid  = expert_ids[eidx];
    const float wt = expert_wts[eidx];
    const int tok  = token_ids[eidx];
    const int N2   = 2 * N_half;

    const float* inp = input + tok * K;
    const float* w1  = all_w1 + (long long)eid * K * N2;
    const float* w2  = all_w2 + (long long)eid * N_half * K;

    extern __shared__ float smem[];

    // --- Phase 1a: Distributed GEMM1 (K/64 = 64 elements per block) ---
    int k_per_block = K / tiles_per_expert;  // 64
    int k_start = tile * k_per_block;

    // Load input slice to SMEM
    for (int i = tid; i < k_per_block; i += BLOCK_SIZE)
        smem[i] = inp[k_start + i];
    __syncthreads();

    // Partial gate/up for N_half columns
    if (tid < N_half) {
        float gate_p = 0.0f, up_p = 0.0f;
        for (int ki = 0; ki < k_per_block; ki++) {
            float inp_k = smem[ki];
            int k = k_start + ki;
            gate_p += inp_k * w1[k * N2 + tid];
            up_p   += inp_k * w1[k * N2 + tid + N_half];
        }
        long long part_base = (long long)eidx * tiles_per_expert * 2 * N_half
                            + (long long)tile * 2 * N_half;
        partials[part_base + tid] = gate_p;
        partials[part_base + N_half + tid] = up_p;
    }

    grid.sync();

    // --- Phase 1b: Reduce + SwiGLU + E4M3 (leader blocks only, tile==0) ---
    if (tile == 0 && tid < N_half) {
        float gate_sum = 0.0f, up_sum = 0.0f;
        long long part_base = (long long)eidx * tiles_per_expert * 2 * N_half;
        for (int t = 0; t < tiles_per_expert; t++) {
            gate_sum += partials[part_base + (long long)t * 2 * N_half + tid];
            up_sum   += partials[part_base + (long long)t * 2 * N_half + N_half + tid];
        }
        float sw = up_sum * d_silu(gate_sum);
        uint8_t e4m3 = float_to_e4m3(sw);
        gmem_inter[eidx * N_half + tid] = c_e4m3_lut[e4m3];
    }

    grid.sync();

    // --- Phase 2: GEMM2 tile (64 cols) + weighted atomicAdd ---
    const float* inter = gmem_inter + eidx * N_half;

    // Load intermediate to SMEM for faster access
    if (tid < N_half) smem[tid] = inter[tid];
    __syncthreads();

    int col_local = tid >> 2;
    int k_quarter = tid & 3;
    int j = tile * COLS_PER_TILE + col_local;

    if (j < K) {
        int k_per = N_half >> 2;
        int k_s = k_quarter * k_per;
        float acc = 0.0f;
        #pragma unroll 8
        for (int k = k_s; k < k_s + k_per; k++)
            acc += smem[k] * w2[k * K + j];

        acc += __shfl_xor_sync(0xFFFFFFFF, acc, 1);
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, 2);

        if (k_quarter == 0)
            atomicAdd(&output[tok * K + j], wt * acc);
    }
}

// ============================================================================
// SINGLE-EXPERT V2 (cooperative, for 10x sequential comparison)
// ============================================================================
__global__ void verdict_fused_v2_single(
    const float* __restrict__ input,
    const float* __restrict__ w1,
    const float* __restrict__ w2,
    float* __restrict__ output,
    float* __restrict__ partials,
    float* __restrict__ gmem_inter,
    int K, int N_half)
{
    cg::grid_group grid = cg::this_grid();
    const int bid = blockIdx.x, tid = threadIdx.x;
    const int num_blocks = gridDim.x;
    const int N2 = 2 * N_half;
    extern __shared__ char smem_raw[];
    float* s_slice = (float*)smem_raw;

    int k_per = K / num_blocks;
    int k_start = bid * k_per;
    for (int i = tid; i < k_per; i += BLOCK_SIZE)
        s_slice[i] = input[k_start + i];
    __syncthreads();

    if (tid < N_half) {
        float gp = 0.0f, up = 0.0f;
        for (int ki = 0; ki < k_per; ki++) {
            float inp = s_slice[ki];
            int k = k_start + ki;
            gp += inp * w1[k * N2 + tid];
            up += inp * w1[k * N2 + tid + N_half];
        }
        partials[bid * 2 * N_half + tid] = gp;
        partials[bid * 2 * N_half + N_half + tid] = up;
    }
    grid.sync();

    if (bid == 0 && tid < N_half) {
        float gs = 0.0f, us = 0.0f;
        for (int b = 0; b < num_blocks; b++) {
            gs += partials[b * 2 * N_half + tid];
            us += partials[b * 2 * N_half + N_half + tid];
        }
        float sw = us * d_silu(gs);
        uint8_t e4m3 = float_to_e4m3(sw);
        gmem_inter[tid] = c_e4m3_lut[e4m3];
    }
    grid.sync();

    int gtid = bid * BLOCK_SIZE + tid;
    int total = num_blocks * BLOCK_SIZE;
    for (int j = gtid; j < K; j += total) {
        float acc = 0.0f;
        for (int k = 0; k < N_half; k++)
            acc += gmem_inter[k] * w2[k * K + j];
        output[j] = acc;
    }
}

// ============================================================================
// 5 SEPARATE KERNELS (baseline)
// ============================================================================
__global__ void kern_gemm1_gate(
    const float* __restrict__ input, const float* __restrict__ w1,
    float* __restrict__ gate, int K, int N_half)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N_half) return;
    int N2 = 2 * N_half;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += input[k] * w1[k * N2 + j];
    gate[j] = acc;
}

__global__ void kern_gemm1_up(
    const float* __restrict__ input, const float* __restrict__ w1,
    float* __restrict__ up, int K, int N_half)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N_half) return;
    int N2 = 2 * N_half;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += input[k] * w1[k * N2 + j + N_half];
    up[j] = acc;
}

__global__ void kern_swiglu(
    const float* __restrict__ gate, const float* __restrict__ up,
    float* __restrict__ inter, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float g = gate[i];
    inter[i] = up[i] * g / (1.0f + __expf(-g));
}

__global__ void kern_requant(float* __restrict__ data, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    uint8_t e4m3 = float_to_e4m3(data[i]);
    data[i] = c_e4m3_lut[e4m3];
}

__global__ void kern_gemm2(
    const float* __restrict__ inter, const float* __restrict__ w2,
    float* __restrict__ output, int K, int N_half)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= K) return;
    float acc = 0.0f;
    for (int k = 0; k < N_half; k++) acc += inter[k] * w2[k * K + j];
    output[j] = acc;
}

__global__ void kern_weighted_add(
    float* __restrict__ output, const float* __restrict__ expert_out,
    float weight, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] += weight * expert_out[i];
}

// ============================================================================
// Host Reference (multi-expert weighted sum)
// ============================================================================
void host_multi_expert_ref(
    const float* input, const float* all_w1, const float* all_w2,
    float* ref_fp32, float* ref_e4m3,
    const int* expert_ids, const float* expert_wts, const int* token_ids,
    int num_active, int K, int N_half)
{
    int N2 = 2 * N_half;
    std::fill(ref_fp32, ref_fp32 + K, 0.0f);
    std::fill(ref_e4m3, ref_e4m3 + K, 0.0f);

    for (int e = 0; e < num_active; e++) {
        int eid = expert_ids[e];
        float wt = expert_wts[e];
        int tok = token_ids[e];
        const float* inp = input + tok * K;
        const float* w1 = all_w1 + (long long)eid * K * N2;
        const float* w2 = all_w2 + (long long)eid * N_half * K;

        std::vector<float> gate(N_half), up(N_half);
        for (int j = 0; j < N_half; j++) {
            float ga = 0, ua = 0;
            for (int k = 0; k < K; k++) {
                ga += inp[k] * w1[k * N2 + j];
                ua += inp[k] * w1[k * N2 + j + N_half];
            }
            gate[j] = ga; up[j] = ua;
        }

        std::vector<float> inter_f(N_half), inter_q(N_half);
        for (int j = 0; j < N_half; j++) {
            inter_f[j] = up[j] * host_silu(gate[j]);
            inter_q[j] = host_e4m3_decode(host_e4m3_encode(inter_f[j]));
        }

        for (int j = 0; j < K; j++) {
            float acc_f = 0, acc_q = 0;
            for (int k = 0; k < N_half; k++) {
                acc_f += inter_f[k] * w2[k * K + j];
                acc_q += inter_q[k] * w2[k * K + j];
            }
            ref_fp32[tok * K + j] += wt * acc_f;
            ref_e4m3[tok * K + j] += wt * acc_q;
        }
    }
}

// ============================================================================
// Error Stats
// ============================================================================
struct ErrorStats {
    float max_abs, avg_abs, rms_ref, norm_pct;
    int nan_count, zero_count, close_count, total;
};

ErrorStats compute_error(const float* actual, const float* ref, int n) {
    ErrorStats s = {};
    s.total = n;
    double sum_err = 0, sum_ref2 = 0;
    for (int i = 0; i < n; i++) {
        if (isnan(actual[i])) { s.nan_count++; continue; }
        if (actual[i] == 0.0f) s.zero_count++;
        float e = fabsf(actual[i] - ref[i]);
        s.max_abs = fmaxf(s.max_abs, e);
        sum_err += e;
        sum_ref2 += (double)ref[i] * ref[i];
        if ((fabsf(ref[i]) > 0.001f && e / fabsf(ref[i]) < 0.05f) ||
            (fabsf(ref[i]) <= 0.001f && fabsf(actual[i]) < 0.05f))
            s.close_count++;
    }
    s.avg_abs = (float)(sum_err / n);
    s.rms_ref = sqrtf((float)(sum_ref2 / n));
    s.norm_pct = s.rms_ref > 0 ? 100.0f * s.avg_abs / s.rms_ref : 0;
    return s;
}

void print_error(const char* label, ErrorStats s) {
    printf("  %s:\n", label);
    printf("    Normalized error: %.4f%%, Max abs: %.6f, Within 5%%: %d/%d (%.1f%%)\n",
           s.norm_pct, s.max_abs, s.close_count, s.total,
           100.0f * s.close_count / s.total);
    if (s.nan_count > 0) printf("    NaN count: %d\n", s.nan_count);
}

// ============================================================================
// Benchmark Helpers
// ============================================================================
struct BenchResult { float avg_us, med_us, p5_us, p95_us; };

BenchResult bench_stats(std::vector<float>& t) {
    std::sort(t.begin(), t.end());
    BenchResult r;
    r.avg_us = std::accumulate(t.begin(), t.end(), 0.0f) / t.size();
    r.med_us = t[t.size() / 2];
    r.p5_us  = t[(int)(t.size() * 0.05f)];
    r.p95_us = t[(int)(t.size() * 0.95f)];
    return r;
}

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Main
// ============================================================================
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM%d%d, %d SMs, %dKB SMEM)\n\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           (int)(prop.sharedMemPerMultiprocessor / 1024));

    const int M = 1, K = 4096, N_half = 256;
    const int N2 = 2 * N_half;
    const int NUM_EXPERTS = 10;
    const int NUM_ACTIVE = 10;
    const int V2_BLOCKS = 16;

    printf("=== Task 4: Multi-Expert Fused Kernel ===\n");
    printf("  %d active experts, M=%d, K=%d, N_half=%d\n", NUM_ACTIVE, M, K, N_half);
    printf("  Grid: %d experts x %d tiles = %d CTAs (%d threads/block)\n",
           NUM_ACTIVE, TILES_PER_EXPERT, NUM_ACTIVE * TILES_PER_EXPERT, BLOCK_SIZE);
    printf("  SMEM/block: %d bytes (input cache)\n", (int)(K * sizeof(float)));
    printf("  Pipeline: GEMM1[1,%d]x[%d,%d] -> SwiGLU -> E4M3 -> GEMM2[1,%d]x[%d,%d] -> atomicAdd\n\n",
           K, K, N2, N_half, N_half, K);

    init_e4m3_lut();

    // --- Allocate host data ---
    srand(42);
    auto randf = []() { return (float)rand() / RAND_MAX * 2.0f - 1.0f; };
    float sw1 = 1.0f / sqrtf((float)K), sw2 = 1.0f / sqrtf((float)N_half);

    size_t w1_size = (size_t)NUM_EXPERTS * K * N2;
    size_t w2_size = (size_t)NUM_EXPERTS * N_half * K;

    std::vector<float> h_input(M * K);
    std::vector<float> h_all_w1(w1_size), h_all_w2(w2_size);
    std::vector<float> h_out_fused(M * K), h_out_coop(M * K), h_out_v2(M * K), h_out_sep(M * K);
    std::vector<float> h_ref_fp32(M * K), h_ref_e4m3(M * K);

    for (auto& v : h_input) v = randf();
    for (auto& v : h_all_w1) v = randf() * sw1;
    for (auto& v : h_all_w2) v = randf() * sw2;

    // Routing table: all 10 experts process token 0
    std::vector<int> h_expert_ids(NUM_ACTIVE);
    std::vector<float> h_expert_wts(NUM_ACTIVE);
    std::vector<int> h_token_ids(NUM_ACTIVE, 0);

    for (int i = 0; i < NUM_ACTIVE; i++) h_expert_ids[i] = i;

    // Softmax of random logits for routing weights
    float logit_sum = 0;
    for (int i = 0; i < NUM_ACTIVE; i++) {
        h_expert_wts[i] = expf(randf());
        logit_sum += h_expert_wts[i];
    }
    for (int i = 0; i < NUM_ACTIVE; i++) h_expert_wts[i] /= logit_sum;

    printf("  Expert weights: [");
    for (int i = 0; i < NUM_ACTIVE; i++)
        printf("%.4f%s", h_expert_wts[i], i < NUM_ACTIVE-1 ? ", " : "");
    printf("]\n\n");

    // --- Allocate device memory ---
    float *d_input, *d_all_w1, *d_all_w2;
    float *d_out_fused, *d_out_coop, *d_out_v2, *d_out_sep;
    int *d_expert_ids, *d_token_ids;
    float *d_expert_wts;
    float *d_gate, *d_up, *d_inter, *d_expert_out;
    float *d_partials, *d_gmem_inter;
    // Cooperative kernel buffers (larger: 10 experts × 64 tiles × 2 × 256)
    float *d_coop_partials, *d_coop_inter;
    size_t coop_part_size = (size_t)NUM_ACTIVE * TILES_PER_EXPERT * 2 * N_half;
    size_t coop_inter_size = (size_t)NUM_ACTIVE * N_half;

    CHECK_CUDA(cudaMalloc(&d_input, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_all_w1, w1_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_all_w2, w2_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_fused, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_coop, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_v2, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_sep, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_expert_ids, NUM_ACTIVE * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_expert_wts, NUM_ACTIVE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_token_ids, NUM_ACTIVE * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_gate, N_half * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_up, N_half * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_inter, N_half * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_expert_out, K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_partials, V2_BLOCKS * 2 * N_half * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gmem_inter, N_half * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_coop_partials, coop_part_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_coop_inter, coop_inter_size * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_all_w1, h_all_w1.data(), w1_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_all_w2, h_all_w2.data(), w2_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_expert_ids, h_expert_ids.data(), NUM_ACTIVE * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_expert_wts, h_expert_wts.data(), NUM_ACTIVE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_token_ids, h_token_ids.data(), NUM_ACTIVE * sizeof(int), cudaMemcpyHostToDevice));

    // ================================================================
    // Host Reference
    // ================================================================
    printf("--- Computing host reference (10 experts, weighted sum) ---\n");
    host_multi_expert_ref(
        h_input.data(), h_all_w1.data(), h_all_w2.data(),
        h_ref_fp32.data(), h_ref_e4m3.data(),
        h_expert_ids.data(), h_expert_wts.data(), h_token_ids.data(),
        NUM_ACTIVE, K, N_half);

    printf("  ref_fp32[0:4] = [%.6f, %.6f, %.6f, %.6f]\n",
           h_ref_fp32[0], h_ref_fp32[1], h_ref_fp32[2], h_ref_fp32[3]);
    printf("  ref_e4m3[0:4] = [%.6f, %.6f, %.6f, %.6f]\n\n",
           h_ref_e4m3[0], h_ref_e4m3[1], h_ref_e4m3[2], h_ref_e4m3[3]);

    // ================================================================
    // Correctness Test 1: Multi-Expert Fused (640 CTAs)
    // ================================================================
    printf("--- Correctness: Multi-Expert Fused (640 CTAs) ---\n");
    int total_blocks = NUM_ACTIVE * TILES_PER_EXPERT;
    int smem_fused = K * sizeof(float);

    CHECK_CUDA(cudaMemset(d_out_fused, 0, M * K * sizeof(float)));
    verdict_fused_multi_expert<<<total_blocks, BLOCK_SIZE, smem_fused>>>(
        d_input, d_all_w1, d_all_w2, d_out_fused,
        d_expert_ids, d_expert_wts, d_token_ids,
        K, N_half, NUM_ACTIVE, TILES_PER_EXPERT);
    cudaError_t err = cudaDeviceSynchronize();
    printf("  Launch: %s (%d blocks x %d threads)\n",
           err == cudaSuccess ? "OK" : cudaGetErrorString(err), total_blocks, BLOCK_SIZE);

    CHECK_CUDA(cudaMemcpy(h_out_fused.data(), d_out_fused, M * K * sizeof(float), cudaMemcpyDeviceToHost));

    printf("  fused[0:4]    = [%.6f, %.6f, %.6f, %.6f]\n",
           h_out_fused[0], h_out_fused[1], h_out_fused[2], h_out_fused[3]);

    auto ef_e4m3 = compute_error(h_out_fused.data(), h_ref_e4m3.data(), K);
    auto ef_fp32 = compute_error(h_out_fused.data(), h_ref_fp32.data(), K);
    print_error("Fused vs E4M3 ref (should be ~0)", ef_e4m3);
    print_error("Fused vs FP32 ref (E4M3 quant error)", ef_fp32);

    // ================================================================
    // Correctness Test 2: Multi-Expert Cooperative (640 CTAs, grid.sync)
    // ================================================================
    printf("\n--- Correctness: Multi-Expert Cooperative (640 CTAs, grid.sync) ---\n");
    {
        int smem_coop = std::max((int)(K / TILES_PER_EXPERT * sizeof(float)),
                                 (int)(N_half * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_out_coop, 0, K * sizeof(float)));
        CHECK_CUDA(cudaMemset(d_coop_partials, 0, coop_part_size * sizeof(float)));
        int coop_K = K, coop_Nh = N_half, coop_NA = NUM_ACTIVE, coop_TPE = TILES_PER_EXPERT;
        void* args[] = { (void*)&d_input, (void*)&d_all_w1, (void*)&d_all_w2,
                         (void*)&d_out_coop, (void*)&d_expert_ids, (void*)&d_expert_wts,
                         (void*)&d_token_ids, (void*)&d_coop_partials, (void*)&d_coop_inter,
                         (void*)&coop_K, (void*)&coop_Nh, (void*)&coop_NA, (void*)&coop_TPE };
        err = cudaLaunchCooperativeKernel(
            (void*)verdict_fused_multi_expert_coop, total_blocks, BLOCK_SIZE, args, smem_coop);
        if (err != cudaSuccess) {
            printf("  Coop launch FAILED: %s\n", cudaGetErrorString(err));
        } else {
            err = cudaDeviceSynchronize();
            printf("  Launch: %s (%d blocks, cooperative)\n",
                   err == cudaSuccess ? "OK" : cudaGetErrorString(err), total_blocks);
        }
        CHECK_CUDA(cudaMemcpy(h_out_coop.data(), d_out_coop, K * sizeof(float), cudaMemcpyDeviceToHost));
        printf("  coop[0:4]     = [%.6f, %.6f, %.6f, %.6f]\n",
               h_out_coop[0], h_out_coop[1], h_out_coop[2], h_out_coop[3]);
        auto ec_e4m3 = compute_error(h_out_coop.data(), h_ref_e4m3.data(), K);
        print_error("Coop vs E4M3 ref", ec_e4m3);
    }

    // ================================================================
    // Correctness Test 3: 10x V2 Cooperative (sequential)
    // ================================================================
    printf("\n--- Correctness: 10x V2 Cooperative ---\n");
    int smem_v2 = (K / V2_BLOCKS) * sizeof(float);
    CHECK_CUDA(cudaMemset(d_out_v2, 0, K * sizeof(float)));

    for (int e = 0; e < NUM_ACTIVE; e++) {
        const float* w1_ptr = d_all_w1 + (long long)h_expert_ids[e] * K * N2;
        const float* w2_ptr = d_all_w2 + (long long)h_expert_ids[e] * N_half * K;

        CHECK_CUDA(cudaMemset(d_partials, 0, V2_BLOCKS * 2 * N_half * sizeof(float)));
        void* args[] = { (void*)&d_input, (void*)&w1_ptr, (void*)&w2_ptr,
                         (void*)&d_expert_out, (void*)&d_partials, (void*)&d_gmem_inter,
                         (void*)&K, (void*)&N_half };
        err = cudaLaunchCooperativeKernel(
            (void*)verdict_fused_v2_single, V2_BLOCKS, BLOCK_SIZE, args, smem_v2);
        if (err != cudaSuccess) {
            printf("  V2 launch failed for expert %d: %s\n", e, cudaGetErrorString(err));
            break;
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        float wt = h_expert_wts[e];
        kern_weighted_add<<<(K + 255) / 256, 256>>>(d_out_v2, d_expert_out, wt, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out_v2.data(), d_out_v2, K * sizeof(float), cudaMemcpyDeviceToHost));
    printf("  v2[0:4]       = [%.6f, %.6f, %.6f, %.6f]\n",
           h_out_v2[0], h_out_v2[1], h_out_v2[2], h_out_v2[3]);

    auto ev2_e4m3 = compute_error(h_out_v2.data(), h_ref_e4m3.data(), K);
    print_error("10xV2 vs E4M3 ref", ev2_e4m3);

    // ================================================================
    // Correctness Test 3: 10x 5-Kernel Baseline
    // ================================================================
    printf("\n--- Correctness: 10x 5-Kernel Baseline ---\n");
    dim3 grid1((N_half + 255) / 256);
    dim3 grid2((K + 255) / 256);
    CHECK_CUDA(cudaMemset(d_out_sep, 0, K * sizeof(float)));

    for (int e = 0; e < NUM_ACTIVE; e++) {
        const float* w1_ptr = d_all_w1 + (long long)h_expert_ids[e] * K * N2;
        const float* w2_ptr = d_all_w2 + (long long)h_expert_ids[e] * N_half * K;
        float wt = h_expert_wts[e];

        kern_gemm1_gate<<<grid1, 256>>>(d_input, w1_ptr, d_gate, K, N_half);
        kern_gemm1_up<<<grid1, 256>>>(d_input, w1_ptr, d_up, K, N_half);
        kern_swiglu<<<(N_half + 255) / 256, 256>>>(d_gate, d_up, d_inter, N_half);
        kern_requant<<<(N_half + 255) / 256, 256>>>(d_inter, N_half);
        kern_gemm2<<<grid2, 256>>>(d_inter, w2_ptr, d_expert_out, K, N_half);
        kern_weighted_add<<<(K + 255) / 256, 256>>>(d_out_sep, d_expert_out, wt, K);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out_sep.data(), d_out_sep, K * sizeof(float), cudaMemcpyDeviceToHost));
    printf("  sep[0:4]      = [%.6f, %.6f, %.6f, %.6f]\n",
           h_out_sep[0], h_out_sep[1], h_out_sep[2], h_out_sep[3]);

    auto esep_e4m3 = compute_error(h_out_sep.data(), h_ref_e4m3.data(), K);
    print_error("10x5-kernel vs E4M3 ref", esep_e4m3);

    // Cross-check fused vs other paths
    auto ef_v2  = compute_error(h_out_fused.data(), h_out_v2.data(), K);
    auto ef_sep = compute_error(h_out_fused.data(), h_out_sep.data(), K);
    printf("\n");
    print_error("Fused vs 10xV2 (atomicAdd ordering diff)", ef_v2);
    print_error("Fused vs 10x5-kernel (atomicAdd ordering diff)", ef_sep);

    // Pass criteria
    bool pass_fused = (ef_e4m3.nan_count == 0) && (ef_e4m3.norm_pct < 1.0f);
    bool pass_v2    = (ev2_e4m3.nan_count == 0) && (ev2_e4m3.norm_pct < 0.01f);
    bool pass_sep   = (esep_e4m3.nan_count == 0) && (esep_e4m3.norm_pct < 0.01f);
    bool pass_quant = (ef_fp32.nan_count == 0) && (ef_fp32.norm_pct < 10.0f);

    printf("\n  CORRECTNESS:\n");
    printf("    Fused vs E4M3 ref:     %s (%.4f%%)\n", pass_fused ? "PASS" : "FAIL", ef_e4m3.norm_pct);
    printf("    10xV2 vs E4M3 ref:     %s (%.4f%%)\n", pass_v2 ? "PASS" : "FAIL", ev2_e4m3.norm_pct);
    printf("    10x5-kernel vs E4M3:   %s (%.4f%%)\n", pass_sep ? "PASS" : "FAIL", esep_e4m3.norm_pct);
    printf("    E4M3 quant error:      %s (%.2f%%)\n", pass_quant ? "PASS" : "FAIL", ef_fp32.norm_pct);

    // ================================================================
    // Benchmark
    // ================================================================
    printf("\n--- Benchmark (10 experts, M=1) ---\n");
    const int warmup = 50, iters = 200;
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    // --- Benchmark 1: Multi-Expert Fused (1 launch) ---
    for (int i = 0; i < warmup; i++) {
        cudaMemset(d_out_fused, 0, K * sizeof(float));
        verdict_fused_multi_expert<<<total_blocks, BLOCK_SIZE, smem_fused>>>(
            d_input, d_all_w1, d_all_w2, d_out_fused,
            d_expert_ids, d_expert_wts, d_token_ids,
            K, N_half, NUM_ACTIVE, TILES_PER_EXPERT);
    }
    cudaDeviceSynchronize();

    std::vector<float> fused_times, coop_times;
    for (int i = 0; i < iters; i++) {
        cudaMemset(d_out_fused, 0, K * sizeof(float));
        cudaEventRecord(ev_start);
        verdict_fused_multi_expert<<<total_blocks, BLOCK_SIZE, smem_fused>>>(
            d_input, d_all_w1, d_all_w2, d_out_fused,
            d_expert_ids, d_expert_wts, d_token_ids,
            K, N_half, NUM_ACTIVE, TILES_PER_EXPERT);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        float ms; cudaEventElapsedTime(&ms, ev_start, ev_stop);
        fused_times.push_back(ms * 1000.0f);
    }

    // --- Benchmark 2: Multi-Expert Cooperative (1 cooperative launch) ---
    {
        int smem_coop = std::max((int)(K / TILES_PER_EXPERT * sizeof(float)),
                                 (int)(N_half * sizeof(float)));
        int coop_K = K, coop_Nh = N_half, coop_NA = NUM_ACTIVE, coop_TPE = TILES_PER_EXPERT;

        for (int i = 0; i < warmup; i++) {
            cudaMemset(d_out_coop, 0, K * sizeof(float));
            cudaMemset(d_coop_partials, 0, coop_part_size * sizeof(float));
            void* args[] = { (void*)&d_input, (void*)&d_all_w1, (void*)&d_all_w2,
                             (void*)&d_out_coop, (void*)&d_expert_ids, (void*)&d_expert_wts,
                             (void*)&d_token_ids, (void*)&d_coop_partials, (void*)&d_coop_inter,
                             (void*)&coop_K, (void*)&coop_Nh, (void*)&coop_NA, (void*)&coop_TPE };
            cudaLaunchCooperativeKernel((void*)verdict_fused_multi_expert_coop,
                                        total_blocks, BLOCK_SIZE, args, smem_coop);
        }
        cudaDeviceSynchronize();

        for (int i = 0; i < iters; i++) {
            cudaMemset(d_out_coop, 0, K * sizeof(float));
            cudaMemset(d_coop_partials, 0, coop_part_size * sizeof(float));
            cudaEventRecord(ev_start);
            void* args[] = { (void*)&d_input, (void*)&d_all_w1, (void*)&d_all_w2,
                             (void*)&d_out_coop, (void*)&d_expert_ids, (void*)&d_expert_wts,
                             (void*)&d_token_ids, (void*)&d_coop_partials, (void*)&d_coop_inter,
                             (void*)&coop_K, (void*)&coop_Nh, (void*)&coop_NA, (void*)&coop_TPE };
            cudaLaunchCooperativeKernel((void*)verdict_fused_multi_expert_coop,
                                        total_blocks, BLOCK_SIZE, args, smem_coop);
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            float ms; cudaEventElapsedTime(&ms, ev_start, ev_stop);
            coop_times.push_back(ms * 1000.0f);
        }
    }

    // --- Benchmark 3: 10x V2 Cooperative (10+10 launches) ---
    for (int i = 0; i < warmup; i++) {
        cudaMemset(d_out_v2, 0, K * sizeof(float));
        for (int e = 0; e < NUM_ACTIVE; e++) {
            const float* w1_ptr = d_all_w1 + (long long)e * K * N2;
            const float* w2_ptr = d_all_w2 + (long long)e * N_half * K;
            float wt = h_expert_wts[e];
            cudaMemset(d_partials, 0, V2_BLOCKS * 2 * N_half * sizeof(float));
            void* args[] = { (void*)&d_input, (void*)&w1_ptr, (void*)&w2_ptr,
                             (void*)&d_expert_out, (void*)&d_partials, (void*)&d_gmem_inter,
                             (void*)&K, (void*)&N_half };
            cudaLaunchCooperativeKernel((void*)verdict_fused_v2_single, V2_BLOCKS, BLOCK_SIZE, args, smem_v2);
            kern_weighted_add<<<(K+255)/256, 256>>>(d_out_v2, d_expert_out, wt, K);
        }
    }
    cudaDeviceSynchronize();

    std::vector<float> v2_times;
    for (int i = 0; i < iters; i++) {
        cudaMemset(d_out_v2, 0, K * sizeof(float));
        cudaEventRecord(ev_start);
        for (int e = 0; e < NUM_ACTIVE; e++) {
            const float* w1_ptr = d_all_w1 + (long long)e * K * N2;
            const float* w2_ptr = d_all_w2 + (long long)e * N_half * K;
            float wt = h_expert_wts[e];
            cudaMemset(d_partials, 0, V2_BLOCKS * 2 * N_half * sizeof(float));
            void* args[] = { (void*)&d_input, (void*)&w1_ptr, (void*)&w2_ptr,
                             (void*)&d_expert_out, (void*)&d_partials, (void*)&d_gmem_inter,
                             (void*)&K, (void*)&N_half };
            cudaLaunchCooperativeKernel((void*)verdict_fused_v2_single, V2_BLOCKS, BLOCK_SIZE, args, smem_v2);
            kern_weighted_add<<<(K+255)/256, 256>>>(d_out_v2, d_expert_out, wt, K);
        }
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        float ms; cudaEventElapsedTime(&ms, ev_start, ev_stop);
        v2_times.push_back(ms * 1000.0f);
    }

    // --- Benchmark 3: 10x 5-Kernel Baseline (60 launches) ---
    for (int i = 0; i < warmup; i++) {
        cudaMemset(d_out_sep, 0, K * sizeof(float));
        for (int e = 0; e < NUM_ACTIVE; e++) {
            const float* w1_ptr = d_all_w1 + (long long)e * K * N2;
            const float* w2_ptr = d_all_w2 + (long long)e * N_half * K;
            float wt = h_expert_wts[e];
            kern_gemm1_gate<<<grid1, 256>>>(d_input, w1_ptr, d_gate, K, N_half);
            kern_gemm1_up<<<grid1, 256>>>(d_input, w1_ptr, d_up, K, N_half);
            kern_swiglu<<<1, 256>>>(d_gate, d_up, d_inter, N_half);
            kern_requant<<<1, 256>>>(d_inter, N_half);
            kern_gemm2<<<grid2, 256>>>(d_inter, w2_ptr, d_expert_out, K, N_half);
            kern_weighted_add<<<(K+255)/256, 256>>>(d_out_sep, d_expert_out, wt, K);
        }
    }
    cudaDeviceSynchronize();

    std::vector<float> sep_times;
    for (int i = 0; i < iters; i++) {
        cudaMemset(d_out_sep, 0, K * sizeof(float));
        cudaEventRecord(ev_start);
        for (int e = 0; e < NUM_ACTIVE; e++) {
            const float* w1_ptr = d_all_w1 + (long long)e * K * N2;
            const float* w2_ptr = d_all_w2 + (long long)e * N_half * K;
            float wt = h_expert_wts[e];
            kern_gemm1_gate<<<grid1, 256>>>(d_input, w1_ptr, d_gate, K, N_half);
            kern_gemm1_up<<<grid1, 256>>>(d_input, w1_ptr, d_up, K, N_half);
            kern_swiglu<<<1, 256>>>(d_gate, d_up, d_inter, N_half);
            kern_requant<<<1, 256>>>(d_inter, N_half);
            kern_gemm2<<<grid2, 256>>>(d_inter, w2_ptr, d_expert_out, K, N_half);
            kern_weighted_add<<<(K+255)/256, 256>>>(d_out_sep, d_expert_out, wt, K);
        }
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        float ms; cudaEventElapsedTime(&ms, ev_start, ev_stop);
        sep_times.push_back(ms * 1000.0f);
    }

    // --- Results ---
    auto r_fused = bench_stats(fused_times);
    auto r_coop  = bench_stats(coop_times);
    auto r_v2    = bench_stats(v2_times);
    auto r_sep   = bench_stats(sep_times);

    // Find best multi-expert time
    float best_med = fminf(r_fused.med_us, r_coop.med_us);
    const char* best_name = (r_fused.med_us <= r_coop.med_us) ? "Independent" : "Cooperative";

    printf("\n  %-38s  Launches  avg       med       p5        p95\n", "Path");
    printf("  %-38s  %-8d  %.1f us  %.1f us  %.1f us  %.1f us\n",
           "Fused Independent (640 CTAs)", 1,
           r_fused.avg_us, r_fused.med_us, r_fused.p5_us, r_fused.p95_us);
    printf("  %-38s  %-8d  %.1f us  %.1f us  %.1f us  %.1f us\n",
           "Fused Cooperative (640 CTAs, grid.sync)", 1,
           r_coop.avg_us, r_coop.med_us, r_coop.p5_us, r_coop.p95_us);
    printf("  %-38s  %-8d  %.1f us  %.1f us  %.1f us  %.1f us\n",
           "10x V2 Cooperative (16 blocks each)", 20,
           r_v2.avg_us, r_v2.med_us, r_v2.p5_us, r_v2.p95_us);
    printf("  %-38s  %-8d  %.1f us  %.1f us  %.1f us  %.1f us\n",
           "10x 5-Kernel Baseline", 60,
           r_sep.avg_us, r_sep.med_us, r_sep.p5_us, r_sep.p95_us);

    float speedup_vs_v2  = r_v2.med_us / best_med;
    float speedup_vs_sep = r_sep.med_us / best_med;
    printf("\n  Best multi-expert: %s (%.1f us)\n", best_name, best_med);
    printf("  Speedup vs 10xV2:     %.2fx\n", speedup_vs_v2);
    printf("  Speedup vs 10x5-kern: %.2fx\n", speedup_vs_sep);

    // ================================================================
    // Summary
    // ================================================================
    bool all_pass = pass_fused && pass_v2 && pass_sep && pass_quant;
    printf("\n=== TASK 4 SUMMARY ===\n");
    printf("  Grid: %d experts x %d tiles = %d CTAs (%d threads/block)\n",
           NUM_ACTIVE, TILES_PER_EXPERT, total_blocks, BLOCK_SIZE);
    printf("  SMEM: %d bytes/block, atomicAdd scatter for output accumulation\n", smem_fused);
    printf("  Fused Independent:   %.1f us (1 launch, %d CTAs, redundant GEMM1)\n", r_fused.med_us, total_blocks);
    printf("  Fused Cooperative:   %.1f us (1 launch, %d CTAs, distributed GEMM1)\n", r_coop.med_us, total_blocks);
    printf("  10x V2 Cooperative:  %.1f us (20 launches)\n", r_v2.med_us);
    printf("  10x 5-Kernel:        %.1f us (60 launches)\n", r_sep.med_us);
    printf("  Best: %s at %.1f us\n", best_name, best_med);
    printf("  Speedup: %.2fx vs 10xV2, %.2fx vs 10x5-kernel\n", speedup_vs_v2, speedup_vs_sep);
    printf("  Fused error vs E4M3 ref: %.4f%%\n", ef_e4m3.norm_pct);
    printf("  VERDICT: %s\n", all_pass ? "PASSED" : "FAILED");

    // Cleanup
    cudaFree(d_input); cudaFree(d_all_w1); cudaFree(d_all_w2);
    cudaFree(d_out_fused); cudaFree(d_out_coop); cudaFree(d_out_v2); cudaFree(d_out_sep);
    cudaFree(d_expert_ids); cudaFree(d_expert_wts); cudaFree(d_token_ids);
    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_inter); cudaFree(d_expert_out);
    cudaFree(d_partials); cudaFree(d_gmem_inter);
    cudaFree(d_coop_partials); cudaFree(d_coop_inter);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);

    return all_pass ? 0 : 1;
}
