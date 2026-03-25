/**
 * verdict_consec_k_fused.cu — Sprint 5 Task 0
 *
 * Fused cooperative MMA kernel: GEMM1 → SwiGLU → E4M3 requant → GEMM2
 *
 * KEY INNOVATION: Consecutive-K packing makes each MMA register hold K
 * positions from a single SF_BLOCK=16 block. This aligns scale_vec::4X's
 * per-register-pair scaling with per-K-block scaling.
 * Result: raw E4M3FN checkpoint scales pass directly to MMA. ZERO rescaling.
 *
 * Validated: consec_k_probe2 confirms 0.0000% error (bit-exact) at MMA level.
 *
 * CUDA-graph safe: atomic barriers, no cooperative_groups, no -rdc=true.
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a \
 *     --expt-relaxed-constexpr --compiler-options '-fPIC' \
 *     -o verdict_consec_k_fused verdict_consec_k_fused.cu
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <numeric>

// ============================================================================
// Constants (Qwen3.5-397B-A17B, EP=4)
// ============================================================================
static constexpr int HIDDEN      = 4096;
static constexpr int N_HALF      = 1024;
static constexpr int N2          = 2 * N_HALF;
static constexpr int NUM_ACTIVE  = 10;
static constexpr int TILES       = 64;
static constexpr int K_PER_TILE  = HIDDEN / TILES;  // 64

static constexpr int BM = 16;
static constexpr int BN = 64;
static constexpr int BK = 64;

static constexpr int SF_BLOCK   = 16;
static constexpr int SF_PER_K   = BK / SF_BLOCK;       // 4

static constexpr int NUM_WARPS  = 8;
static constexpr int WARP_SIZE  = 32;
static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;  // 256

static constexpr int K_PACKED      = HIDDEN / 2;
static constexpr int N_HALF_PACKED = N_HALF / 2;
static constexpr int SF_COLS_W1    = HIDDEN / SF_BLOCK;   // 256
static constexpr int SF_COLS_W2    = N_HALF / SF_BLOCK;   // 64

static constexpr int SMEM_A       = BM * (BK / 2);         // 512
static constexpr int SMEM_B       = BN * (BK / 2);         // 2048
static constexpr int SMEM_SFA_PAD = 16;
static constexpr int SMEM_SFB     = BN * SF_PER_K;         // 256
static constexpr int SMEM_TOTAL   = SMEM_A + SMEM_B + SMEM_SFA_PAD + SMEM_SFB + 128;

static const float E2M1_TABLE[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// ============================================================================
// E4M3FN Host Utilities
// ============================================================================
float h_e4m3fn_decode(uint8_t x) {
    int s = (x >> 7) & 1, e = (x >> 3) & 0xF, m = x & 7;
    if (e == 15 && m == 7) return s ? -NAN : NAN;
    float val = (e == 0) ? ldexpf((float)m, -9) : ldexpf((float)(8 + m), e - 10);
    return s ? -val : val;
}

uint8_t h_e4m3fn_encode(float v) {
    if (isnan(v)) return 0x7F;
    int s = v < 0 ? 1 : 0;
    float av = fabsf(v);
    if (av > 448.0f) av = 448.0f;
    uint8_t best = 0; float best_err = FLT_MAX;
    for (int e = 0; e <= 15; e++)
        for (int m = 0; m <= 7; m++) {
            if (e == 15 && m == 7) continue;
            float repr = (e == 0) ? ldexpf((float)m, -9) : ldexpf((float)(8+m), e-10);
            float err = fabsf(av - repr);
            if (err < best_err) { best_err = err; best = (e << 3) | m; }
        }
    return (s << 7) | best;
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

float host_silu(float x) { return x / (1.0f + expf(-x)); }

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

__device__ __forceinline__ uint32_t swizzle_343(uint32_t off) {
    return off ^ ((off >> 3) & 0x70u);
}

__device__ __forceinline__ uint32_t get_nibble_swz(const uint8_t* smem, int rbo, int k) {
    int addr = rbo + k / 2;
    uint8_t bv = smem[swizzle_343(addr)];
    return (k & 1) ? ((bv >> 4) & 0xFu) : (bv & 0xFu);
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
    if (threadIdx.x == 0) {
        int target = total_ctas * (gen + 1);
        atomicAdd((int*)counter, 1);
        while (atomicAdd((int*)counter, 0) < target) {}
    }
    __syncthreads();
}

// ============================================================================
// Consecutive-K packing helpers
// ============================================================================
__device__ __forceinline__ void pack_a_consec(
    uint32_t (&a)[4], const uint8_t* s_A, int lane_id)
{
    a[0] = a[1] = a[2] = a[3] = 0;
    if (lane_id / 4 == 0) {
        int t0 = lane_id % 4;
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            a[0] |= get_nibble_swz(s_A, 0, t0*8 + p)      << (p*4);
            a[2] |= get_nibble_swz(s_A, 0, 32 + t0*8 + p) << (p*4);
        }
    }
}

__device__ __forceinline__ void pack_b_consec(
    uint32_t (&b)[2], const uint8_t* s_B, int rbo, int lane_id)
{
    b[0] = b[1] = 0;
    int t0 = lane_id % 4;
    #pragma unroll
    for (int p = 0; p < 8; p++) {
        b[0] |= get_nibble_swz(s_B, rbo, t0*8 + p)      << (p*4);
        b[1] |= get_nibble_swz(s_B, rbo, 32 + t0*8 + p) << (p*4);
    }
}

__device__ __forceinline__ uint32_t pack_sf4(const uint8_t* sf) {
    return (uint32_t)sf[0] | ((uint32_t)sf[1]<<8) | ((uint32_t)sf[2]<<16) | ((uint32_t)sf[3]<<24);
}

// ============================================================================
// FUSED COOPERATIVE KERNEL
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
verdict_consec_k_fused(
    const uint8_t* __restrict__ input_fp4,
    const uint8_t* __restrict__ input_sf,
    const uint8_t* __restrict__ all_w1_fp4,
    const uint8_t* __restrict__ all_w1_sf,
    const uint8_t* __restrict__ all_w2_fp4,
    const uint8_t* __restrict__ all_w2_sf,
    const int*     __restrict__ expert_ids,
    const float*   __restrict__ expert_wts,
    float*         __restrict__ output,
    float*         __restrict__ partials,
    uint8_t*       __restrict__ gmem_inter_fp4,
    uint8_t*       __restrict__ gmem_inter_sf,
    volatile int*  __restrict__ barrier_counter,
    int num_active)
{
    const int eidx = blockIdx.x / TILES, tile = blockIdx.x % TILES;
    const int tid = threadIdx.x, warp_id = tid / WARP_SIZE, lane_id = tid % WARP_SIZE;
    const int total_ctas = num_active * TILES;
    if (eidx >= num_active) return;

    const int eid = expert_ids[eidx];
    const float wt = expert_wts[eidx];

    extern __shared__ char smem_raw[];
    uint8_t* s_A   = (uint8_t*)smem_raw;
    uint8_t* s_B   = s_A + SMEM_A;
    uint8_t* s_SFA = s_B + SMEM_B;
    uint8_t* s_SFB = s_SFA + SMEM_SFA_PAD;

    // ================================================================
    // PHASE 1a: GEMM1 — K-distributed, 32 N-passes
    // ================================================================
    {
        const int k_start = tile * K_PER_TILE;
        const int k_start_pk = k_start / 2;
        const int k_start_sf = k_start / SF_BLOCK;
        const uint8_t* w1_fp4 = all_w1_fp4 + (long long)eid * N2 * K_PACKED;
        const uint8_t* w1_sf  = all_w1_sf  + (long long)eid * N2 * SF_COLS_W1;

        // Load A ONCE — reused across all 32 N-passes
        for (int i = tid; i < SMEM_A; i += BLOCK_SIZE) {
            int row = i / (BK/2), col = i % (BK/2);
            s_A[swizzle_343(i)] = (row == 0) ? input_fp4[k_start_pk + col] : 0;
        }
        if (tid < SF_PER_K) s_SFA[tid] = input_sf[k_start_sf + tid];
        __syncthreads();

        // Pack A ONCE
        uint32_t a_regs[4];
        pack_a_consec(a_regs, s_A, lane_id);
        uint32_t sfa_pk = pack_sf4(s_SFA);

        for (int np = 0; np < N2 / BN; np++) {
            const int n_off = np * BN;

            // Load B tile + SFB
            for (int i = tid; i < SMEM_B; i += BLOCK_SIZE) {
                int row = i / (BK/2), col = i % (BK/2);
                s_B[swizzle_343(i)] = w1_fp4[(long long)(n_off+row)*K_PACKED + k_start_pk + col];
            }
            for (int i = tid; i < BN*SF_PER_K; i += BLOCK_SIZE) {
                int row = i/SF_PER_K, col = i%SF_PER_K;
                s_SFB[i] = w1_sf[(long long)(n_off+row)*SF_COLS_W1 + k_start_sf + col];
            }
            __syncthreads();

            // Pack B + SFB per lane group
            int g = lane_id/4, Nl = 4*(g&1)+(g>>1), sn = warp_id*8+Nl;
            int rbo = sn * (BK/2);

            uint32_t b_regs[2];
            pack_b_consec(b_regs, s_B, rbo, lane_id);

            uint32_t sfb_pk = pack_sf4(&s_SFB[sn * SF_PER_K]);

            float acc[4] = {0,0,0,0};
            mma_nvf4_e4m3_m16n8k64(acc, a_regs, b_regs, acc, sfa_pk, sfb_pk);

            if (lane_id < 4) {
                long long pb = (long long)eidx*TILES*N2 + (long long)tile*N2;
                partials[pb + n_off + warp_id*8 + lane_id]   = acc[0];
                partials[pb + n_off + warp_id*8 + lane_id+4] = acc[1];
            }
            __syncthreads();
        }
    }

    grid_barrier_atomic(barrier_counter, total_ctas, 0);

    // ================================================================
    // PHASE 1b: Reduce + SwiGLU + FP4 requant (leaders only)
    // ================================================================
    if (tile == 0) {
        constexpr int CPT = N_HALF / BLOCK_SIZE;  // 4
        float sw[CPT];
        long long pb = (long long)eidx * TILES * N2;
        for (int ci = 0; ci < CPT; ci++) {
            int col = tid*CPT + ci;
            float gs = 0, us = 0;
            for (int t = 0; t < TILES; t++) {
                long long tb = pb + (long long)t*N2;
                gs += partials[tb + col];
                us += partials[tb + N_HALF + col];
            }
            sw[ci] = us * d_silu(gs);
        }

        // Group-max across SF_BLOCK=16 columns
        float lm = 0;
        for (int ci = 0; ci < CPT; ci++) lm = fmaxf(lm, fabsf(sw[ci]));
        float gm = lm;
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 1));
        gm = fmaxf(gm, __shfl_xor_sync(0xFFFFFFFF, gm, 2));

        float st = fmaxf(gm / 6.0f, 1e-30f);
        uint8_t sfb = d_e4m3fn_encode(st);
        float as = d_e4m3fn_decode(sfb);
        if (as < 1e-30f) as = 1e-30f;

        int bc = tid * CPT, b0 = bc / 2;
        uint8_t n0 = d_quantize_e2m1(sw[0]/as), n1 = d_quantize_e2m1(sw[1]/as);
        uint8_t n2 = d_quantize_e2m1(sw[2]/as), n3 = d_quantize_e2m1(sw[3]/as);
        gmem_inter_fp4[eidx*N_HALF_PACKED + b0]   = n0 | (n1 << 4);
        gmem_inter_fp4[eidx*N_HALF_PACKED + b0+1] = n2 | (n3 << 4);
        if ((tid % 4) == 0)
            gmem_inter_sf[eidx*(N_HALF/SF_BLOCK) + tid/4] = sfb;
    }

    grid_barrier_atomic(barrier_counter, total_ctas, 1);

    // ================================================================
    // PHASE 2: GEMM2 — N-distributed, 16 K-passes
    // ================================================================
    {
        const int j_start = tile * 64;
        const uint8_t* w2_fp4 = all_w2_fp4 + (long long)eid*HIDDEN*N_HALF_PACKED;
        const uint8_t* w2_sf  = all_w2_sf  + (long long)eid*HIDDEN*SF_COLS_W2;
        float acc[4] = {0,0,0,0};

        int g2 = lane_id/4, Nl2 = 4*(g2&1)+(g2>>1), sn2 = warp_id*8+Nl2;

        for (int kp = 0; kp < N_HALF/BK; kp++) {
            int ko = kp*BK, kpk = ko/2, ksf = ko/SF_BLOCK;

            // Load A: intermediate FP4
            for (int i = tid; i < SMEM_A; i += BLOCK_SIZE) {
                int row = i/(BK/2), col = i%(BK/2);
                s_A[swizzle_343(i)] = (row==0) ? gmem_inter_fp4[eidx*N_HALF_PACKED+kpk+col] : 0;
            }
            // Load B: w2 weights
            for (int i = tid; i < SMEM_B; i += BLOCK_SIZE) {
                int row = i/(BK/2), col = i%(BK/2), oc = j_start+row;
                s_B[swizzle_343(i)] = (oc<HIDDEN) ? w2_fp4[(long long)oc*N_HALF_PACKED+kpk+col] : 0;
            }
            // Load scales
            if (tid < SF_PER_K) s_SFA[tid] = gmem_inter_sf[eidx*(N_HALF/SF_BLOCK)+ksf+tid];
            for (int i = tid; i < BN*SF_PER_K; i += BLOCK_SIZE) {
                int row = i/SF_PER_K, col = i%SF_PER_K, oc = j_start+row;
                s_SFB[i] = (oc<HIDDEN) ? w2_sf[(long long)oc*SF_COLS_W2+ksf+col] : 0;
            }
            __syncthreads();

            // Pack and compute
            uint32_t ar[4]; pack_a_consec(ar, s_A, lane_id);
            int rbo2 = sn2 * (BK/2);
            uint32_t br[2]; pack_b_consec(br, s_B, rbo2, lane_id);
            uint32_t sfap = pack_sf4(s_SFA);
            uint32_t sfbp = pack_sf4(&s_SFB[sn2 * SF_PER_K]);
            mma_nvf4_e4m3_m16n8k64(acc, ar, br, acc, sfap, sfbp);
            __syncthreads();
        }

        // Scatter output with expert weight
        if (lane_id < 4) {
            int j0 = j_start + warp_id*8 + lane_id;
            int j1 = j0 + 4;
            if (j0 < HIDDEN) atomicAdd(&output[j0], wt * acc[0]);
            if (j1 < HIDDEN) atomicAdd(&output[j1], wt * acc[1]);
        }
    }
}

// ============================================================================
// Host: Quantization
// ============================================================================
void quantize_to_nvfp4_e4m3(const float* data, int numel,
                             uint8_t* packed, uint8_t* sf_out) {
    int nb = numel / SF_BLOCK;
    memset(packed, 0, numel / 2);
    for (int b = 0; b < nb; b++) {
        int s = b * SF_BLOCK;
        float bmax = 0;
        for (int i = s; i < s+SF_BLOCK; i++) bmax = std::max(bmax, fabsf(data[i]));
        uint8_t sf = h_e4m3fn_encode_ceil(std::max(bmax/6.0f, 1e-30f));
        sf_out[b] = sf;
        float as = h_e4m3fn_decode(sf);
        if (as < 1e-30f) as = 1e-30f;
        for (int i = s; i < s+SF_BLOCK; i++) {
            float sc = data[i]/as, av = fabsf(sc);
            int sign = (sc < 0) ? 1 : 0, idx = 0;
            float bd = av;
            for (int j = 1; j < 8; j++) { float d = fabsf(av-E2M1_TABLE[j]); if (d<bd){bd=d;idx=j;} }
            uint8_t fp4 = (uint8_t)((sign<<3)|idx);
            int bi = i/2;
            if (i%2==0) packed[bi] = fp4; else packed[bi] |= (fp4<<4);
        }
    }
}

float dequant_fp4_e4m3(const uint8_t* pk, const uint8_t* sf, int idx) {
    uint8_t bv = pk[idx/2];
    uint8_t nib = (idx&1) ? (bv>>4) : (bv&0xF);
    int sign = (nib>>3)&1, mag = nib&7;
    float val = E2M1_TABLE[mag] * h_e4m3fn_decode(sf[idx/SF_BLOCK]);
    return sign ? -val : val;
}

// ============================================================================
// Host References
// ============================================================================
void host_reference(const float* input, const float* w1, const float* w2,
                    const int* eids, const float* ewts, float* out, int na) {
    memset(out, 0, HIDDEN*sizeof(float));
    for (int e = 0; e < na; e++) {
        int eid = eids[e]; float wt = ewts[e];
        const float* ew1 = w1+(long long)eid*N2*HIDDEN;
        const float* ew2 = w2+(long long)eid*HIDDEN*N_HALF;
        float gate[N_HALF], up[N_HALF];
        for (int n = 0; n < N_HALF; n++) {
            float s = 0; for (int k = 0; k < HIDDEN; k++) s += input[k]*ew1[(long long)n*HIDDEN+k];
            gate[n] = s;
        }
        for (int n = 0; n < N_HALF; n++) {
            float s = 0; for (int k = 0; k < HIDDEN; k++) s += input[k]*ew1[(long long)(n+N_HALF)*HIDDEN+k];
            up[n] = s;
        }
        float inter[N_HALF];
        for (int n = 0; n < N_HALF; n++) inter[n] = up[n]*host_silu(gate[n]);
        for (int j = 0; j < HIDDEN; j++) {
            float s = 0; for (int n = 0; n < N_HALF; n++) s += inter[n]*ew2[(long long)j*N_HALF+n];
            out[j] += wt*s;
        }
    }
}

void host_quantized_reference(
    const uint8_t* ifp4, const uint8_t* isf,
    const uint8_t* w1f, const uint8_t* w1s,
    const uint8_t* w2f, const uint8_t* w2s,
    const int* eids, const float* ewts, float* out, int na) {
    memset(out, 0, HIDDEN*sizeof(float));
    for (int e = 0; e < na; e++) {
        int eid = eids[e]; float wt = ewts[e];
        const uint8_t* ew1f = w1f+(long long)eid*N2*K_PACKED;
        const uint8_t* ew1s = w1s+(long long)eid*N2*SF_COLS_W1;
        const uint8_t* ew2f = w2f+(long long)eid*HIDDEN*N_HALF_PACKED;
        const uint8_t* ew2s = w2s+(long long)eid*HIDDEN*SF_COLS_W2;
        float gate[N_HALF], up_a[N_HALF];
        for (int n = 0; n < N_HALF; n++) {
            float s = 0;
            for (int k = 0; k < HIDDEN; k++)
                s += dequant_fp4_e4m3(ifp4,isf,k) * dequant_fp4_e4m3(ew1f+(long long)n*K_PACKED, ew1s+(long long)n*SF_COLS_W1, k);
            gate[n] = s;
        }
        for (int n = 0; n < N_HALF; n++) {
            float s = 0;
            for (int k = 0; k < HIDDEN; k++)
                s += dequant_fp4_e4m3(ifp4,isf,k) * dequant_fp4_e4m3(ew1f+(long long)(n+N_HALF)*K_PACKED, ew1s+(long long)(n+N_HALF)*SF_COLS_W1, k);
            up_a[n] = s;
        }
        float sw[N_HALF];
        for (int n = 0; n < N_HALF; n++) sw[n] = up_a[n]*host_silu(gate[n]);
        uint8_t ifp[N_HALF_PACKED], isf2[N_HALF/SF_BLOCK];
        memset(ifp, 0, sizeof(ifp));
        quantize_to_nvfp4_e4m3(sw, N_HALF, ifp, isf2);
        for (int j = 0; j < HIDDEN; j++) {
            float s = 0;
            for (int n = 0; n < N_HALF; n++)
                s += dequant_fp4_e4m3(ifp,isf2,n) * dequant_fp4_e4m3(ew2f+(long long)j*N_HALF_PACKED, ew2s+(long long)j*SF_COLS_W2, n);
            out[j] += wt*s;
        }
    }
}

// ============================================================================
// Error + Main
// ============================================================================
struct ErrS { double rmse, rel; int nan_c; };
ErrS compute_error(const float* a, const float* r, int n) {
    ErrS s={}; double es=0,rs=0;
    for (int i=0;i<n;i++) { if(isnan(a[i])||isinf(a[i])){s.nan_c++;continue;}
        double d=a[i]-r[i]; es+=d*d; rs+=(double)r[i]*r[i]; }
    s.rmse=sqrt(es/n); s.rel=(rs>0)?sqrt(es/rs):0; return s;
}
#define CHECK_CUDA(c) do{cudaError_t _e=(c);if(_e!=cudaSuccess){printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);}}while(0)

int main() {
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM%d%d, %d SMs, %dKB SMEM)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           (int)(prop.sharedMemPerMultiprocessor/1024));
    printf("\n=== Verdict Consecutive-K Fused Cooperative Kernel ===\n");
    printf("MMA: mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.ue4m3\n");
    printf("Packing: consecutive-K (t0*8+p), zero rescaling\n\n");

    const int NE=16, TC=NUM_ACTIVE*TILES;
    printf("Config: K=%d, N_half=%d, %d active/%d experts, %d tiles, %d CTAs\n",
           HIDDEN, N_HALF, NUM_ACTIVE, NE, TILES, TC);

    int mb=0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, verdict_consec_k_fused, BLOCK_SIZE, SMEM_TOTAL);
    int cap = mb*prop.multiProcessorCount;
    printf("Occupancy: %d CTAs/SM x %d SMs = %d (need %d) %s\n\n", mb, prop.multiProcessorCount, cap, TC, cap>=TC?"OK":"FAIL");
    if (cap < TC) { printf("ERROR: deadlock risk\n"); return 1; }

    printf("Generating Xavier-scaled data...\n");
    srand(42);
    auto rf = [](){return((float)rand()/RAND_MAX-0.5f)*2.0f;};
    float* hi = new float[HIDDEN]; for(int i=0;i<HIDDEN;i++) hi[i]=rf();
    float w1s = 1.0f/sqrtf((float)HIDDEN), w2s = 1.0f/sqrtf((float)N_HALF);
    float* hw1 = new float[(long long)NE*N2*HIDDEN];
    for(long long i=0;i<(long long)NE*N2*HIDDEN;i++) hw1[i]=rf()*w1s;
    float* hw2 = new float[(long long)NE*HIDDEN*N_HALF];
    for(long long i=0;i<(long long)NE*HIDDEN*N_HALF;i++) hw2[i]=rf()*w2s;
    int heids[NUM_ACTIVE]; float hewts[NUM_ACTIVE];
    for(int i=0;i<NUM_ACTIVE;i++){heids[i]=i%NE; hewts[i]=1.0f/NUM_ACTIVE;}

    printf("Quantizing to NVFP4...\n");
    size_t ifs=HIDDEN/2, iss=HIDDEN/SF_BLOCK;
    uint8_t *hif=new uint8_t[ifs](), *his=new uint8_t[iss]();
    quantize_to_nvfp4_e4m3(hi, HIDDEN, hif, his);
    size_t w1fs=(size_t)NE*N2*K_PACKED, w1ss_sz=(size_t)NE*N2*SF_COLS_W1;
    uint8_t *hw1f=new uint8_t[w1fs](), *hw1s_d=new uint8_t[w1ss_sz]();
    for(int e=0;e<NE;e++) for(int n=0;n<N2;n++)
        quantize_to_nvfp4_e4m3(&hw1[(long long)e*N2*HIDDEN+(long long)n*HIDDEN], HIDDEN,
            &hw1f[(long long)e*N2*K_PACKED+(long long)n*K_PACKED],
            &hw1s_d[(long long)e*N2*SF_COLS_W1+(long long)n*SF_COLS_W1]);
    size_t w2fs=(size_t)NE*HIDDEN*N_HALF_PACKED, w2ss_sz=(size_t)NE*HIDDEN*SF_COLS_W2;
    uint8_t *hw2f=new uint8_t[w2fs](), *hw2s_d=new uint8_t[w2ss_sz]();
    for(int e=0;e<NE;e++) for(int j=0;j<HIDDEN;j++)
        quantize_to_nvfp4_e4m3(&hw2[(long long)e*HIDDEN*N_HALF+(long long)j*N_HALF], N_HALF,
            &hw2f[(long long)e*HIDDEN*N_HALF_PACKED+(long long)j*N_HALF_PACKED],
            &hw2s_d[(long long)e*HIDDEN*SF_COLS_W2+(long long)j*SF_COLS_W2]);

    printf("Computing references...\n");
    float* hrf=new float[HIDDEN](); host_reference(hi,hw1,hw2,heids,hewts,hrf,NUM_ACTIVE);
    float* hrq=new float[HIDDEN]();
    host_quantized_reference(hif,his,hw1f,hw1s_d,hw2f,hw2s_d,heids,hewts,hrq,NUM_ACTIVE);
    ErrS qf0=compute_error(hrq,hrf,HIDDEN);
    printf("  QRef vs FP32 baseline: %.4f%%\n", qf0.rel*100);

    printf("Uploading to GPU...\n");
    uint8_t *dif,*dis,*dw1f,*dw1s,*dw2f,*dw2s,*dif2,*dis2;
    int *deids; float *dewts,*dout,*dpart; int *dbar;
    CHECK_CUDA(cudaMalloc(&dif,ifs)); CHECK_CUDA(cudaMalloc(&dis,iss));
    CHECK_CUDA(cudaMalloc(&dw1f,w1fs)); CHECK_CUDA(cudaMalloc(&dw1s,w1ss_sz));
    CHECK_CUDA(cudaMalloc(&dw2f,w2fs)); CHECK_CUDA(cudaMalloc(&dw2s,w2ss_sz));
    CHECK_CUDA(cudaMalloc(&deids,NUM_ACTIVE*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dewts,NUM_ACTIVE*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dout,HIDDEN*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dpart,(size_t)NUM_ACTIVE*TILES*N2*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dif2,NUM_ACTIVE*N_HALF_PACKED));
    CHECK_CUDA(cudaMalloc(&dis2,NUM_ACTIVE*(N_HALF/SF_BLOCK)));
    CHECK_CUDA(cudaMalloc(&dbar,sizeof(int)));
    CHECK_CUDA(cudaMemcpy(dif,hif,ifs,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dis,his,iss,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw1f,hw1f,w1fs,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw1s,hw1s_d,w1ss_sz,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw2f,hw2f,w2fs,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw2s,hw2s_d,w2ss_sz,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deids,heids,NUM_ACTIVE*sizeof(int),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dewts,hewts,NUM_ACTIVE*sizeof(float),cudaMemcpyHostToDevice));

    cudaFuncSetAttribute(verdict_consec_k_fused, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);

    // === Run kernel with real scales ===
    printf("\nLaunching kernel (%d CTAs)...\n", TC);
    CHECK_CUDA(cudaMemset(dout,0,HIDDEN*sizeof(float)));
    CHECK_CUDA(cudaMemset(dbar,0,sizeof(int)));
    verdict_consec_k_fused<<<TC,BLOCK_SIZE,SMEM_TOTAL>>>(
        dif,dis,dw1f,dw1s,dw2f,dw2s,deids,dewts,dout,dpart,dif2,dis2,dbar,NUM_ACTIVE);
    cudaError_t err=cudaDeviceSynchronize();
    if(err!=cudaSuccess){printf("Kernel error: %s\n",cudaGetErrorString(err));return 1;}

    float* hgo=new float[HIDDEN];
    CHECK_CUDA(cudaMemcpy(hgo,dout,HIDDEN*sizeof(float),cudaMemcpyDeviceToHost));

    printf("\n=== Correctness ===\n");
    printf("GPU[0:8]:  "); for(int i=0;i<8;i++) printf("%10.4f ",hgo[i]); printf("\n");
    printf("QRef[0:8]: "); for(int i=0;i<8;i++) printf("%10.4f ",hrq[i]); printf("\n");
    printf("FP32[0:8]: "); for(int i=0;i<8;i++) printf("%10.4f ",hrf[i]); printf("\n");

    ErrS eq=compute_error(hgo,hrq,HIDDEN), ef=compute_error(hgo,hrf,HIDDEN);
    ErrS qf=compute_error(hrq,hrf,HIDDEN);
    printf("\nGPU vs QRef:  RMSE=%.6f RelErr=%.4f%% NaN=%d\n", eq.rmse, eq.rel*100, eq.nan_c);
    printf("GPU vs FP32:  RMSE=%.6f RelErr=%.4f%% NaN=%d\n", ef.rmse, ef.rel*100, ef.nan_c);
    printf("QRef vs FP32: RMSE=%.6f RelErr=%.4f%% (baseline)\n", qf.rmse, qf.rel*100);

    bool pq = (eq.nan_c==0)&&(eq.rel<0.05), pf = (ef.nan_c==0)&&(ef.rel<0.50);
    printf("\nvs QRef (<5%%): %s\nvs FP32 (<50%%): %s\n", pq?"PASS":"FAIL", pf?"PASS":"FAIL");

    // === Benchmark ===
    printf("\n=== Benchmark ===\n");
    for(int i=0;i<20;i++){
        CHECK_CUDA(cudaMemset(dout,0,HIDDEN*sizeof(float)));
        CHECK_CUDA(cudaMemset(dbar,0,sizeof(int)));
        verdict_consec_k_fused<<<TC,BLOCK_SIZE,SMEM_TOTAL>>>(
            dif,dis,dw1f,dw1s,dw2f,dw2s,deids,dewts,dout,dpart,dif2,dis2,dbar,NUM_ACTIVE);
    }
    cudaDeviceSynchronize();

    std::vector<float> times;
    for(int i=0;i<100;i++){
        cudaEvent_t st,sp; cudaEventCreate(&st); cudaEventCreate(&sp);
        CHECK_CUDA(cudaMemset(dout,0,HIDDEN*sizeof(float)));
        CHECK_CUDA(cudaMemset(dbar,0,sizeof(int)));
        cudaEventRecord(st);
        verdict_consec_k_fused<<<TC,BLOCK_SIZE,SMEM_TOTAL>>>(
            dif,dis,dw1f,dw1s,dw2f,dw2s,deids,dewts,dout,dpart,dif2,dis2,dbar,NUM_ACTIVE);
        cudaEventRecord(sp); cudaEventSynchronize(sp);
        float ms; cudaEventElapsedTime(&ms,st,sp); times.push_back(ms*1000.0f);
        cudaEventDestroy(st); cudaEventDestroy(sp);
    }
    std::sort(times.begin(),times.end());
    float med=times[50], mn=std::accumulate(times.begin(),times.end(),0.0f)/100;
    printf("Latency: median=%.1f us, mean=%.1f us, p10=%.1f, p90=%.1f\n",
           med, mn, times[10], times[90]);
    printf("\nComparison:\n");
    printf("  VLLM_CUTLASS baseline:                 98.0 us\n");
    printf("  Sprint 5 backup (rescaling, no -rdc):  625.9 us\n");
    printf("  THIS (consec-K, no rescaling):         %.1f us\n", med);
    if(med>0) printf("  Speedup vs VLLM_CUTLASS: %.2fx\n", 98.0f/med);

    delete[] hi; delete[] hw1; delete[] hw2;
    delete[] hif; delete[] his; delete[] hw1f; delete[] hw1s_d; delete[] hw2f; delete[] hw2s_d;
    delete[] hrf; delete[] hrq; delete[] hgo;
    cudaFree(dif); cudaFree(dis); cudaFree(dw1f); cudaFree(dw1s);
    cudaFree(dw2f); cudaFree(dw2s); cudaFree(deids); cudaFree(dewts);
    cudaFree(dout); cudaFree(dpart); cudaFree(dif2); cudaFree(dis2); cudaFree(dbar);
    printf("\nDone.\n");
    return (pq&&pf) ? 0 : 1;
}
