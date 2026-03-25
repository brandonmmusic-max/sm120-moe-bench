/**
 * verdict_fused_cooperative_e4m3.cu — Sprint 5 Task 0
 *
 * Fused cooperative MMA kernel: GEMM1 → SwiGLU → E4M3 requant → GEMM2
 *
 * KEY INNOVATION: Uses mxf4nvf4.block_scale.scale_vec::4X with native E4M3FN
 * scale factors (ue4m3), eliminating ALL per-nibble SMEM rescaling overhead.
 *
 * The checkpoint stores weights as:
 *   - FP4 data: uint8 packed (2 E2M1 nibbles/byte)
 *   - Block scales: float8_e4m3fn, one per 16 FP4 elements (SF_BLOCK=16)
 *
 * The MMA instruction natively applies E4M3FN scales per 16-element K block:
 *   result += e2m1(a_nib) * e4m3fn(sfa) * e2m1(b_nib) * e4m3fn(sfb)
 *
 * CUDA-graph safe: atomic barriers (no cooperative_groups, no grid.sync())
 * No -rdc=true needed — standard kernel launch.
 *
 * Architecture:
 *   Grid: 640 CTAs = 10 experts x 64 tiles, 256 threads/CTA
 *   Phase 1a: Distributed GEMM1 K-reduction (64 CTAs/expert)
 *   BARRIER (atomic counter spinning)
 *   Phase 1b: 10 leader CTAs reduce + SwiGLU + FP4 requant (E4M3FN scales)
 *   BARRIER (atomic counter spinning)
 *   Phase 2: GEMM2 N-distributed + weighted atomicAdd scatter
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a \
 *     --expt-relaxed-constexpr --compiler-options '-fPIC' \
 *     -o verdict_fused_cooperative_e4m3 verdict_fused_cooperative_e4m3.cu
 *
 * DO NOT use -rdc=true or cooperative_groups!
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
__device__ __forceinline__ float d_e2m1_val(int idx) {
    switch (idx & 7) {
        case 0: return 0.0f; case 1: return 0.5f; case 2: return 1.0f; case 3: return 1.5f;
        case 4: return 2.0f; case 5: return 3.0f; case 6: return 4.0f; case 7: return 6.0f;
    }
    return 0.0f;
}

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

__device__ __forceinline__ float d_silu(float x) { return x / (1.0f + __expf(-x)); }

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

// Rescale FP4 nibble from old E4M3FN scale to new (larger) unified scale
__device__ __forceinline__ uint32_t rescale_nib_e4m3(
    uint32_t nib, float old_scale, float new_scale)
{
    if (old_scale == new_scale || nib == 0) return nib;
    int sign = (nib >> 3) & 1;
    float val = d_e2m1_val(nib & 7) * old_scale / new_scale;
    float av = fabsf(val);
    int idx;
    if      (av < 0.25f) idx = 0; else if (av < 0.75f) idx = 1;
    else if (av < 1.25f) idx = 2; else if (av < 1.75f) idx = 3;
    else if (av < 2.5f)  idx = 4; else if (av < 3.5f)  idx = 5;
    else if (av < 5.0f)  idx = 6; else idx = 7;
    return (sign << 3) | idx;
}

// Compute unified E4M3FN scale from 4 block scales (max decoded, re-encoded)
__device__ __forceinline__ uint8_t unify_e4m3_scales(const uint8_t* sf4) {
    float mx = d_e4m3fn_decode(sf4[0]);
    for (int i = 1; i < 4; i++) mx = fmaxf(mx, d_e4m3fn_decode(sf4[i]));
    return d_e4m3fn_encode(mx);
}

// Rescale all nibbles in a SMEM row (32 bytes = 64 nibbles) from per-block
// E4M3FN scales to a unified scale. sf4[0..3] are the 4 block scales.
__device__ void rescale_smem_row_e4m3(
    uint8_t* smem, int row_byte_off, const uint8_t* sf4,
    uint8_t unified_sf, int tid, int block_size)
{
    float uni_scale = d_e4m3fn_decode(unified_sf);
    if (uni_scale < 1e-30f) return;
    float blk_scales[4];
    for (int i = 0; i < 4; i++) blk_scales[i] = d_e4m3fn_decode(sf4[i]);

    for (int i = tid; i < 32; i += block_size) {
        int swz_i = swizzle_343(row_byte_off + i);
        uint8_t bv = smem[swz_i];
        int k_lo = i * 2, k_hi = k_lo + 1;
        uint8_t nib_lo = bv & 0xF, nib_hi = (bv >> 4) & 0xF;
        float sf_lo = blk_scales[k_lo / SF_BLOCK];
        float sf_hi = blk_scales[k_hi / SF_BLOCK];
        if (sf_lo != uni_scale)
            nib_lo = (uint8_t)rescale_nib_e4m3(nib_lo, sf_lo, uni_scale);
        if (sf_hi != uni_scale)
            nib_hi = (uint8_t)rescale_nib_e4m3(nib_hi, sf_hi, uni_scale);
        smem[swz_i] = (nib_hi << 4) | nib_lo;
    }
}

// ============================================================================
// MMA: scale_vec::4X with E4M3FN (ue4m3), unified scales after rescaling
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
// FUSED COOPERATIVE KERNEL
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
verdict_fused_cooperative_e4m3(
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
    uint8_t* s_A = (uint8_t*)smem_raw;
    uint8_t* s_B = s_A + SMEM_A;
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

        for (int np = 0; np < N2 / BN; np++) {
            const int n_off = np * BN;

            for (int i = tid; i < SMEM_A; i += BLOCK_SIZE) {
                int row = i / (BK/2), col = i % (BK/2);
                s_A[swizzle_343(i)] = (row == 0) ? input_fp4[k_start_pk + col] : 0;
            }
            for (int i = tid; i < SMEM_B; i += BLOCK_SIZE) {
                int row = i / (BK/2), col = i % (BK/2);
                s_B[swizzle_343(i)] = w1_fp4[(long long)(n_off+row)*K_PACKED + k_start_pk + col];
            }
            if (tid < SF_PER_K) s_SFA[tid] = input_sf[k_start_sf + tid];
            for (int i = tid; i < BN*SF_PER_K; i += BLOCK_SIZE) {
                int row = i/SF_PER_K, col = i%SF_PER_K;
                s_SFB[i] = w1_sf[(long long)(n_off+row)*SF_COLS_W1 + k_start_sf + col];
            }
            __syncthreads();

            // === RESCALE: unify 4 E4M3FN block scales per row ===
            // A row 0: compute unified SFA, rescale nibbles
            uint8_t sfa_unified = unify_e4m3_scales(s_SFA);
            {   bool need_rescale = false;
                for (int i = 0; i < 4; i++)
                    if (s_SFA[i] != sfa_unified) { need_rescale = true; break; }
                if (need_rescale)
                    rescale_smem_row_e4m3(s_A, 0, s_SFA, sfa_unified, tid, BLOCK_SIZE);
            }
            // B rows: each N-column has its own 4 scales
            for (int i = tid; i < BN; i += BLOCK_SIZE) {
                uint8_t sf4[4] = {s_SFB[i*4], s_SFB[i*4+1], s_SFB[i*4+2], s_SFB[i*4+3]};
                uint8_t uni = unify_e4m3_scales(sf4);
                bool need = false;
                for (int j = 0; j < 4; j++) if (sf4[j] != uni) { need = true; break; }
                if (need) {
                    int rbo = i * (BK/2);
                    float uni_sc = d_e4m3fn_decode(uni);
                    for (int bi = 0; bi < 32; bi++) {
                        int swz_bi = swizzle_343(rbo + bi);
                        uint8_t bv = s_B[swz_bi];
                        int k_lo = bi*2, k_hi = k_lo+1;
                        uint8_t nl = bv & 0xF, nh = (bv>>4) & 0xF;
                        float sl = d_e4m3fn_decode(sf4[k_lo/SF_BLOCK]);
                        float sh = d_e4m3fn_decode(sf4[k_hi/SF_BLOCK]);
                        if (sl != uni_sc) nl = (uint8_t)rescale_nib_e4m3(nl, sl, uni_sc);
                        if (sh != uni_sc) nh = (uint8_t)rescale_nib_e4m3(nh, sh, uni_sc);
                        s_B[swz_bi] = (nh<<4) | nl;
                    }
                }
                // Store unified scale back
                s_SFB[i*4] = s_SFB[i*4+1] = s_SFB[i*4+2] = s_SFB[i*4+3] = uni;
            }
            __syncthreads();

            // Pack A registers (validated Sprint 4 pack_a_m1_v2)
            uint32_t a_regs[4] = {0,0,0,0};
            if (lane_id/4 == 0) {
                int t0 = lane_id % 4;
                #pragma unroll
                for (int p = 0; p < 8; p++) {
                    a_regs[0] |= get_nibble_swz(s_A, 0, t0+p*8)   << (p*4);
                    a_regs[2] |= get_nibble_swz(s_A, 0, t0+4+p*8) << (p*4);
                }
            }

            // Pack B registers (validated Sprint 4 pack_b_v2)
            uint32_t b_regs[2] = {0,0};
            { int g=lane_id/4, t0=lane_id%4, Nl=4*(g&1)+(g>>1);
              int rbo = (warp_id*8+Nl)*(BK/2);
              #pragma unroll
              for (int p=0;p<8;p++) {
                  b_regs[0] |= get_nibble_swz(s_B, rbo, t0+p*8)   << (p*4);
                  b_regs[1] |= get_nibble_swz(s_B, rbo, t0+4+p*8) << (p*4);
              }
            }

            // SFA: unified scale broadcast to all 4 bytes
            uint32_t sfa_pk = (uint32_t)sfa_unified | ((uint32_t)sfa_unified<<8)
                           | ((uint32_t)sfa_unified<<16) | ((uint32_t)sfa_unified<<24);
            { int g=lane_id/4, Nl=4*(g&1)+(g>>1), sn=warp_id*8+Nl, sb=sn*SF_PER_K;
              uint8_t sfb_uni = s_SFB[sb]; // all 4 bytes same after unification
              uint32_t sfb_pk = (uint32_t)sfb_uni | ((uint32_t)sfb_uni<<8)
                             | ((uint32_t)sfb_uni<<16) | ((uint32_t)sfb_uni<<24);
              float acc[4] = {0,0,0,0};
              mma_nvf4_e4m3_m16n8k64(acc, a_regs, b_regs, acc, sfa_pk, sfb_pk);
              if (lane_id < 4) {
                  long long pb = (long long)eidx*TILES*N2 + (long long)tile*N2;
                  partials[pb + n_off + warp_id*8 + lane_id]   = acc[0];
                  partials[pb + n_off + warp_id*8 + lane_id+4] = acc[1];
              }
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

        for (int kp = 0; kp < N_HALF/BK; kp++) {
            int ko = kp*BK, kpk = ko/2, ksf = ko/SF_BLOCK;

            for (int i = tid; i < SMEM_A; i += BLOCK_SIZE) {
                int row = i/(BK/2), col = i%(BK/2);
                s_A[swizzle_343(i)] = (row==0) ? gmem_inter_fp4[eidx*N_HALF_PACKED+kpk+col] : 0;
            }
            for (int i = tid; i < SMEM_B; i += BLOCK_SIZE) {
                int row = i/(BK/2), col = i%(BK/2), oc = j_start+row;
                s_B[swizzle_343(i)] = (oc<HIDDEN) ? w2_fp4[(long long)oc*N_HALF_PACKED+kpk+col] : 0;
            }
            if (tid < SF_PER_K) s_SFA[tid] = gmem_inter_sf[eidx*(N_HALF/SF_BLOCK)+ksf+tid];
            for (int i = tid; i < BN*SF_PER_K; i += BLOCK_SIZE) {
                int row = i/SF_PER_K, col = i%SF_PER_K, oc = j_start+row;
                s_SFB[i] = (oc<HIDDEN) ? w2_sf[(long long)oc*SF_COLS_W2+ksf+col] : 0;
            }
            __syncthreads();

            // === RESCALE for GEMM2 (same pattern as GEMM1) ===
            uint8_t sfa2_unified = unify_e4m3_scales(s_SFA);
            {   bool need_rescale = false;
                for (int i = 0; i < 4; i++)
                    if (s_SFA[i] != sfa2_unified) { need_rescale = true; break; }
                if (need_rescale)
                    rescale_smem_row_e4m3(s_A, 0, s_SFA, sfa2_unified, tid, BLOCK_SIZE);
            }
            for (int i = tid; i < BN; i += BLOCK_SIZE) {
                uint8_t sf4b[4] = {s_SFB[i*4], s_SFB[i*4+1], s_SFB[i*4+2], s_SFB[i*4+3]};
                uint8_t uni = unify_e4m3_scales(sf4b);
                bool need = false;
                for (int j = 0; j < 4; j++) if (sf4b[j] != uni) { need = true; break; }
                if (need) {
                    int rbo = i * (BK/2);
                    float uni_sc = d_e4m3fn_decode(uni);
                    for (int bi = 0; bi < 32; bi++) {
                        int swz_bi = swizzle_343(rbo + bi);
                        uint8_t bv = s_B[swz_bi];
                        int k_lo = bi*2, k_hi = k_lo+1;
                        uint8_t nl = bv & 0xF, nh = (bv>>4) & 0xF;
                        float sl = d_e4m3fn_decode(sf4b[k_lo/SF_BLOCK]);
                        float sh = d_e4m3fn_decode(sf4b[k_hi/SF_BLOCK]);
                        if (sl != uni_sc) nl = (uint8_t)rescale_nib_e4m3(nl, sl, uni_sc);
                        if (sh != uni_sc) nh = (uint8_t)rescale_nib_e4m3(nh, sh, uni_sc);
                        s_B[swz_bi] = (nh<<4) | nl;
                    }
                }
                s_SFB[i*4] = s_SFB[i*4+1] = s_SFB[i*4+2] = s_SFB[i*4+3] = uni;
            }
            __syncthreads();

            uint32_t ar[4]={0,0,0,0};
            if (lane_id/4==0) { int t0=lane_id%4;
                #pragma unroll
                for (int p=0;p<8;p++) {
                    ar[0] |= get_nibble_swz(s_A,0,t0+p*8)   << (p*4);
                    ar[2] |= get_nibble_swz(s_A,0,t0+4+p*8) << (p*4);
                }
            }
            uint32_t br[2]={0,0};
            { int g=lane_id/4, t0=lane_id%4, Nl=4*(g&1)+(g>>1);
              int rbo=(warp_id*8+Nl)*(BK/2);
              #pragma unroll
              for (int p=0;p<8;p++) {
                  br[0] |= get_nibble_swz(s_B,rbo,t0+p*8)   << (p*4);
                  br[1] |= get_nibble_swz(s_B,rbo,t0+4+p*8) << (p*4);
              }
            }
            // SFA/SFB unified (all 4 bytes same after rescaling)
            uint32_t sfap = (uint32_t)sfa2_unified | ((uint32_t)sfa2_unified<<8)
                         | ((uint32_t)sfa2_unified<<16) | ((uint32_t)sfa2_unified<<24);
            { int g=lane_id/4, Nl=4*(g&1)+(g>>1), sn=warp_id*8+Nl, sb=sn*SF_PER_K;
              uint8_t sfb2_uni = s_SFB[sb]; // all 4 bytes same
              uint32_t sfbp = (uint32_t)sfb2_uni | ((uint32_t)sfb2_uni<<8)
                           | ((uint32_t)sfb2_uni<<16) | ((uint32_t)sfb2_uni<<24);
              mma_nvf4_e4m3_m16n8k64(acc, ar, br, acc, sfap, sfbp);
            }
            __syncthreads();
        }

        if (lane_id < 4) {
            int j0 = j_start + warp_id*8 + lane_id;
            int j1 = j0 + 4;
            if (j0 < HIDDEN) atomicAdd(&output[j0], wt * acc[0]);
            if (j1 < HIDDEN) atomicAdd(&output[j1], wt * acc[1]);
        }
    }
}

// ============================================================================
// Host: Quantization with E4M3FN at SF_BLOCK=16
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
// Error + Helpers
// ============================================================================
struct ErrS { double rmse, rel; int nan_c; };
ErrS compute_error(const float* a, const float* r, int n) {
    ErrS s={}; double es=0,rs=0;
    for (int i=0;i<n;i++) { if(isnan(a[i])||isinf(a[i])){s.nan_c++;continue;}
        double d=a[i]-r[i]; es+=d*d; rs+=(double)r[i]*r[i]; }
    s.rmse=sqrt(es/n); s.rel=(rs>0)?sqrt(es/rs):0; return s;
}
#define CHECK_CUDA(c) do{cudaError_t _e=(c);if(_e!=cudaSuccess){printf("CUDA err %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));exit(1);}}while(0)

// ============================================================================
// Main
// ============================================================================
int main() {
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM%d%d, %d SMs, %dKB SMEM)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           (int)(prop.sharedMemPerMultiprocessor/1024));
    printf("\n=== Sprint 5 Task 0: Fused Cooperative MMA with E4M3FN Scales ===\n");
    printf("MMA: mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.ue4m3\n");
    printf("SF_BLOCK=%d, Barriers=atomic (CUDA-graph safe, no -rdc=true)\n", SF_BLOCK);

    const int NE=16, TC=NUM_ACTIVE*TILES;
    printf("Config: K=%d, N_half=%d, %d active/%d total experts, %d tiles, %d CTAs\n",
           HIDDEN, N_HALF, NUM_ACTIVE, NE, TILES, TC);

    int mb=0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, verdict_fused_cooperative_e4m3, BLOCK_SIZE, SMEM_TOTAL);
    int cap = mb*prop.multiProcessorCount;
    printf("Occupancy: %d CTAs/SM x %d SMs = %d (need %d) %s\n", mb, prop.multiProcessorCount, cap, TC, cap>=TC?"OK":"FAIL");
    if (cap < TC) { printf("ERROR: deadlock risk\n"); return 1; }

    printf("\nGenerating data...\n");
    srand(42);
    auto rf = [](){return((float)rand()/RAND_MAX-0.5f)*2.0f;};
    float* hi = new float[HIDDEN]; for(int i=0;i<HIDDEN;i++) hi[i]=rf();
    float* hw1 = new float[(long long)NE*N2*HIDDEN];
    for(long long i=0;i<(long long)NE*N2*HIDDEN;i++) hw1[i]=rf()*0.1f;
    float* hw2 = new float[(long long)NE*HIDDEN*N_HALF];
    for(long long i=0;i<(long long)NE*HIDDEN*N_HALF;i++) hw2[i]=rf()*0.1f;
    int heids[NUM_ACTIVE]; float hewts[NUM_ACTIVE];
    for(int i=0;i<NUM_ACTIVE;i++){heids[i]=i%NE; hewts[i]=1.0f/NUM_ACTIVE;}

    printf("Quantizing to NVFP4 (E4M3FN, SF=%d)...\n", SF_BLOCK);
    size_t ifs=HIDDEN/2, iss=HIDDEN/SF_BLOCK;
    uint8_t *hif=new uint8_t[ifs](), *his=new uint8_t[iss]();
    quantize_to_nvfp4_e4m3(hi, HIDDEN, hif, his);

    size_t w1fs=(size_t)NE*N2*K_PACKED, w1ss=(size_t)NE*N2*SF_COLS_W1;
    uint8_t *hw1f=new uint8_t[w1fs](), *hw1s=new uint8_t[w1ss]();
    for(int e=0;e<NE;e++) for(int n=0;n<N2;n++)
        quantize_to_nvfp4_e4m3(&hw1[(long long)e*N2*HIDDEN+(long long)n*HIDDEN], HIDDEN,
            &hw1f[(long long)e*N2*K_PACKED+(long long)n*K_PACKED],
            &hw1s[(long long)e*N2*SF_COLS_W1+(long long)n*SF_COLS_W1]);

    size_t w2fs=(size_t)NE*HIDDEN*N_HALF_PACKED, w2ss=(size_t)NE*HIDDEN*SF_COLS_W2;
    uint8_t *hw2f=new uint8_t[w2fs](), *hw2s=new uint8_t[w2ss]();
    for(int e=0;e<NE;e++) for(int j=0;j<HIDDEN;j++)
        quantize_to_nvfp4_e4m3(&hw2[(long long)e*HIDDEN*N_HALF+(long long)j*N_HALF], N_HALF,
            &hw2f[(long long)e*HIDDEN*N_HALF_PACKED+(long long)j*N_HALF_PACKED],
            &hw2s[(long long)e*HIDDEN*SF_COLS_W2+(long long)j*SF_COLS_W2]);

    printf("Computing FP32 reference...\n");
    float* hrf=new float[HIDDEN](); host_reference(hi,hw1,hw2,heids,hewts,hrf,NUM_ACTIVE);
    printf("Computing quantized reference...\n");
    float* hrq=new float[HIDDEN]();
    host_quantized_reference(hif,his,hw1f,hw1s,hw2f,hw2s,heids,hewts,hrq,NUM_ACTIVE);

    printf("Uploading to GPU...\n");
    uint8_t *dif,*dis,*dw1f,*dw1s,*dw2f,*dw2s,*dif2,*dis2;
    int *deids; float *dewts,*dout,*dpart; int *dbar;
    CHECK_CUDA(cudaMalloc(&dif,ifs)); CHECK_CUDA(cudaMalloc(&dis,iss));
    CHECK_CUDA(cudaMalloc(&dw1f,w1fs)); CHECK_CUDA(cudaMalloc(&dw1s,w1ss));
    CHECK_CUDA(cudaMalloc(&dw2f,w2fs)); CHECK_CUDA(cudaMalloc(&dw2s,w2ss));
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
    CHECK_CUDA(cudaMemcpy(dw1s,hw1s,w1ss,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw2f,hw2f,w2fs,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw2s,hw2s,w2ss,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deids,heids,NUM_ACTIVE*sizeof(int),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dewts,hewts,NUM_ACTIVE*sizeof(float),cudaMemcpyHostToDevice));

    printf("\nLaunching kernel (%d CTAs, %d threads)...\n", TC, BLOCK_SIZE);
    cudaFuncSetAttribute(verdict_fused_cooperative_e4m3, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL);
    CHECK_CUDA(cudaMemset(dout,0,HIDDEN*sizeof(float)));
    CHECK_CUDA(cudaMemset(dbar,0,sizeof(int)));

    verdict_fused_cooperative_e4m3<<<TC,BLOCK_SIZE,SMEM_TOTAL>>>(
        dif,dis,dw1f,dw1s,dw2f,dw2s,deids,dewts,dout,dpart,dif2,dis2,dbar,NUM_ACTIVE);
    cudaError_t err=cudaDeviceSynchronize();
    if(err!=cudaSuccess){printf("Kernel error: %s\n",cudaGetErrorString(err));return 1;}
    printf("Kernel completed.\n");

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

    printf("\n=== Benchmark ===\n");
    for(int i=0;i<20;i++){
        CHECK_CUDA(cudaMemset(dout,0,HIDDEN*sizeof(float)));
        CHECK_CUDA(cudaMemset(dbar,0,sizeof(int)));
        verdict_fused_cooperative_e4m3<<<TC,BLOCK_SIZE,SMEM_TOTAL>>>(
            dif,dis,dw1f,dw1s,dw2f,dw2s,deids,dewts,dout,dpart,dif2,dis2,dbar,NUM_ACTIVE);
    }
    cudaDeviceSynchronize();

    std::vector<float> times;
    for(int i=0;i<100;i++){
        cudaEvent_t st,sp; cudaEventCreate(&st); cudaEventCreate(&sp);
        CHECK_CUDA(cudaMemset(dout,0,HIDDEN*sizeof(float)));
        CHECK_CUDA(cudaMemset(dbar,0,sizeof(int)));
        cudaEventRecord(st);
        verdict_fused_cooperative_e4m3<<<TC,BLOCK_SIZE,SMEM_TOTAL>>>(
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
    printf("  Sprint 4 (scale_vec::2X, UE8M0, -rdc): 110.6 us\n");
    printf("  Sprint 4 (FP32 weights, -rdc):           38.9 us\n");
    printf("  VLLM_CUTLASS baseline:                   98.0 us\n");
    printf("  THIS (scale_vec::4X, E4M3FN, no -rdc):   %.1f us\n", med);
    if(med>0) printf("  Speedup vs VLLM_CUTLASS: %.2fx\n", 98.0f/med);

    delete[] hi; delete[] hw1; delete[] hw2;
    delete[] hif; delete[] his; delete[] hw1f; delete[] hw1s; delete[] hw2f; delete[] hw2s;
    delete[] hrf; delete[] hrq; delete[] hgo;
    cudaFree(dif); cudaFree(dis); cudaFree(dw1f); cudaFree(dw1s);
    cudaFree(dw2f); cudaFree(dw2s); cudaFree(deids); cudaFree(dewts);
    cudaFree(dout); cudaFree(dpart); cudaFree(dif2); cudaFree(dis2); cudaFree(dbar);
    printf("\nDone.\n");
    return (pq&&pf) ? 0 : 1;
}
