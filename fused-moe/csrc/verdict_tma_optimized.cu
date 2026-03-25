/**
 * verdict_tma_optimized.cu — Sprint 6 Task 1
 *
 * Two-kernel N-distributed MoE pipeline:
 *   K1A: GEMM1 + SwiGLU + E4M3 requant  (160 CTAs, ALL active)
 *   K1B: GEMM2 + weighted scatter         (640 CTAs, ALL active)
 *
 * KEY OPTIMIZATIONS vs Sprint 5 cooperative kernel (116 μs):
 * 1. N-distributed tiles: each CTA covers BN=64 N-columns, iterates K
 *    - Eliminates 32 N-pass serial loop + 96 barriers per CTA
 *    - Eliminates 5.2MB partials buffer + reduction phase
 *    - SwiGLU computed in-register (no GMEM round-trip)
 * 2. Two-kernel pipeline: CUDA stream ordering replaces atomic barrier
 *    - No idle CTA spinning (was 480 CTAs polling atomicAdd = L2 thrash)
 *    - 100% SM utilization in both phases
 * 3. (TODO) cp.async double-buffered pipeline
 * 4. (TODO) ldmatrix.b4x16_p64 for register loads
 *
 * Constraints preserved:
 *   - Consecutive-K packing (bit-exact with scale_vec::4X)
 *   - Native E4M3FN scales (ue4m3) — zero rescaling
 *   - CUDA graph compatible (no cooperative_groups, no -rdc=true)
 *
 * Build:
 *   nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a \
 *     --expt-relaxed-constexpr --compiler-options '-fPIC' \
 *     -o verdict_tma_optimized verdict_tma_optimized.cu
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

// N-distributed tile counts
static constexpr int N_TILES_G1   = N_HALF / BN;          // 16
static constexpr int N_TILES_G2   = HIDDEN / BN;           // 64
static constexpr int K_PASSES_G1  = HIDDEN / BK;           // 64
static constexpr int K_PASSES_G2  = N_HALF / BK;           // 16

// SMEM sizes
static constexpr int SMEM_A       = BM * (BK / 2);         // 512
static constexpr int SMEM_B       = BN * (BK / 2);         // 2048
static constexpr int SMEM_SFB     = BN * SF_PER_K;         // 256

// K1A SMEM: double-buffered (A+B_gate+B_up+SFA+SFB_gate+SFB_up) × 2 + swiglu
static constexpr int SMEM_K1A_STAGE = SMEM_A + 2*SMEM_B + 16 + 2*SMEM_SFB;  // 5136 per stage
static constexpr int SMEM_K1A = 2*SMEM_K1A_STAGE + BN*(int)sizeof(float) + 128;  // ~10656
// K1B SMEM: A + B + SFA(16) + SFB
static constexpr int SMEM_K1B = SMEM_A + SMEM_B + 16 + SMEM_SFB + 128;

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
// Fast consecutive-K packing via direct uint32 SMEM loads
// ============================================================================
// KEY INSIGHT: With consecutive-K storage (byte[i] = nib[2i] | nib[2i+1]<<4),
// a 4-byte load at byte offset t0*4 gives nibbles K[t0*8..t0*8+7] in exactly
// the register layout MMA expects for consecutive-K packing.
// The swizzle_343 maps 4-byte-aligned groups to 4-byte-aligned groups,
// so *(uint32_t*)&smem[swizzle_343(addr)] is always valid.
//
// This replaces 16 nibble reads + 16 shift-OR ops with 2 uint32 loads.
//
__device__ __forceinline__ void pack_a_fast(
    uint32_t (&a)[4], const uint8_t* s_A, int lane_id)
{
    a[0] = a[1] = a[2] = a[3] = 0;
    if (lane_id / 4 == 0) {
        int t0 = lane_id % 4;
        a[0] = *(const uint32_t*)&s_A[swizzle_343(t0 * 4)];
        a[2] = *(const uint32_t*)&s_A[swizzle_343(16 + t0 * 4)];
    }
}

__device__ __forceinline__ void pack_b_fast(
    uint32_t (&b)[2], const uint8_t* s_B, int rbo, int lane_id)
{
    int t0 = lane_id % 4;
    b[0] = *(const uint32_t*)&s_B[swizzle_343(rbo + t0 * 4)];
    b[1] = *(const uint32_t*)&s_B[swizzle_343(rbo + 16 + t0 * 4)];
}

__device__ __forceinline__ uint32_t pack_sf_fast(const uint8_t* sf) {
    return *(const uint32_t*)sf;
}

// ============================================================================
// KERNEL 1A: GEMM1 + SwiGLU + E4M3 requant (N-distributed)
// ============================================================================
// Grid: num_active × N_TILES_G1 = num_active × 16
// ALL CTAs active — no idle spinning
//
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
verdict_k1a_gemm1_swiglu(
    const uint8_t* __restrict__ input_fp4,
    const uint8_t* __restrict__ input_sf,
    const uint8_t* __restrict__ all_w1_fp4,
    const uint8_t* __restrict__ all_w1_sf,
    const int*     __restrict__ expert_ids,
    uint8_t*       __restrict__ gmem_inter_fp4,
    uint8_t*       __restrict__ gmem_inter_sf,
    int num_active)
{
    const int eidx = blockIdx.x / N_TILES_G1;
    const int tile = blockIdx.x % N_TILES_G1;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    if (eidx >= num_active) return;

    const int eid = expert_ids[eidx];
    const int n_base = tile * BN;

    extern __shared__ char smem_raw[];
    // Double-buffered SMEM: two stages
    uint8_t* stage[2];
    stage[0] = (uint8_t*)smem_raw;
    stage[1] = stage[0] + SMEM_K1A_STAGE;
    float* s_swiglu = (float*)(stage[1] + SMEM_K1A_STAGE);

    // Offsets within each stage
    #define S_A(s)        (stage[s])
    #define S_B_GATE(s)   (stage[s] + SMEM_A)
    #define S_B_UP(s)     (stage[s] + SMEM_A + SMEM_B)
    #define S_SFA(s)      (stage[s] + SMEM_A + 2*SMEM_B)
    #define S_SFB_GATE(s) (stage[s] + SMEM_A + 2*SMEM_B + 16)
    #define S_SFB_UP(s)   (stage[s] + SMEM_A + 2*SMEM_B + 16 + SMEM_SFB)

    const uint8_t* w1_fp4 = all_w1_fp4 + (long long)eid * N2 * K_PACKED;
    const uint8_t* w1_sf  = all_w1_sf  + (long long)eid * N2 * SF_COLS_W1;

    // Warp's B-row assignment
    int g = lane_id / 4, Nl = 4*(g&1) + (g>>1), sn = warp_id*8 + Nl;
    int rbo = sn * (BK/2);

    float gate_acc[4] = {0,0,0,0};
    float up_acc[4]   = {0,0,0,0};

    // === LOAD helper macro ===
    #define LOAD_STAGE(buf, kp_val) do { \
        int _ks = (kp_val) * BK; \
        int _kpk = _ks / 2; \
        int _ksf = _ks / SF_BLOCK; \
        for (int _i = tid; _i < SMEM_A; _i += BLOCK_SIZE) { \
            int _row = _i / (BK/2), _col = _i % (BK/2); \
            S_A(buf)[swizzle_343(_i)] = (_row == 0) ? input_fp4[_kpk + _col] : 0; \
        } \
        for (int _i = tid; _i < SMEM_B; _i += BLOCK_SIZE) { \
            int _row = _i / (BK/2), _col = _i % (BK/2); \
            S_B_GATE(buf)[swizzle_343(_i)] = w1_fp4[(long long)(n_base+_row)*K_PACKED + _kpk + _col]; \
        } \
        for (int _i = tid; _i < SMEM_B; _i += BLOCK_SIZE) { \
            int _row = _i / (BK/2), _col = _i % (BK/2); \
            S_B_UP(buf)[swizzle_343(_i)] = w1_fp4[(long long)(N_HALF+n_base+_row)*K_PACKED + _kpk + _col]; \
        } \
        if (tid < SF_PER_K) S_SFA(buf)[tid] = input_sf[_ksf + tid]; \
        for (int _i = tid; _i < BN*SF_PER_K; _i += BLOCK_SIZE) { \
            int _row = _i / SF_PER_K, _col = _i % SF_PER_K; \
            S_SFB_GATE(buf)[_i] = w1_sf[(long long)(n_base+_row)*SF_COLS_W1 + _ksf + _col]; \
        } \
        for (int _i = tid; _i < BN*SF_PER_K; _i += BLOCK_SIZE) { \
            int _row = _i / SF_PER_K, _col = _i % SF_PER_K; \
            S_SFB_UP(buf)[_i] = w1_sf[(long long)(N_HALF+n_base+_row)*SF_COLS_W1 + _ksf + _col]; \
        } \
    } while(0)

    // Pre-fill stage 0
    LOAD_STAGE(0, 0);
    __syncthreads();

    for (int kp = 0; kp < K_PASSES_G1; kp++) {
        int cur = kp & 1;
        int nxt = cur ^ 1;

        // Compute from current stage
        uint32_t a_regs[4];
        pack_a_fast(a_regs, S_A(cur), lane_id);
        uint32_t sfa_pk = pack_sf_fast(S_SFA(cur));

        uint32_t b_gate[2];
        pack_b_fast(b_gate, S_B_GATE(cur), rbo, lane_id);
        uint32_t sfb_gate_pk = pack_sf_fast(&S_SFB_GATE(cur)[sn * SF_PER_K]);
        mma_nvf4_e4m3_m16n8k64(gate_acc, a_regs, b_gate, gate_acc, sfa_pk, sfb_gate_pk);

        uint32_t b_up[2];
        pack_b_fast(b_up, S_B_UP(cur), rbo, lane_id);
        uint32_t sfb_up_pk = pack_sf_fast(&S_SFB_UP(cur)[sn * SF_PER_K]);
        mma_nvf4_e4m3_m16n8k64(up_acc, a_regs, b_up, up_acc, sfa_pk, sfb_up_pk);

        // Load next stage (overlaps nothing with regular loads, but saves one sync)
        if (kp + 1 < K_PASSES_G1)
            LOAD_STAGE(nxt, kp + 1);

        __syncthreads();  // ensures both compute reads and next load writes complete
    }

    #undef LOAD_STAGE
    #undef S_A
    #undef S_B_GATE
    #undef S_B_UP
    #undef S_SFA
    #undef S_SFB_GATE
    #undef S_SFB_UP

    // SwiGLU in-register, write to SMEM for requantization
    if (lane_id < 4) {
        int c0 = warp_id*8 + lane_id;
        int c1 = c0 + 4;
        s_swiglu[c0] = up_acc[0] * d_silu(gate_acc[0]);
        s_swiglu[c1] = up_acc[1] * d_silu(gate_acc[1]);
    }
    __syncthreads();

    // Requantize 64 SwiGLU values → E4M3 FP4 (4 SF groups of 16)
    if (tid < BN / SF_BLOCK) {
        int sf_base = tid * SF_BLOCK;
        float gm = 0;
        for (int i = 0; i < SF_BLOCK; i++)
            gm = fmaxf(gm, fabsf(s_swiglu[sf_base + i]));

        float st = fmaxf(gm / 6.0f, 1e-30f);
        uint8_t sfb = d_e4m3fn_encode(st);
        float as = d_e4m3fn_decode(sfb);
        if (as < 1e-30f) as = 1e-30f;

        gmem_inter_sf[eidx*(N_HALF/SF_BLOCK) + n_base/SF_BLOCK + tid] = sfb;

        int fp4_base = eidx*N_HALF_PACKED + (n_base + sf_base)/2;
        for (int i = 0; i < SF_BLOCK; i += 2) {
            uint8_t n0 = d_quantize_e2m1(s_swiglu[sf_base+i] / as);
            uint8_t n1 = d_quantize_e2m1(s_swiglu[sf_base+i+1] / as);
            gmem_inter_fp4[fp4_base + i/2] = n0 | (n1 << 4);
        }
    }
}

// ============================================================================
// KERNEL 1B: GEMM2 + weighted atomicAdd scatter (N-distributed)
// ============================================================================
// Grid: num_active × N_TILES_G2 = num_active × 64
// ALL CTAs active
//
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
verdict_k1b_gemm2_scatter(
    const uint8_t* __restrict__ gmem_inter_fp4,
    const uint8_t* __restrict__ gmem_inter_sf,
    const uint8_t* __restrict__ all_w2_fp4,
    const uint8_t* __restrict__ all_w2_sf,
    const int*     __restrict__ expert_ids,
    const float*   __restrict__ expert_wts,
    float*         __restrict__ output,
    int num_active)
{
    const int eidx = blockIdx.x / N_TILES_G2;
    const int tile = blockIdx.x % N_TILES_G2;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    if (eidx >= num_active) return;

    const int eid = expert_ids[eidx];
    const float wt = expert_wts[eidx];
    const int j_start = tile * BN;

    extern __shared__ char smem_raw[];
    uint8_t* s_A   = (uint8_t*)smem_raw;
    uint8_t* s_B   = s_A + SMEM_A;
    uint8_t* s_SFA = s_B + SMEM_B;
    uint8_t* s_SFB = s_SFA + 16;

    const uint8_t* w2_fp4 = all_w2_fp4 + (long long)eid*HIDDEN*N_HALF_PACKED;
    const uint8_t* w2_sf  = all_w2_sf  + (long long)eid*HIDDEN*SF_COLS_W2;

    int g2 = lane_id/4, Nl2 = 4*(g2&1)+(g2>>1), sn2 = warp_id*8+Nl2;
    float acc[4] = {0,0,0,0};

    for (int kp = 0; kp < K_PASSES_G2; kp++) {
        int ko = kp*BK, kpk = ko/2, ksf = ko/SF_BLOCK;

        // Load A row 0 only (32 bytes)
        if (tid < 8) {
            *(uint32_t*)&s_A[swizzle_343(tid * 4)] =
                *(const uint32_t*)&gmem_inter_fp4[eidx*N_HALF_PACKED+kpk+tid*4];
        }
        // Load B: 2048 bytes as 512 uint32_t (vectorized)
        for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
            int byte_off = i * 4;
            int row = byte_off / (BK/2), col = byte_off % (BK/2);
            int oc = j_start + row;
            *(uint32_t*)&s_B[swizzle_343(byte_off)] =
                (oc < HIDDEN) ? *(const uint32_t*)&w2_fp4[(long long)oc*N_HALF_PACKED+kpk+col] : 0;
        }
        // Load SFA
        if (tid == 0) *(uint32_t*)s_SFA = *(const uint32_t*)&gmem_inter_sf[eidx*(N_HALF/SF_BLOCK)+ksf];
        // Load SFB: 64 rows × 4 bytes
        if (tid < BN) {
            int oc = j_start + tid;
            *(uint32_t*)&s_SFB[tid * SF_PER_K] =
                (oc < HIDDEN) ? *(const uint32_t*)&w2_sf[(long long)oc*SF_COLS_W2+ksf] : 0;
        }
        __syncthreads();

        uint32_t ar[4]; pack_a_fast(ar, s_A, lane_id);
        int rbo2 = sn2 * (BK/2);
        uint32_t br[2]; pack_b_fast(br, s_B, rbo2, lane_id);
        uint32_t sfap = pack_sf_fast(s_SFA);
        uint32_t sfbp = pack_sf_fast(&s_SFB[sn2 * SF_PER_K]);
        mma_nvf4_e4m3_m16n8k64(acc, ar, br, acc, sfap, sfbp);
        __syncthreads();
    }

    if (lane_id < 4) {
        int j0 = j_start + warp_id*8 + lane_id;
        int j1 = j0 + 4;
        if (j0 < HIDDEN) atomicAdd(&output[j0], wt * acc[0]);
        if (j1 < HIDDEN) atomicAdd(&output[j1], wt * acc[1]);
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
    printf("\n=== Verdict N-Distributed Two-Kernel Pipeline ===\n");
    printf("MMA: mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.ue4m3\n");
    printf("Packing: consecutive-K, zero rescaling\n");
    printf("K1A grid: %d × %d = %d CTAs (GEMM1+SwiGLU+requant)\n", NUM_ACTIVE, N_TILES_G1, NUM_ACTIVE*N_TILES_G1);
    printf("K1B grid: %d × %d = %d CTAs (GEMM2+scatter)\n", NUM_ACTIVE, N_TILES_G2, NUM_ACTIVE*N_TILES_G2);
    printf("K1A SMEM: %d bytes, K1B SMEM: %d bytes\n\n", SMEM_K1A, SMEM_K1B);

    // Check occupancy
    int mb_k1a=0, mb_k1b=0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb_k1a, verdict_k1a_gemm1_swiglu, BLOCK_SIZE, SMEM_K1A);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb_k1b, verdict_k1b_gemm2_scatter, BLOCK_SIZE, SMEM_K1B);
    printf("K1A occupancy: %d CTAs/SM (grid %d)\n", mb_k1a, NUM_ACTIVE*N_TILES_G1);
    printf("K1B occupancy: %d CTAs/SM (grid %d)\n\n", mb_k1b, NUM_ACTIVE*N_TILES_G2);

    printf("Generating Xavier-scaled data...\n");
    srand(42);
    auto rf = [](){return((float)rand()/RAND_MAX-0.5f)*2.0f;};
    float* hi = new float[HIDDEN]; for(int i=0;i<HIDDEN;i++) hi[i]=rf();
    float w1s = 1.0f/sqrtf((float)HIDDEN), w2s = 1.0f/sqrtf((float)N_HALF);
    const int NE=16;
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
    int *deids; float *dewts,*dout;
    CHECK_CUDA(cudaMalloc(&dif,ifs)); CHECK_CUDA(cudaMalloc(&dis,iss));
    CHECK_CUDA(cudaMalloc(&dw1f,w1fs)); CHECK_CUDA(cudaMalloc(&dw1s,w1ss_sz));
    CHECK_CUDA(cudaMalloc(&dw2f,w2fs)); CHECK_CUDA(cudaMalloc(&dw2s,w2ss_sz));
    CHECK_CUDA(cudaMalloc(&deids,NUM_ACTIVE*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&dewts,NUM_ACTIVE*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dout,HIDDEN*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dif2,(size_t)NUM_ACTIVE*N_HALF_PACKED));
    CHECK_CUDA(cudaMalloc(&dis2,(size_t)NUM_ACTIVE*(N_HALF/SF_BLOCK)));
    CHECK_CUDA(cudaMemcpy(dif,hif,ifs,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dis,his,iss,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw1f,hw1f,w1fs,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw1s,hw1s_d,w1ss_sz,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw2f,hw2f,w2fs,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dw2s,hw2s_d,w2ss_sz,cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deids,heids,NUM_ACTIVE*sizeof(int),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dewts,hewts,NUM_ACTIVE*sizeof(float),cudaMemcpyHostToDevice));

    cudaFuncSetAttribute(verdict_k1a_gemm1_swiglu, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_K1A);
    cudaFuncSetAttribute(verdict_k1b_gemm2_scatter, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_K1B);

    int grid_k1a = NUM_ACTIVE * N_TILES_G1;
    int grid_k1b = NUM_ACTIVE * N_TILES_G2;

    // === Correctness run ===
    printf("\nLaunching kernels (K1A: %d, K1B: %d CTAs)...\n", grid_k1a, grid_k1b);
    CHECK_CUDA(cudaMemset(dout,0,HIDDEN*sizeof(float)));
    verdict_k1a_gemm1_swiglu<<<grid_k1a,BLOCK_SIZE,SMEM_K1A>>>(
        dif,dis,dw1f,dw1s,deids,dif2,dis2,NUM_ACTIVE);
    verdict_k1b_gemm2_scatter<<<grid_k1b,BLOCK_SIZE,SMEM_K1B>>>(
        dif2,dis2,dw2f,dw2s,deids,dewts,dout,NUM_ACTIVE);
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

    bool pq = (eq.nan_c==0)&&(eq.rel<0.15), pf = (ef.nan_c==0)&&(ef.rel<0.50);
    printf("\nvs QRef (<15%%): %s\nvs FP32 (<50%%): %s\n", pq?"PASS":"FAIL", pf?"PASS":"FAIL");

    // === Benchmark: combined K1A+K1B ===
    printf("\n=== Benchmark (K1A+K1B combined) ===\n");
    for(int i=0;i<20;i++){
        CHECK_CUDA(cudaMemset(dout,0,HIDDEN*sizeof(float)));
        verdict_k1a_gemm1_swiglu<<<grid_k1a,BLOCK_SIZE,SMEM_K1A>>>(
            dif,dis,dw1f,dw1s,deids,dif2,dis2,NUM_ACTIVE);
        verdict_k1b_gemm2_scatter<<<grid_k1b,BLOCK_SIZE,SMEM_K1B>>>(
            dif2,dis2,dw2f,dw2s,deids,dewts,dout,NUM_ACTIVE);
    }
    cudaDeviceSynchronize();

    std::vector<float> times, times_k1a, times_k1b;
    for(int i=0;i<100;i++){
        cudaEvent_t t0,t1,t2; cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventCreate(&t2);
        CHECK_CUDA(cudaMemset(dout,0,HIDDEN*sizeof(float)));
        cudaEventRecord(t0);
        verdict_k1a_gemm1_swiglu<<<grid_k1a,BLOCK_SIZE,SMEM_K1A>>>(
            dif,dis,dw1f,dw1s,deids,dif2,dis2,NUM_ACTIVE);
        cudaEventRecord(t1);
        verdict_k1b_gemm2_scatter<<<grid_k1b,BLOCK_SIZE,SMEM_K1B>>>(
            dif2,dis2,dw2f,dw2s,deids,dewts,dout,NUM_ACTIVE);
        cudaEventRecord(t2); cudaEventSynchronize(t2);
        float ms01, ms12;
        cudaEventElapsedTime(&ms01,t0,t1);
        cudaEventElapsedTime(&ms12,t1,t2);
        times_k1a.push_back(ms01*1000.0f);
        times_k1b.push_back(ms12*1000.0f);
        times.push_back((ms01+ms12)*1000.0f);
        cudaEventDestroy(t0); cudaEventDestroy(t1); cudaEventDestroy(t2);
    }
    std::sort(times.begin(),times.end());
    std::sort(times_k1a.begin(),times_k1a.end());
    std::sort(times_k1b.begin(),times_k1b.end());
    float med=times[50], med_k1a=times_k1a[50], med_k1b=times_k1b[50];
    float mn=std::accumulate(times.begin(),times.end(),0.0f)/100;
    printf("K1A (GEMM1+SwiGLU):  median=%.1f us\n", med_k1a);
    printf("K1B (GEMM2+scatter): median=%.1f us\n", med_k1b);
    printf("Total:               median=%.1f us, mean=%.1f us, p10=%.1f, p90=%.1f\n",
           med, mn, times[10], times[90]);
    printf("\nComparison:\n");
    printf("  VLLM_CUTLASS baseline:                   98.0 us\n");
    printf("  Sprint 5 cooperative (K-distributed):    116.1 us\n");
    printf("  THIS (N-distributed two-kernel):         %.1f us\n", med);
    if(med>0) printf("  Speedup vs CUTLASS: %.2fx\n", 98.0f/med);
    if(med>0) printf("  Speedup vs Sprint 5: %.2fx\n", 116.1f/med);

    // Cleanup
    delete[] hi; delete[] hw1; delete[] hw2;
    delete[] hif; delete[] his; delete[] hw1f; delete[] hw1s_d; delete[] hw2f; delete[] hw2s_d;
    delete[] hrf; delete[] hrq; delete[] hgo;
    cudaFree(dif); cudaFree(dis); cudaFree(dw1f); cudaFree(dw1s);
    cudaFree(dw2f); cudaFree(dw2s); cudaFree(deids); cudaFree(dewts);
    cudaFree(dout); cudaFree(dif2); cudaFree(dis2);
    printf("\nDone.\n");
    return (pq&&pf) ? 0 : 1;
}
