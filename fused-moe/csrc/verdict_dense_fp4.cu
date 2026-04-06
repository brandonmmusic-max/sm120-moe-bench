// VerdictDense FP4 GEMM — fused BF16→FP4 quantize + GEMM (single kernel)
// Numerically IDENTICAL to VerdictMoE: same quantize (warp-shuffle max,
// E4M3FN encode via PTX, E2M1 quantize), same MMA (mxf4nvf4 m16n8k64),
// same BK=64, same swizzle_343, same Kahan reduction, same pack_sf4.
//
// C[M, N] = A[M, K] (BF16) @ B[N, K/2]^T (FP4 packed, E4M3FN block-scaled)
//
// Single kernel launch with atomic grid barrier:
//   Phase 0: Cooperative BF16→FP4 quantize of A (grid-stride)
//   Phase 1: Per-(m, n_chunk, k_group) GEMM tile + Kahan reduce
//
// Optimizations over v1:
//   1. Single kernel launch (fused quantize + GEMM via atomic barrier)
//   2. Vectorized uint32 SMEM loads for MMA operands (24.6% speedup finding)
//   3. Fixed partial write to match VerdictMoE lane mapping (lane_id < 4)
//   4. Proper atomic grid barrier between k_group writers and reducer
//   5. Partials buffer passed as kernel arg (host-allocated once, not per-call)
//   6. k_groups=1 fast path: skip partials, write output directly
//
// Weight format: [N, K/2] packed FP4, [N, K/SF_BLOCK] E4M3FN scales
// (same swizzle_blockscale format as CUTLASS, scales stored as raw bytes)

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int BN = 64;
constexpr int BK = 64;
constexpr int SF_BLOCK = 16;
constexpr int SF_PER_K = BK / SF_BLOCK;  // 4

constexpr int SMEM_A   = BK / 2;             // 32 bytes
constexpr int SMEM_B   = BN * (BK / 2);      // 2048 bytes
constexpr int SMEM_SFA = SF_PER_K;            // 4 bytes (padded to 4)
constexpr int SMEM_SFB = BN * SF_PER_K;      // 256 bytes

// ============================================================================
// Device helpers — IDENTICAL to VerdictMoE
// ============================================================================

__device__ __forceinline__ float d_e4m3fn_decode(uint8_t v) {
    constexpr float LUT[256] = {
        0.0f, 0.001953125f, 0.00390625f, 0.005859375f, 0.0078125f, 0.009765625f,
        0.01171875f, 0.013671875f, 0.015625f, 0.01953125f, 0.0234375f, 0.02734375f,
        0.03125f, 0.0390625f, 0.046875f, 0.0546875f, 0.0625f, 0.078125f, 0.09375f,
        0.109375f, 0.125f, 0.15625f, 0.1875f, 0.21875f, 0.25f, 0.3125f, 0.375f,
        0.4375f, 0.5f, 0.625f, 0.75f, 0.875f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f,
        2.5f, 3.0f, 3.5f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 10.0f, 12.0f, 14.0f,
        16.0f, 20.0f, 24.0f, 28.0f, 32.0f, 40.0f, 48.0f, 56.0f, 64.0f, 80.0f,
        96.0f, 112.0f, 128.0f, 160.0f, 192.0f, 224.0f, 256.0f, 320.0f, 384.0f,
        448.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        // Negative mirror
        0.0f, -0.001953125f, -0.00390625f, -0.005859375f, -0.0078125f, -0.009765625f,
        -0.01171875f, -0.013671875f, -0.015625f, -0.01953125f, -0.0234375f, -0.02734375f,
        -0.03125f, -0.0390625f, -0.046875f, -0.0546875f, -0.0625f, -0.078125f, -0.09375f,
        -0.109375f, -0.125f, -0.15625f, -0.1875f, -0.21875f, -0.25f, -0.3125f, -0.375f,
        -0.4375f, -0.5f, -0.625f, -0.75f, -0.875f, -1.0f, -1.25f, -1.5f, -1.75f, -2.0f,
        -2.5f, -3.0f, -3.5f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f, -10.0f, -12.0f, -14.0f,
        -16.0f, -20.0f, -24.0f, -28.0f, -32.0f, -40.0f, -48.0f, -56.0f, -64.0f, -80.0f,
        -96.0f, -112.0f, -128.0f, -160.0f, -192.0f, -224.0f, -256.0f, -320.0f, -384.0f,
        -448.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
    return LUT[v];
}

__device__ __forceinline__ uint8_t d_e4m3fn_encode(float val) {
    uint16_t packed;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
        : "=h"(packed) : "f"(val), "f"(0.0f));
    return (uint8_t)((packed >> 8) & 0xFF);
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

__device__ __forceinline__ uint32_t pack_sf4(const uint8_t* sf) {
    return (uint32_t)sf[0] | ((uint32_t)sf[1]<<8)
         | ((uint32_t)sf[2]<<16) | ((uint32_t)sf[3]<<24);
}

// ============================================================================
// Atomic Grid Barrier (CUDA-graph safe) — IDENTICAL to VerdictMoE
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

// ============================================================================
// Fused Dense FP4 GEMM — single kernel launch
//
// Phase 0: Cooperative BF16→FP4 quantize (all CTAs, grid-stride)
// Phase 1: Per-(m, n_chunk, k_group) GEMM tile
// Phase 2: Kahan reduction (k_group==0 CTAs only, after barrier)
//
// Grid: (M * n_chunks * k_groups) — flat 1D for barrier compatibility
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 4)
verdict_dense_fused_kernel(
    const __nv_bfloat16* __restrict__ input_bf16,  // [M, K]
    uint8_t*             __restrict__ a_fp4,        // [M, K/2] workspace
    uint8_t*             __restrict__ a_sf,         // [M, K/SF_BLOCK] workspace
    const uint8_t*       __restrict__ b_fp4,        // [N, K/2]
    const uint8_t*       __restrict__ b_sf,         // [N, K/SF_BLOCK]
    float*               __restrict__ partials,     // [M * n_chunks * k_groups * BN]
    __nv_bfloat16*       __restrict__ output,       // [M, N]
    volatile int*        __restrict__ barrier_counter,
    float                alpha,
    int M, int N, int K,
    int n_chunks, int k_groups)
{
    const int total_ctas = M * n_chunks * k_groups;
    const int tid = threadIdx.x;

    // ================================================================
    // PHASE 0: BF16→FP4 quantize (cooperative, grid-stride)
    // IDENTICAL to VerdictMoE prologue
    // ================================================================
    {
        const int K_PACKED = K / 2;
        const int SF_COLS = K / SF_BLOCK;
        const int num_sf_groups = K / SF_BLOCK;
        const int total_sf_groups = M * num_sf_groups;
        const int global_tid = blockIdx.x * BLOCK_SIZE + tid;
        const int total_threads = gridDim.x * BLOCK_SIZE;
        const int half_warp_id = global_tid / 16;
        const int hw_lane = tid % 16;

        // Process one SF group (16 elements) per half-warp, grid-stride
        for (int hwi = half_warp_id; hwi < total_sf_groups; hwi += total_threads / 16) {
            const int m = hwi / num_sf_groups;
            const int g = hwi % num_sf_groups;
            const int kb = g * SF_BLOCK;

            float val = __bfloat162float(input_bf16[m * K + kb + hw_lane]);
            float aval = fabsf(val);

            // Warp-shuffle max across 16 lanes (identical to VerdictMoE)
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
                a_fp4[m * K_PACKED + kb / 2 + hw_lane / 2] = nib | (partner_nib << 4);
            if (hw_lane == 0)
                a_sf[m * SF_COLS + g] = sf_byte;
        }
    }

    // Barrier: all CTAs must finish quantization before GEMM reads a_fp4/a_sf
    grid_barrier_atomic(barrier_counter, total_ctas, 0);

    // ================================================================
    // Decode grid indices from flat blockIdx.x
    // Layout: blockIdx.x = m * (n_chunks * k_groups) + n_chunk * k_groups + k_group
    // ================================================================
    const int flat = blockIdx.x;
    const int m       = flat / (n_chunks * k_groups);
    const int rem     = flat % (n_chunks * k_groups);
    const int n_chunk = rem / k_groups;
    const int k_group = rem % k_groups;

    if (m >= M) return;

    const int K_PACKED    = K / 2;
    const int SF_COLS     = K / SF_BLOCK;
    const int n_start     = n_chunk * BN;
    const int k_per_group = K / k_groups;
    const int k_tiles     = k_per_group / BK;
    const int k_base      = k_group * k_per_group;

    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // ================================================================
    // SMEM layout — identical to VerdictMoE
    // ================================================================
    extern __shared__ char smem_raw[];
    uint8_t* s_A   = (uint8_t*)smem_raw;
    uint8_t* s_B   = s_A + 32;
    uint8_t* s_SFA = s_B + SMEM_B;
    uint8_t* s_SFB = s_SFA + ((SF_PER_K + 3) & ~3);

    // ================================================================
    // Warp→output mapping — IDENTICAL to VerdictMoE
    // ================================================================
    const int g_lane = lane_id / 4;
    const int Nl = 4 * (g_lane & 1) + (g_lane >> 1);
    const int sn = warp_id * 8 + Nl;
    const int t0 = lane_id % 4;
    const int rbo = sn * (BK / 2);

    float acc[4] = {0, 0, 0, 0};

    for (int kt = 0; kt < k_tiles; kt++) {
        const int k_off = k_base + kt * BK;
        const int k_pk  = k_off / 2;
        const int k_sf  = k_off / SF_BLOCK;

        // Load A (one row, BK/2 = 32 bytes) — vectorized uint32
        for (int i = tid; i < 8; i += BLOCK_SIZE) {
            *(uint32_t*)(s_A + i * 4) =
                *(const uint32_t*)&a_fp4[m * K_PACKED + k_pk + i * 4];
        }

        // Load B (BN rows x BK/2 cols) — vectorized uint32 + swizzle_343
        for (int i = tid; i < SMEM_B / 4; i += BLOCK_SIZE) {
            int boff = i * 4;
            int row = boff / (BK / 2), col = boff % (BK / 2);
            int global_n = n_start + row;
            if (global_n < N)
                *(uint32_t*)&s_B[swizzle_343(boff)] =
                    *(const uint32_t*)&b_fp4[(long long)global_n * K_PACKED + k_pk + col];
            else
                *(uint32_t*)&s_B[swizzle_343(boff)] = 0;
        }

        // Load SFA — single uint32 load (4 bytes, always aligned)
        // k_sf = k_off / SF_BLOCK = (k_base + kt * 64) / 16 = always 4-aligned
        if (tid == 0) {
            *(uint32_t*)s_SFA =
                *(const uint32_t*)&a_sf[m * SF_COLS + k_sf];
        }

        // Load SFB — vectorized uint32 loads (4 scale bytes per row, contiguous)
        // k_sf is 4-aligned so &b_sf[n * SF_COLS + k_sf] is 4-byte aligned
        for (int i = tid; i < BN; i += BLOCK_SIZE) {
            int global_n = n_start + i;
            if (global_n < N)
                *(uint32_t*)&s_SFB[i * SF_PER_K] =
                    *(const uint32_t*)&b_sf[(long long)global_n * SF_COLS + k_sf];
            else
                *(uint32_t*)&s_SFB[i * SF_PER_K] = 0;
        }

        __syncthreads();

        // MMA — IDENTICAL operand layout to VerdictMoE
        // Vectorized uint32 SMEM loads (key optimization: 24.6% speedup)
        uint32_t b_reg[2];
        b_reg[0] = *(uint32_t*)&s_B[swizzle_343(rbo + t0 * 4)];
        b_reg[1] = *(uint32_t*)&s_B[swizzle_343(rbo + 16 + t0 * 4)];
        // Direct uint32 SMEM load for SFB (4 bytes always aligned at sn*4)
        uint32_t sfb_pk = *(const uint32_t*)&s_SFB[sn * SF_PER_K];

        uint32_t a_reg[4] = {0, 0, 0, 0};
        if (lane_id / 4 == 0) {
            a_reg[0] = *(uint32_t*)(s_A + t0 * 4);
            a_reg[2] = *(uint32_t*)(s_A + 16 + t0 * 4);
        }
        // Direct uint32 SMEM load for SFA (4 bytes at SMEM base, always aligned)
        uint32_t sfa_pk = *(const uint32_t*)s_SFA;

        mma_nvf4_e4m3_m16n8k64(acc, a_reg, b_reg, acc, sfa_pk, sfb_pk);

        __syncthreads();
    }

    // ================================================================
    // PHASE 2: Write partials / output — using VerdictMoE lane mapping
    //
    // For M=1, only lane_id < 4 in each warp hold valid M=0 data.
    // c0 = warp_id * 8 + lane_id maps to BN columns (8 per warp, 4 threads).
    // c1 = c0 + 4 covers the second half of the warp's 8-column tile.
    // This is IDENTICAL to VerdictMoE's partial write (lines 320-327).
    // ================================================================
    if (k_groups == 1) {
        // Fast path: single k_group, write directly to output (no Kahan needed)
        if (lane_id < 4) {
            int c0 = warp_id * 8 + lane_id;
            int c1 = c0 + 4;
            int gn0 = n_start + c0;
            int gn1 = n_start + c1;
            if (gn0 < N) output[m * N + gn0] = __float2bfloat16(alpha * acc[0]);
            if (gn1 < N) output[m * N + gn1] = __float2bfloat16(alpha * acc[1]);
        }
    } else {
        // Multi k_group: write partials, then reduce
        if (lane_id < 4) {
            long long pb = ((long long)m * n_chunks + n_chunk) * k_groups + k_group;
            pb *= BN;
            int c0 = warp_id * 8 + lane_id;
            int c1 = c0 + 4;
            partials[pb + c0] = acc[0];
            partials[pb + c1] = acc[1];
        }

        // Barrier: all k_groups must finish writing before reduction
        grid_barrier_atomic(barrier_counter, total_ctas, 1);

        // Leader CTA (k_group == 0) reduces with Kahan summation
        if (k_group == 0 && tid < BN) {
            int col = tid;
            int gn = n_start + col;
            if (gn < N) {
                float sum = 0.0f, comp = 0.0f;
                for (int kg = 0; kg < k_groups; kg++) {
                    long long base = ((long long)m * n_chunks + n_chunk) * k_groups + kg;
                    base *= BN;
                    float y = partials[base + col] - comp;
                    float t = sum + y;
                    comp = (t - sum) - y;
                    sum = t;
                }
                output[m * N + gn] = __float2bfloat16(alpha * sum);
            }
        }
    }
}

// ============================================================================
// Python entry point
// ============================================================================
torch::Tensor verdict_dense_fp4(
    torch::Tensor input_bf16,   // [M, K] BF16
    torch::Tensor b_fp4,        // [N, K/2] uint8
    torch::Tensor b_sf,         // [N, K/SF_BLOCK] uint8 (E4M3FN raw bytes)
    float alpha,
    int output_size)            // actual N before padding
{
    TORCH_CHECK(input_bf16.dtype() == torch::kBFloat16, "input must be BF16");
    TORCH_CHECK(b_fp4.dtype() == torch::kUInt8, "weights must be uint8");

    const int M = input_bf16.size(0);
    const int K = input_bf16.size(1);
    const int N = b_fp4.size(0);
    const int K_PACKED = K / 2;
    const int SF_COLS = K / SF_BLOCK;
    const int n_chunks = (N + BN - 1) / BN;

    // Allocate quantized activation workspace
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(input_bf16.device());
    auto a_fp4 = torch::empty({M, K_PACKED}, opts_u8);
    auto a_sf = torch::empty({M, SF_COLS}, opts_u8);

    // Adaptive k_groups — same heuristic as VerdictMoE, targeting ~640 CTAs
    int k_groups = 1;
    int total_k_tiles = K / BK;
    while (k_groups * 2 <= total_k_tiles &&
           M * n_chunks * k_groups * 2 <= 752) {
        k_groups *= 2;
    }

    int total_ctas = M * n_chunks * k_groups;

    // Allocate partials only when k_groups > 1
    torch::Tensor partials;
    if (k_groups > 1) {
        partials = torch::empty({(long long)M * n_chunks * k_groups * BN},
            torch::TensorOptions().dtype(torch::kFloat32).device(input_bf16.device()));
    } else {
        // Dummy 1-element tensor (kernel won't use it)
        partials = torch::empty({1},
            torch::TensorOptions().dtype(torch::kFloat32).device(input_bf16.device()));
    }

    auto output = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(input_bf16.device()));

    // Barrier counter — single int, zeroed async on current stream
    auto barrier = torch::empty({1},
        torch::TensorOptions().dtype(torch::kInt32).device(input_bf16.device()));
    auto stream = c10::cuda::getCurrentCUDAStream();
    cudaMemsetAsync(barrier.data_ptr(), 0, sizeof(int), stream);

    // SMEM: s_A(32) + s_B(2048) + s_SFA(4 padded) + s_SFB(256)
    int smem_size = 32 + SMEM_B + ((SF_PER_K + 3) & ~3) + SMEM_SFB;

    dim3 grid(total_ctas);
    dim3 block(BLOCK_SIZE);

    verdict_dense_fused_kernel<<<grid, block, smem_size, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input_bf16.data_ptr<torch::BFloat16>()),
        a_fp4.data_ptr<uint8_t>(),
        a_sf.data_ptr<uint8_t>(),
        b_fp4.data_ptr<uint8_t>(),
        b_sf.data_ptr<uint8_t>(),
        partials.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<torch::BFloat16>()),
        reinterpret_cast<volatile int*>(barrier.data_ptr<int>()),
        alpha,
        M, N, K,
        n_chunks, k_groups);

    // Slice to actual output size
    if (N != output_size) {
        output = output.index({torch::indexing::Slice(), torch::indexing::Slice(0, output_size)}).contiguous();
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("verdict_dense_fp4", &verdict_dense_fp4,
          "VerdictDense FP4 GEMM — fused BF16->FP4 quantize + GEMM, "
          "numerically identical to VerdictMoE");
}
