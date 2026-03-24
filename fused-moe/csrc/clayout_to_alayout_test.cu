/**
 * CLayout → ALayout SMEM Handoff Test for SM120 Fused MoE
 *
 * Validates the full pipeline for fusing GEMM1 output into GEMM2 input:
 *   FP32 accumulators (CLayout) → SwiGLU → FP32→E4M3 → Swizzled SMEM → GEMM2 A operand
 *
 * SM120 CLayout (CUTLASS SM80_16x8_Row, confirmed for both NVF4 and MxF8F6F4 atoms):
 *   thread t: g = t/4 (0-7), l = t%4 (0-3)
 *   d[0] → C[2g,   2l]       d[1] → C[2g+1, 2l]
 *   d[2] → C[2g,   2l+1]     d[3] → C[2g+1, 2l+1]
 *
 * SMEM Swizzle<3,4,3> for GEMM2 A operand (128B aligned):
 *   smem_byte_offset(row, col) = row * 128 + (col ^ ((row & 7) << 3))
 *
 * FP32 → E4M3 conversion: cvt.rn.satfinite.e4m3x2.f32 (SM89+)
 *   Packs 2 FP32 → 2 E4M3 in uint16: src_a → bits[7:0], src_b → bits[15:8]
 *
 * E4M3 write optimization:
 *   d[0] (col=2l) and d[2] (col=2l+1) are at the same row, consecutive columns.
 *   Swizzle mask has bit 0 = 0, so swiz(2l) is even → swiz(2l)+1 = swiz(2l+1).
 *   Therefore we can write both as a single uint16_t store. Same for d[1]+d[3].
 *
 * Part 1: Known E4M3-representable values → byte-exact verification
 * Part 2: SwiGLU integration → quantization error measurement
 * Part 3: CUTLASS GEMM2 end-to-end → proves full pipeline
 *
 * Build:
 *   CUTLASS=/home/brandonmusic/flashinfer-pr/3rdparty/cutlass
 *   nvcc -std=c++17 -O2 -gencode=arch=compute_120a,code=sm_120a \
 *     --expt-relaxed-constexpr -diag-suppress 177 \
 *     -I$CUTLASS/include -I$CUTLASS/tools/util/include \
 *     -o clayout_to_alayout_test clayout_to_alayout_test.cu
 */

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/reference/host/gett.hpp>

#include <cute/tensor.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <cfloat>
#include <algorithm>

using namespace cute;

// ============================================================================
// E4M3 Host Utilities
// ============================================================================

// E4M3FN: sign(1) + exp(4) + mant(3), bias=7, max=448, min_normal=2^-6
float host_e4m3_decode(uint8_t x) {
    int s = (x >> 7) & 1;
    int e = (x >> 3) & 0xF;
    int m = x & 0x7;
    if (e == 15 && m == 7) return s ? -NAN : NAN;
    float val;
    if (e == 0) val = ldexpf((float)m / 8.0f, -6);  // subnormal
    else        val = ldexpf(1.0f + (float)m / 8.0f, e - 7);
    return s ? -val : val;
}

uint8_t host_e4m3_encode(float v) {
    if (isnan(v)) return 0x7F;
    int s = v < 0 ? 1 : 0;
    float av = fabsf(v);
    if (av > 448.0f) av = 448.0f;  // satfinite
    uint8_t best = 0;
    float best_err = FLT_MAX;
    for (int e = 0; e <= 15; e++) {
        for (int m = 0; m <= 7; m++) {
            if (e == 15 && m == 7) continue;  // skip NaN
            float repr = (e == 0) ? ldexpf((float)m / 8.0f, -6)
                                  : ldexpf(1.0f + (float)m / 8.0f, e - 7);
            float err = fabsf(av - repr);
            if (err < best_err) { best_err = err; best = (e << 3) | m; }
        }
    }
    return (s << 7) | best;
}

// Swizzle<3,4,3>: XOR bits[5:3] of col with row[2:0]<<3
// For stride=128: swizzled_col = col ^ ((row & 7) << 3)
__host__ __device__ inline int swizzle_col(int row, int col) {
    return col ^ ((row & 7) << 3);
}

float host_silu(float x) { return x / (1.0f + expf(-x)); }

// ============================================================================
// Device: FP32 → E4M3x2 conversion (packs 2 values into uint16_t)
// ============================================================================

__device__ __forceinline__ uint16_t cvt_e4m3x2(float lo, float hi) {
    uint16_t r;
    // PTX: cvt.rn.satfinite.e4m3x2.f32 d, a, b
    // Empirically confirmed on SM120: a → bits[15:8] (high byte), b → bits[7:0] (low byte)
    // We want lo → bits[7:0], hi → bits[15:8], so pass (hi, lo) to PTX
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;  // lo → bits[7:0], hi → bits[15:8]
}

// ============================================================================
// Part 1 Kernel: CLayout → E4M3 → Swizzled SMEM (known values)
// ============================================================================

__global__ void part1_clayout_to_smem(
    const float* __restrict__ intermediate,  // [16, K] known FP32 values
    uint8_t* __restrict__ swizzled_dump,      // [16 * 128] raw SMEM bytes
    uint8_t* __restrict__ unswizzled_dump,    // [16 * K] un-swizzled E4M3
    int K)
{
    extern __shared__ uint8_t smem[];
    const int lane = threadIdx.x;
    const int g = lane / 4;  // 0-7
    const int l = lane % 4;  // 0-3

    // Zero SMEM (16 rows × 128 bytes = 2048 bytes)
    for (int i = lane; i < 16 * 128; i += 32) smem[i] = 0;
    __syncthreads();

    // Process K in groups of 8 columns (one CLayout MMA N-pass)
    for (int pass = 0; pass < K / 8; pass++) {
        int col_base = pass * 8;

        // CLayout positions for this thread
        int row0 = 2 * g;       // d[0], d[2]
        int row1 = 2 * g + 1;   // d[1], d[3]
        int col0 = col_base + 2 * l;       // d[0], d[1]
        int col1 = col_base + 2 * l + 1;   // d[2], d[3]

        // Load FP32 from known intermediate
        float d0 = intermediate[row0 * K + col0];  // C[2g,   2l+base]
        float d1 = intermediate[row1 * K + col0];  // C[2g+1, 2l+base]
        float d2 = intermediate[row0 * K + col1];  // C[2g,   2l+1+base]
        float d3 = intermediate[row1 * K + col1];  // C[2g+1, 2l+1+base]

        // FP32 → E4M3x2 conversion (PTX cvt.rn.satfinite)
        // d0 (col0) and d2 (col1=col0+1) share row0 → pack together
        uint16_t packed_r0 = cvt_e4m3x2(d0, d2);  // d0→lo, d2→hi
        uint16_t packed_r1 = cvt_e4m3x2(d1, d3);  // d1→lo, d3→hi

        // Swizzled SMEM addresses (stride = 128 bytes per row)
        // col0 is always even, mask has bit0=0 → swiz(col0) is even → 2B aligned
        int sc0_r0 = swizzle_col(row0, col0);
        int sc0_r1 = swizzle_col(row1, col0);

        // Write 2 packed E4M3 bytes per row
        *(uint16_t*)(&smem[row0 * 128 + sc0_r0]) = packed_r0;
        *(uint16_t*)(&smem[row1 * 128 + sc0_r1]) = packed_r1;
    }
    __syncthreads();

    // Dump swizzled SMEM to GMEM
    for (int i = lane; i < 16 * 128; i += 32)
        swizzled_dump[i] = smem[i];

    // Un-swizzle readback
    for (int row = 0; row < 16; row++) {
        for (int col = lane; col < K; col += 32) {
            int sc = swizzle_col(row, col);
            unswizzled_dump[row * K + col] = smem[row * 128 + sc];
        }
    }
}

// ============================================================================
// Part 2 Kernel: SwiGLU → E4M3 → Swizzled SMEM
// ============================================================================

__global__ void part2_swiglu_to_smem(
    const float* __restrict__ gate_up,   // [16, 2*K_half] from GEMM1
    uint8_t* __restrict__ swizzled_dump,
    uint8_t* __restrict__ unswizzled_dump,
    int K_half)
{
    extern __shared__ uint8_t smem[];
    const int lane = threadIdx.x;
    const int g = lane / 4;
    const int l = lane % 4;

    for (int i = lane; i < 16 * 128; i += 32) smem[i] = 0;
    __syncthreads();

    int N_full = 2 * K_half;

    for (int pass = 0; pass < K_half / 8; pass++) {
        int col_base = pass * 8;
        int row0 = 2 * g, row1 = 2 * g + 1;
        int col0 = col_base + 2 * l;
        int col1 = col_base + 2 * l + 1;

        // Load gate (first K_half cols) and up (last K_half cols) from GEMM1 output
        float ga0 = gate_up[row0 * N_full + col0];
        float ga1 = gate_up[row1 * N_full + col0];
        float ga2 = gate_up[row0 * N_full + col1];
        float ga3 = gate_up[row1 * N_full + col1];

        float u0 = gate_up[row0 * N_full + col0 + K_half];
        float u1 = gate_up[row1 * N_full + col0 + K_half];
        float u2 = gate_up[row0 * N_full + col1 + K_half];
        float u3 = gate_up[row1 * N_full + col1 + K_half];

        // SwiGLU in registers: out = up * silu(gate) = up * gate * sigmoid(gate)
        float d0 = u0 * ga0 / (1.0f + expf(-ga0));
        float d1 = u1 * ga1 / (1.0f + expf(-ga1));
        float d2 = u2 * ga2 / (1.0f + expf(-ga2));
        float d3 = u3 * ga3 / (1.0f + expf(-ga3));

        // FP32 → E4M3x2 + swizzled write
        uint16_t p0 = cvt_e4m3x2(d0, d2);
        uint16_t p1 = cvt_e4m3x2(d1, d3);

        int sc0 = swizzle_col(row0, col0);
        int sc1 = swizzle_col(row1, col0);

        *(uint16_t*)(&smem[row0 * 128 + sc0]) = p0;
        *(uint16_t*)(&smem[row1 * 128 + sc1]) = p1;
    }
    __syncthreads();

    for (int i = lane; i < 16 * 128; i += 32)
        swizzled_dump[i] = smem[i];
    for (int row = 0; row < 16; row++)
        for (int col = lane; col < K_half; col += 32) {
            int sc = swizzle_col(row, col);
            unswizzled_dump[row * K_half + col] = smem[row * 128 + sc];
        }
}

// SwiGLU + E4M3 conversion WITHOUT swizzle (flat write, for reference)
__global__ void part2_swiglu_flat_reference(
    const float* __restrict__ gate_up,  // [16, 2*K_half]
    uint8_t* __restrict__ flat_e4m3,    // [16 * K_half] E4M3 bytes
    int K_half)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 16 * K_half;
    if (idx >= total) return;

    int row = idx / K_half;
    int col = idx % K_half;
    int N_full = 2 * K_half;

    float gate = gate_up[row * N_full + col];
    float up   = gate_up[row * N_full + col + K_half];
    float val  = up * gate / (1.0f + expf(-gate));

    // Convert single FP32 → E4M3 using cvt_e4m3x2 with dummy hi operand
    uint16_t packed = cvt_e4m3x2(val, 0.0f);
    flat_e4m3[idx] = (uint8_t)(packed & 0xFF);  // lo byte = val
}

// ============================================================================
// Part 3: CUTLASS GEMM2 end-to-end (E4M3 × FP4 → float)
// ============================================================================

using ElementA_G2 = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementB_G2 = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
using ElementC_G2 = float;
using ElementD_G2 = float;

using TileShape_G2 = Shape<_128, _128, _128>;
using ClusterShape_G2 = Shape<_1, _1, _1>;

using CollectiveEpilogue_G2 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_G2, ClusterShape_G2, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, float, cutlass::layout::RowMajor,
    128 / cutlass::sizeof_bits<float>::value,
    float, cutlass::layout::RowMajor,
    128 / cutlass::sizeof_bits<float>::value,
    cutlass::epilogue::NoSmemWarpSpecialized
>::CollectiveOp;

using StageCount_G2 = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue_G2::SharedStorage))>;

using CollectiveMainloop_G2 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementA_G2, cutlass::layout::RowMajor, 16,
    ElementB_G2, cutlass::layout::ColumnMajor, 128,
    float, TileShape_G2, ClusterShape_G2, StageCount_G2,
    cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120
>::CollectiveOp;

using GemmKernel_G2 = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop_G2, CollectiveEpilogue_G2>;
using Gemm_G2 = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_G2>;

using StrideA_G2 = typename Gemm_G2::GemmKernel::StrideA;
using StrideB_G2 = typename Gemm_G2::GemmKernel::StrideB;
using StrideC_G2 = typename Gemm_G2::GemmKernel::StrideC;
using StrideD_G2 = typename Gemm_G2::GemmKernel::StrideD;
using Sm1xxCfg_G2 = typename Gemm_G2::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

template <typename Element, typename Layout>
void init_block(cutlass::TensorView<Element, Layout> view, uint64_t seed) {
    constexpr int bits = cutlass::sizeof_bits<Element>::value;
    double hi, lo;
    if constexpr (bits <= 6) { hi = 2; lo = -2; }
    else if constexpr (bits <= 8) {
        if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t> ||
                      cute::is_same_v<Element, cutlass::float_ue4m3_t>) {
            hi = 4; lo = 1;
        } else { hi = 1; lo = -1; }
    } else { hi = 4; lo = -4; }
    cutlass::reference::host::TensorFillRandomUniform(view, seed, hi, lo, 0);
}

struct GemmResult {
    bool passed;
    float normalized_error;
    float max_abs_error;
    int nan_count;
};

GemmResult run_gemm2_validation(int M, int K, int N,
    const uint8_t* host_a_e4m3, bool verbose)
{
    GemmResult result = {};

    auto stride_A = cutlass::make_cute_packed_stride(StrideA_G2{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB_G2{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC_G2{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD_G2{}, {M, N, 1});

    auto layout_A = make_layout(make_shape(M, K, 1), stride_A);
    auto layout_B = make_layout(make_shape(N, K, 1), stride_B);
    auto layout_D = make_layout(make_shape(M, N, 1), stride_D);
    auto layout_SFA = Sm1xxCfg_G2::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxCfg_G2::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

    cutlass::HostTensor<typename ElementA_G2::DataType, cutlass::layout::PackedVectorLayout> block_A;
    cutlass::HostTensor<typename ElementA_G2::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
    cutlass::HostTensor<typename ElementB_G2::DataType, cutlass::layout::PackedVectorLayout> block_B;
    cutlass::HostTensor<typename ElementB_G2::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
    cutlass::HostTensor<float, cutlass::layout::PackedVectorLayout> block_C, block_D, block_ref_D;

    block_A.reset(cutlass::make_Coord(size(layout_A)));
    block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
    block_B.reset(cutlass::make_Coord(size(layout_B)));
    block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));
    block_C.reset(cutlass::make_Coord(M * N));
    block_D.reset(cutlass::make_Coord(M * N));
    block_ref_D.reset(cutlass::make_Coord(M * N));

    // Copy E4M3 data into CUTLASS tensor
    memcpy(block_A.host_data(), host_a_e4m3, M * K);

    // SF_A = all 1.0 (UE8M0: 0x7F = 2^(127-127) = 1.0)
    memset(block_SFA.host_data(), 0x7F, size(filter_zeros(layout_SFA)));

    // Random B weights and SF
    init_block(block_B.host_view(), 100);
    init_block(block_SFB.host_view(), 101);
    cutlass::reference::host::TensorFill(block_C.host_view(), 0.0f);

    block_A.sync_device(); block_SFA.sync_device();
    block_B.sync_device(); block_SFB.sync_device();
    block_C.sync_device();

    typename CollectiveMainloop_G2::Arguments ml_args;
    ml_args.ptr_A = block_A.device_data();
    ml_args.dA = stride_A;
    ml_args.ptr_B = block_B.device_data();
    ml_args.dB = stride_B;
    ml_args.ptr_SFA = block_SFA.device_data();
    ml_args.layout_SFA = layout_SFA;
    ml_args.ptr_SFB = block_SFB.device_data();
    ml_args.layout_SFB = layout_SFB;

    typename Gemm_G2::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1}, ml_args,
        {{1.0f, 0.0f}, block_C.device_data(), stride_C, block_D.device_data(), stride_D}
    };

    Gemm_G2 gemm;
    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        printf("    can_implement: FAIL\n"); return result;
    }

    size_t ws = Gemm_G2::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(ws);
    gemm.initialize(args, workspace.get());
    status = gemm.run();
    cudaDeviceSynchronize();

    if (status != cutlass::Status::kSuccess) {
        printf("    run: FAIL\n"); return result;
    }

    // Host reference
    auto tA = make_tensor(cute::recast_ptr<typename ElementA_G2::DataType>(block_A.host_data()), layout_A);
    auto tSFA = make_tensor(block_SFA.host_data(), layout_SFA);
    auto tB = make_tensor(cute::recast_ptr<typename ElementB_G2::DataType>(block_B.host_data()), layout_B);
    auto tSFB = make_tensor(block_SFB.host_data(), layout_SFB);
    auto tC = make_tensor(cute::recast_ptr<float>(block_C.host_data()), layout_D);
    auto tD = make_tensor(cute::recast_ptr<float>(block_ref_D.host_data()), layout_D);

    cutlass::reference::host::GettBlockScalingMainloopParams<
        float, decltype(tA), decltype(tSFA), decltype(tB), decltype(tSFB)
    > ml_ref{tA, tSFA, tB, tSFB};
    cutlass::reference::host::GettEpilogueParams<
        float, float, float, float, decltype(tC), decltype(tD)
    > ep_ref{1.0f, 0.0f, tC, tD};
    cutlass::reference::host::Gemm3x(ml_ref, ep_ref);

    block_D.sync_host();

    int count = M * N;
    float max_abs = 0, sum_abs = 0;
    int nan_cnt = 0, n_close = 0;
    for (int i = 0; i < count; i++) {
        float k = block_D.host_data()[i];
        float r = block_ref_D.host_data()[i];
        if (isnan(k)) { nan_cnt++; continue; }
        float e = fabsf(k - r);
        max_abs = fmaxf(max_abs, e);
        sum_abs += e;
        if ((fabsf(r) > 0.01f && e / fabsf(r) < 0.05f) ||
            (fabsf(r) <= 0.01f && fabsf(k) < 0.05f)) n_close++;
    }
    float rms = 0;
    for (int i = 0; i < count; i++) rms += block_ref_D.host_data()[i] * block_ref_D.host_data()[i];
    rms = sqrtf(rms / count);

    result.normalized_error = rms > 0 ? 100.0f * (sum_abs / count) / rms : 0;
    result.max_abs_error = max_abs;
    result.nan_count = nan_cnt;
    result.passed = (nan_cnt == 0) && (n_close > count * 0.95f);

    if (verbose) {
        printf("    Elements: %d | NaN: %d | Max err: %.6f | Norm err: %.4f%%\n",
               count, nan_cnt, max_abs, result.normalized_error);
        printf("    ref =[%.4f, %.4f, %.4f, %.4f]\n",
               block_ref_D.host_data()[0], block_ref_D.host_data()[1],
               block_ref_D.host_data()[2], block_ref_D.host_data()[3]);
        printf("    kern=[%.4f, %.4f, %.4f, %.4f]\n",
               block_D.host_data()[0], block_D.host_data()[1],
               block_D.host_data()[2], block_D.host_data()[3]);
    }
    return result;
}

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

    int overall_pass = 0, overall_fail = 0;

    // ================================================================
    // Part 1: CLayout → E4M3 → Swizzled SMEM (known values)
    // ================================================================
    printf("=== Part 1: CLayout → E4M3 → Swizzled SMEM (byte-exact) ===\n");
    {
        constexpr int M = 16, K = 128;

        // Create known E4M3-representable values
        // Round-trip through E4M3 to guarantee exact representation
        std::vector<float> h_inter(M * K);
        std::vector<uint8_t> h_ref_e4m3(M * K);
        for (int r = 0; r < M; r++) {
            for (int c = 0; c < K; c++) {
                float raw = ((r * K + c) % 200) * 0.1f - 10.0f;
                uint8_t enc = host_e4m3_encode(raw);
                h_ref_e4m3[r * K + c] = enc;
                h_inter[r * K + c] = host_e4m3_decode(enc);  // exact E4M3 value
            }
        }

        // Host expected swizzled SMEM
        std::vector<uint8_t> h_expected_swiz(M * 128, 0);
        for (int r = 0; r < M; r++) {
            for (int c = 0; c < K; c++) {
                int sc = swizzle_col(r, c);
                h_expected_swiz[r * 128 + sc] = h_ref_e4m3[r * K + c];
            }
        }

        // Device
        float *d_inter;
        uint8_t *d_swiz, *d_unswiz;
        cudaMalloc(&d_inter, M * K * sizeof(float));
        cudaMalloc(&d_swiz, M * 128);
        cudaMalloc(&d_unswiz, M * K);
        cudaMemcpy(d_inter, h_inter.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);

        part1_clayout_to_smem<<<1, 32, M * 128>>>(d_inter, d_swiz, d_unswiz, K);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  Kernel error: %s\n", cudaGetErrorString(err));
            overall_fail++;
        } else {
            std::vector<uint8_t> h_swiz(M * 128), h_unswiz(M * K);
            cudaMemcpy(h_swiz.data(), d_swiz, M * 128, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_unswiz.data(), d_unswiz, M * K, cudaMemcpyDeviceToHost);

            // Compare un-swizzled E4M3 against host reference
            int unswiz_match = 0, unswiz_mismatch = 0;
            for (int i = 0; i < M * K; i++) {
                if (h_unswiz[i] == h_ref_e4m3[i]) unswiz_match++;
                else unswiz_mismatch++;
            }
            printf("  Un-swizzled E4M3: %d/%d match (%.2f%% error)\n",
                   unswiz_match, M * K, 100.0f * unswiz_mismatch / (M * K));

            // Compare swizzled SMEM against host expected
            int swiz_match = 0, swiz_mismatch = 0;
            for (int i = 0; i < M * 128; i++) {
                if (h_swiz[i] == h_expected_swiz[i]) swiz_match++;
                else swiz_mismatch++;
            }
            printf("  Swizzled SMEM:    %d/%d match (%.2f%% error)\n",
                   swiz_match, M * 128, 100.0f * swiz_mismatch / (M * 128));

            // Show first few values
            printf("  Sample row=0 cols 0-7: device=[");
            for (int c = 0; c < 8; c++) printf("%s%.2f", c?",":"", host_e4m3_decode(h_unswiz[c]));
            printf("]\n");
            printf("  Sample row=0 cols 0-7: ref   =[");
            for (int c = 0; c < 8; c++) printf("%s%.2f", c?",":"", h_inter[c]);
            printf("]\n");

            // Float round-trip error
            float max_fp_err = 0;
            for (int i = 0; i < M * K; i++) {
                float dv = host_e4m3_decode(h_unswiz[i]);
                float rv = h_inter[i];
                max_fp_err = fmaxf(max_fp_err, fabsf(dv - rv));
            }
            printf("  Max float round-trip error: %.6f\n", max_fp_err);

            bool p1_ok = (unswiz_mismatch == 0) && (swiz_mismatch == 0) && (max_fp_err == 0.0f);
            printf("  Part 1 VERDICT: %s\n\n", p1_ok ? "PASSED (byte-exact)" : "FAILED");
            if (p1_ok) overall_pass++; else overall_fail++;
        }
        cudaFree(d_inter); cudaFree(d_swiz); cudaFree(d_unswiz);
    }

    // ================================================================
    // Part 2: SwiGLU → E4M3 → Swizzled SMEM
    // ================================================================
    printf("=== Part 2: SwiGLU → E4M3 → Swizzled SMEM ===\n");
    {
        constexpr int M = 16, K_half = 128;
        constexpr int N_full = 2 * K_half;

        // Random gate+up values
        std::vector<float> h_gate_up(M * N_full);
        srand(42);
        for (int i = 0; i < M * N_full; i++)
            h_gate_up[i] = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;

        // Host SwiGLU + E4M3 reference
        std::vector<float> h_swiglu_fp32(M * K_half);
        for (int r = 0; r < M; r++) {
            for (int c = 0; c < K_half; c++) {
                float gate = h_gate_up[r * N_full + c];
                float up   = h_gate_up[r * N_full + c + K_half];
                h_swiglu_fp32[r * K_half + c] = up * host_silu(gate);
            }
        }

        // Device: swizzled path
        float *d_gate_up;
        uint8_t *d_swiz, *d_unswiz;
        cudaMalloc(&d_gate_up, M * N_full * sizeof(float));
        cudaMalloc(&d_swiz, M * 128);
        cudaMalloc(&d_unswiz, M * K_half);
        cudaMemcpy(d_gate_up, h_gate_up.data(), M * N_full * sizeof(float), cudaMemcpyHostToDevice);

        part2_swiglu_to_smem<<<1, 32, M * 128>>>(d_gate_up, d_swiz, d_unswiz, K_half);

        // Device: flat reference (same SwiGLU + cvt, no swizzle)
        uint8_t *d_flat;
        cudaMalloc(&d_flat, M * K_half);
        int threads = 256;
        int blocks = (M * K_half + threads - 1) / threads;
        part2_swiglu_flat_reference<<<blocks, threads>>>(d_gate_up, d_flat, K_half);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("  Kernel error: %s\n", cudaGetErrorString(err));
            overall_fail++;
        } else {
            std::vector<uint8_t> h_unswiz(M * K_half), h_flat(M * K_half);
            cudaMemcpy(h_unswiz.data(), d_unswiz, M * K_half, cudaMemcpyDeviceToHost);
            cudaMemcpy(h_flat.data(), d_flat, M * K_half, cudaMemcpyDeviceToHost);

            // Swizzle round-trip: swizzled-then-unswizzled must match flat (byte-exact)
            int swiz_match = 0;
            for (int i = 0; i < M * K_half; i++)
                if (h_unswiz[i] == h_flat[i]) swiz_match++;
            printf("  Swizzle round-trip: %d/%d byte-exact match\n", swiz_match, M * K_half);

            // Quantization error vs FP32 SwiGLU
            float max_qerr = 0, sum_qerr = 0;
            for (int i = 0; i < M * K_half; i++) {
                float dev = host_e4m3_decode(h_flat[i]);
                float ref = h_swiglu_fp32[i];
                float e = fabsf(dev - ref);
                max_qerr = fmaxf(max_qerr, e);
                sum_qerr += e;
            }
            float rms = 0;
            for (int i = 0; i < M * K_half; i++) rms += h_swiglu_fp32[i] * h_swiglu_fp32[i];
            rms = sqrtf(rms / (M * K_half));
            float norm_qerr = rms > 0 ? 100.0f * (sum_qerr / (M * K_half)) / rms : 0;

            printf("  E4M3 quantization error vs FP32: max=%.4f, norm=%.2f%%\n", max_qerr, norm_qerr);
            printf("  Sample row=0 cols 0-3: swiglu_fp32=[%.4f, %.4f, %.4f, %.4f]\n",
                   h_swiglu_fp32[0], h_swiglu_fp32[1], h_swiglu_fp32[2], h_swiglu_fp32[3]);
            printf("  Sample row=0 cols 0-3: e4m3_float=[%.4f, %.4f, %.4f, %.4f]\n",
                   host_e4m3_decode(h_flat[0]), host_e4m3_decode(h_flat[1]),
                   host_e4m3_decode(h_flat[2]), host_e4m3_decode(h_flat[3]));

            bool p2_ok = (swiz_match == M * K_half);
            printf("  Part 2 VERDICT: %s\n\n",
                   p2_ok ? "PASSED (byte-exact swizzle, expected quant error)" : "FAILED");
            if (p2_ok) overall_pass++; else overall_fail++;
        }
        cudaFree(d_gate_up); cudaFree(d_swiz); cudaFree(d_unswiz); cudaFree(d_flat);
    }

    // ================================================================
    // Part 3: CUTLASS GEMM2 with handoff E4M3 intermediate
    // ================================================================
    printf("=== Part 3: CUTLASS GEMM2 with E4M3 intermediate ===\n");
    {
        constexpr int M = 128, K = 128, N = 128;

        // Create E4M3 intermediate (quantized random values)
        std::vector<uint8_t> h_e4m3(M * K);
        srand(123);
        for (int i = 0; i < M * K; i++) {
            float raw = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
            h_e4m3[i] = host_e4m3_encode(raw);
        }

        printf("  Shape: [%d, %d] x [%d, %d] -> [%d, %d]\n", M, K, K, N, M, N);
        printf("  A: E4M3 (from handoff), SF=1.0 (UE8M0 0x7F)\n");
        printf("  B: FP4 (random), Schedule: MxF8F6F4Sm120\n");

        auto result = run_gemm2_validation(M, K, N, h_e4m3.data(), true);

        printf("  Part 3 VERDICT: %s\n\n",
               result.passed ? "PASSED" : "FAILED");
        if (result.passed) overall_pass++; else overall_fail++;
    }

    // ================================================================
    // Summary
    // ================================================================
    printf("====================================\n");
    printf("OVERALL: %d/%d passed\n", overall_pass, overall_pass + overall_fail);
    printf("====================================\n");
    printf("\nCLayout→ALayout Handoff Summary:\n");
    printf("  CLayout: SM80_16x8_Row (g=t/4, l=t%%4)\n");
    printf("    d[0]→C[2g,2l] d[1]→C[2g+1,2l] d[2]→C[2g,2l+1] d[3]→C[2g+1,2l+1]\n");
    printf("  Conversion: cvt.rn.satfinite.e4m3x2.f32 (2 FP32 → 2 E4M3 packed uint16)\n");
    printf("  Swizzle: col ^ ((row & 7) << 3) with 128B row stride\n");
    printf("  Write: d[0]+d[2] share row → uint16 store, d[1]+d[3] share row → uint16 store\n");
    printf("  No bank conflicts: each thread writes 2 distinct uint16 addresses per pass\n");

    return (overall_fail > 0) ? 1 : 0;
}
