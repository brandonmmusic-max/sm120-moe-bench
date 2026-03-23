/**
 * SF Encoding Probe: Determine what scale factor encoding CUTLASS uses
 * for SM120 NVFP4 (NVF4) block-scaled GEMM.
 *
 * Hypothesis: NVF4 uses E4M3FN scale factors (not UE8M0), since the
 * ElementPairA tuple has Int<16> (sf_vec=16) which selects NVF4 path.
 *
 * Test: All-ones FP4 data, sweep SF byte values, measure output.
 * D[0] = K * 1.0 * 1.0 * SFA * SFB = K * SFA * SFB
 * From the output and known K, we can derive SF_actual = sqrt(D[0]/K)
 */

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/util/packed_stride.hpp>

#include <cute/tensor.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

using namespace cute;

using ElementA = cutlass::float_e2m1_t;
using ElementB = cutlass::float_e2m1_t;
using ElementSF = cutlass::float_ue8m0_t;  // declared type — actual encoding TBD
using ElementAcc = float;
using ElementC = float;
using ElementD = float;

static constexpr int SFVec = 16;

using ElementPairA = cute::tuple<ElementA, ElementSF, cute::Int<SFVec>>;
using ElementPairB = cute::tuple<ElementB, ElementSF, cute::Int<SFVec>>;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using TileShape = Shape<_128, _128, _64>;
using ClusterShape = Shape<_1, _1, _1>;

static constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;
static constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;
static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;

using EpilogueSchedule = cutlass::epilogue::NoSmemWarpSpecialized;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc,
    ElementC, LayoutC, AlignC,
    ElementD, LayoutD, AlignD,
    EpilogueSchedule
>::CollectiveOp;

using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedNvf4Sm120;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassBlockScaledTensorOp,
    ElementPairA, LayoutA, AlignA,
    ElementPairB, LayoutB, AlignB,
    ElementAcc,
    TileShape, ClusterShape,
    StageCount,
    KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

float run_with_sf(uint8_t sf_val, int M, int N, int K) {
    uint8_t *dA, *dB, *dSFA, *dSFB;
    float *dC, *dD;

    cudaMalloc(&dA, M * K / 2);
    cudaMalloc(&dB, N * K / 2);
    cudaMalloc(&dSFA, (M * K) / SFVec);
    cudaMalloc(&dSFB, (N * K) / SFVec);
    cudaMalloc(&dC, M * N * sizeof(float));
    cudaMalloc(&dD, M * N * sizeof(float));

    // All 1.0 in E2M1: nibble 0x2 = 1.0, byte 0x22 = (1.0, 1.0)
    cudaMemset(dA, 0x22, M * K / 2);
    cudaMemset(dB, 0x22, N * K / 2);
    cudaMemset(dSFA, sf_val, (M * K) / SFVec);
    cudaMemset(dSFB, sf_val, (N * K) / SFVec);
    cudaMemset(dC, 0, M * N * sizeof(float));
    cudaMemset(dD, 0, M * N * sizeof(float));

    auto stride_A = cutlass::make_cute_packed_stride(typename CollectiveMainloop::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename CollectiveMainloop::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename CollectiveEpilogue::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename CollectiveEpilogue::StrideD{}, {M, N, 1});

    using BlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVec>;
    int K_sf = K / SFVec;

    auto layout_sfa = blocked_product(
        typename BlkScaledConfig::SfAtom{},
        make_layout(make_shape(M, K, 1),
                    make_stride(K_sf, cute::_1{}, M * K_sf)));
    auto layout_sfb = blocked_product(
        typename BlkScaledConfig::SfAtom{},
        make_layout(make_shape(N, K, 1),
                    make_stride(K_sf, cute::_1{}, N * K_sf)));

    typename CollectiveMainloop::Arguments mainloop_args;
    mainloop_args.ptr_A = (ElementA*)dA;
    mainloop_args.dA = stride_A;
    mainloop_args.ptr_B = (ElementB*)dB;
    mainloop_args.dB = stride_B;
    mainloop_args.ptr_SFA = (ElementSF*)dSFA;
    mainloop_args.layout_SFA = layout_sfa;
    mainloop_args.ptr_SFB = (ElementSF*)dSFB;
    mainloop_args.layout_SFB = layout_sfb;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        mainloop_args,
        {{1.0f, 0.0f}, dC, stride_C, dD, stride_D}
    };

    Gemm gemm;
    auto status = gemm.can_implement(args);
    float result = -1.0f;

    if (status == cutlass::Status::kSuccess) {
        size_t ws = Gemm::get_workspace_size(args);
        void* workspace = nullptr;
        if (ws > 0) cudaMalloc(&workspace, ws);

        gemm.initialize(args, workspace);
        status = gemm.run();
        cudaDeviceSynchronize();

        if (status == cutlass::Status::kSuccess) {
            cudaMemcpy(&result, dD, sizeof(float), cudaMemcpyDeviceToHost);
        }

        if (workspace) cudaFree(workspace);
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dSFA); cudaFree(dSFB);
    cudaFree(dC); cudaFree(dD);
    return result;
}

// E4M3FN decode (FP8 E4M3, no NaN/Inf)
float decode_e4m3fn(uint8_t bits) {
    int sign = (bits >> 7) & 1;
    int exp = (bits >> 3) & 0xF;
    int mant = bits & 0x7;

    float val;
    if (exp == 0) {
        // Subnormal: 2^(-6) * (mant/8)
        val = ldexpf((float)mant, -9);  // mant * 2^-9 = mant/8 * 2^-6
    } else {
        // Normal: 2^(exp-7) * (1 + mant/8)
        val = ldexpf(1.0f + (float)mant / 8.0f, exp - 7);
    }
    return sign ? -val : val;
}

// UE8M0 decode
float decode_ue8m0(uint8_t val, int bias) {
    return powf(2.0f, (float)((int)val - bias));
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM%d%d)\n\n", prop.name, prop.major, prop.minor);

    // Print type info
    printf("=== Type Info ===\n");
    printf("ElementSF declared as: float_ue8m0_t\n");
    printf("SFVec = %d (selects NVF4 path if 16, MXF4 if 32)\n", SFVec);
    printf("sizeof(ElementSF) = %zu\n", sizeof(ElementSF));
    printf("sizeof(SharedStorage) = %zu bytes\n\n", sizeof(typename GemmKernel::SharedStorage));

    const int M = 128, N = 128, K = 4096;
    printf("GEMM: [%d,%d] x [%d,%d], all FP4=1.0\n", M, K, K, N);
    printf("Expected D[0] = K * SFA * SFB = %d * SFA * SFB\n\n", K);

    // Sweep SF byte values
    printf("=== SF Byte Sweep ===\n");
    printf("SF_byte  Output      SF_actual   E4M3FN_decode  UE8M0_b127  UE8M0_b128\n");
    printf("-------  ----------  ----------  -------------  ----------  ----------\n");

    uint8_t test_vals[] = {
        0x00,  // E4M3: 0.0,       UE8M0_127: 2^-127
        0x08,  // E4M3: -0.0,      UE8M0: 2^-119
        0x10,  // E4M3: 0.015625,  UE8M0: 2^-111
        0x20,  // E4M3: 0.0625,    UE8M0: 2^-95
        0x30,  // E4M3: 0.25,      UE8M0: 2^-79
        0x38,  // E4M3: 0.4375 (=7/16), UE8M0: 2^-71
        0x3C,  // E4M3: 0.625,     UE8M0: 2^-67
        0x3E,  // E4M3: 0.75,      UE8M0: 2^-65
        0x3F,  // E4M3: 0.875,     UE8M0: 2^-64
        0x40,  // E4M3: 1.0,       UE8M0: 2^-63
        0x41,  // E4M3: 1.125,     UE8M0: 2^-62
        0x42,  // E4M3: 1.25,      UE8M0: 2^-62
        0x44,  // E4M3: 1.5,       UE8M0: 2^-59
        0x48,  // E4M3: 2.0,       UE8M0: 2^-55
        0x50,  // E4M3: 4.0,       UE8M0: 2^-47
        0x58,  // E4M3: 8.0,       UE8M0: 2^-39
        0x60,  // E4M3: 16.0,      UE8M0: 2^-31
        0x70,  // E4M3: 64.0,      UE8M0: 2^-15
        0x76,  // E4M3: 384.0,     UE8M0: 2^-9
        0x7E,  // E4M3: 448.0,     UE8M0: 2^-1 = 0.5
        0x7F,  // E4M3: 448.0 (max normal, same as 0x7E in E4M3FN), UE8M0: 1.0
        0x80,  // E4M3: -0.0 (sign bit!), UE8M0: 2.0
    };
    int n_tests = sizeof(test_vals) / sizeof(test_vals[0]);

    for (int i = 0; i < n_tests; i++) {
        uint8_t sv = test_vals[i];
        float out = run_with_sf(sv, M, N, K);

        float sf_actual = -1;
        if (out > 0) sf_actual = sqrtf(out / K);

        float e4m3 = decode_e4m3fn(sv);
        float ue8m0_127 = decode_ue8m0(sv, 127);
        float ue8m0_128 = decode_ue8m0(sv, 128);

        printf("0x%02X     %10.2f  %10.6f  %13.6f  %10.6f  %10.6f",
               sv, out, sf_actual, e4m3, ue8m0_127, ue8m0_128);

        // Mark if SF_actual matches any encoding
        if (sf_actual > 0) {
            if (fabsf(sf_actual - e4m3) / fmaxf(fabsf(e4m3), 1e-10f) < 0.01f)
                printf("  ← E4M3FN MATCH!");
            if (fabsf(sf_actual - ue8m0_127) / fmaxf(fabsf(ue8m0_127), 1e-10f) < 0.01f)
                printf("  ← UE8M0(b=127) MATCH!");
            if (fabsf(sf_actual - ue8m0_128) / fmaxf(fabsf(ue8m0_128), 1e-10f) < 0.01f)
                printf("  ← UE8M0(b=128) MATCH!");
        }
        printf("\n");
    }

    // Also check: what's 0x80 in the original test (M=16, N=64, K=4096)?
    printf("\n=== Original dimensions (M=16, N=64, K=4096) ===\n");
    float out_small = run_with_sf(0x80, 128, 128, 4096);  // tile-aligned
    printf("0x80 tile-aligned: %.2f (expect same as above)\n", out_small);

    return 0;
}
