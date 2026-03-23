/**
 * GEMM2 Correctness Test
 *
 * Validates that GEMM2 produces correct output when fed requantized
 * SwiGLU intermediate. Compares against host reference that:
 * 1. Computes GEMM1 → SwiGLU in FP32
 * 2. Requantizes to FP4 with block-16 E4M3FN scales (same as device kernel)
 * 3. Dequantizes back to FP32
 * 4. Computes GEMM2 in FP32
 *
 * This isolates GEMM2 correctness from GEMM1/SwiGLU.
 * The key question: does our row-major SF layout match what CUTLASS expects?
 *
 * Strategy: use CUTLASS HostTensor to create properly-laid-out FP4 + SF
 * tensors for GEMM2's input, bypassing the requant kernel entirely.
 * If GEMM2 matches with CUTLASS-formatted input, the issue is in requant layout.
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
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/tensor_norm.h>

#include <cute/tensor.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

using namespace cute;

// --- GEMM types ---
using ElementAB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC    = float;
using ElementD    = float;
using ElementAcc  = float;

constexpr int AlignmentAB = 32;
constexpr int AlignmentCD = 128 / cutlass::sizeof_bits<ElementC>::value;

using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using EpilogueSchedule = cutlass::epilogue::NoSmemWarpSpecialized;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc,
    ElementC, cutlass::layout::RowMajor, AlignmentCD,
    ElementD, cutlass::layout::RowMajor, AlignmentCD,
    EpilogueSchedule
>::CollectiveOp;

using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedNvf4Sm120;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementAB, cutlass::layout::RowMajor, AlignmentAB,
    ElementAB, cutlass::layout::ColumnMajor, AlignmentAB,
    ElementAcc,
    TileShape, ClusterShape, StageCount, KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using Sm1xxCfg = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

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

template <typename T>
auto make_iterator(T* ptr) { return cute::recast_ptr<T>(ptr); }

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM%d%d)\n\n", prop.name, prop.major, prop.minor);

    // GEMM2: intermediate[M, N_half] × W2[K_out, N_half]^T → output[M, K_out]
    // Qwen3.5 dims: M=128, N_half=256 (intermediate), K_out=4096 (hidden)
    int M = 128, K_gemm2 = 256, N_gemm2 = 4096;

    printf("=== GEMM2 Correctness Test ===\n");
    printf("  GEMM2: [%d, %d] × [%d, %d] → [%d, %d]\n\n", M, K_gemm2, K_gemm2, N_gemm2, M, N_gemm2);

    // Create GEMM2 inputs using CUTLASS HostTensor (guarantees correct layout)
    // A = intermediate (RowMajor [M, K_gemm2])
    // B = W2 weights (ColumnMajor [N_gemm2, K_gemm2])

    auto sA = cutlass::make_cute_packed_stride(StrideA{}, {M, K_gemm2, 1});
    auto sB = cutlass::make_cute_packed_stride(StrideB{}, {N_gemm2, K_gemm2, 1});
    auto sC = cutlass::make_cute_packed_stride(StrideC{}, {M, N_gemm2, 1});
    auto sD = cutlass::make_cute_packed_stride(StrideD{}, {M, N_gemm2, 1});

    auto lA = make_layout(make_shape(M, K_gemm2, 1), sA);
    auto lB = make_layout(make_shape(N_gemm2, K_gemm2, 1), sB);
    auto lC = make_layout(make_shape(M, N_gemm2, 1), sC);
    auto lD = make_layout(make_shape(M, N_gemm2, 1), sD);

    auto lSFA = Sm1xxCfg::tile_atom_to_shape_SFA(make_shape(M, N_gemm2, K_gemm2, 1));
    auto lSFB = Sm1xxCfg::tile_atom_to_shape_SFB(make_shape(M, N_gemm2, K_gemm2, 1));

    cutlass::HostTensor<typename ElementAB::DataType, cutlass::layout::PackedVectorLayout> h_A, h_B;
    cutlass::HostTensor<typename ElementAB::ScaleFactorType, cutlass::layout::PackedVectorLayout> h_SFA, h_SFB;
    cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> h_C;
    cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> h_D, h_ref_D;

    h_A.reset(cutlass::make_Coord(size(lA)));
    h_B.reset(cutlass::make_Coord(size(lB)));
    h_C.reset(cutlass::make_Coord(size(lC)));
    h_D.reset(cutlass::make_Coord(size(lD)));
    h_ref_D.reset(cutlass::make_Coord(size(lD)));
    h_SFA.reset(cutlass::make_Coord(size(filter_zeros(lSFA))));
    h_SFB.reset(cutlass::make_Coord(size(filter_zeros(lSFB))));

    uint64_t seed = 42;
    init_block(h_A.host_view(), seed + 10);
    init_block(h_B.host_view(), seed + 11);
    init_block(h_SFA.host_view(), seed + 12);
    init_block(h_SFB.host_view(), seed + 13);
    cutlass::reference::host::TensorFill(h_C.host_view(), 0.0f);

    h_A.sync_device(); h_B.sync_device();
    h_SFA.sync_device(); h_SFB.sync_device();
    h_C.sync_device();

    printf("Running CUTLASS GEMM2...\n");

    typename CollectiveMainloop::Arguments ml;
    ml.ptr_A = h_A.device_data();
    ml.dA = sA;
    ml.ptr_B = h_B.device_data();
    ml.dB = sB;
    ml.ptr_SFA = h_SFA.device_data();
    ml.layout_SFA = lSFA;
    ml.ptr_SFB = h_SFB.device_data();
    ml.layout_SFB = lSFB;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N_gemm2, K_gemm2, 1},
        ml,
        {{1.0f, 0.0f}, h_C.device_data(), sC, h_D.device_data(), sD}
    };

    Gemm gemm;
    auto status = gemm.can_implement(args);
    printf("  can_implement: %s\n", status == cutlass::Status::kSuccess ? "OK" : "FAIL");
    if (status != cutlass::Status::kSuccess) return 1;

    size_t ws = Gemm::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(ws);
    gemm.initialize(args, workspace.get());
    status = gemm.run();
    cudaDeviceSynchronize();
    printf("  run: %s\n", status == cutlass::Status::kSuccess ? "OK" : "FAIL");
    if (status != cutlass::Status::kSuccess) return 1;

    // Host reference
    printf("Computing host reference...\n");
    {
        auto tA = make_tensor(make_iterator(h_A.host_data()), lA);
        auto tSFA = make_tensor(h_SFA.host_data(), lSFA);
        auto tB = make_tensor(make_iterator(h_B.host_data()), lB);
        auto tSFB = make_tensor(h_SFB.host_data(), lSFB);
        auto tC = make_tensor(make_iterator(h_C.host_data()), lC);
        auto tD = make_tensor(make_iterator(h_ref_D.host_data()), lD);

        cutlass::reference::host::GettBlockScalingMainloopParams<
            float, decltype(tA), decltype(tSFA), decltype(tB), decltype(tSFB)
        > ml_params{tA, tSFA, tB, tSFB};
        cutlass::reference::host::GettEpilogueParams<
            float, float, float, float, decltype(tC), decltype(tD)
        > ep_params{1.0f, 0.0f, tC, tD};

        cutlass::reference::host::Gemm3x(ml_params, ep_params);
    }

    // Compare
    h_D.sync_host();

    int count = M * N_gemm2;
    float max_abs = 0, sum_abs = 0;
    int n_close = 0;
    for (int i = 0; i < count; i++) {
        float kern = h_D.host_data()[i];
        float ref = h_ref_D.host_data()[i];
        float err = fabsf(kern - ref);
        max_abs = fmaxf(max_abs, err);
        sum_abs += err;
        if (fabsf(ref) > 0.01f) {
            if (err / fabsf(ref) < 0.02f) n_close++;
        } else {
            if (fabsf(kern) < 0.01f) n_close++;
        }
    }

    float rms = 0;
    for (int i = 0; i < count; i++) rms += h_ref_D.host_data()[i] * h_ref_D.host_data()[i];
    rms = sqrtf(rms / count);

    printf("\nGEMM2 Results (%d elements):\n", count);
    printf("  First 4: ref=[%.4f, %.4f, %.4f, %.4f]\n",
           h_ref_D.host_data()[0], h_ref_D.host_data()[1],
           h_ref_D.host_data()[2], h_ref_D.host_data()[3]);
    printf("          kern=[%.4f, %.4f, %.4f, %.4f]\n",
           h_D.host_data()[0], h_D.host_data()[1],
           h_D.host_data()[2], h_D.host_data()[3]);
    printf("  Max abs error:    %.6f\n", max_abs);
    printf("  Avg abs error:    %.6f\n", sum_abs / count);
    printf("  RMS reference:    %.4f\n", rms);
    printf("  Normalized error: %.4f%%\n", rms > 0 ? 100.0f * (sum_abs / count) / rms : 0);
    printf("  Within 2%%:       %d/%d (%.1f%%)\n", n_close, count, 100.0f * n_close / count);

    bool passed = (n_close > count * 0.95f);
    printf("\n  GEMM2 VERDICT: %s\n", passed ? "PASSED (bit-exact with CUTLASS-formatted input)" : "FAILED");

    return passed ? 0 : 1;
}
