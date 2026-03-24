/**
 * GEMM2 Standalone Validation: E4M3 activations × FP4 weights on SM120
 *
 * Shape: [M, 256] × [256, 4096] → [M, 4096]
 * MMA instruction: f8f6f4.m16n8k32 (FP8 × FP4)
 *
 * Schedule selection:
 *   - KernelTmaWarpSpecializedNvf4Sm120 requires BOTH operands to be
 *     nv_float4_t<float_e2m1_t> (FP4×FP4). Cannot mix E4M3 × FP4.
 *   - KernelTmaWarpSpecializedMxf8f6f4Sm120 supports mixed-precision:
 *     mx_float8_t<float_e4m3_t> × mx_float4_t<float_e2m1_t> via f8f6f4 MMA.
 *   → Using MxF8F6F4 schedule (the only valid option for E4M3 × FP4).
 *
 * Types:
 *   ElementA = mx_float8_t<float_e4m3_t>  (E4M3 with UE8M0 SF, SfVec=32)
 *   ElementB = mx_float4_t<float_e2m1_t>  (FP4 with UE8M0 SF, SfVec=32)
 *   Note: MX format uses UE8M0 scale factors (bias=127, 0x7F=1.0)
 *         Unlike NVF4 which uses E4M3FN (bias=7, 0x38=1.0)
 *
 * Compile:
 *   nvcc -std=c++17 -O2 \
 *     -gencode=arch=compute_120a,code=sm_120a \
 *     --expt-relaxed-constexpr \
 *     -I<cutlass_include_path> \
 *     -o gemm2_test gemm2_test.cu
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
#include <stdlib.h>
#include <math.h>

using namespace cute;

// ====================================================================
// GEMM2 Configuration: E4M3 × FP4 via MXF8F6F4 on SM120
// ====================================================================

// MX block-scaled types (UE8M0 scale factors, SfVec=32)
using ElementA = cutlass::mx_float8_t<cutlass::float_e4m3_t>;  // E4M3 activation
using ElementB = cutlass::mx_float4_t<cutlass::float_e2m1_t>;  // FP4 weight

using ElementC   = float;
using ElementD   = float;
using ElementAcc = float;

using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;

// MXF8F6F4 alignment requirements
constexpr int AlignmentA = 16;   // 8-bit × 16 = 128 bits
constexpr int AlignmentB = 128;  // 4-bit × 128 = 512 bits (FP4 sub-byte requirement)
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

// SM90-compatible epilogue (no SMEM, float output)
using EpilogueSchedule = cutlass::epilogue::NoSmemWarpSpecialized;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    EpilogueSchedule
>::CollectiveOp;

using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

// MXF8F6F4 schedule — the only valid schedule for E4M3 × FP4 on SM120
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedMxf8f6f4Sm120;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAcc,
    TileShape, ClusterShape,
    StageCount,
    KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Type aliases
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// ====================================================================
// Data initialization (from validated collective_mma_test.cu)
// ====================================================================

template <typename Element, typename Layout>
bool initialize_block(cutlass::TensorView<Element, Layout> view, uint64_t seed) {
    constexpr int bits = cutlass::sizeof_bits<Element>::value;
    double scope_max, scope_min;
    if constexpr (bits <= 6) { scope_max = 2; scope_min = -2; }
    else if constexpr (bits <= 8) {
        if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t> ||
                      cute::is_same_v<Element, cutlass::float_ue4m3_t>) {
            scope_max = 4; scope_min = 1;
        } else { scope_max = 1; scope_min = -1; }
    } else { scope_max = 4; scope_min = -4; }
    cutlass::reference::host::TensorFillRandomUniform(view, seed, scope_max, scope_min, 0);
    return true;
}

template <typename T>
auto make_iterator(T* ptr) { return cute::recast_ptr<T>(ptr); }

// ====================================================================
// Run GEMM2 for a given M value
// ====================================================================

struct GemmResult {
    bool passed;
    float normalized_error;
    float max_abs_error;
    float pct_within_5;
    int nan_count;
    float us_per_call;
};

GemmResult run_gemm2(int M, int K, int N, bool benchmark) {
    GemmResult result = {};

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    auto layout_A = make_layout(make_shape(M, K, 1), stride_A);
    auto layout_B = make_layout(make_shape(N, K, 1), stride_B);
    auto layout_C = make_layout(make_shape(M, N, 1), stride_C);
    auto layout_D = make_layout(make_shape(M, N, 1), stride_D);

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

    // Allocate tensors
    cutlass::HostTensor<typename ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A;
    cutlass::HostTensor<typename ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
    cutlass::HostTensor<typename ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B;
    cutlass::HostTensor<typename ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
    cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
    cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_D;
    cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_ref_D;

    block_A.reset(cutlass::make_Coord(size(layout_A)));
    block_B.reset(cutlass::make_Coord(size(layout_B)));
    block_C.reset(cutlass::make_Coord(size(layout_C)));
    block_D.reset(cutlass::make_Coord(size(layout_D)));
    block_ref_D.reset(cutlass::make_Coord(size(layout_D)));
    block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
    block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));

    // Initialize with random data
    uint64_t seed = 42;
    initialize_block(block_A.host_view(), seed + 1);
    initialize_block(block_B.host_view(), seed + 2);
    initialize_block(block_SFA.host_view(), seed + 3);
    initialize_block(block_SFB.host_view(), seed + 4);
    cutlass::reference::host::TensorFill(block_C.host_view(), ElementC(0));

    block_A.sync_device();
    block_B.sync_device();
    block_C.sync_device();
    block_SFA.sync_device();
    block_SFB.sync_device();

    // Build arguments
    typename CollectiveMainloop::Arguments mainloop_args;
    mainloop_args.ptr_A = block_A.device_data();
    mainloop_args.dA = stride_A;
    mainloop_args.ptr_B = block_B.device_data();
    mainloop_args.dB = stride_B;
    mainloop_args.ptr_SFA = block_SFA.device_data();
    mainloop_args.layout_SFA = layout_SFA;
    mainloop_args.ptr_SFB = block_SFB.device_data();
    mainloop_args.layout_SFB = layout_SFB;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        mainloop_args,
        {{1.0f, 0.0f}, block_C.device_data(), stride_C, block_D.device_data(), stride_D}
    };

    Gemm gemm;
    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        printf("    can_implement: FAIL\n");
        return result;
    }

    size_t ws = Gemm::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(ws);

    status = gemm.initialize(args, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        printf("    initialize: FAIL\n");
        return result;
    }

    status = gemm.run();
    cudaError_t cuda_err = cudaDeviceSynchronize();
    if (status != cutlass::Status::kSuccess || cuda_err != cudaSuccess) {
        printf("    run: FAIL (cuda: %s)\n",
               cuda_err == cudaSuccess ? "OK" : cudaGetErrorString(cuda_err));
        return result;
    }

    // Host reference via Gemm3x
    auto tensor_A = make_tensor(make_iterator(block_A.host_data()), layout_A);
    auto tensor_SFA = make_tensor(block_SFA.host_data(), layout_SFA);
    auto tensor_B = make_tensor(make_iterator(block_B.host_data()), layout_B);
    auto tensor_SFB = make_tensor(block_SFB.host_data(), layout_SFB);
    auto tensor_C = make_tensor(make_iterator(block_C.host_data()), layout_C);
    auto tensor_D = make_tensor(make_iterator(block_ref_D.host_data()), layout_D);

    cutlass::reference::host::GettBlockScalingMainloopParams<
        ElementAcc,
        decltype(tensor_A),
        decltype(tensor_SFA),
        decltype(tensor_B),
        decltype(tensor_SFB)
    > mainloop_params{tensor_A, tensor_SFA, tensor_B, tensor_SFB};

    cutlass::reference::host::GettEpilogueParams<
        ElementAcc, ElementAcc, ElementAcc, ElementAcc,
        decltype(tensor_C),
        decltype(tensor_D)
    > epilogue_params{1.0f, 0.0f, tensor_C, tensor_D};

    cutlass::reference::host::Gemm3x(mainloop_params, epilogue_params);

    // Compare results
    block_D.sync_host();

    int count = M * N;
    float max_abs = 0, sum_abs = 0;
    int n_close = 0;
    int nan_count = 0;

    for (int i = 0; i < count; i++) {
        float kern = block_D.host_data()[i];
        float ref = block_ref_D.host_data()[i];

        if (isnan(kern)) { nan_count++; continue; }

        float err = fabsf(kern - ref);
        max_abs = fmaxf(max_abs, err);
        sum_abs += err;

        if (fabsf(ref) > 0.01f) {
            if (err / fabsf(ref) < 0.05f) n_close++;
        } else {
            if (fabsf(kern) < 0.05f) n_close++;
        }
    }

    float rms = 0;
    for (int i = 0; i < count; i++) rms += block_ref_D.host_data()[i] * block_ref_D.host_data()[i];
    rms = sqrtf(rms / count);

    result.normalized_error = rms > 0 ? 100.0f * (sum_abs / count) / rms : 0;
    result.max_abs_error = max_abs;
    result.pct_within_5 = 100.0f * n_close / count;
    result.nan_count = nan_count;
    result.passed = (nan_count == 0) && (n_close > count * 0.95f);

    printf("    Elements: %d | NaN: %d | Max err: %.6f | Norm err: %.4f%% | Within 5%%: %.1f%%\n",
           count, nan_count, max_abs, result.normalized_error, result.pct_within_5);

    // Show first 4 values
    printf("    ref =[%.4f, %.4f, %.4f, %.4f]\n",
           block_ref_D.host_data()[0], block_ref_D.host_data()[1],
           block_ref_D.host_data()[2], block_ref_D.host_data()[3]);
    printf("    kern=[%.4f, %.4f, %.4f, %.4f]\n",
           block_D.host_data()[0], block_D.host_data()[1],
           block_D.host_data()[2], block_D.host_data()[3]);

    // Benchmark
    if (result.passed && benchmark) {
        for (int i = 0; i < 50; i++) gemm.run();
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int iters = 200;
        cudaEventRecord(start);
        for (int i = 0; i < iters; i++) gemm.run();
        cudaEventRecord(stop);
        cudaDeviceSynchronize();

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        result.us_per_call = ms * 1000.0f / iters;
        printf("    Benchmark: %.1f us/call (%d iters)\n", result.us_per_call, iters);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return result;
}

// ====================================================================
// Main: test multiple M values
// ====================================================================

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM%d%d, %d SMs, %dKB SMEM)\n\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           (int)(prop.sharedMemPerMultiprocessor / 1024));

    // GEMM2 shape: [M, K] × [K, N] = [M, N]
    // K=256 (intermediate after SwiGLU), N=4096 (hidden dim)
    int K = 256, N = 4096;

    printf("=== GEMM2 E4M3 x FP4 Validation (MXF8F6F4 on SM120) ===\n");
    printf("Schedule: KernelTmaWarpSpecializedMxf8f6f4Sm120\n");
    printf("  NvF4 schedule rejected: requires both operands nv_float4_t (FP4xFP4 only)\n");
    printf("  MxF8F6F4 schedule: supports mx_float8_t<e4m3> x mx_float4_t<e2m1>\n");
    printf("  MMA: f8f6f4.m16n8k32 | TileShape: 128x128x128\n");
    printf("  SF: UE8M0 (bias=127, 0x7F=1.0), SfVec=32\n\n");

    // Test M values: decode (1,16) and batch (128)
    int M_values[] = {128, 16, 1};
    int n_tests = 3;
    if (argc > 1) {
        // Custom M from command line
        M_values[0] = atoi(argv[1]);
        n_tests = 1;
    }

    int total_pass = 0, total_fail = 0;

    for (int t = 0; t < n_tests; t++) {
        int M = M_values[t];
        printf("--- M=%d: [%d, %d] x [%d, %d] -> [%d, %d] ---\n", M, M, K, K, N, M, N);
        auto r = run_gemm2(M, K, N, true);
        printf("    VERDICT: %s\n\n", r.passed ? "PASSED" : "FAILED");
        if (r.passed) total_pass++; else total_fail++;
    }

    printf("=== SUMMARY: %d/%d passed ===\n", total_pass, total_pass + total_fail);
    return total_fail > 0 ? 1 : 0;
}
