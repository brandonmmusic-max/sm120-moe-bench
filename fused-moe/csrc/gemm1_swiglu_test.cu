/**
 * SM120 NVF4 GEMM1 + SwiGLU Validation
 *
 * Pipeline:
 *   1. GEMM1: gate_up[M, 2*N_half] = input[M, K] × W1^T[K, 2*N_half]
 *   2. SwiGLU: output[M, N_half] = up * silu(gate)
 *      where gate = gate_up[:, 0:N_half], up = gate_up[:, N_half:2*N_half]
 *      silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * For Qwen3.5 at TP=4: K=4096, N_half=256, so gate_up width = 512
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

// --- NVF4 GEMM kernel (same as validated collective_mma_test) ---

using ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC    = float;
using ElementD    = float;
using ElementAcc  = float;

using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;

constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

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

using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedNvf4Sm120;

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
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// --- SwiGLU CUDA kernel ---

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// gate_up: [M, 2*N_half] row-major
// output:  [M, N_half] row-major
// SwiGLU: output[m,n] = gate_up[m, n + N_half] * silu(gate_up[m, n])
__global__ void swiglu_kernel(
    const float* __restrict__ gate_up,  // [M, 2*N_half]
    float* __restrict__ output,          // [M, N_half]
    int M, int N_half)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N_half;
    if (idx >= total) return;

    int m = idx / N_half;
    int n = idx % N_half;

    float gate = gate_up[m * (2 * N_half) + n];           // first half
    float up   = gate_up[m * (2 * N_half) + n + N_half];  // second half

    output[idx] = up * silu(gate);
}

// --- Host reference ---

float host_silu(float x) {
    return x / (1.0f + expf(-x));
}

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

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM%d%d)\n\n", prop.name, prop.major, prop.minor);

    // Qwen3.5 MoE dims at TP=4: K=4096, gate_up=512 (gate=256, up=256)
    int M = 128;       // tile-aligned batch
    int K = 4096;      // hidden_size
    int N_half = 256;  // moe_intermediate_size / TP
    int N_full = 2 * N_half;  // gate + up = 512

    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) K = atoi(argv[2]);
    if (argc > 3) N_half = atoi(argv[3]);
    N_full = 2 * N_half;

    printf("GEMM1 + SwiGLU Test\n");
    printf("  Input:    [%d, %d]\n", M, K);
    printf("  W1:       [%d, %d] (gate+up)\n", N_full, K);
    printf("  gate_up:  [%d, %d]\n", M, N_full);
    printf("  SwiGLU:   [%d, %d] -> [%d, %d]\n", M, N_full, M, N_half);
    printf("\n");

    // --- Step 1: Setup GEMM1 ---

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N_full, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N_full, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N_full, 1});

    auto layout_A = make_layout(make_shape(M, K, 1), stride_A);
    auto layout_B = make_layout(make_shape(N_full, K, 1), stride_B);
    auto layout_C = make_layout(make_shape(M, N_full, 1), stride_C);
    auto layout_D = make_layout(make_shape(M, N_full, 1), stride_D);

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N_full, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N_full, K, 1));

    // Allocate
    cutlass::HostTensor<typename ElementA::DataType, cutlass::layout::PackedVectorLayout> block_A;
    cutlass::HostTensor<typename ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFA;
    cutlass::HostTensor<typename ElementB::DataType, cutlass::layout::PackedVectorLayout> block_B;
    cutlass::HostTensor<typename ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> block_SFB;
    cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> block_C;
    cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_gate_up;  // GEMM1 output
    cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> block_ref_gate_up;

    block_A.reset(cutlass::make_Coord(size(layout_A)));
    block_B.reset(cutlass::make_Coord(size(layout_B)));
    block_C.reset(cutlass::make_Coord(size(layout_C)));
    block_gate_up.reset(cutlass::make_Coord(size(layout_D)));
    block_ref_gate_up.reset(cutlass::make_Coord(size(layout_D)));
    block_SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
    block_SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));

    // Initialize
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

    // --- Step 2: Run GEMM1 ---

    printf("Running GEMM1 [%d,%d] x [%d,%d] -> [%d,%d]...\n", M, K, K, N_full, M, N_full);

    typename CollectiveMainloop::Arguments mainloop_args;
    mainloop_args.ptr_A = block_A.device_data();
    mainloop_args.dA = stride_A;
    mainloop_args.ptr_B = block_B.device_data();
    mainloop_args.dB = stride_B;
    mainloop_args.ptr_SFA = block_SFA.device_data();
    mainloop_args.layout_SFA = layout_SFA;
    mainloop_args.ptr_SFB = block_SFB.device_data();
    mainloop_args.layout_SFB = layout_SFB;

    typename Gemm::Arguments gemm_args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N_full, K, 1},
        mainloop_args,
        {{1.0f, 0.0f}, block_C.device_data(), stride_C,
         block_gate_up.device_data(), stride_D}
    };

    Gemm gemm;
    auto status = gemm.can_implement(gemm_args);
    if (status != cutlass::Status::kSuccess) {
        printf("  GEMM1 can_implement FAILED: %d\n", (int)status);
        return 1;
    }

    size_t ws = Gemm::get_workspace_size(gemm_args);
    cutlass::device_memory::allocation<uint8_t> workspace(ws);

    gemm.initialize(gemm_args, workspace.get());
    status = gemm.run();
    cudaDeviceSynchronize();
    printf("  GEMM1: %s\n", status == cutlass::Status::kSuccess ? "OK" : "FAIL");
    if (status != cutlass::Status::kSuccess) return 1;

    // --- Step 3: Run SwiGLU kernel ---

    printf("Running SwiGLU [%d,%d] -> [%d,%d]...\n", M, N_full, M, N_half);

    float* d_swiglu_out;
    cudaMalloc(&d_swiglu_out, M * N_half * sizeof(float));

    int total_elems = M * N_half;
    int threads = 256;
    int blocks = (total_elems + threads - 1) / threads;

    swiglu_kernel<<<blocks, threads>>>(
        block_gate_up.device_data(),
        d_swiglu_out,
        M, N_half);
    cudaDeviceSynchronize();
    printf("  SwiGLU: %s\n",
           cudaGetLastError() == cudaSuccess ? "OK" : cudaGetErrorString(cudaGetLastError()));

    // --- Step 4: Host reference ---

    printf("Computing host reference...\n");

    // GEMM1 reference
    {
        auto tensor_A = make_tensor(make_iterator(block_A.host_data()), layout_A);
        auto tensor_SFA = make_tensor(block_SFA.host_data(), layout_SFA);
        auto tensor_B = make_tensor(make_iterator(block_B.host_data()), layout_B);
        auto tensor_SFB = make_tensor(block_SFB.host_data(), layout_SFB);
        auto tensor_C = make_tensor(make_iterator(block_C.host_data()), layout_C);
        auto tensor_D = make_tensor(make_iterator(block_ref_gate_up.host_data()), layout_D);

        cutlass::reference::host::GettBlockScalingMainloopParams<
            ElementAcc,
            decltype(tensor_A), decltype(tensor_SFA),
            decltype(tensor_B), decltype(tensor_SFB)
        > ml_params{tensor_A, tensor_SFA, tensor_B, tensor_SFB};

        cutlass::reference::host::GettEpilogueParams<
            ElementAcc, ElementAcc, ElementAcc, ElementAcc,
            decltype(tensor_C), decltype(tensor_D)
        > ep_params{1.0f, 0.0f, tensor_C, tensor_D};

        cutlass::reference::host::Gemm3x(ml_params, ep_params);
    }

    // SwiGLU reference on host
    std::vector<float> ref_swiglu(M * N_half);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N_half; n++) {
            float gate = block_ref_gate_up.host_data()[m * N_full + n];
            float up   = block_ref_gate_up.host_data()[m * N_full + n + N_half];
            ref_swiglu[m * N_half + n] = up * host_silu(gate);
        }
    }

    // --- Step 5: Compare ---

    // First verify GEMM1 matches
    block_gate_up.sync_host();
    {
        int count = M * N_full;
        float max_err = 0;
        int match = 0;
        for (int i = 0; i < count; i++) {
            float err = fabsf(block_gate_up.host_data()[i] - block_ref_gate_up.host_data()[i]);
            max_err = fmaxf(max_err, err);
            if (err < 0.01f) match++;
        }
        printf("\nGEMM1 check: max_err=%.6f, match=%d/%d (%.1f%%)\n",
               max_err, match, count, 100.0f * match / count);
    }

    // Compare SwiGLU output
    std::vector<float> h_swiglu_out(M * N_half);
    cudaMemcpy(h_swiglu_out.data(), d_swiglu_out, M * N_half * sizeof(float), cudaMemcpyDeviceToHost);

    {
        int count = M * N_half;
        float max_abs = 0, sum_abs = 0;
        int n_close = 0;
        for (int i = 0; i < count; i++) {
            float err = fabsf(h_swiglu_out[i] - ref_swiglu[i]);
            max_abs = fmaxf(max_abs, err);
            sum_abs += err;
            if (fabsf(ref_swiglu[i]) > 0.01f) {
                if (err / fabsf(ref_swiglu[i]) < 0.02f) n_close++;
            } else {
                if (fabsf(h_swiglu_out[i]) < 0.01f) n_close++;
            }
        }
        float rms = 0;
        for (int i = 0; i < count; i++) rms += ref_swiglu[i] * ref_swiglu[i];
        rms = sqrtf(rms / count);

        printf("\nSwiGLU Results (%d elements):\n", count);
        printf("  First 4: ref=[%.4f, %.4f, %.4f, %.4f]\n",
               ref_swiglu[0], ref_swiglu[1], ref_swiglu[2], ref_swiglu[3]);
        printf("          kern=[%.4f, %.4f, %.4f, %.4f]\n",
               h_swiglu_out[0], h_swiglu_out[1], h_swiglu_out[2], h_swiglu_out[3]);
        printf("  Max abs error:    %.6f\n", max_abs);
        printf("  Avg abs error:    %.6f\n", sum_abs / count);
        printf("  RMS reference:    %.4f\n", rms);
        printf("  Normalized error: %.4f%%\n", rms > 0 ? 100.0f * (sum_abs / count) / rms : 0);
        printf("  Within 2%%:       %d/%d (%.1f%%)\n", n_close, count, 100.0f * n_close / count);

        bool passed = (n_close > count * 0.95f);
        printf("\n  VERDICT: %s\n", passed ? "PASSED" : "FAILED");
    }

    cudaFree(d_swiglu_out);
    return 0;
}
