/**
 * SM120 FP4 GEMM using CUTLASS CollectiveBuilder
 * Minimal: just compile + verify the CollectiveBuilder produces a CollectiveOp
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

using namespace cute;

// NVFP4 types
using ElementA = cutlass::float_e2m1_t;
using ElementB = cutlass::float_e2m1_t;
// NVFP4 scale factors: ue8m0 with sf_vec=16 for SM120
using ElementSF = cutlass::float_ue8m0_t;
using ElementAcc = float;
using ElementC = float;
using ElementD = float;

static constexpr int SFVec = 16;  // NVFP4 block-16

// 3-element flat tuple: (data_type, sf_type, sf_vec_size)
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

// Epilogue: use SM90 NoSmem style (simpler, works for identity epilogue)
using EpilogueSchedule = cutlass::epilogue::NoSmemWarpSpecialized;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,  // SM90 epilogue builder works for SM120
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

// SM120 NVFP4 (E2M1×E2M1) kernel schedule
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

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM%d%d)\n\n", prop.name, prop.major, prop.minor);

    printf("CUTLASS CollectiveBuilder for SM120 NVFP4 GEMM compiled!\n");
    printf("TileShape: 128x128x64\n");
    printf("KernelSchedule: CooperativeBlockScaledSm120<3>\n");
    printf("sizeof(SharedStorage) = %zu bytes\n\n", sizeof(typename GemmKernel::SharedStorage));

    // Try running a small GEMM
    const int M = 16, N = 64, K = 4096;

    // Allocate
    cutlass::float_e2m1_t *dA, *dB;
    cutlass::float_ue8m0_t *dSFA, *dSFB;
    float *dC, *dD;

    cudaMalloc(&dA, M * K / 2);  // FP4 packed
    cudaMalloc(&dB, N * K / 2);
    cudaMalloc(&dSFA, (M * K) / SFVec);
    cudaMalloc(&dSFB, (N * K) / SFVec);
    cudaMalloc(&dC, M * N * sizeof(float));
    cudaMalloc(&dD, M * N * sizeof(float));

    // Zero init
    cudaMemset(dA, 0x22, M * K / 2);  // all 1.0 in FP4
    cudaMemset(dB, 0x22, N * K / 2);
    cudaMemset(dSFA, 0x80, (M * K) / SFVec);  // SF = 1.0
    cudaMemset(dSFB, 0x80, (N * K) / SFVec);
    cudaMemset(dC, 0, M * N * sizeof(float));
    cudaMemset(dD, 0, M * N * sizeof(float));

    printf("Running GEMM [%d, %d] x [%d, %d]...\n", M, K, K, N);

    // Use the StrideA type directly (data stride only, SF stride auto-derived)
    auto stride_A = cutlass::make_cute_packed_stride(typename CollectiveMainloop::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename CollectiveMainloop::StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(typename CollectiveEpilogue::StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(typename CollectiveEpilogue::StrideD{}, {M, N, 1});

    // Construct SF layouts using the block-scaled layout config
    // LayoutSFA = blocked_product(SfAtom, make_layout(shape(M,K,L), stride(M_stride,1,L_stride)))
    using BlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVec>;
    int K_sf = K / SFVec;  // number of SF blocks along K
    int M_sf_stride = K_sf;  // SF elements per M-row
    int L_sf_stride = M * K_sf;  // batch stride

    auto layout_sfa = blocked_product(
        typename BlkScaledConfig::SfAtom{},
        make_layout(make_shape(M, K, 1),
                    make_stride(M_sf_stride, cute::_1{}, L_sf_stride)));
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
    if (status != cutlass::Status::kSuccess) {
        printf("can_implement failed: %d\n", (int)status);
    } else {
        printf("can_implement: OK\n");

        size_t workspace_size = Gemm::get_workspace_size(args);
        printf("workspace: %zu bytes\n", workspace_size);

        void* workspace = nullptr;
        if (workspace_size > 0) cudaMalloc(&workspace, workspace_size);

        status = gemm.initialize(args, workspace);
        printf("initialize: %s\n", status == cutlass::Status::kSuccess ? "OK" : "FAIL");

        if (status == cutlass::Status::kSuccess) {
            status = gemm.run();
            cudaDeviceSynchronize();
            printf("run: %s\n", status == cutlass::Status::kSuccess ? "OK" : "FAIL");

            // Read output
            float h_out[16];
            cudaMemcpy(h_out, dD, 16 * sizeof(float), cudaMemcpyDeviceToHost);
            printf("\nOutput[0:8]: ");
            for (int i = 0; i < 8; i++) printf("%.2f ", h_out[i]);
            printf("\n");
            printf("Expected (all 1s, K=4096, SF=1): %.2f\n", 4096.0f);
        }

        if (workspace) cudaFree(workspace);
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dSFA); cudaFree(dSFB);
    cudaFree(dC); cudaFree(dD);
    return 0;
}
