/**
 * Dump the actual stride/layout types CUTLASS uses for SM120 NVFP4
 * to understand what GMEM layout TMA expects.
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

using ElementA = cutlass::float_e2m1_t;
using ElementB = cutlass::float_e2m1_t;
using ElementSF = cutlass::float_ue8m0_t;
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

int main() {
    printf("=== CUTLASS SM120 NVFP4 Type/Layout Dump ===\n\n");

    // Print stride types
    printf("StrideA type: ");
    print(typename CollectiveMainloop::StrideA{}); printf("\n");
    printf("StrideB type: ");
    print(typename CollectiveMainloop::StrideB{}); printf("\n");

    // Print with actual dimensions
    const int M = 128, N = 128, K = 256;
    auto stride_A = cutlass::make_cute_packed_stride(typename CollectiveMainloop::StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(typename CollectiveMainloop::StrideB{}, {N, K, 1});

    printf("\nstride_A (M=%d, K=%d): ", M, K);
    print(stride_A); printf("\n");
    printf("stride_B (N=%d, K=%d): ", N, K);
    print(stride_B); printf("\n");

    // Print mainloop internal types
    printf("\nCollectiveMainloop::TiledMma: ");
    print(typename CollectiveMainloop::TiledMma{}); printf("\n");

    // Print SmemLayoutA
    printf("\nSmemLayoutAtomA: ");
    print(typename CollectiveMainloop::SmemLayoutAtomA{}); printf("\n");

    printf("SmemLayoutAtomB: ");
    print(typename CollectiveMainloop::SmemLayoutAtomB{}); printf("\n");

    // Print SF layout info
    using BlkScaledConfig = cutlass::detail::Sm1xxBlockScaledConfig<SFVec>;
    printf("\nSfAtom: ");
    print(typename BlkScaledConfig::SfAtom{}); printf("\n");

    int K_sf = K / SFVec;
    auto layout_sfa = blocked_product(
        typename BlkScaledConfig::SfAtom{},
        make_layout(make_shape(M, K, 1),
                    make_stride(K_sf, cute::_1{}, M * K_sf)));
    printf("layout_sfa: ");
    print(layout_sfa); printf("\n");
    printf("layout_sfa size: %d\n", (int)size(layout_sfa));

    // Check: does TMA expect interleaved FP4 layout?
    // Print the GmemTiledCopy type if accessible
    printf("\nInternalElementA (after builder transform): sizeof=%zu bits\n",
           (size_t)cutlass::sizeof_bits<typename CollectiveMainloop::ElementA>::value);
    printf("InternalElementB: sizeof=%zu bits\n",
           (size_t)cutlass::sizeof_bits<typename CollectiveMainloop::ElementB>::value);

    // Check if there's a GMEM layout transform
    printf("\nsizeof(SharedStorage) = %zu bytes\n", sizeof(typename GemmKernel::SharedStorage));

    printf("\n=== Check CUTLASS FP4 example for reference layout ===\n");
    printf("For NVFP4 (E2M1 block-scaled), CUTLASS may expect:\n");
    printf("  - Standard row-major byte layout (2 elements per byte)\n");
    printf("  - Or an interleaved/reordered layout for TMA efficiency\n");
    printf("The all-ones test works because uniform data is invariant to reordering.\n");
    printf("Need to check CUTLASS examples or tests for FP4 data preparation.\n");

    return 0;
}
