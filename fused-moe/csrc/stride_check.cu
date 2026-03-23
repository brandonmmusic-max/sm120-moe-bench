/**
 * Quick stride comparison: nv_float4_t vs tuple approach.
 * Also check if we need to pass K/2 to make_cute_packed_stride for nv_float4_t.
 */
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cute/tensor.hpp>
#include <stdio.h>

using namespace cute;

// NVF4 types (example 79b style)
using ElementA_nvf4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB_nvf4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

using TileShape = Shape<_128, _128, _64>;
using ClusterShape = Shape<_1, _1, _1>;

using EpilogueSchedule = cutlass::epilogue::NoSmemWarpSpecialized;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
    float, float, float, cutlass::layout::RowMajor, 4,
    float, cutlass::layout::RowMajor, 4, EpilogueSchedule
>::CollectiveOp;

using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedNvf4Sm120;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementA_nvf4, cutlass::layout::RowMajor, 32,
    ElementB_nvf4, cutlass::layout::ColumnMajor, 32,
    float, TileShape, ClusterShape, StageCount, KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue>;

int main() {
    using StrideA = typename GemmKernel::StrideA;
    using StrideB = typename GemmKernel::StrideB;
    using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    printf("=== NVF4 Stride Analysis ===\n\n");

    printf("StrideA type: "); print(StrideA{}); printf("\n");
    printf("sizeof_bits<ElementA::DataType> = %d\n",
           (int)cutlass::sizeof_bits<typename ElementA_nvf4::DataType>::value);
    printf("sizeof_bits<ElementA::ScaleFactorType> = %d\n",
           (int)cutlass::sizeof_bits<typename ElementA_nvf4::ScaleFactorType>::value);

    // Check InternalElementA
    printf("sizeof_bits<CollectiveMainloop::ElementA> = %d\n",
           (int)cutlass::sizeof_bits<typename CollectiveMainloop::ElementA>::value);

    int M = 128, K = 256;

    // Standard stride with K=256
    auto stride_A_256 = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    printf("\nstride_A (M=%d, K=%d): ", M, K); print(stride_A_256); printf("\n");

    // If we pass K/2
    auto stride_A_128 = cutlass::make_cute_packed_stride(StrideA{}, {M, K/2, 1});
    printf("stride_A (M=%d, K=%d/2=%d): ", M, K, K/2); print(stride_A_128); printf("\n");

    // Layout with K
    auto layout_A_256 = make_layout(make_shape(M, K, 1), stride_A_256);
    auto layout_A_128 = make_layout(make_shape(M, K/2, 1), stride_A_128);
    printf("\nlayout_A(K=%d) size = %d\n", K, (int)size(layout_A_256));
    printf("layout_A(K=%d) size = %d\n", K/2, (int)size(layout_A_128));

    // SF layout
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, 128, K, 1));
    printf("\nlayout_SFA(K=%d): ", K); print(layout_SFA); printf("\n");
    printf("layout_SFA size (filtered): %d\n", (int)size(filter_zeros(layout_SFA)));

    // What the example does: passes {m, n, k, 1} to tile_atom_to_shape_SFA
    // NOT {m, k, 1}! It uses the full 4D problem shape.

    printf("\n=== Key Question ===\n");
    printf("For FP4 packed data with %d elements per row:\n", K);
    printf("  Bytes per row = %d\n", K / 2);
    printf("  stride_A = (%d, 1, 0) in element units\n", K);
    printf("  TMA will advance K-tile by TileK * stride_K = 64 * 1 = 64 elements\n");
    printf("  = 32 bytes (if TMA handles sub-byte correctly)\n");
    printf("  = 64 bytes (if TMA treats stride as byte-based!)\n");

    return 0;
}
