/**
 * K-tile isolation: sweep byte positions with K=64 (single tile)
 * and K=128 (two tiles) to isolate whether the issue is
 * within-tile or cross-tile.
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
#include <vector>
#include <math.h>

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

using TileShape = Shape<_128, _128, _64>;
using ClusterShape = Shape<_1, _1, _1>;

static constexpr int AlignA = 128 / cutlass::sizeof_bits<ElementA>::value;
static constexpr int AlignB = 128 / cutlass::sizeof_bits<ElementB>::value;
static constexpr int AlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
static constexpr int AlignD = 128 / cutlass::sizeof_bits<ElementD>::value;

using EpilogueSchedule = cutlass::epilogue::NoSmemWarpSpecialized;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc, ElementC, cutlass::layout::RowMajor, AlignC,
    ElementD, cutlass::layout::RowMajor, AlignD, EpilogueSchedule
>::CollectiveOp;

using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedNvf4Sm120;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementPairA, cutlass::layout::RowMajor, AlignA,
    ElementPairB, cutlass::layout::ColumnMajor, AlignB,
    ElementAcc, TileShape, ClusterShape, StageCount, KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

float run_probe(int M, int N, int K, const uint8_t* hA, const uint8_t* hB,
                const uint8_t* hSFA, const uint8_t* hSFB) {
    int A_sz = M * K / 2, B_sz = N * K / 2;
    int sfA_sz = M * K / SFVec, sfB_sz = N * K / SFVec;
    uint8_t *dA, *dB, *dSFA, *dSFB; float *dC, *dD;
    cudaMalloc(&dA, A_sz); cudaMalloc(&dB, B_sz);
    cudaMalloc(&dSFA, sfA_sz); cudaMalloc(&dSFB, sfB_sz);
    cudaMalloc(&dC, M*N*4); cudaMalloc(&dD, M*N*4);
    cudaMemcpy(dA, hA, A_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, B_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(dSFA, hSFA, sfA_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(dSFB, hSFB, sfB_sz, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M*N*4); cudaMemset(dD, 0, M*N*4);

    auto sA = cutlass::make_cute_packed_stride(typename CollectiveMainloop::StrideA{}, {M,K,1});
    auto sB = cutlass::make_cute_packed_stride(typename CollectiveMainloop::StrideB{}, {N,K,1});
    auto sC = cutlass::make_cute_packed_stride(typename CollectiveEpilogue::StrideC{}, {M,N,1});
    auto sD = cutlass::make_cute_packed_stride(typename CollectiveEpilogue::StrideD{}, {M,N,1});

    using Cfg = cutlass::detail::Sm1xxBlockScaledConfig<SFVec>;
    int Ksf = K / SFVec;
    auto lsfa = blocked_product(typename Cfg::SfAtom{},
        make_layout(make_shape(M,K,1), make_stride(Ksf, cute::_1{}, M*Ksf)));
    auto lsfb = blocked_product(typename Cfg::SfAtom{},
        make_layout(make_shape(N,K,1), make_stride(Ksf, cute::_1{}, N*Ksf)));

    typename CollectiveMainloop::Arguments ml;
    ml.ptr_A=(ElementA*)dA; ml.dA=sA; ml.ptr_B=(ElementB*)dB; ml.dB=sB;
    ml.ptr_SFA=(ElementSF*)dSFA; ml.layout_SFA=lsfa;
    ml.ptr_SFB=(ElementSF*)dSFB; ml.layout_SFB=lsfb;

    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
        {M,N,K,1}, ml, {{1.0f,0.0f}, dC, sC, dD, sD}};

    Gemm gemm; float out = -999;
    if (gemm.can_implement(args) == cutlass::Status::kSuccess) {
        size_t ws = Gemm::get_workspace_size(args);
        void* w = nullptr; if (ws>0) cudaMalloc(&w, ws);
        gemm.initialize(args, w);
        if (gemm.run() == cutlass::Status::kSuccess) {
            cudaDeviceSynchronize();
            cudaMemcpy(&out, dD, 4, cudaMemcpyDeviceToHost);
        }
        if (w) cudaFree(w);
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dSFA); cudaFree(dSFB);
    cudaFree(dC); cudaFree(dD);
    return out;
}

int main() {
    printf("K-tile Isolation Probe\n\n");
    const int M = 128, N = 128;

    // Test each K value
    int K_vals[] = {64, 128, 256};
    for (int ki = 0; ki < 3; ki++) {
        int K = K_vals[ki];
        int K_BYTES = K / 2;
        int sfA_sz = M * K / SFVec, sfB_sz = N * K / SFVec;

        std::vector<uint8_t> B(N * K_BYTES, 0x22);  // all 1.0
        std::vector<uint8_t> sfA(sfA_sz, 0x7F), sfB(sfB_sz, 0x7F);

        printf("=== K=%d (%d bytes/row, %d K-tiles) ===\n", K, K_BYTES, K / 64);
        printf("  byte  D[0,0]  tile  ok?\n");

        for (int bp = 0; bp < K_BYTES; bp++) {
            std::vector<uint8_t> A(M * K_BYTES, 0);
            A[bp] = 0x62;  // lo=1.0 hi=4.0 → sum=5.0

            float d = run_probe(M, N, K, A.data(), B.data(), sfA.data(), sfB.data());
            const char* ok = (fabsf(d - 5.0f) < 0.01f) ? "✓" :
                             (fabsf(d) < 0.01f) ? "ZERO" :
                             (fabsf(d - 10.0f) < 0.01f) ? "2x" : "??";
            // Only print every 4th byte for brevity, plus boundaries
            if (bp % 8 == 0 || bp == K_BYTES - 1 || fabsf(d - 5.0f) > 0.01f)
                printf("  %3d   %6.2f  %d     %s\n", bp, d, bp / 32, ok);
        }
        printf("\n");
    }

    return 0;
}
