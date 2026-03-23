/**
 * Stride Fix Probe: Test if passing K/2 to CUTLASS fixes the stride issue.
 *
 * Hypothesis: CUTLASS TMA descriptor uses stride in byte units for the
 * row dimension, but make_cute_packed_stride creates it in element units.
 * For FP4 (0.5 bytes/element), the row stride is 2x too large.
 *
 * Fix: pass K_actual/2 as K dimension, so stride = K/2 bytes = correct.
 * The data then has K/2 "elements" per row, each "element" being 1 byte
 * (2 FP4 values). TMA loads the right bytes.
 *
 * But wait — this changes the GEMM computation: D = A * B where K is halved.
 * The MMA still processes 64 elements per tile, so K=32 would mean 0.5 tiles.
 * That doesn't work.
 *
 * Alternative: maybe we need to allocate M*K bytes (not M*K/2) and lay out
 * the data with stride K bytes per row, with FP4 pairs spread across K bytes.
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

static const float E2M1_TABLE[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
float decode_e2m1(uint8_t nibble) {
    int sign = (nibble >> 3) & 1;
    return (sign ? -1.0f : 1.0f) * E2M1_TABLE[nibble & 0x7];
}

struct TestResult {
    bool ok;
    std::vector<float> D;
};

TestResult run_gemm(int M, int N, int K_cutlass,
                    const void* hA, int A_bytes,
                    const void* hSFA, int sfA_bytes,
                    const void* hB, int B_bytes,
                    const void* hSFB, int sfB_bytes) {
    uint8_t *dA, *dB, *dSFA, *dSFB; float *dC, *dD;
    cudaMalloc(&dA, A_bytes); cudaMalloc(&dB, B_bytes);
    cudaMalloc(&dSFA, sfA_bytes); cudaMalloc(&dSFB, sfB_bytes);
    cudaMalloc(&dC, M*N*4); cudaMalloc(&dD, M*N*4);
    cudaMemcpy(dA, hA, A_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, B_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dSFA, hSFA, sfA_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dSFB, hSFB, sfB_bytes, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M*N*4); cudaMemset(dD, 0, M*N*4);

    auto sA = cutlass::make_cute_packed_stride(typename CollectiveMainloop::StrideA{}, {M,K_cutlass,1});
    auto sB = cutlass::make_cute_packed_stride(typename CollectiveMainloop::StrideB{}, {N,K_cutlass,1});
    auto sC = cutlass::make_cute_packed_stride(typename CollectiveEpilogue::StrideC{}, {M,N,1});
    auto sD = cutlass::make_cute_packed_stride(typename CollectiveEpilogue::StrideD{}, {M,N,1});

    using Cfg = cutlass::detail::Sm1xxBlockScaledConfig<SFVec>;
    int Ksf = K_cutlass / SFVec;
    auto lsfa = blocked_product(typename Cfg::SfAtom{},
        make_layout(make_shape(M,K_cutlass,1), make_stride(Ksf,cute::_1{},M*Ksf)));
    auto lsfb = blocked_product(typename Cfg::SfAtom{},
        make_layout(make_shape(N,K_cutlass,1), make_stride(Ksf,cute::_1{},N*Ksf)));

    typename CollectiveMainloop::Arguments ml;
    ml.ptr_A=(ElementA*)dA; ml.dA=sA; ml.ptr_B=(ElementB*)dB; ml.dB=sB;
    ml.ptr_SFA=(ElementSF*)dSFA; ml.layout_SFA=lsfa;
    ml.ptr_SFB=(ElementSF*)dSFB; ml.layout_SFB=lsfb;

    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,
        {M,N,K_cutlass,1}, ml, {{1.0f,0.0f},dC,sC,dD,sD}};

    TestResult result{false, {}};
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
        printf("  can_implement FAILED for K_cutlass=%d\n", K_cutlass);
        goto out;
    }
    {
        size_t ws = Gemm::get_workspace_size(args);
        void* w = nullptr; if (ws>0) cudaMalloc(&w, ws);
        gemm.initialize(args, w);
        if (gemm.run() == cutlass::Status::kSuccess) {
            cudaDeviceSynchronize();
            result.D.resize(M*N);
            cudaMemcpy(result.D.data(), dD, M*N*4, cudaMemcpyDeviceToHost);
            result.ok = true;
        }
        if (w) cudaFree(w);
    }
out:
    cudaFree(dA); cudaFree(dB); cudaFree(dSFA); cudaFree(dSFB);
    cudaFree(dC); cudaFree(dD);
    return result;
}

int main() {
    printf("Stride Fix Probe\n\n");
    const int M = 128, N = 128;

    // Actual FP4 data: 256 elements = 128 bytes per row
    const int K_actual_elems = 256;
    const int K_row_bytes = K_actual_elems / 2;  // 128

    // =============================================
    // Approach 1: Pass K_cutlass = K_actual (256 elements)
    // Buffer size = M * K/2 = 16384 bytes (standard FP4 packing)
    // This is what we've been doing — has the stride bug
    // =============================================
    printf("=== Approach 1: K_cutlass = %d (standard) ===\n", K_actual_elems);
    {
        int A_bytes = M * K_row_bytes;
        int sfA_bytes = M * K_actual_elems / SFVec;
        std::vector<uint8_t> A(A_bytes, 0x22);  // all 1.0
        std::vector<uint8_t> B(N * K_row_bytes, 0x22);
        std::vector<uint8_t> sfA(sfA_bytes, 0x7F);
        std::vector<uint8_t> sfB(N * K_actual_elems / SFVec, 0x7F);

        auto r = run_gemm(M, N, K_actual_elems,
                          A.data(), A_bytes, sfA.data(), sfA_bytes,
                          B.data(), N*K_row_bytes, sfB.data(), N*K_actual_elems/SFVec);
        if (r.ok) printf("  D[0,0] = %.2f (expect %d)\n", r.D[0], K_actual_elems);
    }

    // =============================================
    // Approach 2: Pass K_cutlass = K_actual / 2 (128 "byte-elements")
    // Buffer size = M * K/2 = 16384 bytes (same data)
    // Stride = 128 bytes per row (correct for TMA)
    // But K=128 means TMA K=64 tile covers 64 bytes = full row? No.
    // K=128 with TileK=64: 2 iterations of 64-element tiles
    // Each "element" is actually 1 byte = 2 FP4 values
    // =============================================
    printf("\n=== Approach 2: K_cutlass = %d (halved) ===\n", K_actual_elems / 2);
    {
        int K_half = K_actual_elems / 2;  // 128
        int A_bytes = M * K_row_bytes;  // same buffer
        int sfA_bytes = M * K_half / SFVec;  // 128 * 128 / 16 = 1024
        std::vector<uint8_t> A(A_bytes, 0x22);
        std::vector<uint8_t> B(N * K_row_bytes, 0x22);
        std::vector<uint8_t> sfA(sfA_bytes, 0x7F);
        std::vector<uint8_t> sfB(N * K_half / SFVec, 0x7F);

        auto r = run_gemm(M, N, K_half,
                          A.data(), A_bytes, sfA.data(), sfA_bytes,
                          B.data(), N*K_row_bytes, sfB.data(), N*K_half/SFVec);
        if (r.ok) printf("  D[0,0] = %.2f (expect ??? — if byte-stride, expect %d)\n",
                         r.D[0], K_actual_elems);
    }

    // =============================================
    // Approach 3: Pass K_cutlass = K_actual (256), but allocate M*K bytes
    // and lay out data with K-byte row stride
    // =============================================
    printf("\n=== Approach 3: K_cutlass = %d, buffer = M*K bytes ===\n", K_actual_elems);
    {
        int A_bytes = M * K_actual_elems;  // 32768 — doubled
        int sfA_bytes = M * K_actual_elems / SFVec;
        std::vector<uint8_t> A(A_bytes, 0);
        std::vector<uint8_t> B(N * K_actual_elems, 0);

        // Layout data with K_actual_elems byte stride per row
        // Only first K/2 bytes of each row are actual FP4 data
        for (int m = 0; m < M; m++) {
            for (int i = 0; i < K_row_bytes; i++) {
                A[m * K_actual_elems + i] = 0x22;
            }
        }
        for (int n = 0; n < N; n++) {
            for (int i = 0; i < K_row_bytes; i++) {
                B[n * K_actual_elems + i] = 0x22;
            }
        }

        std::vector<uint8_t> sfA(sfA_bytes, 0x7F);
        std::vector<uint8_t> sfB(N * K_actual_elems / SFVec, 0x7F);

        auto r = run_gemm(M, N, K_actual_elems,
                          A.data(), A_bytes, sfA.data(), sfA_bytes,
                          B.data(), N*K_actual_elems, sfB.data(), N*K_actual_elems/SFVec);
        if (r.ok) printf("  D[0,0] = %.2f\n", r.D[0]);
    }

    // =============================================
    // Approach 4: Approach 3 but with byte-sweep probe
    // =============================================
    printf("\n=== Approach 4: Doubled buffer, byte sweep ===\n");
    {
        int K = K_actual_elems;
        int A_bytes = M * K;
        int sfA_bytes = M * K / SFVec;
        std::vector<uint8_t> B(N * K, 0);
        // B: fill ALL bytes with 0x22 (1.0 everywhere)
        for (int n = 0; n < N; n++)
            for (int i = 0; i < K; i++)
                B[n * K + i] = 0x22;
        std::vector<uint8_t> sfA(sfA_bytes, 0x7F);
        std::vector<uint8_t> sfB(N * K / SFVec, 0x7F);

        printf("  byte  D[0,0]  ok?\n");
        for (int bp = 0; bp < 256; bp += 16) {
            std::vector<uint8_t> A(A_bytes, 0);
            A[bp] = 0x62;  // row 0, byte bp

            auto r = run_gemm(M, N, K,
                              A.data(), A_bytes, sfA.data(), sfA_bytes,
                              B.data(), N*K, sfB.data(), N*K/SFVec);
            if (r.ok) {
                const char* ok = (fabsf(r.D[0] - 5.0f) < 0.01f) ? "✓" :
                                 (fabsf(r.D[0]) < 0.01f) ? "ZERO" : "??";
                printf("  %3d   %6.2f  %s\n", bp, r.D[0], ok);
            }
        }
    }

    return 0;
}
