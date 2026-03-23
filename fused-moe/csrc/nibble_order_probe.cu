/**
 * Nibble Order Probe: Determine exact element-to-byte mapping.
 *
 * Strategy: A has ONE non-zero element at position (0, k_target).
 * B = all 1.0, SF = 1.0.
 * D[0, n] = A[0, k_target] * 1.0 = decode(nibble) for all n.
 *
 * Set k_target = 0: A byte 0, lo nibble = value, hi nibble = 0
 * If kernel reads lo-first: D = decode(value)
 * If kernel reads hi-first: D = 0 (wrong nibble is zero)
 *
 * Then k_target = 1: A byte 0, hi nibble = value, lo nibble = 0
 * Should produce the opposite.
 *
 * Also test: are there interleaving/reordering within the 128-byte
 * cache lines that TMA uses?
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
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <string.h>

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

static const float E2M1_TABLE[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

float decode_e2m1(uint8_t nibble) {
    int sign = (nibble >> 3) & 1;
    float val = E2M1_TABLE[nibble & 0x7];
    return sign ? -val : val;
}

float run_single(int M, int N, int K,
                 const uint8_t* h_A, const uint8_t* h_sf_A,
                 const uint8_t* h_B, const uint8_t* h_sf_B) {
    uint8_t *dA, *dB, *dSFA, *dSFB;
    float *dC, *dD;

    cudaMalloc(&dA, M * K / 2);
    cudaMalloc(&dB, N * K / 2);
    cudaMalloc(&dSFA, (M * K) / SFVec);
    cudaMalloc(&dSFB, (N * K) / SFVec);
    cudaMalloc(&dC, M * N * sizeof(float));
    cudaMalloc(&dD, M * N * sizeof(float));

    cudaMemcpy(dA, h_A, M * K / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, h_B, N * K / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dSFA, h_sf_A, (M * K) / SFVec, cudaMemcpyHostToDevice);
    cudaMemcpy(dSFB, h_sf_B, (N * K) / SFVec, cudaMemcpyHostToDevice);
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
    float result = -999.0f;

    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) return result;

    size_t ws = Gemm::get_workspace_size(args);
    void* workspace = nullptr;
    if (ws > 0) cudaMalloc(&workspace, ws);
    gemm.initialize(args, workspace);
    status = gemm.run();
    cudaDeviceSynchronize();
    if (workspace) cudaFree(workspace);

    if (status == cutlass::Status::kSuccess) {
        cudaMemcpy(&result, dD, sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dSFA); cudaFree(dSFB);
    cudaFree(dC); cudaFree(dD);
    return result;
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM%d%d)\n\n", prop.name, prop.major, prop.minor);

    const int M = 128, N = 128, K = 256;
    const int K_BYTES = K / 2;  // 128 bytes per row
    printf("Nibble Order Probe: [%d,%d] x [%d,%d]\n\n", M, K, K, N);

    // B = all 1.0, SF = all 1.0
    std::vector<uint8_t> packed_B(N * K_BYTES, 0x22);
    std::vector<uint8_t> sf_A(M * K / SFVec, 0x7F);
    std::vector<uint8_t> sf_B(N * K / SFVec, 0x7F);

    // =============================================
    // Probe: Set A[row 0] to have exactly ONE non-zero byte at position byte_pos
    // That byte = 0x02 (lo nibble = 1.0, hi nibble = 0.0)
    // B = all 1.0, SF = 1.0
    // Then D[0,0] = contribution from elements at positions 2*byte_pos and 2*byte_pos+1
    // Expected: decode(0x2)*1 + decode(0x0)*1 = 1.0 + 0.0 = 1.0
    // If we get 1.0 for byte_pos=p, that byte contributes to k=2p and k=2p+1
    // =============================================

    printf("=== Probe 1: Single byte position sweep ===\n");
    printf("A = all zeros except byte_pos in row 0 = 0x62 (lo=1.0, hi=4.0)\n");
    printf("D[0,0] should be 5.0 for the byte that the kernel reads at some k position\n\n");

    // We'll set byte to 0x62: lo = 0x2 = 1.0, hi = 0x6 = 4.0
    // If byte contributes 2 elements, D += 1.0*1.0 + 4.0*1.0 = 5.0
    printf("byte_pos  D[0,0]  (should be 5.0 if this byte contributes correctly)\n");

    // Only check a few positions to understand the pattern
    int positions[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 31, 32, 63, 64, 95, 96, 127};
    int npos = sizeof(positions) / sizeof(positions[0]);

    for (int pi = 0; pi < npos; pi++) {
        int byte_pos = positions[pi];
        std::vector<uint8_t> packed_A(M * K_BYTES, 0);
        packed_A[byte_pos] = 0x62;  // lo=1.0, hi=4.0 → sum=5.0

        float d = run_single(M, N, K, packed_A.data(), sf_A.data(), packed_B.data(), sf_B.data());
        printf("  %3d     %.2f\n", byte_pos, d);
    }

    // =============================================
    // Probe 2: More precise — use different values per byte
    // Fill A[row 0] with unique byte per position:
    //   byte_pos 0 = 0x12 (lo=1.0, hi=0.5), byte_pos 1 = 0x32 (lo=1.0, hi=1.5), etc.
    // Then B[row n] = all zeros except one specific k position has nibble 0x2 (1.0)
    // D[0,n] = A[0,k] * 1.0 = A's value at position k
    // This reveals the exact k→byte mapping
    // =============================================
    printf("\n=== Probe 2: Which k does each byte contribute to? ===\n");
    printf("A[row 0]: byte b has unique nibble pair. B[row n]: 1.0 at k=n, 0 elsewhere\n");
    printf("D[0,n] reveals A's dequantized value at k=n → which byte provided it\n\n");

    // Create A: each byte has a unique pair of values
    // Use the byte index modulo values: byte b → lo = E2M1[(b%6)+1], hi = E2M1[((b+3)%6)+1]
    std::vector<uint8_t> packed_A(M * K_BYTES, 0);
    std::vector<float> byte_lo_vals(K_BYTES), byte_hi_vals(K_BYTES);
    for (int b = 0; b < K_BYTES; b++) {
        uint8_t lo_nib = (b % 6) + 1;      // 1-6
        uint8_t hi_nib = ((b + 3) % 6) + 1; // offset pattern
        packed_A[b] = lo_nib | (hi_nib << 4);
        byte_lo_vals[b] = decode_e2m1(lo_nib);
        byte_hi_vals[b] = decode_e2m1(hi_nib);
    }

    // Create B: row n has exactly one 1.0 at element position n
    // B is ColumnMajor stored as [N, K], so B[n, k] is at packed_B[n*(K/2) + k/2]
    std::vector<uint8_t> packed_B2(N * K_BYTES, 0);
    for (int n = 0; n < N && n < K; n++) {
        int byte_idx = n * K_BYTES + n / 2;
        if (n % 2 == 0)
            packed_B2[byte_idx] = 0x02;  // lo nibble = 1.0
        else
            packed_B2[byte_idx] = 0x20;  // hi nibble = 1.0
    }

    // Now D[0, n] = sum_k A[0,k] * B[n,k] = A[0,n] (since B[n,:] is 1 at k=n)
    // This directly reads out A's dequantized value at position n
    std::vector<uint8_t> sf_B2(N * K / SFVec, 0x7F);

    // Run multiple GEMMs is expensive. Instead, run ONE GEMM and read D[0,:]
    {
        uint8_t *dA, *dB, *dSFA, *dSFB;
        float *dC, *dD;

        cudaMalloc(&dA, M * K_BYTES);
        cudaMalloc(&dB, N * K_BYTES);
        cudaMalloc(&dSFA, (M * K) / SFVec);
        cudaMalloc(&dSFB, (N * K) / SFVec);
        cudaMalloc(&dC, M * N * sizeof(float));
        cudaMalloc(&dD, M * N * sizeof(float));

        cudaMemcpy(dA, packed_A.data(), M * K_BYTES, cudaMemcpyHostToDevice);
        cudaMemcpy(dB, packed_B2.data(), N * K_BYTES, cudaMemcpyHostToDevice);
        cudaMemcpy(dSFA, sf_A.data(), sf_A.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dSFB, sf_B2.data(), sf_B2.size(), cudaMemcpyHostToDevice);
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
        if (status == cutlass::Status::kSuccess) {
            size_t ws = Gemm::get_workspace_size(args);
            void* workspace = nullptr;
            if (ws > 0) cudaMalloc(&workspace, ws);
            gemm.initialize(args, workspace);
            status = gemm.run();
            cudaDeviceSynchronize();
            if (workspace) cudaFree(workspace);

            if (status == cutlass::Status::kSuccess) {
                std::vector<float> D(M * N);
                cudaMemcpy(D.data(), dD, M * N * sizeof(float), cudaMemcpyDeviceToHost);

                // D[0, n] should be A's dequantized value at element position n
                printf("k   D[0,k]  expected_lo_first  byte_pos  lo_val  hi_val\n");
                for (int k = 0; k < 32 && k < N; k++) {
                    int byte_pos = k / 2;
                    float exp_val = (k % 2 == 0) ? byte_lo_vals[byte_pos] : byte_hi_vals[byte_pos];
                    printf("%3d  %6.2f  %6.2f (byte %d %s)  [lo=%.1f hi=%.1f]\n",
                           k, D[k], exp_val, byte_pos,
                           (k % 2 == 0) ? "lo" : "hi",
                           byte_lo_vals[byte_pos], byte_hi_vals[byte_pos]);
                }

                // Check how many match lo-first vs hi-first
                int match_lo = 0, match_hi = 0, match_neither = 0;
                for (int k = 0; k < K && k < N; k++) {
                    int bp = k / 2;
                    float lo_exp = (k % 2 == 0) ? byte_lo_vals[bp] : byte_hi_vals[bp];
                    float hi_exp = (k % 2 == 0) ? byte_hi_vals[bp] : byte_lo_vals[bp];
                    if (fabsf(D[k] - lo_exp) < 0.01f) match_lo++;
                    else if (fabsf(D[k] - hi_exp) < 0.01f) match_hi++;
                    else match_neither++;
                }
                printf("\nMatches lo-first: %d, hi-first: %d, neither: %d (of %d)\n",
                       match_lo, match_hi, match_neither, K < N ? K : N);
            }
        }

        cudaFree(dA); cudaFree(dB); cudaFree(dSFA); cudaFree(dSFB);
        cudaFree(dC); cudaFree(dD);
    }

    printf("\n=== DONE ===\n");
    return 0;
}
