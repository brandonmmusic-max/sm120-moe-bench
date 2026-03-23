/**
 * Minimal FP4 data layout probe.
 *
 * Strategy: Fill A[row 0] with a UNIQUE known pattern per-byte,
 * B = all 1.0, SF = all 1.0. Then D[0,n] = dot(A_row0, B_row_n).
 * Since B is all 1.0 and SF=1.0: D[0,n] = sum of dequantized A_row0 values.
 * All D[0,:] should be identical. If not, data is being reordered.
 *
 * Then: fill A with DIFFERENT rows and check that D[m,0] varies correctly.
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

std::vector<float> run_gemm(int M, int N, int K,
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
    std::vector<float> D;

    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) { printf("FAIL\n"); return D; }

    size_t ws = Gemm::get_workspace_size(args);
    void* workspace = nullptr;
    if (ws > 0) cudaMalloc(&workspace, ws);
    gemm.initialize(args, workspace);
    status = gemm.run();
    cudaDeviceSynchronize();
    if (workspace) cudaFree(workspace);

    if (status == cutlass::Status::kSuccess) {
        D.resize(M * N);
        cudaMemcpy(D.data(), dD, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dSFA); cudaFree(dSFB);
    cudaFree(dC); cudaFree(dD);
    return D;
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM%d%d)\n\n", prop.name, prop.major, prop.minor);

    const int M = 128, N = 128, K = 256;
    printf("Data Layout Probe: [%d,%d] x [%d,%d], SF=1.0 everywhere\n\n", M, K, K, N);

    srand(42);

    // =============================================
    // Test 1: A has ONE non-zero row (row 0), B = all 1.0
    // D[0,n] should all be the same = sum of A_row0 dequantized values
    // D[m,n] for m>0 should be 0
    // =============================================
    printf("=== Test 1: Single non-zero row in A ===\n");
    {
        std::vector<uint8_t> packed_A(M * K / 2, 0);  // All zeros
        std::vector<uint8_t> packed_B(N * K / 2, 0x22);  // All 1.0

        // Fill A[row 0] with pattern: byte i = ((i%7)+1) | (((i%5)+1) << 4)
        for (int i = 0; i < K / 2; i++) {
            uint8_t lo = (i % 7) + 1;  // 1-7 positive nibbles
            uint8_t hi = (i % 5) + 1;
            packed_A[i] = lo | (hi << 4);
        }

        // Compute expected sum of A_row0
        float expected_sum = 0;
        for (int k = 0; k < K; k++) {
            int byte_idx = k / 2;
            uint8_t nib = (k % 2 == 0) ? (packed_A[byte_idx] & 0xF) : (packed_A[byte_idx] >> 4);
            expected_sum += decode_e2m1(nib);
        }

        std::vector<uint8_t> sf_A(M * K / SFVec, 0x7F);
        std::vector<uint8_t> sf_B(N * K / SFVec, 0x7F);

        auto D = run_gemm(M, N, K, packed_A.data(), sf_A.data(), packed_B.data(), sf_B.data());
        if (!D.empty()) {
            printf("  Expected: D[0,n] = %.2f for all n, D[m>0,n] = 0\n", expected_sum);
            printf("  D[0,0]=%.2f  D[0,1]=%.2f  D[0,63]=%.2f  D[0,127]=%.2f\n",
                   D[0], D[1], D[63], D[127]);
            printf("  D[1,0]=%.2f  D[2,0]=%.2f  D[127,0]=%.2f\n",
                   D[N], D[2*N], D[127*N]);

            // Check consistency of D[0,:]
            float min_d0 = D[0], max_d0 = D[0];
            for (int n = 0; n < N; n++) {
                min_d0 = fminf(min_d0, D[n]);
                max_d0 = fmaxf(max_d0, D[n]);
            }
            printf("  D[0,:] range: [%.2f, %.2f] (should be constant)\n", min_d0, max_d0);
            printf("  D[0,:] spread: %.4f\n", max_d0 - min_d0);
        }
    }

    // =============================================
    // Test 2: Each row of A has a DIFFERENT constant value
    // Row m: all nibbles = (m%7)+1
    // B = all 1.0, SF=1.0
    // D[m,n] = K * decode_e2m1((m%7)+1) for all n
    // =============================================
    printf("\n=== Test 2: Each A row = different constant ===\n");
    {
        std::vector<uint8_t> packed_A(M * K / 2);
        for (int m = 0; m < M; m++) {
            uint8_t nib = (m % 7) + 1;
            uint8_t byte_val = nib | (nib << 4);
            for (int i = 0; i < K / 2; i++)
                packed_A[m * (K / 2) + i] = byte_val;
        }

        std::vector<uint8_t> packed_B(N * K / 2, 0x22);  // All 1.0
        std::vector<uint8_t> sf_A(M * K / SFVec, 0x7F);
        std::vector<uint8_t> sf_B(N * K / SFVec, 0x7F);

        auto D = run_gemm(M, N, K, packed_A.data(), sf_A.data(), packed_B.data(), sf_B.data());
        if (!D.empty()) {
            printf("  Row | Nibble | E2M1  | Expected    | D[m,0]      | Match\n");
            bool all_match = true;
            for (int m = 0; m < 8; m++) {
                uint8_t nib = (m % 7) + 1;
                float e2m1 = decode_e2m1(nib);
                float expected = K * e2m1;
                bool match = fabsf(D[m * N] - expected) < 0.01f;
                if (!match) all_match = false;
                printf("  %3d |   0x%X  | %5.2f | %10.2f  | %10.2f  | %s\n",
                       m, nib, e2m1, expected, D[m * N], match ? "OK" : "FAIL");
            }
            // Check a few more
            for (int m = 120; m < 128; m++) {
                uint8_t nib = (m % 7) + 1;
                float expected = K * decode_e2m1(nib);
                bool match = fabsf(D[m * N] - expected) < 0.01f;
                if (!match) all_match = false;
                printf("  %3d |   0x%X  | %5.2f | %10.2f  | %10.2f  | %s\n",
                       m, nib, decode_e2m1(nib), expected, D[m * N], match ? "OK" : "FAIL");
            }
            printf("  Overall: %s\n", all_match ? "ALL ROWS MATCH" : "SOME ROWS FAIL");
        }
    }

    // =============================================
    // Test 3: A = all 1.0, B rows have different constants
    // Same idea but testing B's column-major layout
    // =============================================
    printf("\n=== Test 3: Each B row = different constant ===\n");
    {
        std::vector<uint8_t> packed_A(M * K / 2, 0x22);
        std::vector<uint8_t> packed_B(N * K / 2);
        for (int n = 0; n < N; n++) {
            uint8_t nib = (n % 7) + 1;
            uint8_t byte_val = nib | (nib << 4);
            for (int i = 0; i < K / 2; i++)
                packed_B[n * (K / 2) + i] = byte_val;
        }

        std::vector<uint8_t> sf_A(M * K / SFVec, 0x7F);
        std::vector<uint8_t> sf_B(N * K / SFVec, 0x7F);

        auto D = run_gemm(M, N, K, packed_A.data(), sf_A.data(), packed_B.data(), sf_B.data());
        if (!D.empty()) {
            printf("  Col | Nibble | Expected    | D[0,n]      | Match\n");
            bool all_match = true;
            for (int n = 0; n < 8; n++) {
                uint8_t nib = (n % 7) + 1;
                float expected = K * 1.0f * decode_e2m1(nib);
                bool match = fabsf(D[n] - expected) < 0.01f;
                if (!match) all_match = false;
                printf("  %3d |   0x%X  | %10.2f  | %10.2f  | %s\n",
                       n, nib, expected, D[n], match ? "OK" : "FAIL");
            }
            printf("  Overall: %s\n", all_match ? "ALL COLS MATCH" : "SOME COLS FAIL");
        }
    }

    // =============================================
    // Test 4: Full random with POSITIVE-ONLY nibbles, SF=1.0
    // If decode_e2m1 is correct, this should match the reference
    // =============================================
    printf("\n=== Test 4: Random positive FP4 (nibbles 1-7), SF=1.0 ===\n");
    {
        std::vector<uint8_t> packed_A(M * K / 2), packed_B(N * K / 2);
        for (auto& b : packed_A) {
            uint8_t lo = (rand() % 7) + 1;  // 1-7 only
            uint8_t hi = (rand() % 7) + 1;
            b = lo | (hi << 4);
        }
        for (auto& b : packed_B) {
            uint8_t lo = (rand() % 7) + 1;
            uint8_t hi = (rand() % 7) + 1;
            b = lo | (hi << 4);
        }

        std::vector<uint8_t> sf_A(M * K / SFVec, 0x7F);
        std::vector<uint8_t> sf_B(N * K / SFVec, 0x7F);

        // Host reference
        std::vector<float> ref(M * N, 0.0f);
        for (int m = 0; m < M; m++)
            for (int n = 0; n < N; n++) {
                float sum = 0;
                for (int k = 0; k < K; k++) {
                    uint8_t a_nib = (k % 2 == 0) ?
                        (packed_A[m*(K/2)+k/2] & 0xF) : (packed_A[m*(K/2)+k/2] >> 4);
                    uint8_t b_nib = (k % 2 == 0) ?
                        (packed_B[n*(K/2)+k/2] & 0xF) : (packed_B[n*(K/2)+k/2] >> 4);
                    sum += decode_e2m1(a_nib) * decode_e2m1(b_nib);
                }
                ref[m * N + n] = sum;
            }

        auto D = run_gemm(M, N, K, packed_A.data(), sf_A.data(), packed_B.data(), sf_B.data());
        if (!D.empty()) {
            printf("  First 4: ref=[%.2f, %.2f, %.2f, %.2f]\n", ref[0], ref[1], ref[2], ref[3]);
            printf("          kern=[%.2f, %.2f, %.2f, %.2f]\n", D[0], D[1], D[2], D[3]);

            float max_abs = 0, sum_abs = 0;
            int n_close = 0;
            for (int i = 0; i < M * N; i++) {
                float err = fabsf(D[i] - ref[i]);
                max_abs = fmaxf(max_abs, err);
                sum_abs += err;
                if (fabsf(ref[i]) > 0.1f && err / fabsf(ref[i]) < 0.02f) n_close++;
            }
            printf("  Max abs err: %.4f, Avg abs err: %.4f\n", max_abs, sum_abs/(M*N));
            printf("  Within 2%%: %d/%d (%.1f%%)\n", n_close, M*N, 100.0f*n_close/(M*N));
        }
    }

    printf("\n=== DONE ===\n");
    return 0;
}
