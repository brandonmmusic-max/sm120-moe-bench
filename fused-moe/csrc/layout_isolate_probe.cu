/**
 * Layout Isolation Probe: Is the error in FP4 data packing or SF layout?
 *
 * Test 1: Random FP4 data, all SF=1.0 (0x7F) — isolates FP4 packing
 * Test 2: All FP4=1.0, random SF — isolates SF layout
 * Test 3: Structured FP4 data (ascending nibbles) — checks element ordering
 *
 * If Test 1 fails: our FP4 byte packing doesn't match CUTLASS TMA expectations
 * If Test 2 fails: our SF blocked_product layout is wrong
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

float decode_ue8m0(uint8_t val) {
    if (val == 0) return 0.0f;
    return powf(2.0f, (float)((int)val - 127));
}

struct GemmResult {
    bool ok;
    std::vector<float> D;
};

GemmResult run_gemm(int M, int N, int K,
                    const uint8_t* h_packed_A, const uint8_t* h_sf_A,
                    const uint8_t* h_packed_B, const uint8_t* h_sf_B) {
    uint8_t *dA, *dB, *dSFA, *dSFB;
    float *dC, *dD;

    int sf_A_size = (M * K) / SFVec;
    int sf_B_size = (N * K) / SFVec;

    cudaMalloc(&dA, M * K / 2);
    cudaMalloc(&dB, N * K / 2);
    cudaMalloc(&dSFA, sf_A_size);
    cudaMalloc(&dSFB, sf_B_size);
    cudaMalloc(&dC, M * N * sizeof(float));
    cudaMalloc(&dD, M * N * sizeof(float));

    cudaMemcpy(dA, h_packed_A, M * K / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, h_packed_B, N * K / 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dSFA, h_sf_A, sf_A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dSFB, h_sf_B, sf_B_size, cudaMemcpyHostToDevice);
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

    GemmResult result;
    result.ok = false;

    Gemm gemm;
    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        printf("  can_implement FAILED: %d\n", (int)status);
        goto cleanup;
    }

    {
        size_t ws = Gemm::get_workspace_size(args);
        void* workspace = nullptr;
        if (ws > 0) cudaMalloc(&workspace, ws);
        gemm.initialize(args, workspace);
        status = gemm.run();
        cudaDeviceSynchronize();
        if (workspace) cudaFree(workspace);

        if (status == cutlass::Status::kSuccess) {
            result.D.resize(M * N);
            cudaMemcpy(result.D.data(), dD, M * N * sizeof(float), cudaMemcpyDeviceToHost);
            result.ok = true;
        } else {
            printf("  run FAILED\n");
        }
    }

cleanup:
    cudaFree(dA); cudaFree(dB); cudaFree(dSFA); cudaFree(dSFB);
    cudaFree(dC); cudaFree(dD);
    return result;
}

void print_stats(const std::vector<float>& kernel, const std::vector<float>& ref, int M, int N) {
    printf("  First 8:\n");
    printf("    idx     ref          kernel       diff\n");
    for (int i = 0; i < 8 && i < (int)ref.size(); i++) {
        printf("    [%d]  %10.4f  %10.4f  %+8.4f\n", i, ref[i], kernel[i], kernel[i]-ref[i]);
    }

    float max_abs = 0, sum_abs = 0, sum_ref_sq = 0;
    int n_close = 0;
    for (int i = 0; i < M * N; i++) {
        float err = fabsf(kernel[i] - ref[i]);
        max_abs = fmaxf(max_abs, err);
        sum_abs += err;
        sum_ref_sq += ref[i] * ref[i];
        if (fabsf(ref[i]) > 1e-6f) {
            if (err / fabsf(ref[i]) < 0.02f) n_close++;
        } else {
            if (fabsf(kernel[i]) < 1e-3f) n_close++;
        }
    }
    float rms = sqrtf(sum_ref_sq / (M * N));
    printf("  Max abs: %.4f, Avg abs: %.4f, RMS ref: %.4f\n", max_abs, sum_abs/(M*N), rms);
    printf("  Norm err: %.2f%%, Within 2%%: %d/%d (%.1f%%)\n",
           rms > 0 ? 100.0f*(sum_abs/(M*N))/rms : 0,
           n_close, M*N, 100.0f*n_close/(M*N));
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM%d%d)\n\n", prop.name, prop.major, prop.minor);

    const int M = 128, N = 128, K = 256;  // Small K for fast host ref
    printf("Layout Isolation Probe: [%d,%d] x [%d,%d]\n\n", M, K, K, N);

    srand(42);

    // =============================================
    // Test 1: Random FP4 data, ALL SF = 1.0 (0x7F)
    // This isolates FP4 data packing correctness
    // =============================================
    printf("=== Test 1: Random FP4, SF=1.0 everywhere ===\n");
    {
        // Generate random FP4 nibbles directly (skip float quantization)
        std::vector<uint8_t> packed_A(M * K / 2), packed_B(N * K / 2);
        for (auto& b : packed_A) b = rand() & 0xFF;  // random nibble pairs
        for (auto& b : packed_B) b = rand() & 0xFF;

        // All SF = 0x7F = 1.0
        std::vector<uint8_t> sf_A(M * K / SFVec, 0x7F);
        std::vector<uint8_t> sf_B(N * K / SFVec, 0x7F);

        // Host reference: dequantize and multiply
        // A is RowMajor [M, K], B is ColumnMajor stored as [N, K]
        // Element (m, k) of A: packed_A[m*(K/2) + k/2], nibble k%2
        std::vector<float> ref(M * N, 0.0f);
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0;
                for (int k = 0; k < K; k++) {
                    // A[m,k]: row-major
                    int a_byte = m * (K/2) + k/2;
                    uint8_t a_nib = (k % 2 == 0) ? (packed_A[a_byte] & 0xF) : (packed_A[a_byte] >> 4);
                    float a_val = decode_e2m1(a_nib) * 1.0f;  // SF=1.0

                    // B[n,k]: col-major stored as [N,K] row-major
                    int b_byte = n * (K/2) + k/2;
                    uint8_t b_nib = (k % 2 == 0) ? (packed_B[b_byte] & 0xF) : (packed_B[b_byte] >> 4);
                    float b_val = decode_e2m1(b_nib) * 1.0f;  // SF=1.0

                    sum += a_val * b_val;
                }
                ref[m * N + n] = sum;
            }
        }

        auto result = run_gemm(M, N, K, packed_A.data(), sf_A.data(), packed_B.data(), sf_B.data());
        if (result.ok) {
            print_stats(result.D, ref, M, N);
        }
    }

    // =============================================
    // Test 2: All FP4 = 1.0, random SF values
    // This isolates SF layout correctness
    // =============================================
    printf("\n=== Test 2: FP4=1.0 everywhere, random SF ===\n");
    {
        std::vector<uint8_t> packed_A(M * K / 2, 0x22);  // all 1.0
        std::vector<uint8_t> packed_B(N * K / 2, 0x22);

        // Random SF: use values in range 0x75-0x85 to stay in reasonable scale range
        std::vector<uint8_t> sf_A(M * K / SFVec), sf_B(N * K / SFVec);
        for (auto& s : sf_A) s = 0x75 + (rand() % 0x10);
        for (auto& s : sf_B) s = 0x75 + (rand() % 0x10);

        // Host reference: each block of 16 elements along K has its own SF
        // D[m,n] = sum over k-blocks: 16 * 1.0 * 1.0 * SFA[m,block] * SFB[n,block]
        std::vector<float> ref(M * N, 0.0f);
        int num_k_blocks = K / SFVec;
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0;
                for (int kb = 0; kb < num_k_blocks; kb++) {
                    float sfa = decode_ue8m0(sf_A[m * num_k_blocks + kb]);
                    float sfb = decode_ue8m0(sf_B[n * num_k_blocks + kb]);
                    sum += SFVec * 1.0f * sfa * sfb;  // 16 elements, all 1.0
                }
                ref[m * N + n] = sum;
            }
        }

        auto result = run_gemm(M, N, K, packed_A.data(), sf_A.data(), packed_B.data(), sf_B.data());
        if (result.ok) {
            print_stats(result.D, ref, M, N);
        }
    }

    // =============================================
    // Test 3: Structured FP4 — ascending pattern per row
    // Each row has a repeating pattern: nibbles 1,2,3,4,5,6,7,1,2,...
    // This checks element ordering
    // =============================================
    printf("\n=== Test 3: Ascending FP4 pattern, SF=1.0 ===\n");
    {
        std::vector<uint8_t> packed_A(M * K / 2, 0), packed_B(N * K / 2, 0);

        // A: row m, col k -> nibble = (k % 7) + 1 = {1,2,3,4,5,6,7,1,...}
        for (int m = 0; m < M; m++) {
            for (int k = 0; k < K; k++) {
                uint8_t nib = (k % 7) + 1;  // values 1-7 (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
                int byte_idx = m * (K/2) + k/2;
                if (k % 2 == 0) packed_A[byte_idx] = nib;
                else packed_A[byte_idx] |= (nib << 4);
            }
        }
        // B: same pattern per row
        for (int n = 0; n < N; n++) {
            for (int k = 0; k < K; k++) {
                uint8_t nib = ((k + n) % 7) + 1;  // offset by n for variation
                int byte_idx = n * (K/2) + k/2;
                if (k % 2 == 0) packed_B[byte_idx] = nib;
                else packed_B[byte_idx] |= (nib << 4);
            }
        }

        std::vector<uint8_t> sf_A(M * K / SFVec, 0x7F);
        std::vector<uint8_t> sf_B(N * K / SFVec, 0x7F);

        // Host reference
        std::vector<float> ref(M * N, 0.0f);
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                float sum = 0;
                for (int k = 0; k < K; k++) {
                    uint8_t a_nib = (k % 7) + 1;
                    uint8_t b_nib = ((k + n) % 7) + 1;
                    sum += decode_e2m1(a_nib) * decode_e2m1(b_nib);
                }
                ref[m * N + n] = sum;
            }
        }

        auto result = run_gemm(M, N, K, packed_A.data(), sf_A.data(), packed_B.data(), sf_B.data());
        if (result.ok) {
            print_stats(result.D, ref, M, N);
        }
    }

    // =============================================
    // Test 4: Check nibble order — is it lo-first or hi-first?
    // Row 0 of A: alternating (0.5, 6.0) = nibbles (1, 7)
    // If lo-first: byte = 0x71 → elements (0.5, 6.0)
    // If hi-first: byte = 0x71 → elements (6.0, 0.5)
    // B: all 1.0 (0x22), SF=1.0 (0x7F)
    // D[0,0] should be K/2 * (0.5 + 6.0) = K/2 * 6.5 = 832 (if lo-first)
    // Or K/2 * (6.0 + 0.5) = 832 (same — need asymmetric test)
    // Better: A row 0: (0.5, 0, 0.5, 0, ...) = nibble pairs (1, 0) = byte 0x01
    // Then D[0,n] = K/2 * 0.5 = 64  (lo-first)
    // vs  D[0,n] = 0 (hi-first, because all even positions are 0)
    // =============================================
    printf("\n=== Test 4: Nibble order test ===\n");
    {
        // A: byte 0x01 everywhere → lo nibble=1 (0.5), hi nibble=0 (0.0)
        // If CUTLASS interprets as lo-first: elements = (0.5, 0.0, 0.5, 0.0, ...)
        // If CUTLASS interprets as hi-first: elements = (0.0, 0.5, 0.0, 0.5, ...)
        std::vector<uint8_t> packed_A(M * K / 2, 0x01);

        // B: all 1.0
        std::vector<uint8_t> packed_B(N * K / 2, 0x22);

        std::vector<uint8_t> sf_A(M * K / SFVec, 0x7F);
        std::vector<uint8_t> sf_B(N * K / SFVec, 0x7F);

        // If lo-first: D[m,n] = sum_k A[m,k]*B[n,k] = (K/2)*0.5*1.0 + (K/2)*0.0*1.0
        float expected_lo_first = (K / 2) * 0.5f;  // = 64
        // If hi-first: D[m,n] = (K/2)*0.0*1.0 + (K/2)*0.5*1.0
        float expected_hi_first = (K / 2) * 0.5f;  // = 64 (same! both halves contribute same count)

        // Need asymmetric: A = 0x01 = (lo=0.5, hi=0.0), B = 0x02 = (lo=1.0, hi=0.0)
        // If both lo-first: D = (K/2)*(0.5*1.0) + (K/2)*(0.0*0.0) = K/2*0.5 = 64
        // If both hi-first: D = (K/2)*(0.0*0.0) + (K/2)*(0.5*1.0) = K/2*0.5 = 64 (same!)

        // Better: A = 0x01 (lo=0.5, hi=0.0), B = 0x20 (lo=0.0, hi=1.0)
        // If lo-first: k=0: A=0.5, B=0.0; k=1: A=0.0, B=1.0 → sum = 0
        // If hi-first: k=0: A=0.0, B=1.0; k=1: A=0.5, B=0.0 → sum = 0
        // Hmm, need different approach.

        // A = 0x21 = (lo=1, hi=2) = (0.5, 1.0)
        // B = 0x22 = (lo=2, hi=2) = (1.0, 1.0)
        // If lo-first: D = (K/2)*(0.5*1.0 + 1.0*1.0) = K/2*1.5 = 192
        // If hi-first: D = (K/2)*(1.0*1.0 + 0.5*1.0) = K/2*1.5 = 192 (same again!)
        // The sum is commutative — can't distinguish with dot product.

        // Use DIFFERENT patterns for A:
        // A = 0x31 = (lo=1, hi=3) = (0.5, 1.5)
        // B = 0x02 = (lo=2, hi=0) = (1.0, 0.0)
        // lo-first: k=0: 0.5*1.0=0.5, k=1: 1.5*0.0=0 → per-pair = 0.5 → total = K/2 * 0.5 = 64
        // hi-first: k=0: 1.5*0.0=0, k=1: 0.5*1.0=0.5 → per-pair = 0.5 → total = 64 (SAME!)

        // OK the dot product makes this symmetric. Let me use a non-uniform B:
        // A = 0x31, B row 0 different from B row 1
        // Actually, let me just test with a known non-symmetric case:
        // Fill A[row 0] with specific bytes, B with specific bytes, check exact output

        // Simpler approach: use ONLY the low nibble, set high nibble to 0
        // A = 0x01 everywhere → even k=0.5, odd k=0.0
        // B = 0x01 everywhere → even k=0.5, odd k=0.0
        // lo-first: D = (K/2)*(0.5*0.5) + (K/2)*(0.0*0.0) = K/2 * 0.25 = 32
        // hi-first: D = (K/2)*(0.0*0.0) + (K/2)*(0.5*0.5) = K/2 * 0.25 = 32 (same again)

        // The fundamental issue: dot product = sum of products, commutative.
        // So Test 4 can't distinguish nibble order with dot product alone.
        // But it CAN detect if some elements are garbled.

        // Let's just run it with 0x01 and check output = expected
        auto result = run_gemm(M, N, K, packed_A.data(), sf_A.data(), packed_B.data(), sf_B.data());
        if (result.ok) {
            printf("  A = 0x01 (lo=0.5, hi=0.0), B = 0x22 (lo=1.0, hi=1.0), SF=1.0\n");
            printf("  Expected (lo-first): D = K/2*(0.5*1.0 + 0.0*1.0) = %.1f\n", (K/2)*0.5f);
            printf("  Expected (hi-first): D = K/2*(0.0*1.0 + 0.5*1.0) = %.1f\n", (K/2)*0.5f);
            printf("  Actual D[0]: %.4f, D[1]: %.4f, D[M*N-1]: %.4f\n",
                   result.D[0], result.D[1], result.D[M*N-1]);

            // All outputs should be identical since all rows are the same
            float expected = (K/2) * 0.5f;
            bool all_match = true;
            for (int i = 0; i < M * N; i++) {
                if (fabsf(result.D[i] - expected) > 0.01f) {
                    all_match = false;
                    printf("  MISMATCH at [%d,%d]: %.4f (expected %.4f)\n",
                           i/N, i%N, result.D[i], expected);
                    if (i > 10) { printf("  ... (more mismatches)\n"); break; }
                }
            }
            if (all_match) printf("  ALL %d elements match expected value %.1f ✓\n", M*N, expected);
        }
    }

    printf("\n=== ISOLATION COMPLETE ===\n");
    return 0;
}
