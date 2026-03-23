/**
 * SM120 Fused MoE Pipeline Benchmark
 *
 * Measures CUDA-event latency of:
 *   GEMM1 [M,4096] × [4096,512] → SwiGLU → requant → GEMM2 [M,256] × [256,4096]
 *
 * Baseline: 52μs/layer (FLASHINFER_CUTLASS, 5 kernel launches)
 *   GEMM1: 25μs, Activation: 8μs, GEMM2: 9μs, Reduce: 9μs, Launch overhead: 32μs
 *
 * Our pipeline: 3 kernel launches (GEMM1, SwiGLU+requant, GEMM2)
 * Target: eliminate launch overhead + GMEM round-trip
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

#include <cute/tensor.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace cute;

// --- GEMM types (same as validated test) ---

using ElementAB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC    = float;
using ElementD    = float;
using ElementAcc  = float;

constexpr int AlignmentAB = 32;
constexpr int AlignmentCD = 128 / cutlass::sizeof_bits<ElementC>::value;

using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using EpilogueSchedule = cutlass::epilogue::NoSmemWarpSpecialized;
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc,
    ElementC, cutlass::layout::RowMajor, AlignmentCD,
    ElementD, cutlass::layout::RowMajor, AlignmentCD,
    EpilogueSchedule
>::CollectiveOp;

using StageCount = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;
using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedNvf4Sm120;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementAB, cutlass::layout::RowMajor, AlignmentAB,
    ElementAB, cutlass::layout::ColumnMajor, AlignmentAB,
    ElementAcc,
    TileShape, ClusterShape, StageCount, KernelSchedule
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using Sm1xxCfg = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

static constexpr int SFVec = 16;

// --- SwiGLU + Requant kernel ---

__constant__ float c_e2m1_table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

__global__ void swiglu_requant_kernel(
    const float* __restrict__ gate_up,
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_sf,
    int M, int N_half)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sf_blocks_per_row = N_half / SFVec;
    int total_sf_blocks = M * sf_blocks_per_row;
    if (idx >= total_sf_blocks) return;

    int m = idx / sf_blocks_per_row;
    int sf_block = idx % sf_blocks_per_row;
    int n_start = sf_block * SFVec;

    float vals[SFVec];
    float max_abs = 0.0f;
    for (int i = 0; i < SFVec; i++) {
        int n = n_start + i;
        float gate = gate_up[m * (2 * N_half) + n];
        float up   = gate_up[m * (2 * N_half) + n + N_half];
        float x = gate / (1.0f + expf(-gate));  // silu(gate)
        vals[i] = up * x;
        max_abs = fmaxf(max_abs, fabsf(vals[i]));
    }

    float scale = max_abs / 6.0f;
    if (scale < 1.17549435e-38f) scale = 1.17549435e-38f;

    int exp_bits;
    frexpf(scale, &exp_bits);
    int encoded_exp = exp_bits - 1 + 7;
    encoded_exp = max(0, min(15, encoded_exp));
    uint8_t sf_byte = (uint8_t)(encoded_exp << 3);

    float actual_scale = (encoded_exp == 0) ? 1e-30f : ldexpf(1.0f, encoded_exp - 7);

    out_sf[idx] = sf_byte;

    int byte_base = m * (N_half / 2) + n_start / 2;
    for (int i = 0; i < SFVec; i += 2) {
        float v0 = vals[i] / actual_scale;
        float v1 = vals[i + 1] / actual_scale;

        auto quant = [](float v) -> uint8_t {
            int sign = v < 0 ? 1 : 0;
            float av = fabsf(v);
            int best = 0; float bd = av;
            for (int j = 1; j < 8; j++) {
                float d = fabsf(av - c_e2m1_table[j]);
                if (d < bd) { bd = d; best = j; }
            }
            return (sign << 3) | best;
        };

        out_fp4[byte_base + i / 2] = quant(v0) | (quant(v1) << 4);
    }
}

template <typename Element, typename Layout>
void init_block(cutlass::TensorView<Element, Layout> view, uint64_t seed) {
    constexpr int bits = cutlass::sizeof_bits<Element>::value;
    double hi, lo;
    if constexpr (bits <= 6) { hi = 2; lo = -2; }
    else if constexpr (bits <= 8) {
        if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t> ||
                      cute::is_same_v<Element, cutlass::float_ue4m3_t>) {
            hi = 4; lo = 1;
        } else { hi = 1; lo = -1; }
    } else { hi = 4; lo = -4; }
    cutlass::reference::host::TensorFillRandomUniform(view, seed, hi, lo, 0);
}

// Pre-initialized GEMM runner (avoids repeated setup in timing loop)
struct GemmRunner {
    Gemm gemm;
    typename Gemm::Arguments args;
    cutlass::device_memory::allocation<uint8_t> workspace;

    bool init(int M, int N, int K,
              void* dA, void* dSFA, void* dB, void* dSFB,
              void* dC, void* dD)
    {
        auto sA = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
        auto sB = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
        auto sC = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
        auto sD = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

        auto lSFA = Sm1xxCfg::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
        auto lSFB = Sm1xxCfg::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));

        typename CollectiveMainloop::Arguments ml;
        ml.ptr_A = (typename ElementAB::DataType*)dA;
        ml.dA = sA;
        ml.ptr_B = (typename ElementAB::DataType*)dB;
        ml.dB = sB;
        ml.ptr_SFA = (typename ElementAB::ScaleFactorType*)dSFA;
        ml.layout_SFA = lSFA;
        ml.ptr_SFB = (typename ElementAB::ScaleFactorType*)dSFB;
        ml.layout_SFB = lSFB;

        args = typename Gemm::Arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1}, ml,
            {{1.0f, 0.0f}, (ElementC*)dC, sC, (ElementD*)dD, sD}
        };

        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return false;

        size_t ws = Gemm::get_workspace_size(args);
        workspace = cutlass::device_memory::allocation<uint8_t>(ws);
        return gemm.initialize(args, workspace.get()) == cutlass::Status::kSuccess;
    }

    cutlass::Status run() { return gemm.run(); }
};

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM%d%d, %d SMs)\n\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);

    printf("============================================================\n");
    printf("  SM120 Fused MoE Pipeline Benchmark\n");
    printf("  GEMM1 -> SwiGLU+Requant -> GEMM2\n");
    printf("  Baseline: 52μs/layer (FLASHINFER_CUTLASS, 5 launches)\n");
    printf("============================================================\n\n");

    int K = 4096;       // hidden_size
    int N_half = 256;   // moe_intermediate / TP
    int N_full = 512;   // gate + up
    int warmup = 50;
    int iters = 200;
    uint64_t seed = 42;

    // Benchmark multiple M values (decode=1, small batch)
    int M_values[] = {1, 2, 4, 8, 16, 128};
    int n_M = sizeof(M_values) / sizeof(M_values[0]);

    for (int mi = 0; mi < n_M; mi++) {
        int M = M_values[mi];
        // Round up to tile alignment (128)
        int M_padded = ((M + 127) / 128) * 128;

        printf("--- M=%d (padded to %d) ---\n", M, M_padded);

        // Allocate and initialize all tensors
        // GEMM1: input[M_padded, K] × W1[N_full, K] → gate_up[M_padded, N_full]
        auto lA1 = make_layout(make_shape(M_padded, K, 1),
                   cutlass::make_cute_packed_stride(StrideA{}, {M_padded, K, 1}));
        auto lB1 = make_layout(make_shape(N_full, K, 1),
                   cutlass::make_cute_packed_stride(StrideB{}, {N_full, K, 1}));
        auto lSFA1 = Sm1xxCfg::tile_atom_to_shape_SFA(make_shape(M_padded, N_full, K, 1));
        auto lSFB1 = Sm1xxCfg::tile_atom_to_shape_SFB(make_shape(M_padded, N_full, K, 1));

        cutlass::HostTensor<typename ElementAB::DataType, cutlass::layout::PackedVectorLayout> h_input, h_W1;
        cutlass::HostTensor<typename ElementAB::ScaleFactorType, cutlass::layout::PackedVectorLayout> h_sfI, h_sfW1;

        h_input.reset(cutlass::make_Coord(size(lA1)));
        h_W1.reset(cutlass::make_Coord(size(lB1)));
        h_sfI.reset(cutlass::make_Coord(size(filter_zeros(lSFA1))));
        h_sfW1.reset(cutlass::make_Coord(size(filter_zeros(lSFB1))));

        init_block(h_input.host_view(), seed+1);
        init_block(h_W1.host_view(), seed+2);
        init_block(h_sfI.host_view(), seed+3);
        init_block(h_sfW1.host_view(), seed+4);
        h_input.sync_device(); h_W1.sync_device();
        h_sfI.sync_device(); h_sfW1.sync_device();

        // GEMM2: intermediate[M_padded, N_half] × W2[K, N_half] → output[M_padded, K]
        auto lB2 = make_layout(make_shape(K, N_half, 1),
                   cutlass::make_cute_packed_stride(StrideB{}, {K, N_half, 1}));
        auto lSFB2 = Sm1xxCfg::tile_atom_to_shape_SFB(make_shape(M_padded, K, N_half, 1));

        cutlass::HostTensor<typename ElementAB::DataType, cutlass::layout::PackedVectorLayout> h_W2;
        cutlass::HostTensor<typename ElementAB::ScaleFactorType, cutlass::layout::PackedVectorLayout> h_sfW2;
        h_W2.reset(cutlass::make_Coord(size(lB2)));
        h_sfW2.reset(cutlass::make_Coord(size(filter_zeros(lSFB2))));
        init_block(h_W2.host_view(), seed+5);
        init_block(h_sfW2.host_view(), seed+6);
        h_W2.sync_device(); h_sfW2.sync_device();

        // Intermediate buffers (device only)
        float *d_gate_up, *d_zero;
        uint8_t *d_inter_fp4, *d_inter_sf;
        float *d_output;

        cudaMalloc(&d_gate_up, M_padded * N_full * sizeof(float));
        cudaMalloc(&d_zero, max(M_padded * N_full, M_padded * K) * (int)sizeof(float));
        cudaMemset(d_zero, 0, max(M_padded * N_full, M_padded * K) * sizeof(float));
        cudaMalloc(&d_inter_fp4, M_padded * N_half / 2);
        cudaMalloc(&d_inter_sf, M_padded * (N_half / SFVec));
        cudaMalloc(&d_output, M_padded * K * sizeof(float));

        // Initialize GEMM runners (one-time setup)
        GemmRunner gemm1, gemm2;

        bool ok1 = gemm1.init(M_padded, N_full, K,
            h_input.device_data(), h_sfI.device_data(),
            h_W1.device_data(), h_sfW1.device_data(),
            d_zero, d_gate_up);

        // For GEMM2, we need SFA for the intermediate
        auto lSFA2 = Sm1xxCfg::tile_atom_to_shape_SFA(make_shape(M_padded, K, N_half, 1));

        bool ok2 = gemm2.init(M_padded, K, N_half,
            d_inter_fp4, d_inter_sf,
            h_W2.device_data(), h_sfW2.device_data(),
            d_zero, d_output);

        if (!ok1 || !ok2) {
            printf("  GEMM init failed (gemm1=%d, gemm2=%d)\n\n", ok1, ok2);
            cudaFree(d_gate_up); cudaFree(d_zero); cudaFree(d_inter_fp4);
            cudaFree(d_inter_sf); cudaFree(d_output);
            continue;
        }

        // SwiGLU kernel config
        int total_sf_blocks = M_padded * (N_half / SFVec);
        int swiglu_threads = 256;
        int swiglu_blocks = (total_sf_blocks + swiglu_threads - 1) / swiglu_threads;

        // Warmup
        for (int i = 0; i < warmup; i++) {
            gemm1.run();
            swiglu_requant_kernel<<<swiglu_blocks, swiglu_threads>>>(
                d_gate_up, d_inter_fp4, d_inter_sf, M_padded, N_half);
            gemm2.run();
        }
        cudaDeviceSynchronize();

        // Benchmark: measure each phase separately AND end-to-end
        std::vector<float> t_gemm1(iters), t_swiglu(iters), t_gemm2(iters), t_total(iters);

        for (int i = 0; i < iters; i++) {
            cudaEvent_t e0, e1, e2, e3;
            cudaEventCreate(&e0); cudaEventCreate(&e1);
            cudaEventCreate(&e2); cudaEventCreate(&e3);

            cudaEventRecord(e0);
            gemm1.run();
            cudaEventRecord(e1);
            swiglu_requant_kernel<<<swiglu_blocks, swiglu_threads>>>(
                d_gate_up, d_inter_fp4, d_inter_sf, M_padded, N_half);
            cudaEventRecord(e2);
            gemm2.run();
            cudaEventRecord(e3);
            cudaDeviceSynchronize();

            cudaEventElapsedTime(&t_gemm1[i], e0, e1);
            cudaEventElapsedTime(&t_swiglu[i], e1, e2);
            cudaEventElapsedTime(&t_gemm2[i], e2, e3);
            cudaEventElapsedTime(&t_total[i], e0, e3);

            cudaEventDestroy(e0); cudaEventDestroy(e1);
            cudaEventDestroy(e2); cudaEventDestroy(e3);
        }

        // Compute stats (skip first 20 for warmup)
        auto stats = [](std::vector<float>& v, int skip) {
            std::vector<float> d(v.begin() + skip, v.end());
            std::sort(d.begin(), d.end());
            float sum = std::accumulate(d.begin(), d.end(), 0.0f);
            int n = d.size();
            struct { float avg, med, p5, p95; } s;
            s.avg = sum / n;
            s.med = d[n/2];
            s.p5 = d[(int)(n * 0.05f)];
            s.p95 = d[(int)(n * 0.95f)];
            return s;
        };

        auto sg1 = stats(t_gemm1, 20);
        auto ssw = stats(t_swiglu, 20);
        auto sg2 = stats(t_gemm2, 20);
        auto stot = stats(t_total, 20);

        // Convert to μs
        printf("  GEMM1:     %6.1f μs (med %6.1f, p5-p95: %.1f-%.1f)\n",
               sg1.avg * 1000, sg1.med * 1000, sg1.p5 * 1000, sg1.p95 * 1000);
        printf("  SwiGLU+RQ: %6.1f μs (med %6.1f)\n", ssw.avg * 1000, ssw.med * 1000);
        printf("  GEMM2:     %6.1f μs (med %6.1f, p5-p95: %.1f-%.1f)\n",
               sg2.avg * 1000, sg2.med * 1000, sg2.p5 * 1000, sg2.p95 * 1000);
        printf("  ─────────────────────────────────\n");
        printf("  TOTAL:     %6.1f μs (med %6.1f)\n", stot.avg * 1000, stot.med * 1000);

        // Comparison with baseline
        if (M == 1 || M_padded == 128) {
            float baseline_us = 52.0f;  // from probe
            float speedup = baseline_us / (stot.med * 1000);
            printf("\n  vs BASELINE (52μs): %.1fμs → %.2fx %s\n",
                   stot.med * 1000, speedup,
                   speedup > 1.0f ? "FASTER" : "SLOWER");
            if (M == 1) {
                float per_token_ms = 60.0f * stot.med;  // 60 MoE layers
                float baseline_per_token_ms = 60.0f * 0.052f;
                printf("  Per-token MoE (60 layers): %.2f ms (baseline %.2f ms)\n",
                       per_token_ms, baseline_per_token_ms);
            }
        }
        printf("\n");

        cudaFree(d_gate_up); cudaFree(d_zero);
        cudaFree(d_inter_fp4); cudaFree(d_inter_sf);
        cudaFree(d_output);
    }

    printf("============================================================\n");
    printf("  Baseline decode: 142.6 tok/s (FLASHINFER_CUTLASS)\n");
    printf("  Target: >180 tok/s (26%% speedup from fused MoE)\n");
    printf("============================================================\n");

    return 0;
}
