/**
 * SM120 FA v3 — MMA for Q@K^T only, scalar for softmax + P@V
 *
 * Hybrid kernel: validates MMA score computation against scalar reference.
 * Once QK scores match, we'll add MMA for P@V.
 *
 * Tile: BLOCK_M=64, BLOCK_N=64, HEAD_DIM=128
 * 4 warps × 32 threads = 128 threads
 * Each warp handles 16 rows of output
 */

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

#define BLOCK_M 64
#define BLOCK_N 64
#define HEAD_DIM 128
#define NUM_STAGES 2
#define WARP_SIZE 32
#define NUM_WARPS 4
#define BLOCK_SIZE (NUM_WARPS * WARP_SIZE)
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_M MMA_M

#define Q_ELEMS (BLOCK_M * HEAD_DIM)
#define KV_ELEMS (BLOCK_N * HEAD_DIM)
#define SMEM_BYTES ((Q_ELEMS + NUM_STAGES * 2 * KV_ELEMS) * 2)  // 80KB

__device__ __forceinline__ uint32_t pack2(const __nv_bfloat16& a, const __nv_bfloat16& b) {
    uint32_t r;
    asm("mov.b32 %0, {%1, %2};" : "=r"(r) : "h"(*(const uint16_t*)&a), "h"(*(const uint16_t*)&b));
    return r;
}

__device__ __forceinline__ void cp_async_16B(void* s, const void* g) {
    uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(s));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(g));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n"); }
__device__ __forceinline__ void cp_async_wait_one() { asm volatile("cp.async.wait_group 1;\n"); }

__global__ void __launch_bounds__(BLOCK_SIZE, 1)
sm120_fa_v3(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    float* __restrict__ LSE,
    int Sq, int Skv, int Hq, int Hkv, float scale
) {
    const int bm = blockIdx.x;
    const int head = blockIdx.y;
    const int kv_head = head / (Hq / Hkv);
    const int m_start = bm * BLOCK_M;
    const int tid = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int g = lane / 4;
    const int t = lane % 4;

    const __nv_bfloat16* q_ptr = Q + head * Sq * HEAD_DIM;
    const __nv_bfloat16* k_ptr = K + kv_head * Skv * HEAD_DIM;
    const __nv_bfloat16* v_ptr = V + kv_head * Skv * HEAD_DIM;

    extern __shared__ char smem[];
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* k_s[2], *v_s[2];
    k_s[0] = q_s + Q_ELEMS;
    v_s[0] = k_s[0] + KV_ELEMS;
    k_s[1] = v_s[0] + KV_ELEMS;
    v_s[1] = k_s[1] + KV_ELEMS;

    // Load Q
    for (int i = tid; i < Q_ELEMS / 8; i += BLOCK_SIZE) {
        int row = (i * 8) / HEAD_DIM;
        int col = (i * 8) % HEAD_DIM;
        int qr = m_start + row;
        if (qr < Sq) cp_async_16B(&q_s[row * HEAD_DIM + col], &q_ptr[qr * HEAD_DIM + col]);
    }
    cp_async_commit(); cp_async_wait_all(); __syncthreads();

    // This thread owns 2 rows: row_a = warp*16 + g, row_b = warp*16 + g + 8
    const int own_row_a = warp * WARP_M + g;       // 0..7 within warp tile
    const int own_row_b = warp * WARP_M + g + 8;   // 8..15 within warp tile
    const int global_row_a = m_start + own_row_a;
    const int global_row_b = m_start + own_row_b;

    // Output accumulators per thread: 2 rows × HEAD_DIM cols
    // Thread owns cols: t*2 and t*2+1 for each MMA_N=8 tile
    // 16 tiles × 2 cols × 2 rows = 64 floats
    float o_a[16][2], o_b[16][2];  // row_a and row_b outputs
    float max_a = -FLT_MAX, max_b = -FLT_MAX;
    float sum_a = 0.0f, sum_b = 0.0f;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        o_a[i][0] = o_a[i][1] = 0.0f;
        o_b[i][0] = o_b[i][1] = 0.0f;
    }

    // KV loop
    const int num_kv = (Skv + BLOCK_N - 1) / BLOCK_N;

    auto load_kv = [&](int blk, int stage) {
        int ns = blk * BLOCK_N;
        for (int i = tid; i < KV_ELEMS / 8; i += BLOCK_SIZE) {
            int row = (i * 8) / HEAD_DIM;
            int col = (i * 8) % HEAD_DIM;
            int kvr = ns + row;
            if (kvr < Skv) {
                cp_async_16B(&k_s[stage][row * HEAD_DIM + col], &k_ptr[kvr * HEAD_DIM + col]);
                cp_async_16B(&v_s[stage][row * HEAD_DIM + col], &v_ptr[kvr * HEAD_DIM + col]);
            }
        }
        cp_async_commit();
    };

    load_kv(0, 0);

    for (int kv = 0; kv < num_kv; kv++) {
        int cs = kv % 2;
        if (kv + 1 < num_kv) load_kv(kv + 1, 1 - cs);
        if (kv + 1 < num_kv) cp_async_wait_one(); else cp_async_wait_all();
        __syncthreads();

        // ================================================================
        // Q@K^T via MMA: compute scores for this warp's 16 rows × BLOCK_N cols
        //
        // S[16, 64] = Q[16, 128] @ K^T[128, 64]
        // MMA tiles: 8 along N (BLOCK_N/MMA_N), 8 along K (HEAD_DIM/MMA_K)
        //
        // Each MMA produces c[0..3] for 2 rows × 2 cols.
        // Score accumulators: s_a[8][2] for row_a, s_b[8][2] for row_b
        // ================================================================
        float s_a[8][2], s_b[8][2];  // [n_tile][col_pair]
        #pragma unroll
        for (int i = 0; i < 8; i++)
            s_a[i][0] = s_a[i][1] = s_b[i][0] = s_b[i][1] = 0.0f;

        #pragma unroll
        for (int ki = 0; ki < HEAD_DIM / MMA_K; ki++) {
            // Load Q fragment (A matrix)
            // A[g, 2t], A[g+8, 2t], A[g, 2t+8], A[g+8, 2t+8]
            int qr0 = own_row_a, qr1 = own_row_b;
            int qc0 = ki * MMA_K + t * 2;
            int qc1 = ki * MMA_K + t * 2 + 8;

            uint32_t a_frag[4];
            a_frag[0] = pack2(q_s[qr0 * HEAD_DIM + qc0],     q_s[qr0 * HEAD_DIM + qc0 + 1]);
            a_frag[1] = pack2(q_s[qr1 * HEAD_DIM + qc0],     q_s[qr1 * HEAD_DIM + qc0 + 1]);
            a_frag[2] = pack2(q_s[qr0 * HEAD_DIM + qc1],     q_s[qr0 * HEAD_DIM + qc1 + 1]);
            a_frag[3] = pack2(q_s[qr1 * HEAD_DIM + qc1],     q_s[qr1 * HEAD_DIM + qc1 + 1]);

            #pragma unroll
            for (int ni = 0; ni < BLOCK_N / MMA_N; ni++) {
                // Load K fragment (B matrix): K[N, K] row-major
                // Rb0 = [K[g, 2t], K[g, 2t+1]], Rb1 = [K[g, 2t+8], K[g, 2t+9]]
                // where g=n_index, t=k_pair
                int kn = ni * MMA_N + g;  // g reused as n-index for B
                int kk0 = ki * MMA_K + t * 2;
                int kk1 = ki * MMA_K + t * 2 + 8;

                uint32_t b_frag[2];
                if (kn < BLOCK_N) {
                    b_frag[0] = pack2(k_s[cs][kn * HEAD_DIM + kk0],
                                      k_s[cs][kn * HEAD_DIM + kk0 + 1]);
                    b_frag[1] = pack2(k_s[cs][kn * HEAD_DIM + kk1],
                                      k_s[cs][kn * HEAD_DIM + kk1 + 1]);
                } else {
                    b_frag[0] = b_frag[1] = 0;
                }

                // MMA accumulate
                float tile[4] = {s_a[ni][0], s_a[ni][1], s_b[ni][0], s_b[ni][1]};
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\n"
                    "    {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(tile[0]), "=f"(tile[1]), "=f"(tile[2]), "=f"(tile[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                      "r"(b_frag[0]), "r"(b_frag[1]),
                      "f"(tile[0]), "f"(tile[1]), "f"(tile[2]), "f"(tile[3])
                );
                s_a[ni][0] = tile[0]; s_a[ni][1] = tile[1];
                s_b[ni][0] = tile[2]; s_b[ni][1] = tile[3];
            }
        }

        // Scale
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            s_a[i][0] *= scale; s_a[i][1] *= scale;
            s_b[i][0] *= scale; s_b[i][1] *= scale;
        }

        // ================================================================
        // Online softmax (scalar, per-thread)
        // ================================================================
        // Row A
        {
            float nm = max_a;
            for (int i = 0; i < 8; i++) { nm = fmaxf(nm, s_a[i][0]); nm = fmaxf(nm, s_a[i][1]); }
            float rs = __expf(max_a - nm);
            sum_a *= rs;
            for (int i = 0; i < 16; i++) { o_a[i][0] *= rs; o_a[i][1] *= rs; }
            float ls = 0.0f;
            float p_a[8][2];
            for (int i = 0; i < 8; i++) {
                p_a[i][0] = __expf(s_a[i][0] - nm);
                p_a[i][1] = __expf(s_a[i][1] - nm);
                ls += p_a[i][0] + p_a[i][1];
            }
            sum_a += ls; max_a = nm;

            // P@V scalar: o_a[di][c] += sum_n p_a[ni][c_n] * V[n, d]
            // Thread owns output cols: di*8 + t*2 and +1 for each di
            for (int di = 0; di < 16; di++) {
                int d0 = di * MMA_N + t * 2;
                int d1 = d0 + 1;
                float sum0 = 0.0f, sum1 = 0.0f;
                for (int ni = 0; ni < 8; ni++) {
                    int n0 = ni * MMA_N + t * 2;     // col indices this thread owns in S
                    int n1 = n0 + 1;
                    // But P has different col mapping than V!
                    // P[row_a, n0] and P[row_a, n1] are the softmax weights
                    // V[n0, d0] and V[n1, d0] are the values
                    // Actually: n0/n1 are the KV indices for this thread's S columns
                    // We need ALL KV indices (0..BLOCK_N-1) for the dot product
                    // This per-thread approach only has 16 of the 64 S values!
                    // Need to share P across threads via SMEM or reduce differently.

                    // For now: WRONG but let's just get the QK part validated
                    sum0 += p_a[ni][0] * __bfloat162float(v_s[cs][n0 * HEAD_DIM + d0]);
                    sum0 += p_a[ni][1] * __bfloat162float(v_s[cs][n1 * HEAD_DIM + d0]);
                    sum1 += p_a[ni][0] * __bfloat162float(v_s[cs][n0 * HEAD_DIM + d1]);
                    sum1 += p_a[ni][1] * __bfloat162float(v_s[cs][n1 * HEAD_DIM + d1]);
                }
                o_a[di][0] += sum0;
                o_a[di][1] += sum1;
            }
        }

        // Row B (same logic)
        {
            float nm = max_b;
            for (int i = 0; i < 8; i++) { nm = fmaxf(nm, s_b[i][0]); nm = fmaxf(nm, s_b[i][1]); }
            float rs = __expf(max_b - nm);
            sum_b *= rs;
            for (int i = 0; i < 16; i++) { o_b[i][0] *= rs; o_b[i][1] *= rs; }
            float ls = 0.0f;
            float p_b[8][2];
            for (int i = 0; i < 8; i++) {
                p_b[i][0] = __expf(s_b[i][0] - nm);
                p_b[i][1] = __expf(s_b[i][1] - nm);
                ls += p_b[i][0] + p_b[i][1];
            }
            sum_b += ls; max_b = nm;

            for (int di = 0; di < 16; di++) {
                int d0 = di * MMA_N + t * 2;
                int d1 = d0 + 1;
                float sum0 = 0.0f, sum1 = 0.0f;
                for (int ni = 0; ni < 8; ni++) {
                    int n0 = ni * MMA_N + t * 2;
                    int n1 = n0 + 1;
                    sum0 += p_b[ni][0] * __bfloat162float(v_s[cs][n0 * HEAD_DIM + d0]);
                    sum0 += p_b[ni][1] * __bfloat162float(v_s[cs][n1 * HEAD_DIM + d0]);
                    sum1 += p_b[ni][0] * __bfloat162float(v_s[cs][n0 * HEAD_DIM + d1]);
                    sum1 += p_b[ni][1] * __bfloat162float(v_s[cs][n1 * HEAD_DIM + d1]);
                }
                o_b[di][0] += sum0;
                o_b[di][1] += sum1;
            }
        }

        __syncthreads();
    }

    // Normalize and write
    __nv_bfloat16* o_ptr = O + head * Sq * HEAD_DIM;
    float inv_a = (sum_a > 0) ? 1.0f / sum_a : 0.0f;
    float inv_b = (sum_b > 0) ? 1.0f / sum_b : 0.0f;

    if (global_row_a < Sq) {
        for (int di = 0; di < 16; di++) {
            int c0 = di * MMA_N + t * 2;
            o_ptr[global_row_a * HEAD_DIM + c0]     = __float2bfloat16(o_a[di][0] * inv_a);
            o_ptr[global_row_a * HEAD_DIM + c0 + 1] = __float2bfloat16(o_a[di][1] * inv_a);
        }
    }
    if (global_row_b < Sq) {
        for (int di = 0; di < 16; di++) {
            int c0 = di * MMA_N + t * 2;
            o_ptr[global_row_b * HEAD_DIM + c0]     = __float2bfloat16(o_b[di][0] * inv_b);
            o_ptr[global_row_b * HEAD_DIM + c0 + 1] = __float2bfloat16(o_b[di][1] * inv_b);
        }
    }

    if (LSE && t == 0) {
        if (global_row_a < Sq) LSE[head * Sq + global_row_a] = max_a + logf(sum_a);
        if (global_row_b < Sq) LSE[head * Sq + global_row_b] = max_b + logf(sum_b);
    }
}

extern "C" void sm120_flash_attn_forward(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    int batch, int Hq, int Hkv, int Sq, int Skv, int hd, cudaStream_t stream
) {
    float sc = 1.0f / sqrtf((float)hd);
    dim3 grid((Sq + BLOCK_M - 1) / BLOCK_M, batch * Hq);
    cudaFuncSetAttribute(sm120_fa_v3, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    sm120_fa_v3<<<grid, BLOCK_SIZE, SMEM_BYTES, stream>>>(Q, K, V, O, L, Sq, Skv, Hq, Hkv, sc);
}
