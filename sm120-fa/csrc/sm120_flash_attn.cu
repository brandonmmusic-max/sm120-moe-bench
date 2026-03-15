/**
 * SM120 Flash Attention — BF16 Forward Kernel v3 (MMA-accelerated)
 *
 * Uses mma.sync.m16n8k16 with empirically validated fragment layout.
 * Key insight: A fragment registers are interleaved row/row+8, NOT k/k+8.
 *   Ra0 = [A[g, 2t], A[g, 2t+1]]
 *   Ra1 = [A[g+8, 2t], A[g+8, 2t+1]]
 *   Ra2 = [A[g, 2t+8], A[g, 2t+9]]
 *   Ra3 = [A[g+8, 2t+8], A[g+8, 2t+9]]
 *
 * Tile: BLOCK_M=64, BLOCK_N=64, HEAD_DIM=128
 * 4 warps, each warp handles 16 rows of M via MMA
 * Double-buffered K/V via cp.async
 * Online softmax with FP32 accumulators
 * SMEM: Q(16KB) + K×2(32KB) + V×2(32KB) + P_staging(8KB) = 88KB < 99KB
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

// Each warp handles 16 rows (one MMA_M tile along M)
#define WARP_M MMA_M

// SMEM layout (all row-major, no swizzle for now)
#define Q_ELEMS (BLOCK_M * HEAD_DIM)          // 64×128 = 8192 bf16 = 16KB
#define KV_ELEMS (BLOCK_N * HEAD_DIM)         // 64×128 = 8192 bf16 = 16KB
#define P_ELEMS (BLOCK_M * BLOCK_N)           // 64×64 = 4096 bf16 = 8KB (P staging)
#define SMEM_BYTES ((Q_ELEMS + NUM_STAGES * 2 * KV_ELEMS + P_ELEMS) * 2)  // 88KB

// Helper: pack two bf16 into uint32
__device__ __forceinline__ uint32_t pack2(const __nv_bfloat16& a, const __nv_bfloat16& b) {
    uint32_t r;
    asm("mov.b32 %0, {%1, %2};" : "=r"(r) : "h"(*(const uint16_t*)&a), "h"(*(const uint16_t*)&b));
    return r;
}

// cp.async
__device__ __forceinline__ void cp_async_16B(void* s, const void* g) {
    uint32_t sa = static_cast<uint32_t>(__cvta_generic_to_shared(s));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(g));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n"); }
__device__ __forceinline__ void cp_async_wait_one() { asm volatile("cp.async.wait_group 1;\n"); }

// Warp reductions
__device__ __forceinline__ float warp_max(float v) {
    for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, m));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffff, v, m);
    return v;
}

// ============================================================================
// Load A fragment from SMEM using ldmatrix.x4 + register swap
// ldmatrix output: [top-left, top-right, bottom-left, bottom-right]
// MMA expects:     [top-left, bottom-left, top-right, bottom-right]
// Fix: swap frag[1] and frag[2] after ldmatrix
// ============================================================================
__device__ __forceinline__ void load_A_frag(
    uint32_t frag[4],
    const __nv_bfloat16* smem,  // row-major matrix
    int row_offset,              // base row for this warp
    int k_offset,                // which k-block (0, 16, 32, ...)
    int stride,                  // row stride in bf16 elements
    int lane
) {
    // ldmatrix.x4 thread-to-address mapping:
    //   sub = lane / 8 (0..3), sublane = lane % 8
    //   row = row_offset + (sub/2)*8 + sublane
    //   col = k_offset + (sub%2)*8
    int sub = lane / 8;
    int sublane = lane % 8;
    int row = row_offset + (sub / 2) * 8 + sublane;
    int col = k_offset + (sub % 2) * 8;

    uint32_t addr = static_cast<uint32_t>(
        __cvta_generic_to_shared(&smem[row * stride + col]));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3])
        : "r"(addr)
    );

    // Swap frag[1] and frag[2]: ldmatrix gives [TL,TR,BL,BR], MMA wants [TL,BL,TR,BR]
    uint32_t tmp = frag[1];
    frag[1] = frag[2];
    frag[2] = tmp;
}

// ============================================================================
// Load B fragment from SMEM
// B is K stored as [N, K] row-major (= col-major [K, N])
// Rb0=[K[g, 2t], K[g, 2t+1]], Rb1=[K[g, 2t+8], K[g, 2t+9]]
// Using pack2 (ldmatrix.x2 for B needs .trans or transposed storage)
// Keep pack2 for B since it's only 2 registers — low overhead
// ============================================================================
__device__ __forceinline__ void load_B_frag(
    uint32_t frag[2],
    const __nv_bfloat16* k_smem,  // K[N, K] row-major, stride = HEAD_DIM
    int n_offset,                  // which n-block (0, 8, 16, ...)
    int k_offset,                  // which k-block
    int stride,                    // row stride (HEAD_DIM for K, BLOCK_N for P)
    int lane
) {
    int g = lane / 4;
    int t = lane % 4;
    int n = n_offset + g;
    int k0 = k_offset + t * 2;
    int k1 = k_offset + t * 2 + 8;

    frag[0] = pack2(k_smem[n * stride + k0],     k_smem[n * stride + k0 + 1]);
    frag[1] = pack2(k_smem[n * stride + k1],     k_smem[n * stride + k1 + 1]);
}

// ============================================================================
// MMA: C += A × B
// ============================================================================
__device__ __forceinline__ void mma_m16n8k16(float c[4], uint32_t a[4], uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\n"
        "    {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

// ============================================================================
// Main kernel
// ============================================================================
__global__ void __launch_bounds__(BLOCK_SIZE, 1)
sm120_flash_attn_fwd_bf16(
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

    // SMEM allocation
    extern __shared__ char smem[];
    __nv_bfloat16* q_s = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* k_s[2], *v_s[2];
    k_s[0] = q_s + Q_ELEMS;
    v_s[0] = k_s[0] + KV_ELEMS;
    k_s[1] = v_s[0] + KV_ELEMS;
    v_s[1] = k_s[1] + KV_ELEMS;
    __nv_bfloat16* p_s = v_s[1] + KV_ELEMS;  // P staging buffer

    // Zero-fill Q SMEM first (for partial tiles at sequence end)
    for (int i = tid; i < Q_ELEMS; i += BLOCK_SIZE)
        q_s[i] = __float2bfloat16(0.0f);
    __syncthreads();

    // Load Q to SMEM
    for (int i = tid; i < Q_ELEMS / 8; i += BLOCK_SIZE) {
        int row = (i * 8) / HEAD_DIM;
        int col = (i * 8) % HEAD_DIM;
        int qr = m_start + row;
        if (qr < Sq)
            cp_async_16B(&q_s[row * HEAD_DIM + col], &q_ptr[qr * HEAD_DIM + col]);
    }
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();

    // Per-thread output accumulators
    // Each warp: 16 rows × HEAD_DIM cols
    // MMA output: c[0..3] → 2 rows × 2 cols per MMA tile
    // HEAD_DIM/MMA_N = 16 MMA tiles along output dim
    // 2 rows × 16 tiles × 2 cols = 64 floats per thread
    float o_acc[2][16][2];  // [row_half: g, g+8][n_tile][col_pair]
    float rowmax[2] = {-FLT_MAX, -FLT_MAX};
    float rowsum[2] = {0.0f, 0.0f};

    #pragma unroll
    for (int rh = 0; rh < 2; rh++)
        #pragma unroll
        for (int nt = 0; nt < 16; nt++)
            o_acc[rh][nt][0] = o_acc[rh][nt][1] = 0.0f;

    // KV loop
    const int num_kv = (Skv + BLOCK_N - 1) / BLOCK_N;

    // Persistent zero buffer in SMEM for zero-fill cp_async source
    // (cp_async needs a valid source address — use SMEM itself)
    __shared__ __nv_bfloat16 zero_buf[8];  // 16 bytes of zeros
    if (tid < 8) zero_buf[tid] = __float2bfloat16(0.0f);
    __syncthreads();

    auto load_kv = [&](int blk, int stage) {
        int ns = blk * BLOCK_N;
        for (int i = tid; i < KV_ELEMS / 8; i += BLOCK_SIZE) {
            int row = (i * 8) / HEAD_DIM;
            int col = (i * 8) % HEAD_DIM;
            int kvr = ns + row;
            if (kvr < Skv) {
                cp_async_16B(&k_s[stage][row * HEAD_DIM + col], &k_ptr[kvr * HEAD_DIM + col]);
                cp_async_16B(&v_s[stage][row * HEAD_DIM + col], &v_ptr[kvr * HEAD_DIM + col]);
            } else {
                // Zero-fill via cp_async from our zero buffer (SMEM→SMEM not valid)
                // Use regular store instead — cp_async_commit will wait for these too
                for (int j = 0; j < 8; j++) {
                    k_s[stage][(row * HEAD_DIM + col) + j] = __float2bfloat16(0.0f);
                    v_s[stage][(row * HEAD_DIM + col) + j] = __float2bfloat16(0.0f);
                }
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
        // Step 1: S = Q @ K^T via MMA
        // Q is [BLOCK_M, HEAD_DIM] in q_s
        // K is [BLOCK_N, HEAD_DIM] in k_s[cs]
        // S is [BLOCK_M, BLOCK_N] — this warp computes S[warp*16 : (warp+1)*16, :]
        //
        // S[16, 64] = Q[16, 128] @ K^T[128, 64]
        // = sum over k=0..127 in steps of 16:
        //   Q_tile[16, 16] @ K_tile^T[16, 8] for each of 8 N-tiles
        // ================================================================

        // Score accumulators: 2 row halves × (BLOCK_N/MMA_N=8) n-tiles × 2 cols
        float s_acc[2][8][2];
        #pragma unroll
        for (int rh = 0; rh < 2; rh++)
            #pragma unroll
            for (int nt = 0; nt < 8; nt++)
                s_acc[rh][nt][0] = s_acc[rh][nt][1] = 0.0f;

        // Iterate over K dimension in steps of MMA_K=16
        #pragma unroll
        for (int ki = 0; ki < HEAD_DIM / MMA_K; ki++) {
            // Load Q fragment for this k-block
            uint32_t q_frag[4];
            load_A_frag(q_frag, q_s, warp * WARP_M, ki * MMA_K, HEAD_DIM, lane);

            // For each N-tile of K
            #pragma unroll
            for (int ni = 0; ni < BLOCK_N / MMA_N; ni++) {
                // Load K fragment: K[N, K] with n_offset, k_offset
                uint32_t k_frag[2];
                load_B_frag(k_frag, k_s[cs], ni * MMA_N, ki * MMA_K, HEAD_DIM, lane);

                // MMA: accumulate S += Q × K^T
                // But we need separate accumulators per N-tile
                float s_tile[4] = {s_acc[0][ni][0], s_acc[0][ni][1],
                                   s_acc[1][ni][0], s_acc[1][ni][1]};
                mma_m16n8k16(s_tile, q_frag, k_frag);
                s_acc[0][ni][0] = s_tile[0]; s_acc[0][ni][1] = s_tile[1];
                s_acc[1][ni][0] = s_tile[2]; s_acc[1][ni][1] = s_tile[3];
            }
        }

        // Apply scale and mask invalid KV positions to -INF
        int kv_start = kv * BLOCK_N;
        #pragma unroll
        for (int rh = 0; rh < 2; rh++)
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                s_acc[rh][nt][0] *= scale;
                s_acc[rh][nt][1] *= scale;
                // Each thread owns score columns: nt*MMA_N + t*2 and +1
                // Global KV indices: kv_start + nt*8 + t*2 and +1
                int kv_idx0 = kv_start + nt * MMA_N + t * 2;
                int kv_idx1 = kv_idx0 + 1;
                if (kv_idx0 >= Skv) s_acc[rh][nt][0] = -FLT_MAX;
                if (kv_idx1 >= Skv) s_acc[rh][nt][1] = -FLT_MAX;
            }

        // ================================================================
        // Step 2: Online softmax with cross-thread reduction
        //
        // Each thread owns 16 of 64 score values per row (8 tiles × 2 cols).
        // Threads t=0..3 within each group g share the same row.
        // Must reduce max and sum across t=0..3 via warp shuffle.
        // Shuffle mask: XOR with 1 and 2 reduces across 4 threads.
        // ================================================================
        for (int rh = 0; rh < 2; rh++) {
            // Step 2a: find max across this thread's 16 values
            float thread_max = rowmax[rh];
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                thread_max = fmaxf(thread_max, s_acc[rh][nt][0]);
                thread_max = fmaxf(thread_max, s_acc[rh][nt][1]);
            }

            // Step 2b: reduce max across t=0..3 (same row g)
            // Threads sharing row g are at lanes g*4+0, g*4+1, g*4+2, g*4+3
            // XOR with 1 swaps adjacent threads, XOR with 2 swaps pairs
            float new_max = thread_max;
            new_max = fmaxf(new_max, __shfl_xor_sync(0xffffffff, new_max, 1));
            new_max = fmaxf(new_max, __shfl_xor_sync(0xffffffff, new_max, 2));

            // Step 2c: rescale old accumulators
            float rescale = __expf(rowmax[rh] - new_max);
            rowsum[rh] *= rescale;

            #pragma unroll
            for (int nt = 0; nt < 16; nt++) {
                o_acc[rh][nt][0] *= rescale;
                o_acc[rh][nt][1] *= rescale;
            }

            // Step 2d: compute exp and local sum
            float local_sum = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < 8; nt++) {
                float e0 = __expf(s_acc[rh][nt][0] - new_max);
                float e1 = __expf(s_acc[rh][nt][1] - new_max);
                s_acc[rh][nt][0] = e0;
                s_acc[rh][nt][1] = e1;
                local_sum += e0 + e1;
            }

            // Step 2e: reduce sum across t=0..3
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, 1);
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, 2);

            rowsum[rh] += local_sum;
            rowmax[rh] = new_max;
        }

        // ================================================================
        // Step 3: Write P to SMEM for P@V MMA
        // P[BLOCK_M, BLOCK_N] — this warp writes its 16 rows
        // MMA output layout: c[0]→(g, 2t), c[1]→(g, 2t+1), c[2]→(g+8, 2t), c[3]→(g+8, 2t+1)
        // ================================================================
        #pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            int col0 = nt * MMA_N + t * 2;
            int row0 = warp * WARP_M + g;
            int row1 = warp * WARP_M + g + 8;
            p_s[row0 * BLOCK_N + col0]     = __float2bfloat16(s_acc[0][nt][0]);
            p_s[row0 * BLOCK_N + col0 + 1] = __float2bfloat16(s_acc[0][nt][1]);
            p_s[row1 * BLOCK_N + col0]     = __float2bfloat16(s_acc[1][nt][0]);
            p_s[row1 * BLOCK_N + col0 + 1] = __float2bfloat16(s_acc[1][nt][1]);
        }
        __syncthreads();

        // ================================================================
        // Step 4: O += P @ V via MMA
        // P[BLOCK_M, BLOCK_N] in p_s, V[BLOCK_N, HEAD_DIM] in v_s[cs]
        // P is "A matrix" [M, K=BLOCK_N], V is "B matrix" [N=BLOCK_N, K=HEAD_DIM]
        // Wait — V is [BLOCK_N, HEAD_DIM] = [N_kv, D].
        // We want O[M, D] += P[M, BLOCK_N] @ V[BLOCK_N, D]
        // This is: A=P[M, K=BLOCK_N], B=V[N=BLOCK_N, D] → need B col-major
        // V[BLOCK_N, HEAD_DIM] row-major = col-major [HEAD_DIM, BLOCK_N]... no.
        //
        // Actually: for the P@V matmul:
        //   A = P[16, BLOCK_N] (this warp's rows), stride = BLOCK_N
        //   B = V stored as [BLOCK_N, HEAD_DIM] row-major
        //   We want C[16, HEAD_DIM] = P[16, BLOCK_N] @ V[BLOCK_N, HEAD_DIM]
        //
        //   Using mma.row.col: A row-major, B col-major
        //   V[BLOCK_N, HEAD_DIM] row-major IS col-major [HEAD_DIM, BLOCK_N]? No.
        //   Col-major B[K, N] means B[k, n] at index n*K + k.
        //   V[BLOCK_N, HEAD_DIM] row-major: V[bn, d] at index bn*HEAD_DIM + d.
        //   For B of the MMA: K=BLOCK_N, N=HEAD_DIM (output cols).
        //   Col-major B[K=BLOCK_N, N=HEAD_DIM]: B[k, n] at n*BLOCK_N + k.
        //   But V stores V[bn, d] at bn*HEAD_DIM + d.
        //   These DON'T match. V row-major ≠ B col-major.
        //
        //   Solution: use load_B_frag with V[BLOCK_N, HEAD_DIM] and swap indices.
        //   B[k, n] in col-major = V[k, n] where k=KV_idx, n=head_dim_idx
        //   V row-major: V[k, n] at k*HEAD_DIM + n
        //   For load_B_frag: we need V stored as [N=output_cols, K=BLOCK_N]
        //   which we don't have. So we treat V as the "K matrix" with stride HEAD_DIM.
        //
        //   The B fragment loads: B_in[n_offset+g, k_offset+2t] etc.
        //   If B_in = V with stride HEAD_DIM:
        //     B_in[ni*8+g, ki*16+2t] = V[ni*8+g, ki*16+2t]
        //   But for P@V, the K dimension is BLOCK_N and N dimension is HEAD_DIM.
        //   So the MMA's "n" = head_dim tiles, "k" = BLOCK_N tiles.
        //   load_B_frag(frag, V, n_offset=di*8, k_offset=ki*16, stride=HEAD_DIM, lane)
        //   reads V[di*8+g, ki*16+2t] which is V[head_dim_idx, kv_block_idx]
        //   But V is [BLOCK_N, HEAD_DIM] so V[row, col] = V[kv, dim].
        //   We're reading V[head_dim_idx, kv_block_idx] — that's V transposed!
        //
        //   Fix: we need to access V as V[kv_block_idx, head_dim_idx].
        //   The B fragment for col-major loads B_cm[k, n] = B_stored[n, k] when
        //   B_stored is row-major [N, K].
        //   So load_B_frag with V as B_stored[N=BLOCK_N, K=HEAD_DIM] and
        //   n_offset along BLOCK_N, k_offset along HEAD_DIM would give:
        //   B_cm[k, n] = V[n, k] — but that's V transposed!
        //
        //   We actually want: C[m, d] = sum_kv P[m, kv] * V[kv, d]
        //   This is: A=P, B_cm[kv, d] = V[kv, d]
        //   Col-major B[kv, d] stored as V[d, kv]? No...
        //
        //   OK let me just think of it simply:
        //   MMA computes C[m,n] = sum_k A[m,k] * B_cm[k,n]
        //   We want C[m,d] = sum_kv P[m,kv] * V[kv,d]
        //   So: A=P (stride BLOCK_N), k=kv, n=d
        //   B_cm[kv, d] needs to be col-major stored.
        //   Col-major B_cm[kv, d] = storage at d*K + kv = d*BLOCK_N + kv
        //   But V stores V[kv, d] at kv*HEAD_DIM + d (row-major)
        //
        //   So B_cm[kv, d] ≠ V[kv, d] in memory!
        //   The fix: load_B_frag needs to read from V using the col-major convention.
        //   For B_cm[k=kv, n=d]:
        //     fragment reads B_cm[2t, g] and B_cm[2t+1, g]
        //     where 2t is the k index (kv_offset + 2t) and g is the n index (d_offset + g)
        //     B_cm[kv, d] is stored at V[kv, d] = V[kv * HEAD_DIM + d]
        //     So: V[(kv_offset+2t) * HEAD_DIM + (d_offset+g)]
        //     and V[(kv_offset+2t+1) * HEAD_DIM + (d_offset+g)]
        // ================================================================

        // Iterate over K(=BLOCK_N) dimension in steps of 16
        #pragma unroll
        for (int ki = 0; ki < BLOCK_N / MMA_K; ki++) {
            // P fragment (A): P[warp*16, BLOCK_N], stride=BLOCK_N
            uint32_t p_frag[4];
            load_A_frag(p_frag, p_s, warp * WARP_M, ki * MMA_K, BLOCK_N, lane);

            // V fragment (B): V[BLOCK_N, HEAD_DIM] row-major
            // For each output HEAD_DIM tile:
            #pragma unroll
            for (int di = 0; di < HEAD_DIM / MMA_N; di++) {
                // B fragment: B_cm[kv, d] = V[kv, d]
                // kv_offset = ki * 16, d_offset = di * 8
                // Thread reads: V[(ki*16 + 2t), (di*8 + g)] and V[(ki*16 + 2t + 1), (di*8 + g)]
                //               V[(ki*16 + 2t + 8), (di*8 + g)] and V[(ki*16 + 2t + 9), (di*8 + g)]
                uint32_t v_frag[2];
                int vk0 = ki * MMA_K + t * 2;
                int vk1 = ki * MMA_K + t * 2 + 8;
                int vn = di * MMA_N + g;

                v_frag[0] = pack2(v_s[cs][vk0 * HEAD_DIM + vn],
                                  v_s[cs][(vk0 + 1) * HEAD_DIM + vn]);
                v_frag[1] = pack2(v_s[cs][vk1 * HEAD_DIM + vn],
                                  v_s[cs][(vk1 + 1) * HEAD_DIM + vn]);

                float o_tile[4] = {o_acc[0][di][0], o_acc[0][di][1],
                                   o_acc[1][di][0], o_acc[1][di][1]};
                mma_m16n8k16(o_tile, p_frag, v_frag);
                o_acc[0][di][0] = o_tile[0]; o_acc[0][di][1] = o_tile[1];
                o_acc[1][di][0] = o_tile[2]; o_acc[1][di][1] = o_tile[3];
            }
        }

        __syncthreads();
    }  // end KV loop

    // ================================================================
    // Normalize and write output
    // ================================================================
    __nv_bfloat16* o_ptr = O + head * Sq * HEAD_DIM;

    for (int rh = 0; rh < 2; rh++) {
        float inv = (rowsum[rh] > 0.0f) ? 1.0f / rowsum[rh] : 0.0f;
        int row = m_start + warp * WARP_M + g + rh * 8;

        if (row < Sq) {
            #pragma unroll
            for (int di = 0; di < 16; di++) {
                int col0 = di * MMA_N + t * 2;
                o_ptr[row * HEAD_DIM + col0]     = __float2bfloat16(o_acc[rh][di][0] * inv);
                o_ptr[row * HEAD_DIM + col0 + 1] = __float2bfloat16(o_acc[rh][di][1] * inv);
            }
        }
    }

    // LSE
    if (LSE && t == 0) {
        for (int rh = 0; rh < 2; rh++) {
            int row = m_start + warp * WARP_M + g + rh * 8;
            if (row < Sq)
                LSE[head * Sq + row] = rowmax[rh] + logf(rowsum[rh]);
        }
    }
}

// Host launch
extern "C" void sm120_flash_attn_forward(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    int batch, int Hq, int Hkv, int Sq, int Skv, int head_dim,
    cudaStream_t stream
) {
    float sc = 1.0f / sqrtf((float)head_dim);
    dim3 grid((Sq + BLOCK_M - 1) / BLOCK_M, batch * Hq);
    dim3 block(BLOCK_SIZE);
    cudaFuncSetAttribute(sm120_flash_attn_fwd_bf16,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
    sm120_flash_attn_fwd_bf16<<<grid, block, SMEM_BYTES, stream>>>(
        Q, K, V, O, L, Sq, Skv, Hq, Hkv, sc);
}
