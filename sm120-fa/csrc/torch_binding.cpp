/**
 * PyTorch C++ binding for SM120 Flash Attention
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

// Forward declaration of CUDA launch function
extern "C" void sm120_flash_attn_forward(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    __nv_bfloat16* O,
    float* L,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int seq_len_q,
    int seq_len_kv,
    int head_dim,
    cudaStream_t stream
);

/**
 * PyTorch-callable flash attention forward
 *
 * Args:
 *   Q: [batch, num_q_heads, seq_len_q, head_dim] BF16
 *   K: [batch, num_kv_heads, seq_len_kv, head_dim] BF16
 *   V: [batch, num_kv_heads, seq_len_kv, head_dim] BF16
 *   return_lse: if true, also return log-sum-exp
 *
 * Returns:
 *   O: [batch, num_q_heads, seq_len_q, head_dim] BF16
 *   L: [batch, num_q_heads, seq_len_q] FP32 (optional)
 */
std::vector<torch::Tensor> sm120_flash_attn_fwd(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool return_lse
) {
    // Input validation
    TORCH_CHECK(Q.is_cuda(), "Q must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(V.is_cuda(), "V must be a CUDA tensor");
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be BFloat16");
    TORCH_CHECK(K.dtype() == torch::kBFloat16, "K must be BFloat16");
    TORCH_CHECK(V.dtype() == torch::kBFloat16, "V must be BFloat16");
    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");
    TORCH_CHECK(Q.dim() == 4, "Q must be 4D [batch, heads, seq, dim]");
    TORCH_CHECK(K.dim() == 4, "K must be 4D [batch, heads, seq, dim]");
    TORCH_CHECK(V.dim() == 4, "V must be 4D [batch, heads, seq, dim]");

    const int batch = Q.size(0);
    const int num_q_heads = Q.size(1);
    const int seq_len_q = Q.size(2);
    const int head_dim = Q.size(3);
    const int num_kv_heads = K.size(1);
    const int seq_len_kv = K.size(2);

    TORCH_CHECK(head_dim == 128, "head_dim must be 128 (got " + std::to_string(head_dim) + ")");
    TORCH_CHECK(K.size(3) == head_dim, "K head_dim mismatch");
    TORCH_CHECK(V.size(3) == head_dim, "V head_dim mismatch");
    TORCH_CHECK(V.size(2) == seq_len_kv, "V seq_len must match K seq_len");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads (GQA)");

    // Allocate output
    auto O = torch::empty_like(Q);

    // Reshape to [batch * heads, seq, dim] for kernel
    auto Q_flat = Q.reshape({batch * num_q_heads, seq_len_q, head_dim});
    auto K_flat = K.reshape({batch * num_kv_heads, seq_len_kv, head_dim});
    auto V_flat = V.reshape({batch * num_kv_heads, seq_len_kv, head_dim});
    auto O_flat = O.reshape({batch * num_q_heads, seq_len_q, head_dim});

    // Optional LSE output
    torch::Tensor L;
    float* L_ptr = nullptr;
    if (return_lse) {
        L = torch::empty({batch * num_q_heads, seq_len_q},
                         torch::dtype(torch::kFloat32).device(Q.device()));
        L_ptr = L.data_ptr<float>();
    }

    // Launch kernel
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sm120_flash_attn_forward(
        reinterpret_cast<const __nv_bfloat16*>(Q_flat.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K_flat.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V_flat.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(O_flat.data_ptr()),
        L_ptr,
        batch, num_q_heads, num_kv_heads,
        seq_len_q, seq_len_kv, head_dim,
        stream
    );

    if (return_lse) {
        return {O, L.reshape({batch, num_q_heads, seq_len_q})};
    }
    return {O};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sm120_flash_attn_fwd,
          "SM120 Flash Attention forward (BF16)",
          py::arg("Q"), py::arg("K"), py::arg("V"),
          py::arg("return_lse") = false);
}
