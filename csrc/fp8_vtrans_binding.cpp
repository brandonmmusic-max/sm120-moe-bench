
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

extern "C" void sm120_flash_attn_fp8_vtrans_forward(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* O, float* L,
    int batch, int Hq, int Hkv, int Sq, int Skv, int hd, cudaStream_t stream
);

std::vector<torch::Tensor> fp8_vtrans_fwd(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, bool return_lse
) {
    TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kBFloat16 && Q.is_contiguous());
    TORCH_CHECK(K.is_cuda() && K.dtype() == torch::kBFloat16 && K.is_contiguous());
    TORCH_CHECK(V.is_cuda() && V.dtype() == torch::kBFloat16 && V.is_contiguous());

    const int batch = Q.size(0), Hq = Q.size(1), Sq = Q.size(2), hd = Q.size(3);
    const int Hkv = K.size(1), Skv = K.size(2);

    auto O = torch::empty_like(Q);
    torch::Tensor L;
    float* L_ptr = nullptr;
    if (return_lse) {
        L = torch::empty({batch * Hq, Sq}, torch::dtype(torch::kFloat32).device(Q.device()));
        L_ptr = L.data_ptr<float>();
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sm120_flash_attn_fp8_vtrans_forward(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
        L_ptr,
        batch, Hq, Hkv, Sq, Skv, hd, stream
    );

    if (return_lse) return {O, L.reshape({batch, Hq, Sq})};
    return {O};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fp8_vtrans_fwd, "FP8 V^T Flash Attention",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("return_lse") = false);
}
