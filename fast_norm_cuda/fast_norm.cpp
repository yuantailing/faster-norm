#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "fast_norm.hpp"

namespace fast_norm_cuda {

torch::Tensor rms_norm_fwd(torch::Tensor input, torch::Tensor weight, float eps) {
    int64_t b = input.size(0);
    int64_t h = input.size(1);
    torch::Tensor output = torch::empty_like(input);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();
    rms_norm_fwd_cuda(
        reinterpret_cast<__nv_bfloat16 *>(output.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(input.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(weight.data_ptr()),
        eps, b, h, stream);
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> rms_norm_bwd(torch::Tensor input, torch::Tensor weight, torch::Tensor grad_output, float eps) {
    int64_t b = input.size(0);
    int64_t h = input.size(1);
    torch::Tensor grad_input = torch::empty_like(input);
    torch::Tensor grad_weight = torch::empty_like(weight);
    torch::Tensor grad_weight_buffer = torch::empty_like(input, torch::TensorOptions().dtype(torch::kFloat32));
    cudaStream_t stream = c10::cuda::getDefaultCUDAStream().stream();
    rms_norm_bwd_cuda(
        reinterpret_cast<__nv_bfloat16 *>(grad_input.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(grad_weight.data_ptr()),
        reinterpret_cast<float *>(grad_weight_buffer.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const *>(input.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const *>(weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const *>(grad_output.data_ptr()),
        eps, b, h, stream);
    return {grad_input, grad_weight};
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_norm_fwd", &fast_norm_cuda::rms_norm_fwd, "rms_norm forward");
  m.def("rms_norm_bwd", &fast_norm_cuda::rms_norm_bwd, "rms_norm backward");
}
