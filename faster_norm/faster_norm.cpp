#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "faster_norm.hpp"

namespace faster_norm {

torch::Tensor rms_norm_fwd(torch::Tensor input, torch::Tensor weight, float eps) {
    if (input.dtype() != weight.dtype()) {
        throw std::invalid_argument(std::string("dtype mismatch: input ") + c10::toString(input.dtype().toScalarType()) + " vs weight " + c10::toString(weight.dtype().toScalarType()));
    }
    int64_t b = input.size(0);
    int64_t h = input.size(1);
    torch::Tensor output = torch::empty_like(input);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();
    if (input.dtype() == torch::kFloat16) {
        rms_norm_fwd_cuda(
            reinterpret_cast<half *>(output.data_ptr()),
            reinterpret_cast<half *>(input.data_ptr()),
            reinterpret_cast<half *>(weight.data_ptr()),
            eps, b, h, stream);
    } else if (input.dtype() == torch::kBFloat16) {
        rms_norm_fwd_cuda(
            reinterpret_cast<__nv_bfloat16 *>(output.data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(weight.data_ptr()),
            eps, b, h, stream);
    } else {
        throw std::invalid_argument("unsupported dtype " + std::string(c10::toString(input.dtype().toScalarType())));
    }
    return output;
}

torch::Tensor layer_norm_fwd(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, float eps) {
    if (input.dtype() != weight.dtype()) {
        throw std::invalid_argument(std::string("dtype mismatch: input ") + c10::toString(input.dtype().toScalarType()) + " vs weight " + c10::toString(weight.dtype().toScalarType()));
    }
    if (input.dtype() != bias.dtype()) {
        throw std::invalid_argument(std::string("dtype mismatch: input ") + c10::toString(input.dtype().toScalarType()) + " vs bias " + c10::toString(bias.dtype().toScalarType()));
    }
    int64_t b = input.size(0);
    int64_t h = input.size(1);
    torch::Tensor output = torch::empty_like(input);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream().stream();
    if (input.dtype() == torch::kFloat16) {
        layer_norm_fwd_cuda(
            reinterpret_cast<half *>(output.data_ptr()),
            reinterpret_cast<half *>(input.data_ptr()),
            reinterpret_cast<half *>(weight.data_ptr()),
            reinterpret_cast<half *>(bias.data_ptr()),
            eps, b, h, stream);
    } else if (input.dtype() == torch::kBFloat16) {
        layer_norm_fwd_cuda(
            reinterpret_cast<__nv_bfloat16 *>(output.data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(weight.data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(bias.data_ptr()),
            eps, b, h, stream);
    } else {
        throw std::invalid_argument("unsupported dtype " + std::string(c10::toString(input.dtype().toScalarType())));
    }
    return output;

}

std::tuple<torch::Tensor, torch::Tensor> rms_norm_bwd(torch::Tensor input, torch::Tensor weight, torch::Tensor grad_output, float eps) {
    if (input.dtype() != weight.dtype()) {
        throw std::invalid_argument(std::string("dtype mismatch: input ") + c10::toString(input.dtype().toScalarType()) + " vs weight " + c10::toString(weight.dtype().toScalarType()));
    }
    if (input.dtype() != grad_output.dtype()) {
        throw std::invalid_argument(std::string("dtype mismatch: input ") + c10::toString(input.dtype().toScalarType()) + " vs grad_output " + c10::toString(grad_output.dtype().toScalarType()));
    }
    int64_t b = input.size(0);
    int64_t h = input.size(1);
    bool requires_wgrad = weight.requires_grad();
    torch::Tensor grad_input = torch::empty_like(input);
    torch::Tensor grad_weight = torch::empty_like(weight);
    torch::Tensor grad_weight_buffer;
    if (requires_wgrad)
        grad_weight_buffer = torch::empty_like(input, torch::TensorOptions().dtype(torch::kFloat32));
    cudaStream_t stream = c10::cuda::getDefaultCUDAStream().stream();
    if (input.dtype() == torch::kFloat16) {
        rms_norm_bwd_cuda(
            reinterpret_cast<half *>(grad_input.data_ptr()),
            reinterpret_cast<half *>(grad_weight.data_ptr()),
            requires_wgrad ? reinterpret_cast<float *>(grad_weight_buffer.data_ptr()) : nullptr,
            reinterpret_cast<half const *>(input.data_ptr()),
            reinterpret_cast<half const *>(weight.data_ptr()),
            reinterpret_cast<half const *>(grad_output.data_ptr()),
            eps, b, h, requires_wgrad, stream);
    } else if (input.dtype() == torch::kBFloat16) {
        rms_norm_bwd_cuda(
            reinterpret_cast<__nv_bfloat16 *>(grad_input.data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(grad_weight.data_ptr()),
            requires_wgrad ? reinterpret_cast<float *>(grad_weight_buffer.data_ptr()) : nullptr,
            reinterpret_cast<__nv_bfloat16 const *>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const *>(weight.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const *>(grad_output.data_ptr()),
            eps, b, h, requires_wgrad, stream);
    } else {
        throw std::invalid_argument("unsupported dtype " + std::string(c10::toString(input.dtype().toScalarType())));
    }
    return {grad_input, grad_weight};
}

std::tuple<torch::Tensor, torch::Tensor> layer_norm_bwd(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, torch::Tensor grad_output, float eps) {
    if (input.dtype() != weight.dtype()) {
        throw std::invalid_argument(std::string("dtype mismatch: input ") + c10::toString(input.dtype().toScalarType()) + " vs weight " + c10::toString(weight.dtype().toScalarType()));
    }
    if (input.dtype() != bias.dtype()) {
        throw std::invalid_argument(std::string("dtype mismatch: input ") + c10::toString(input.dtype().toScalarType()) + " vs bias " + c10::toString(bias.dtype().toScalarType()));
    }
    if (input.dtype() != grad_output.dtype()) {
        throw std::invalid_argument(std::string("dtype mismatch: input ") + c10::toString(input.dtype().toScalarType()) + " vs grad_output " + c10::toString(grad_output.dtype().toScalarType()));
    }
    int64_t b = input.size(0);
    int64_t h = input.size(1);
    bool requires_wgrad = weight.requires_grad();
    torch::Tensor grad_input = torch::empty_like(input);
    torch::Tensor grad_weight = torch::empty_like(weight);
    torch::Tensor grad_weight_buffer;
    if (requires_wgrad)
        grad_weight_buffer = torch::empty_like(input, torch::TensorOptions().dtype(torch::kFloat32));
    cudaStream_t stream = c10::cuda::getDefaultCUDAStream().stream();
    if (input.dtype() == torch::kFloat16) {
        layer_norm_bwd_cuda(
            reinterpret_cast<half *>(grad_input.data_ptr()),
            reinterpret_cast<half *>(grad_weight.data_ptr()),
            requires_wgrad ? reinterpret_cast<float *>(grad_weight_buffer.data_ptr()) : nullptr,
            reinterpret_cast<half const *>(input.data_ptr()),
            reinterpret_cast<half const *>(weight.data_ptr()),
            reinterpret_cast<half const *>(bias.data_ptr()),
            reinterpret_cast<half const *>(grad_output.data_ptr()),
            eps, b, h, requires_wgrad, stream);
    } else if (input.dtype() == torch::kBFloat16) {
        layer_norm_bwd_cuda(
            reinterpret_cast<__nv_bfloat16 *>(grad_input.data_ptr()),
            reinterpret_cast<__nv_bfloat16 *>(grad_weight.data_ptr()),
            requires_wgrad ? reinterpret_cast<float *>(grad_weight_buffer.data_ptr()) : nullptr,
            reinterpret_cast<__nv_bfloat16 const *>(input.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const *>(weight.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const *>(bias.data_ptr()),
            reinterpret_cast<__nv_bfloat16 const *>(grad_output.data_ptr()),
            eps, b, h, requires_wgrad, stream);
    } else {
        throw std::invalid_argument("unsupported dtype " + std::string(c10::toString(input.dtype().toScalarType())));
    }
    return {grad_input, grad_weight};
}

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rms_norm_fwd", &faster_norm::rms_norm_fwd, "rms_norm forward");
  m.def("layer_norm_fwd", &faster_norm::layer_norm_fwd, "layer_norm forward");
  m.def("rms_norm_bwd", &faster_norm::rms_norm_bwd, "rms_norm backward");
  m.def("layer_norm_bwd", &faster_norm::layer_norm_bwd, "layer_norm backward");
}
