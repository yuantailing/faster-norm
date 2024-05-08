#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace fast_norm_cuda {

template<typename T>
void rms_norm_fwd_cuda(T *output, T const *input, T const *weight, float eps, int64_t b, int64_t h, cudaStream_t stream);

template<typename T>
void rms_norm_bwd_cuda(T *grad_input, T *grad_weight, float *grad_weight_buffer, T const *input, T const *weight, T const *grad_output, float eps, int64_t b, int64_t h, cudaStream_t stream);

}
