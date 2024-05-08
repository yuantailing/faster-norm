#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace fast_norm_cuda {

void rms_norm_fwd_cuda(__nv_bfloat16 *output, __nv_bfloat16 const *input, __nv_bfloat16 const *weight, float eps, int64_t b, int64_t h, cudaStream_t stream);

void rms_norm_bwd_cuda(__nv_bfloat16 *grad_input, __nv_bfloat16 *grad_weight, float *grad_weight_buffer, __nv_bfloat16 const *input, __nv_bfloat16 const *weight, __nv_bfloat16 const *grad_output, float eps, int64_t b, int64_t h, cudaStream_t stream);

}
