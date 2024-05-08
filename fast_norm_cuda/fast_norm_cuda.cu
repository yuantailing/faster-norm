#include "fast_norm.hpp"

#include <stdio.h>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace fast_norm_cuda {

#define FINAL_MASK 0xffffffff

template<typename T>
__inline__ __device__ T warpReduceSum(T val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += (T)__shfl_xor_sync(FINAL_MASK, val, mask, 32);  //__shfl_sync bf16 return float when sm < 80
    return val;
}

/* Calculate the sum of all elements in a block */
template<typename T>
__inline__ __device__ T blockReduceSum(T val)
{
    static __shared__ T shared[32];
    int                 lane = threadIdx.x & 0x1f;
    int                 wid  = threadIdx.x >> 5;

    val = warpReduceSum<T>(val);

    if (lane == 0)
        shared[wid] = val;

    __syncthreads();

    // Modify from blockDim.x << 5 to blockDim.x / 32. to prevent
    // blockDim.x is not divided by 32
    val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : (T)(0.0f);
    val = warpReduceSum<T>(val);

    return val;
}

template<int ROWS_PER_CTA, int BLOCK_DIM_X, int H>
__global__ void rms_norm_fwd_kernel(__nv_bfloat16 *__restrict__ output, __nv_bfloat16 const *__restrict__ input, __nv_bfloat16 const *__restrict__ weight, float eps, int64_t b) {
    __nv_bfloat162 frag_weight[H / 2 / BLOCK_DIM_X];
    for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
        frag_weight[i] = reinterpret_cast<__nv_bfloat162 const *>(weight)[i * BLOCK_DIM_X + threadIdx.x];
    }

    for (int i_b = 0; i_b < ROWS_PER_CTA; i_b++) {
        int b_id = blockIdx.x * ROWS_PER_CTA + i_b;
        float sum_x2 = 0.f;

        __nv_bfloat162 frag_input[H / 2 / BLOCK_DIM_X];
        for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
            __nv_bfloat162 inp = reinterpret_cast<__nv_bfloat162 const *>(input)[b_id * H / 2 + i * BLOCK_DIM_X + threadIdx.x];
            sum_x2 += (float)inp.x * (float)inp.x + (float)inp.y * (float)inp.y;
            frag_input[i] = inp;
        }

        sum_x2 = blockReduceSum(sum_x2);
        __shared__ float shared_multiplier;
        if (threadIdx.x == 0) {
            shared_multiplier = rsqrtf(sum_x2 / H + eps);
        }
        __syncthreads();
        float multiplier = shared_multiplier;

        for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
            __nv_bfloat162 inp = frag_input[i];
            __nv_bfloat162 w = frag_weight[i];
            __nv_bfloat162 o;
            o.x = (float)inp.x * multiplier * (float)w.x;
            o.y = (float)inp.y * multiplier * (float)w.y;
            reinterpret_cast<__nv_bfloat162 *>(output)[b_id * H / 2 + i * BLOCK_DIM_X + threadIdx.x] = o;
        }
    }
}

template<int ROWS_PER_CTA, int BLOCK_DIM_X, int H>
__global__ void rms_norm_bwd_kernel(__nv_bfloat16 *__restrict__ grad_input, float *__restrict__ grad_weight_buffer, __nv_bfloat16 const *__restrict__ input, __nv_bfloat16 const *__restrict__ weight, __nv_bfloat16 const *__restrict__ grad_output, float eps, int64_t b) {
    float frag_grad_weight_buffer[H / BLOCK_DIM_X] = {0};

    for (int i_b = 0; i_b < ROWS_PER_CTA; i_b++) {
        int b_id = blockIdx.x * ROWS_PER_CTA + i_b;

        float sum_x2 = 0.f;
        float sum_xdyw = 0.f;
        for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
            int idx = b_id * H / 2 + i * BLOCK_DIM_X + threadIdx.x;
            int widx = i * BLOCK_DIM_X + threadIdx.x;
            __nv_bfloat162 inp = reinterpret_cast<__nv_bfloat162 const *>(input)[idx];
            __nv_bfloat162 grad_out = reinterpret_cast<__nv_bfloat162 const *>(grad_output)[idx];
            __nv_bfloat162 w = reinterpret_cast<__nv_bfloat162 const *>(weight)[widx];
            sum_x2 += (float)inp.x * (float)inp.x + (float)inp.y * (float)inp.y;
            sum_xdyw += (float)inp.x * (float)grad_out.x * (float)w.x + (float)inp.y * (float)grad_out.y * (float)w.y;
        }

        sum_x2 = blockReduceSum(sum_x2);
        __syncthreads();
        sum_xdyw = blockReduceSum(sum_xdyw);
        __shared__ float shared_rnorm;
        __shared__ float shared_sum_xdyw;
        if (threadIdx.x == 0) {
            shared_rnorm = rsqrtf(sum_x2 / H + eps);
            shared_sum_xdyw = sum_xdyw;
        }
        __syncthreads();
        float rnorm = shared_rnorm;
        sum_xdyw = shared_sum_xdyw;

        for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
            int idx = b_id * H / 2 + i * BLOCK_DIM_X + threadIdx.x;
            int widx = i * BLOCK_DIM_X + threadIdx.x;
            __nv_bfloat162 inp = reinterpret_cast<__nv_bfloat162 const *>(input)[idx];
            __nv_bfloat162 grad_out = reinterpret_cast<__nv_bfloat162 const *>(grad_output)[idx];
            __nv_bfloat162 w = reinterpret_cast<__nv_bfloat162 const *>(weight)[widx];
            __nv_bfloat162 grad_inp;
            grad_inp.x = rnorm * ((float)w.x * (float)grad_out.x - (float)inp.x * rnorm * rnorm * sum_xdyw / H);
            grad_inp.y = rnorm * ((float)w.y * (float)grad_out.y - (float)inp.y * rnorm * rnorm * sum_xdyw / H);
            reinterpret_cast<__nv_bfloat162 *>(grad_input)[idx] = grad_inp;
            frag_grad_weight_buffer[i * 2 + 0] += rnorm * (float)inp.x * (float)grad_out.x;
            frag_grad_weight_buffer[i * 2 + 1] += rnorm * (float)inp.y * (float)grad_out.y;
        }
    }

    for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
        reinterpret_cast<float2 *>(grad_weight_buffer)[blockIdx.x * H / 2 + i * BLOCK_DIM_X + threadIdx.x] =
            reinterpret_cast<float2 const *>(frag_grad_weight_buffer)[i];
    }
}

template<int H>
__global__ void sum_axis_0_kernel(__nv_bfloat16 *__restrict__ output, float const *__restrict__ input, int rows) {
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    float sum = 0.f;
    for (int i = warp; i < rows; i += 32) {
        sum += input[i * H + blockIdx.x * 32 + lane];
    }
    __shared__ float shared_sum[32][32];
    shared_sum[warp][lane ^ warp] = sum;
    __syncthreads();
    sum = 0.f;
    sum = shared_sum[lane][warp ^ lane];
    sum = warpReduceSum(sum);
    if (lane == 0)
        shared_sum[0][warp] = sum;
    __syncthreads();
    if (warp == 0 && lane < 16) {
        __nv_bfloat162 o;
        o.x = shared_sum[0][lane * 2];
        o.y = shared_sum[0][lane * 2 + 1];
        reinterpret_cast<__nv_bfloat162 *>(output)[blockIdx.x * 16 + lane] = o;
    }
}

void rms_norm_fwd_cuda(__nv_bfloat16 *output, __nv_bfloat16 const *input, __nv_bfloat16 const *weight, float eps, int64_t b, int64_t h, cudaStream_t stream) {
    constexpr int ROWS_PER_CTA = 4;
    constexpr int BLOCK_DIM_X = 512;
#define SWITCH_H(H) \
    if (h == H) { \
        rms_norm_fwd_kernel<ROWS_PER_CTA, BLOCK_DIM_X, H><<<b / ROWS_PER_CTA, BLOCK_DIM_X, 0, stream>>>(output, input, weight, eps, b); \
        if (cudaPeekAtLastError() != cudaSuccess) { fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(cudaPeekAtLastError()), __FILE__, __LINE__); abort(); } \
        return; \
    }
    SWITCH_H(4096)
    SWITCH_H(8192)
    SWITCH_H(12288)
    throw std::invalid_argument("unsupported h = " + std::to_string(h));
#undef SWITCH_H
}

void rms_norm_bwd_cuda(__nv_bfloat16 *grad_input, __nv_bfloat16 *grad_weight, float *grad_weight_buffer, __nv_bfloat16 const *input, __nv_bfloat16 const *weight, __nv_bfloat16 const *grad_output, float eps, int64_t b, int64_t h, cudaStream_t stream) {
    constexpr int ROWS_PER_CTA = 8;
    constexpr int BLOCK_DIM_X = 512;
#define SWITCH_H(H) \
    if (h == H) { \
        rms_norm_bwd_kernel<ROWS_PER_CTA, BLOCK_DIM_X, H><<<b / ROWS_PER_CTA, BLOCK_DIM_X, 0, stream>>>(grad_input, grad_weight_buffer, input, weight, grad_output, eps, b); \
        if (cudaPeekAtLastError() != cudaSuccess) { fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(cudaPeekAtLastError()), __FILE__, __LINE__); abort(); } \
        sum_axis_0_kernel<H><<<H / 32, 1024, 0, stream>>>(grad_weight, grad_weight_buffer, b / ROWS_PER_CTA); \
        if (cudaPeekAtLastError() != cudaSuccess) { fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(cudaPeekAtLastError()), __FILE__, __LINE__); abort(); } \
        return; \
    }
    SWITCH_H(4096)
    SWITCH_H(8192)
    SWITCH_H(12288)
    throw std::invalid_argument("unsupported h = " + std::to_string(h));
#undef SWITCH_H
}

}
