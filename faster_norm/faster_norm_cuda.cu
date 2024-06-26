#include "faster_norm.hpp"

#include <stdio.h>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace faster_norm {

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

template<typename T>
class PackTwo;

template<>
class PackTwo<__nv_bfloat16> {
public:
    using type = __nv_bfloat162;
};

template<>
class PackTwo<half> {
public:
    using type = half2;
};

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

template<int ROWS_PER_CTA, int BLOCK_DIM_X, int H, typename T>
__global__ void rms_norm_fwd_kernel(T *__restrict__ output, T const *__restrict__ input, T const *__restrict__ weight, float eps, int64_t b, int64_t h) {
    static_assert(H % (2 * BLOCK_DIM_X) == 0, "not implemented: ceil_div required");
    using T2 = typename PackTwo<T>::type;

    T2 frag_weight[H / 2 / BLOCK_DIM_X];
    for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
        int widx = i * BLOCK_DIM_X + threadIdx.x;
        if (widx * 2 < h) {
            frag_weight[i] = reinterpret_cast<T2 const *>(weight)[widx];
        }
    }

    for (int i_b = 0; i_b < ROWS_PER_CTA; i_b++) {
        int b_id = blockIdx.x * ROWS_PER_CTA + i_b;
        float sum_x2 = 0.f;

        T2 frag_input[H / 2 / BLOCK_DIM_X];
        for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
            int widx = i * BLOCK_DIM_X + threadIdx.x;
            if (widx * 2 < h) {
                T2 inp = reinterpret_cast<T2 const *>(input)[b_id * h / 2 + widx];
                sum_x2 += (float)inp.x * (float)inp.x + (float)inp.y * (float)inp.y;
                frag_input[i] = inp;
            }
        }

        sum_x2 = blockReduceSum(sum_x2);
        __shared__ float shared_multiplier;
        if (threadIdx.x == 0) {
            shared_multiplier = rsqrtf(sum_x2 / h + eps);
        }
        __syncthreads();
        float multiplier = shared_multiplier;

        for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
            int widx = i * BLOCK_DIM_X + threadIdx.x;
            if (widx * 2 < h) {
                T2 inp = frag_input[i];
                T2 w = frag_weight[i];
                T2 o;
                o.x = (float)inp.x * multiplier * (float)w.x;
                o.y = (float)inp.y * multiplier * (float)w.y;
                reinterpret_cast<T2 *>(output)[b_id * h / 2 + i * BLOCK_DIM_X + threadIdx.x] = o;
            }
        }
    }
}

template<int ROWS_PER_CTA, int BLOCK_DIM_X, int H, typename T>
__global__ void layer_norm_fwd_kernel(T *__restrict__ output, T const *__restrict__ input, T const *__restrict__ weight, T const *__restrict__ bias, float eps, int64_t b, int64_t h) {
    static_assert(H % (2 * BLOCK_DIM_X) == 0, "not implemented: ceil_div required");
    using T2 = typename PackTwo<T>::type;

    T2 frag_weight[H / 2 / BLOCK_DIM_X];
    T2 frag_bias[H / 2 / BLOCK_DIM_X];
    for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
        int widx = i * BLOCK_DIM_X + threadIdx.x;
        if (widx * 2 < h) {
            frag_weight[i] = reinterpret_cast<T2 const *>(weight)[widx];
            frag_bias[i] = reinterpret_cast<T2 const *>(bias)[widx];
        }
    }

    for (int i_b = 0; i_b < ROWS_PER_CTA; i_b++) {
        int b_id = blockIdx.x * ROWS_PER_CTA + i_b;
        float sum_x = 0.f;
        float sum_x2 = 0.f;

        T2 frag_input[H / 2 / BLOCK_DIM_X];
        for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
            int widx = i * BLOCK_DIM_X + threadIdx.x;
            if (widx * 2 < h) {
                T2 inp = reinterpret_cast<T2 const *>(input)[b_id * h / 2 + widx];
                sum_x += (float)inp.x + (float)inp.y;
                frag_input[i] = inp;
            }
        }
        sum_x = blockReduceSum(sum_x);
        __shared__ float shared_mean;
        if (threadIdx.x == 0) {
            shared_mean = sum_x / h;
        }
        __syncthreads();
        float mean = shared_mean;

        for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
            int widx = i * BLOCK_DIM_X + threadIdx.x;
            if (widx * 2 < h) {
                T2 inp = frag_input[i];
                sum_x2 += ((float)inp.x - mean) * ((float)inp.x - mean) + ((float)inp.y - mean) * ((float)inp.y - mean);
            }
        }

        sum_x2 = blockReduceSum(sum_x2);
        __shared__ float shared_multiplier;
        if (threadIdx.x == 0) {
            shared_multiplier = rsqrtf(sum_x2 / h + eps);
        }
        __syncthreads();
        float multiplier = shared_multiplier;

        for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
            int widx = i * BLOCK_DIM_X + threadIdx.x;
            if (widx * 2 < h) {
                T2 inp = frag_input[i];
                T2 w = frag_weight[i];
                T2 bi = frag_bias[i];
                T2 o;
                o.x = ((float)inp.x - mean) * multiplier * (float)w.x + (float)bi.x;
                o.y = ((float)inp.y - mean) * multiplier * (float)w.y + (float)bi.y;
                reinterpret_cast<T2 *>(output)[b_id * h / 2 + i * BLOCK_DIM_X + threadIdx.x] = o;
            }
        }
    }
}

template<int ROWS_PER_CTA, int BLOCK_DIM_X, int H, bool REQUIRES_WGRAD, typename T>
__global__ void rms_norm_bwd_kernel(T *__restrict__ grad_input, float *__restrict__ grad_weight_buffer, T const *__restrict__ input, T const *__restrict__ weight, T const *__restrict__ grad_output, float eps, int64_t b, int64_t h) {
    static_assert(H % (2 * BLOCK_DIM_X) == 0, "not implemented: ceil_div required");
    using T2 = typename PackTwo<T>::type;
    float rh = 1.f / h;

    float frag_grad_weight_buffer[H / BLOCK_DIM_X];
    if constexpr (REQUIRES_WGRAD) {
        memset(frag_grad_weight_buffer, 0, sizeof(frag_grad_weight_buffer));
    }

    T2 frag_weight[H / 2 / BLOCK_DIM_X];
    for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
        int widx = i * BLOCK_DIM_X + threadIdx.x;
        if (widx * 2 < h) {
            frag_weight[i] = reinterpret_cast<T2 const *>(weight)[widx];
        }
    }

    for (int i_b = 0; i_b < ROWS_PER_CTA; i_b++) {
        int b_id = blockIdx.x * ROWS_PER_CTA + i_b;

        float sum_x2 = 0.f;
        float sum_xdyw = 0.f;

        T2 frag_input[H / 2 / BLOCK_DIM_X];
        T2 frag_grad_out[H / 2 / BLOCK_DIM_X];
        for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
            int widx = i * BLOCK_DIM_X + threadIdx.x;
            if (widx * 2 < h) {
                int idx = b_id * h / 2 + i * BLOCK_DIM_X + threadIdx.x;
                T2 inp = reinterpret_cast<T2 const *>(input)[idx];
                T2 grad_out = reinterpret_cast<T2 const *>(grad_output)[idx];
                T2 w = frag_weight[i];
                sum_x2 += (float)inp.x * (float)inp.x + (float)inp.y * (float)inp.y;
                sum_xdyw += (float)inp.x * (float)grad_out.x * (float)w.x + (float)inp.y * (float)grad_out.y * (float)w.y;
                frag_input[i] = inp;
                frag_grad_out[i] = grad_out;
            }
        }

        sum_x2 = blockReduceSum(sum_x2);
        __syncthreads();
        sum_xdyw = blockReduceSum(sum_xdyw);
        __shared__ float shared_rnorm;
        __shared__ float shared_sum_xdyw;
        if (threadIdx.x == 0) {
            shared_rnorm = rsqrtf(sum_x2 * rh + eps);
            shared_sum_xdyw = sum_xdyw;
        }
        __syncthreads();
        float rnorm = shared_rnorm;
        sum_xdyw = shared_sum_xdyw;

        for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
            int widx = i * BLOCK_DIM_X + threadIdx.x;
            if (widx * 2 < h) {
                int idx = b_id * h / 2 + i * BLOCK_DIM_X + threadIdx.x;
                T2 inp = frag_input[i];
                T2 grad_out = frag_grad_out[i];
                T2 w = frag_weight[i];
                T2 grad_inp;
                grad_inp.x = rnorm * ((float)w.x * (float)grad_out.x - rnorm * rh * rnorm * sum_xdyw * (float)inp.x);
                grad_inp.y = rnorm * ((float)w.y * (float)grad_out.y - rnorm * rh * rnorm * sum_xdyw * (float)inp.y);
                reinterpret_cast<T2 *>(grad_input)[idx] = grad_inp;
                if constexpr (REQUIRES_WGRAD) {
                    frag_grad_weight_buffer[i * 2 + 0] += rnorm * (float)inp.x * (float)grad_out.x;
                    frag_grad_weight_buffer[i * 2 + 1] += rnorm * (float)inp.y * (float)grad_out.y;
                }
            }
        }
    }

    if constexpr (REQUIRES_WGRAD) {
        for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
            int widx = i * BLOCK_DIM_X + threadIdx.x;
            if (widx * 2 < h) {
                reinterpret_cast<float2 *>(grad_weight_buffer)[blockIdx.x * h / 2 + widx] =
                    reinterpret_cast<float2 const *>(frag_grad_weight_buffer)[i];
            }
        }
    }
}

template<int ROWS_PER_CTA, int BLOCK_DIM_X, int H, bool REQUIRES_WGRAD, typename T>
__global__ void layer_norm_bwd_kernel(T *__restrict__ grad_input, float *__restrict__ grad_weight_buffer, T const *__restrict__ input, T const *__restrict__ weight, T const *__restrict__ bias, T const *__restrict__ grad_output, float eps, int64_t b, int64_t h) {
    static_assert(H % (2 * BLOCK_DIM_X) == 0, "not implemented: ceil_div required");
    using T2 = typename PackTwo<T>::type;
    float rh = 1.f / h;

    float frag_grad_weight_buffer[H / BLOCK_DIM_X];
    if constexpr (REQUIRES_WGRAD) {
        memset(frag_grad_weight_buffer, 0, sizeof(frag_grad_weight_buffer));
    }

    T2 frag_weight[H / 2 / BLOCK_DIM_X];
    for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
        int widx = i * BLOCK_DIM_X + threadIdx.x;
        if (widx * 2 < h) {
            frag_weight[i] = reinterpret_cast<T2 const *>(weight)[widx];
        }
    }

    for (int i_b = 0; i_b < ROWS_PER_CTA; i_b++) {
        int b_id = blockIdx.x * ROWS_PER_CTA + i_b;

        float sum_x = 0.f;
        float sum_x2 = 0.f;
        float sum_dyw = 0.f;
        float sum_xdyw = 0.f;

        T2 frag_input[H / 2 / BLOCK_DIM_X];
        for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
            int widx = i * BLOCK_DIM_X + threadIdx.x;
            if (widx * 2 < h) {
                T2 inp = reinterpret_cast<T2 const *>(input)[b_id * h / 2 + widx];
                sum_x += (float)inp.x + (float)inp.y;
                frag_input[i] = inp;
            }
        }
        sum_x = blockReduceSum(sum_x);
        __shared__ float shared_mean;
        if (threadIdx.x == 0) {
            shared_mean = sum_x * rh;
        }
        __syncthreads();
        float mean = shared_mean;

        T2 frag_grad_out[H / 2 / BLOCK_DIM_X];
        for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
            int widx = i * BLOCK_DIM_X + threadIdx.x;
            if (widx * 2 < h) {
                int idx = b_id * h / 2 + i * BLOCK_DIM_X + threadIdx.x;
                T2 inp = frag_input[i];
                T2 grad_out = reinterpret_cast<T2 const *>(grad_output)[idx];
                T2 w = frag_weight[i];
                sum_x2 += ((float)inp.x - mean) * ((float)inp.x - mean) + ((float)inp.y - mean) * ((float)inp.y - mean);
                sum_dyw += (float)grad_out.x * (float)w.x + (float)grad_out.y * (float)w.y;
                sum_xdyw += ((float)inp.x - mean) * (float)grad_out.x * (float)w.x + ((float)inp.y - mean) * (float)grad_out.y * (float)w.y;
                frag_grad_out[i] = grad_out;
            }
        }

        sum_x2 = blockReduceSum(sum_x2);
        __syncthreads();
        sum_dyw = blockReduceSum(sum_dyw);
        __syncthreads();
        sum_xdyw = blockReduceSum(sum_xdyw);
        __shared__ float shared_rnorm;
        __shared__ float shared_sum_dyw;
        __shared__ float shared_sum_xdyw;
        if (threadIdx.x == 0) {
            shared_rnorm = rsqrtf(sum_x2 * rh + eps);
            shared_sum_dyw = sum_dyw;
            shared_sum_xdyw = sum_xdyw;
        }
        __syncthreads();
        float rnorm = shared_rnorm;
        sum_dyw = shared_sum_dyw;
        sum_xdyw = shared_sum_xdyw;

        for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
            int widx = i * BLOCK_DIM_X + threadIdx.x;
            if (widx * 2 < h) {
                int idx = b_id * h / 2 + i * BLOCK_DIM_X + threadIdx.x;
                T2 inp = frag_input[i];
                T2 grad_out = frag_grad_out[i];
                T2 w = frag_weight[i];
                T2 grad_inp;
                grad_inp.x = rnorm * (float)w.x * (float)grad_out.x - rnorm * rh * sum_dyw - (rnorm * rh * rnorm * rnorm * sum_xdyw) * ((float)inp.x - mean);
                grad_inp.y = rnorm * (float)w.y * (float)grad_out.y - rnorm * rh * sum_dyw - (rnorm * rh * rnorm * rnorm * sum_xdyw) * ((float)inp.y - mean);
                reinterpret_cast<T2 *>(grad_input)[idx] = grad_inp;
                if constexpr (REQUIRES_WGRAD) {
                    frag_grad_weight_buffer[i * 2 + 0] += rnorm * ((float)inp.x - mean) * (float)grad_out.x;
                    frag_grad_weight_buffer[i * 2 + 1] += rnorm * ((float)inp.y - mean) * (float)grad_out.y;
                }
            }
        }
    }

    if constexpr (REQUIRES_WGRAD) {
        for (int i = 0; i * 2 * BLOCK_DIM_X < H; i++) {
            int widx = i * BLOCK_DIM_X + threadIdx.x;
            if (widx * 2 < h) {
                reinterpret_cast<float2 *>(grad_weight_buffer)[blockIdx.x * h / 2 + widx] =
                    reinterpret_cast<float2 const *>(frag_grad_weight_buffer)[i];
            }
        }
    }
}

template<typename T>
__global__ void sum_axis_0_kernel(T *__restrict__ output, float const *__restrict__ input, int rows, int64_t h) {
    using T2 = typename PackTwo<T>::type;
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    float sum = 0.f;
    for (int i = warp; i < rows; i += 32) {
        int widx = blockIdx.x * 32 + lane;
        if (widx < h) {
            sum += input[i * h + blockIdx.x * 32 + lane];
        }
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
        int widx = blockIdx.x * 16 + lane;
        if (widx * 2 < h) {
            T2 o;
            o.x = shared_sum[0][lane * 2];
            o.y = shared_sum[0][lane * 2 + 1];
            reinterpret_cast<T2 *>(output)[blockIdx.x * 16 + lane] = o;
        }
    }
}

template<typename T>
void rms_norm_fwd_cuda(T *output, T const *input, T const *weight, float eps, int64_t b, int64_t h, cudaStream_t stream) {
    if (h % 2 != 0) {
        throw std::invalid_argument("no support odd h (" + std::to_string(h) + ")");
    }
    constexpr int ROWS_PER_CTA = 4;
#define SWITCH_H(H, BLOCK_DIM_X) \
    do { if (h <= (H)) { \
        rms_norm_fwd_kernel<ROWS_PER_CTA, BLOCK_DIM_X, H><<<b / ROWS_PER_CTA, BLOCK_DIM_X, 0, stream>>>(output, input, weight, eps, b, h); \
        if (cudaPeekAtLastError() != cudaSuccess) { fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(cudaPeekAtLastError()), __FILE__, __LINE__); abort(); } \
        return; \
    } } while (0);
    SWITCH_H(128, 64);  // Decrease BLOCK_DIM_X due to line is short
    SWITCH_H(256, 128);  // Decrease BLOCK_DIM_X due to line is short
    SWITCH_H(512, 256);  // Decrease BLOCK_DIM_X due to line is short
    SWITCH_H(1 * 1024, 512);
    SWITCH_H(2 * 1024, 512);
    SWITCH_H(4 * 1024, 512);
    SWITCH_H(6 * 1024, 512);
    SWITCH_H(8 * 1024, 512);
    SWITCH_H(12 * 1024, 512);
    SWITCH_H(16 * 1024, 256);  // Decrease BLOCK_DIM_X due to no enough registers
    throw std::invalid_argument("h is too large (" + std::to_string(h) + ")");
#undef SWITCH_H
}

template<typename T>
void layer_norm_fwd_cuda(T *output, T const *input, T const *weight, T const *bias, float eps, int64_t b, int64_t h, cudaStream_t stream) {
    if (h % 2 != 0) {
        throw std::invalid_argument("no support odd h (" + std::to_string(h) + ")");
    }
    constexpr int ROWS_PER_CTA = 4;
#define SWITCH_H(H, BLOCK_DIM_X) \
    do { if (h <= (H)) { \
        layer_norm_fwd_kernel<ROWS_PER_CTA, BLOCK_DIM_X, H><<<b / ROWS_PER_CTA, BLOCK_DIM_X, 0, stream>>>(output, input, weight, bias, eps, b, h); \
        if (cudaPeekAtLastError() != cudaSuccess) { fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(cudaPeekAtLastError()), __FILE__, __LINE__); abort(); } \
        return; \
    } } while (0)
    SWITCH_H(128, 64);  // Decrease BLOCK_DIM_X due to line is short
    SWITCH_H(256, 128);  // Decrease BLOCK_DIM_X due to line is short
    SWITCH_H(512, 256);  // Decrease BLOCK_DIM_X due to line is short
    SWITCH_H(1 * 1024, 512);
    SWITCH_H(2 * 1024, 512);
    SWITCH_H(4 * 1024, 512);
    SWITCH_H(6 * 1024, 512);
    SWITCH_H(8 * 1024, 512);
    SWITCH_H(12 * 1024, 512);
    SWITCH_H(16 * 1024, 256);  // Decrease BLOCK_DIM_X due to no enough registers
#undef SWITCH_H
}

template<typename T>
void rms_norm_bwd_cuda(T *grad_input, T *grad_weight, float *grad_weight_buffer, T const *input, T const *weight, T const *grad_output, float eps, int64_t b, int64_t h, bool requires_wgrad, cudaStream_t stream) {
    if (h % 2 != 0) {
        throw std::invalid_argument("no support odd h (" + std::to_string(h) + ")");
    }
    constexpr int ROWS_PER_CTA = 8;
#define SWITCH_H(H, BLOCK_DIM_X) \
    do { if (h <= (H)) { \
        BOOL_SWITCH(requires_wgrad, REQUIRES_WGRAD, [&] { \
            rms_norm_bwd_kernel<ROWS_PER_CTA, BLOCK_DIM_X, H, REQUIRES_WGRAD><<<b / ROWS_PER_CTA, BLOCK_DIM_X, 0, stream>>>(grad_input, grad_weight_buffer, input, weight, grad_output, eps, b, h); \
        }); \
        if (cudaPeekAtLastError() != cudaSuccess) { fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(cudaPeekAtLastError()), __FILE__, __LINE__); abort(); } \
        if (requires_wgrad) { \
            sum_axis_0_kernel<<<(H) / 32, 1024, 0, stream>>>(grad_weight, grad_weight_buffer, b / ROWS_PER_CTA, h); \
            if (cudaPeekAtLastError() != cudaSuccess) { fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(cudaPeekAtLastError()), __FILE__, __LINE__); abort(); } \
        } \
        return; \
    } } while (0)
    SWITCH_H(128, 64);  // Decrease BLOCK_DIM_X due to line is short
    SWITCH_H(256, 128);  // Decrease BLOCK_DIM_X due to line is short
    SWITCH_H(512, 256);  // Decrease BLOCK_DIM_X due to line is short
    SWITCH_H(1 * 1024, 512);
    SWITCH_H(2 * 1024, 512);
    SWITCH_H(4 * 1024, 512);
    SWITCH_H(6 * 1024, 512);
    SWITCH_H(8 * 1024, 512);
    SWITCH_H(12 * 1024, 512);
    SWITCH_H(16 * 1024, 256);  // Decrease BLOCK_DIM_X due to no enough registers
    throw std::invalid_argument("h is too large (" + std::to_string(h) + ")");
#undef SWITCH_H
}

template<typename T>
void layer_norm_bwd_cuda(T *grad_input, T *grad_weight, float *grad_weight_buffer, T const *input, T const *weight, T const *bias, T const *grad_output, float eps, int64_t b, int64_t h, bool requires_wgrad, cudaStream_t stream) {
    if (h % 2 != 0) {
        throw std::invalid_argument("no support odd h (" + std::to_string(h) + ")");
    }
    constexpr int ROWS_PER_CTA = 8;
#define SWITCH_H(H, BLOCK_DIM_X) \
    do { if (h <= (H)) { \
        BOOL_SWITCH(requires_wgrad, REQUIRES_WGRAD, [&] { \
            layer_norm_bwd_kernel<ROWS_PER_CTA, BLOCK_DIM_X, H, REQUIRES_WGRAD><<<b / ROWS_PER_CTA, BLOCK_DIM_X, 0, stream>>>(grad_input, grad_weight_buffer, input, weight, bias, grad_output, eps, b, h); \
        }); \
        if (cudaPeekAtLastError() != cudaSuccess) { fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(cudaPeekAtLastError()), __FILE__, __LINE__); abort(); } \
        if (requires_wgrad) { \
            sum_axis_0_kernel<<<(H) / 32, 1024, 0, stream>>>(grad_weight, grad_weight_buffer, b / ROWS_PER_CTA, h); \
            if (cudaPeekAtLastError() != cudaSuccess) { fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(cudaPeekAtLastError()), __FILE__, __LINE__); abort(); } \
        } \
        return; \
    } } while (0)
    SWITCH_H(128, 64);  // Decrease BLOCK_DIM_X due to line is short
    SWITCH_H(256, 128);  // Decrease BLOCK_DIM_X due to line is short
    SWITCH_H(512, 256);  // Decrease BLOCK_DIM_X due to line is short
    SWITCH_H(1 * 1024, 512);
    SWITCH_H(2 * 1024, 512);
    SWITCH_H(4 * 1024, 512);
    SWITCH_H(6 * 1024, 512);
    SWITCH_H(8 * 1024, 512);
    if (requires_wgrad) SWITCH_H(12 * 1024, 256);
    else SWITCH_H(12 * 1024, 512);
    SWITCH_H(16 * 1024, 256);  // Decrease BLOCK_DIM_X due to no enough registers
    throw std::invalid_argument("h is too large (" + std::to_string(h) + ")");
#undef SWITCH_H
}

template void rms_norm_fwd_cuda(__nv_bfloat16 *output, __nv_bfloat16 const *input, __nv_bfloat16 const *weight, float eps, int64_t b, int64_t h, cudaStream_t stream);
template void rms_norm_fwd_cuda(half *output, half const *input, half const *weight, float eps, int64_t b, int64_t h, cudaStream_t stream);

template void layer_norm_fwd_cuda(__nv_bfloat16 *output, __nv_bfloat16 const *input, __nv_bfloat16 const *weight, __nv_bfloat16 const *bias, float eps, int64_t b, int64_t h, cudaStream_t stream);
template void layer_norm_fwd_cuda(half *output, half const *input, half const *weight, half const *bias, float eps, int64_t b, int64_t h, cudaStream_t stream);

template void rms_norm_bwd_cuda(__nv_bfloat16 *grad_input, __nv_bfloat16 *grad_weight, float *grad_weight_buffer, __nv_bfloat16 const *input, __nv_bfloat16 const *weight, __nv_bfloat16 const *grad_output, float eps, int64_t b, int64_t h, bool requires_wgrad, cudaStream_t stream);
template void rms_norm_bwd_cuda(half *grad_input, half *grad_weight, float *grad_weight_buffer, half const *input, half const *weight, half const *grad_output, float eps, int64_t b, int64_t h, bool requires_wgrad, cudaStream_t stream);

template void layer_norm_bwd_cuda(__nv_bfloat16 *grad_input, __nv_bfloat16 *grad_weight, float *grad_weight_buffer, __nv_bfloat16 const *input, __nv_bfloat16 const *weight, __nv_bfloat16 const *bias, __nv_bfloat16 const *grad_output, float eps, int64_t b, int64_t h, bool requires_wgrad, cudaStream_t stream);
template void layer_norm_bwd_cuda(half *grad_input, half *grad_weight, float *grad_weight_buffer, half const *input, half const *weight, half const *bias, half const *grad_output, float eps, int64_t b, int64_t h, bool requires_wgrad, cudaStream_t stream);

}
