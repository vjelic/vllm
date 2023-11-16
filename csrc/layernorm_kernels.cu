#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "reduction_utils.cuh"

namespace vllm {

// TODO(woosuk): Further optimize this kernel.
template<typename scalar_t>
__global__ void rms_norm_kernel(
  scalar_t* __restrict__ out,             // [num_tokens, hidden_size]
  const scalar_t* __restrict__ input,     // [num_tokens, hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float) input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float) input[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] = ((scalar_t) (x * s_variance)) * weight[idx];
  }
}

template<typename scalar_t>
__global__ void res_add_rms_norm_kernel(
  scalar_t* __restrict__ out,             // [num_tokens, hidden_size]
  scalar_t* __restrict__ out_hidden,             // [num_tokens, hidden_size]
  const scalar_t* __restrict__ residual,     // [num_tokens, hidden_size]
  const scalar_t* __restrict__ hidden,     // [num_tokens, hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float) (hidden[blockIdx.x * hidden_size + idx] + residual[blockIdx.x * hidden_size + idx]);
    variance += x * x;
    out_hidden[blockIdx.x * hidden_size + idx] = (scalar_t) x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float) out_hidden[blockIdx.x * hidden_size + idx];
    out[blockIdx.x * hidden_size + idx] = ((scalar_t) (x * s_variance)) * weight[idx];
  }
}
} // namespace vllm

void rms_norm(
  torch::Tensor& out,      // [num_tokens, hidden_size]
  torch::Tensor& input,    // [num_tokens, hidden_size]
  torch::Tensor& weight,   // [hidden_size]
  float epsilon) {
  int num_tokens = input.size(0);
  int hidden_size = input.size(1);

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    input.scalar_type(),
    "rms_norm_kernel",
    [&] {
      vllm::rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        epsilon,
        num_tokens,
        hidden_size);
    });
}

void res_add_rms_norm(
  torch::Tensor& out,      // [num_tokens, hidden_size]
  torch::Tensor& out_hidden,      // [num_tokens, hidden_size]
  torch::Tensor& residual,    // [num_tokens, hidden_size]
  torch::Tensor& hidden,    // [num_tokens, hidden_size]
  torch::Tensor& weight,   // [hidden_size]
  float epsilon) {
  int num_tokens = hidden.size(0);
  int hidden_size = hidden.size(1);

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    hidden.scalar_type(),
    "rms_norm_kernel",
    [&] {
      vllm::res_add_rms_norm_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<scalar_t>(),
        out_hidden.data_ptr<scalar_t>(),
        residual.data_ptr<scalar_t>(),
        hidden.data_ptr<scalar_t>(),
        weight.data_ptr<scalar_t>(),
        epsilon,
        num_tokens,
        hidden_size);
    });
}
