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

template<typename scalar_t, int LOOPS>
__global__ void res_add_rms_norm_kernel(
  scalar_t* __restrict__ out,             // [num_tokens, hidden_size]
  scalar_t* __restrict__ out_hidden,             // [num_tokens, hidden_size]
  const scalar_t* __restrict__ residual,     // [num_tokens, hidden_size]
  const scalar_t* __restrict__ hidden,     // [num_tokens, hidden_size]
  const scalar_t* __restrict__ weight,    // [hidden_size]
  const float epsilon,
  const int num_tokens,
  const int hidden_size,
  const int chunk_offset) {
  __shared__ float s_variance[64];
  float variance = 0.0f;
  float reg_variance = 0.0f;
  scalar_t xval[LOOPS];
  scalar_t yval[LOOPS];
  scalar_t wval[LOOPS];
  //int cnt = 0;
  const int warpid = threadIdx.x/64;
  const int laneid = threadIdx.x%64;
  const int num_warps = blockDim.x/64;


  //for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
#pragma unroll
  for (int idx = 0; idx < LOOPS; idx ++) {
    const int idxl = idx*1024 + threadIdx.x;
    xval[idx] = hidden[blockIdx.x * hidden_size + idxl + chunk_offset];
    yval[idx] = residual[blockIdx.x * hidden_size + idxl + chunk_offset];
    wval[idx] = weight[idxl];
    const float x = (float) (xval[idx] + yval[idx]); 
    variance += x * x;
    xval[idx] = (scalar_t) x;
    //cnt++;
  }
  //cnt=0;
//  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
//#pragma unroll
//  for (int idx = 0; idx < LOOPS; idx ++) {
//    //const int idxl = idx*1024 + threadIdx.x;
//    //const float x = (float) (hidden[blockIdx.x * hidden_size + idx] + residual[blockIdx.x * hidden_size + idx]);
//    const float x = (float) (xval[idx] + yval[idx]); 
//    variance += x * x;
//    xval[idx] = (scalar_t) x;
//    //cnt++;
//  }
#pragma unroll
  for (int mask = 64/2; mask > 0; mask >>= 1) {
    variance += __shfl_xor(variance, mask);
  }
  if (laneid == 0) {
    s_variance[warpid] = variance;
  }
  __syncthreads();
  variance = (laneid < num_warps) ? s_variance[laneid] : 0.f;
#pragma unroll
  for (int mask = 64/2; mask > 0; mask >>= 1) {
    variance += __shfl_xor(variance, mask);
  }

  //variance = blockReduceSum<float>(variance);
  //reg_variance = rsqrtf(variance / hidden_size + epsilon);
  //if (warpid == 0) {
  //  s_variance[laneid] = rsqrtf(variance / hidden_size + epsilon);
  //}
  //__syncthreads();
  variance = rsqrtf(variance / hidden_size + epsilon);
  //if (threadIdx.x%64 == 0) {
  //    reg_variance = s_variance;
  //}
  //reg_variance = __shfl(reg_variance, 0);
  //cnt=0;
  //for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    //wval[cnt] = weight[idx];
    //xval[cnt] = out_hidden[blockIdx.x * hidden_size + idx];
    //cnt++;
  //}
  //cnt=0;
  //for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
#pragma unroll
  for (int idx = 0; idx < LOOPS; idx ++) {
    const int idxl = idx*1024 + threadIdx.x;
    //xval[cnt] = out_hidden[blockIdx.x * hidden_size + idx];
    out_hidden[blockIdx.x * hidden_size + idxl + chunk_offset] = xval[idx];
    float x = (float) xval[idx];
    //out[blockIdx.x * hidden_size + idx] = ((scalar_t) (x * s_variance[laneid])) * weight[idx];
    out[blockIdx.x * hidden_size + idxl + chunk_offset] = ((scalar_t) (x * variance)) * wval[idx];
    //cnt++;
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

#define LAUNCH_RESADD_RMSNORM_KERNEL(LOOPS)                                            \ 
  AT_DISPATCH_FLOATING_TYPES_AND2(                                               \  
    at::ScalarType::Half,                                                        \
    at::ScalarType::BFloat16,                                                    \
    hidden.scalar_type(),                                                        \
    "rms_norm_kernel",                                                           \    
    [&] {                                                                        \ 
      vllm::res_add_rms_norm_kernel<scalar_t,LOOPS><<<grid, block, 0, stream>>>(       \
        out.data_ptr<scalar_t>(),                                                \
        out_hidden.data_ptr<scalar_t>(),                                         \
        residual.data_ptr<scalar_t>(),                                           \
        hidden.data_ptr<scalar_t>(),                                             \
        weight.data_ptr<scalar_t>(),                                             \
        epsilon,                                                                 \
        num_tokens,                                                              \
        hidden_size,                                                             \  
        chunk_size*chunk*hidden_size);                                           \ 
    });                                                                          \  

void res_add_rms_norm(
  torch::Tensor& out,      // [num_tokens, hidden_size]
  torch::Tensor& out_hidden,      // [num_tokens, hidden_size]
  torch::Tensor& residual,    // [num_tokens, hidden_size]
  torch::Tensor& hidden,    // [num_tokens, hidden_size]
  torch::Tensor& weight,   // [hidden_size]
  float epsilon) {
  const int num_tokens = hidden.size(0);
  const int hidden_size = hidden.size(1);

  //dim3 block(std::min(hidden_size, 1024));
  dim3 block(1024);
  assert(hidden_size%1024==0);
  assert(hidden_size<=8192);
  const int num_loops = hidden_size/1024;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int chunk_size = 304;
  if (num_tokens > 800) {
      chunk_size = 800;
  }

  //const int num_chunks = (num_tokens+chunk_size-1)/chunk_size;
  //for (int chunk=0; chunk<num_chunks; chunk++) {
  for (int chunk=0; chunk<2; chunk++) {
      //int grid_size = chunk_size;
      //if (chunk == num_chunks-1) {
      //    grid_size = num_tokens - (num_chunks-1)*chunk_size;
      //}
      int grid_size = chunk_size;
      if (chunk==0) {
          if (num_tokens<chunk_size) {
              grid_size = num_tokens;
          }
      } else {
          if (num_tokens<chunk_size) {
              break;
          } else {
              grid_size=num_tokens-chunk_size;
          }
      }
  dim3 grid(grid_size);
  //AT_DISPATCH_FLOATING_TYPES_AND2(
  //  at::ScalarType::Half,
  //  at::ScalarType::BFloat16,
  //  hidden.scalar_type(),
  //  "rms_norm_kernel",
  //  [&] {
  //    vllm::res_add_rms_norm_kernel<scalar_t,LOOPS><<<grid, block, 0, stream>>>(
  //      out.data_ptr<scalar_t>(),
  //      out_hidden.data_ptr<scalar_t>(),
  //      residual.data_ptr<scalar_t>(),
  //      hidden.data_ptr<scalar_t>(),
  //      weight.data_ptr<scalar_t>(),
  //      epsilon,
  //      num_tokens,
  //      hidden_size,
  //      chunk_size*chunk*hidden_size);
  //  });
  switch (num_loops) {
    case 1:
  	LAUNCH_RESADD_RMSNORM_KERNEL(1);
	break;
    case 2:
  	LAUNCH_RESADD_RMSNORM_KERNEL(2);
	break;
    case 3:
  	LAUNCH_RESADD_RMSNORM_KERNEL(3);
	break;
    case 4:
  	LAUNCH_RESADD_RMSNORM_KERNEL(4);
	break;
    case 5:
  	LAUNCH_RESADD_RMSNORM_KERNEL(5);
	break;
    case 6:
  	LAUNCH_RESADD_RMSNORM_KERNEL(6);
	break;
    case 7:
  	LAUNCH_RESADD_RMSNORM_KERNEL(7);
	break;
    case 8:
  	LAUNCH_RESADD_RMSNORM_KERNEL(8);
	break;
    default:
      TORCH_CHECK(false, "Unsupported hidden size: ", hidden_size);
      break;
  }
   
  }
}
