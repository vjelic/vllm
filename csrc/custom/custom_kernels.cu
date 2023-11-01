#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <algorithm>

constexpr int CUDA_NUM_THREADS = 128;

constexpr int MAXIMUM_NUM_BLOCKS = 4096;

inline int GET_BLOCKS(const int N) {
      return std::max(std::min((N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS,
                             MAXIMUM_NUM_BLOCKS), 1);
}

// define the kernel function:
template <typename T>
__global__ void sum(T *a, T *b, T *c, int N) {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i <= N) {
            //c[i] = a[i] + b[i];
                float2 ftmp;
                ftmp.x = a[i];
                ftmp.y = b[i];
                //__half2 tmp;
                //tmp.x = __float2half(a[i]);
                //tmp.y = __float2half(b[i]);
                __half2 tmp = __float22half2_rn(ftmp);
                __half sum = __hadd(__low2half(tmp),__high2half(tmp));
                c[i] = __half2float(sum);
                  }
}

// define the kernel calling code:
template <typename T>
void AddGPUKernel(T *in_a, T *in_b, T *out_c, int N, cudaStream_t stream) {
      sum<T>
                <<<GET_BLOCKS(N), CUDA_NUM_THREADS, 0, stream>>>(in_a, in_b, out_c, N);

        cudaError_t err = cudaGetLastError();
          if (cudaSuccess != err)
                  throw std::runtime_error("CUDA kernel failed : " + std::to_string(err));
}

// instantiate the kernel template for T=float:
template void AddGPUKernel<float>(float *in_a, float *in_b, float *out_c, int N, cudaStream_t stream);
