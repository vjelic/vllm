#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"
#include "dispatch_utils.h"

#ifdef USE_ROCM
  #include "quantization/fp8/amd/quant_utils.cuh"
#else
  #include "quantization/fp8/nvidia/quant_utils.cuh"
#endif

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
typedef __hip_bfloat16 __nv_bfloat16;
#endif

namespace vllm {

template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding(
    const scalar_t* __restrict__ arr, const scalar_t* __restrict__ cos_ptr,
    const scalar_t* __restrict__ sin_ptr, int rot_offset, int embed_dim,
    int &x_index, scalar_t &x, int &y_index, scalar_t &y) {
  scalar_t cos, sin;
  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;
    cos = VLLM_LDG(cos_ptr + x_index);
    sin = VLLM_LDG(sin_ptr + x_index);
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;
    cos = VLLM_LDG(cos_ptr + x_index / 2);
    sin = VLLM_LDG(sin_ptr + x_index / 2);
  }

  x = x * cos - y * sin;
  y = y * cos + x * sin;
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
inline __device__ void store_value_into_key_cache(
      const int head_size, const int num_kv_heads, 
      const int64_t block_idx, const int block_size, const int64_t block_offset
      const int x,
      cache_t* __restrict__ key_cache, int &idx, scalar_t &val) {
  const int head_idx = idx / head_size;
  const int head_offset = idx % head_size;
  const int x_idx = head_offset / x;
  const int x_offset = head_offset % x;

  const int64_t  tgt_key_idx = 
      block_idx * num_kv_heads * (head_size / x) * block_size * x +
      head_idx * (head_size / x) * block_size * x + x_idx * block_size * x +
      block_offset * x + x_offset;

  if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
    key_cache[tgt_key_idx] = val;
  } else {
    key_cache[tgt_key_idx] =
        fp8::scaled_convert<cache_t, scalar_t, kv_dt>(val, kv_scale);
  }
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
inline __device__ void store_value_into_value_cache(
      const int head_size, const int num_kv_heads, 
      const int64_t block_idx, const int block_size, const int64_t block_offset
      cache_t* __restrict__ key_cache, int idx, scalar_t val) {
  const int head_idx = idx / head_size;
  const int head_offset = idx % head_size;

  const int64_t tgt_value_idx =
      block_idx * num_kv_heads * head_size * block_size +
      head_idx * head_size * block_size + head_offset * block_size +
      block_offset;

  if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
    value_cache[tgt_value_idx] = val;
  } else {
    value_cache[tgt_value_idx] =
        fp8::scaled_convert<cache_t, scalar_t, kv_dt>(val, kv_scale);
  }


template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt, bool IS_NEOX>
__global__ void fused_rotary_embedding_and_reshape_cache_kernel(
        const scalar_t* __restrict__ query,  // [batch_size, seq_len, num_heads, head_size] or 
                                             // [num_tokens, num_heads, head_size]
        const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
        const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
        cache_t* __restrict__ key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
        cache_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size, block_size]
        const scalar_t* __restrict__ cos_sin_cache,  // [max_position, 2, rot_dim // 2]
        const int64_t* __restrict__ positions,  // [batch_size, seq_len] or [num_tokens]
        const int64_t* __restrict__ slot_mapping,  // [num_tokens]
        const int64_t query_stride, const int key_stride, const int value_stride, 
        const int num_heads, const int num_kv_heads, const int head_size, 
        const int rot_dim, const int block_size, const int x, 
        const float k_scale, const float v_scale) {
    
    // Each thread block is responsible for one token.
  const int token_idx = blockIdx.x;
  int64_t pos = positions[token_idx];
  const scalar_t* cache_ptr = cos_sin_cache + pos * rot_dim;

  const int embed_dim = rot_dim / 2;
  const scalar_t* cos_ptr = cache_ptr;
  const scalar_t* sin_ptr = cache_ptr + embed_dim;

  const int nq = num_heads * embed_dim;
  for (int i = threadIdx.x; i < nq; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int rot_offset = i % embed_dim;
    const int64_t token_head = token_idx * query_stride + head_idx * head_size;
    apply_token_rotary_embedding_and_cache<scalar_t, cache_t, IS_NEOX>(
        query + token_head, cos_ptr, sin_ptr, rot_offset, embed_dim);
  }

  const int64_t slot_idx = slot_mapping[token_idx];
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  const int nk = num_kv_heads * embed_dim;
  for (int i = threadIdx.x; i < nk; i += blockDim.x) {
    const int head_idx = i / embed_dim;
    const int rot_offset = i % embed_dim;
    const int64_t token_head = token_idx * key_stride + head_idx * head_size;

    int x_index, y_index;
    scalar_t x_value, y_value; 
    apply_token_rotary_embedding_and_cache<scalar_t, IS_NEOX>(
                                key + token_head, cos_ptr, sin_ptr, 
                                rot_offset, embed_dim, 
                                x_index, x_value, y_index, y_value);

    store_value_into_key_cache(head_size, num_kv_heads, block_idx, 
                                block_size, block_offset, 
                                key_cache, x_index, x_value);
    store_value_into_key_cache(head_size, num_kv_heads, block_idx, 
                                block_size, block_offset, 
                                key_cache, key_cache, y_index, y_value);
  }

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t src_value_idx = token_idx * value_stride + i;
    scalar_t value = value[src_value_idx];

    store_value_into_value_cache(head_size, num_kv_heads, block_idx, 
                                block_size, block_offset, x,
                                value_cache, i, value);
  }
}
} // namespace vllm


 // KV_T is the stored data type of kv-cache.
 // CACHE_T is the data type of key and value tensors.
 // KV_DTYPE is the real data type of kv-cache.
 // IS_NEOX
 #define CALL_ROTARY_EMBEDDING_RESHAPE_AND_CACHE(KV_T, CACHE_T, KV_DTYPE, IS_NEOX)          \
    vllm::fused_rotary_embedding_and_reshape_cache_kernel<KV_T, CACHE_T, KV_DTYPE, IS_NEOX> \
        <<<grid, block, 0, stream>>>(                                       \
                reinterpret_cast<KV_T*>(query.data_ptr()),                  \
                reinterpret_cast<KV_T*>(key.data_ptr()),                    \
                reinterpret_cast<KV_T*>(value.data_ptr()),                  \
                reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),           \
                reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),         \
                reinterpret_cast<KV_T*>(cos_sin_cache.data_ptr()),          \
                positions.data_ptr<int64_t>(),                              \
                slot_mapping.data_ptr<int64_t>(),                           \
                query_stride, key_stride, value_stride,                     \
                num_heads, num_kv_heads, head_size,                         \
                rot_dim, block_size, x, k_scale, v_scale);


  // The following macro is used to dispatch the conversion function based on
  // the data type of the key and value cache. The FN is a macro that calls a
  // function with template<typename scalar_t, typename cache_t,
  // Fp8KVCacheDataType kv_dt, IS_NEOX>.
  #define DISPATCH_ROPE_BY_KV_CACHE_DTYPE(SRC_DTYPE, KV_DTYPE, IS_NEOX, FN)           \
    if (KV_DTYPE == "auto") {                                                       \
      if (SRC_DTYPE == at::ScalarType::Float) {                                     \
        FN(float, float, vllm::Fp8KVCacheDataType::kAuto, IS_NEOX);                 \
      } else if (SRC_DTYPE == at::ScalarType::Half) {                               \
        FN(uint16_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto, IS_NEOX);           \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                           \
        FN(__nv_bfloat16, __nv_bfloat16, vllm::Fp8KVCacheDataType::kAuto, IS_NEOX); \
      } else {                                                                      \
        TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE);      \
      }                                                                             \
    } else {                                                                        \
      if (KV_DTYPE == "fp8" || KV_DTYPE == "fp8_e4m3") {                            \
        if (SRC_DTYPE == at::ScalarType::Float) {                                   \
          FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3, IS_NEOX);          \
        } else if (SRC_DTYPE == at::ScalarType::Half) {                             \
          FN(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3, IS_NEOX);       \
        } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                         \
          FN(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3, IS_NEOX);  \
        } else {                                                                    \
          TORCH_CHECK(false,                                                        \
                      "Unsupported input type of kv cache: ", SRC_DTYPE);           \
        }                                                                           \
      } else if (KV_DTYPE == "fp8_e5m2") {                                          \
        if (SRC_DTYPE == at::ScalarType::Float) {                                   \
          FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2, IS_NEOX);          \
        } else if (SRC_DTYPE == at::ScalarType::Half) {                             \
          FN(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2, IS_NEOX);       \
        } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                         \
          FN(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2, IS_NEOX);  \
        } else {                                                                    \
          TORCH_CHECK(false,                                                        \
                      "Unsupported input type of kv cache: ", SRC_DTYPE);           \
        }                                                                           \
      } else {                                                                      \
        TORCH_CHECK(false, "Unsupported data type of kv cache: ", KV_DTYPE);        \
      }                                                                             \
    }


void fused_rotary_embedding_and_reshape_cache(
        torch::Tensor& query,   // [batch_size, seq_len, num_heads * head_size] or
                                // [num_tokens, num_heads * head_size]
        torch::Tensor& key,     // [batch_size, seq_len, num_kv_heads * head_size] or
                                // [num_tokens, num_kv_heads * head_size]
        torch::Tensor& value,   // [num_tokens, num_heads, head_size]
        torch::Tensor& key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
        torch::Tensor& value_cache,   // [num_blocks, num_heads, head_size, block_size]
        const std::string& kv_cache_dtype,
        torch::Tensor& cos_sin_cache, // [max_position, rot_dim]
        torch::Tensor& positions,     // [batch_size, seq_len] or [num_tokens]
        torch::Tensor& slot_mapping,  // [num_tokens]
        const double k_scale,
        const double v_scale,
        bool is_neox) {
    int64_t num_tokens = query.numel() / query.size(-1);
    int rot_dim = cos_sin_cache.size(1);
    int head_size = key.size(2);
    int num_heads = query.size(-1) / head_size;
    int num_kv_heads = key.size(-1) / head_size;
    int64_t query_stride = query.stride(-2);
    int64_t key_stride = key.stride(-2);
    int64_t value_stride = value.stride(-2);

    int block_size = key_cache.size(3);
    int x = key_cache.size(4);

    dim3 grid(num_tokens);
    dim3 block(std::min<int64_t>(num_heads * rot_dim / 2, 512));
    const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    VLLM_DISPATCH_FLOATING_TYPES(query.scalar_type(), 
        "fused_rotary_embedding_and_reshape_cache_kernel", 
        [&] {
            if (is_neox) {
                DISPATCH_ROPE_BY_KV_CACHE_DTYPE(query.scalar_type(), 
                    kv_cache_dtype, true, CALL_ROTARY_EMBEDDING_RESHAPE_AND_CACHE)
            } else {
                DISPATCH_ROPE_BY_KV_CACHE_DTYPE(query.scalar_type(), 
                    kv_cache_dtype, false, CALL_ROTARY_EMBEDDING_RESHAPE_AND_CACHE)
            }
    });
}