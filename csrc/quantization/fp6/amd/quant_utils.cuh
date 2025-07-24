#pragma once
#if defined(NDEBUG)
  #undef NDEBUG
  #include <assert.h>
  #define NDEBUG
#endif
#include </opt/rocm/include/hip/amd_detail/amd_hip_fp8.h>

#include </opt/rocm/include/hip/amd_detail/amd_hip_fp16.h>
#include </opt/rocm/include/hip/amd_detail/amd_hip_bf16.h>
#include </opt/rocm/include/hip/amd_detail/amd_hip_bfloat16.h>

#include "../../../attention/attention_dtypes.h"

#include </opt/rocm/include/hip/amd_detail/amd_hip_fp6.h>

namespace vllm {
#ifdef USE_ROCM

namespace fp6 {
  #ifdef ENABLE_FP8

// Use hardware cvt instruction for fp8 on rocm
template <typename fp8_type>
__device__ __forceinline__ fp8_type cvt_c10(float const r) {
  return {};
}

// __hip_fp8_e4m3 only exists starting in ROCm 6.3. The macro
// HIP_FP8_TYPE_OCP comes from the hip_fp8.h header and also makes
// its first appearance in ROCm 6.3. Since VLLM_DISPATCH_FP8_TYPES
// on ROCm instantiates both OCP and FNUZ kernels, we need to replace
// the new HW cvt with something reasonable that doesn't rely on the
// ROCm 6.3 feature. This allows compiling on ROCm 6.2 or newer.
template <>
__device__ __forceinline__ c10::Float8_e4m3fn cvt_c10(float const r) {
    #if HIP_FP8_TYPE_OCP
  return c10::Float8_e4m3fn(
      __hip_cvt_float_to_fp8(r, __hip_fp8_e4m3::__default_saturation,
                             __hip_fp8_e4m3::__default_interpret),
      c10::Float8_e4m3fn::from_bits());
    #else
  // Cast implemented by pytorch. Uses bit manipulation instead of HW cvt.
  // HW cvt above is faster when it is available (ROCm 6.3 or newer).
  return static_cast<c10::Float8_e4m3fn>(r);
    #endif
}

template <>
__device__ __forceinline__ c10::Float8_e4m3fnuz cvt_c10(float const r) {
  return c10::Float8_e4m3fnuz(
      __hip_cvt_float_to_fp8(r, __hip_fp8_e4m3_fnuz::__default_saturation,
                             __hip_fp8_e4m3_fnuz::__default_interpret),
      c10::Float8_e4m3fnuz::from_bits());
}

template <typename Tout, typename Tin>
__inline__ __device__ Tout vec_conversion(const Tin& x) {
  assert(false);
  return Tout{};
}

template <typename Tout, typename Tin>
__inline__ __device__ Tout scaled_vec_conversion(const Tin& x,
                                                 const float scale) {
  assert(false);
  return Tout{};
}

    #if HIP_FP8_TYPE_OCP
using fp8_type = __hip_fp8_e4m3;
using fp8x2_type = __hip_fp8x2_e4m3;
    #else
using fp8_type = __hip_fp8_e4m3_fnuz;
using fp8x2_type = __hip_fp8x2_e4m3_fnuz;
    #endif

// bfloat16 -> fp6
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, __hip_bfloat16>(
    const __hip_bfloat16& a, float scale) {
  const float val_float = static_cast<float>(a);

  const float scaled_val = val_float / scale;

  __hip_fp6_e2m3 fp6_val = __hip_fp6_e2m3(scaled_val);

  if (fp6_val.__x == 0b00100000) {
    fp6_val.__x = 0b00000000;
  }

  // printf("test1\n");

  return fp6_val.__x;
}

// fp6 -> bfloat
template <>
__inline__ __device__ __hip_bfloat16 scaled_vec_conversion<__hip_bfloat16, uint8_t>(
    const uint8_t& a, float scale) {
  __hip_fp6_e2m3 fp6_val;
  fp6_val.__x = a;
  float fVal = fp6_val;
  fVal *= scale;
  __hip_bfloat16 res = __hip_bfloat16(fVal);
  return res;
}

// fp6 -> float
template <>
__inline__ __device__ float scaled_vec_conversion<float, uint8_t>(
    const uint8_t& a, float scale) {
  __hip_fp6_e2m3 fp6_val;
  fp6_val.__x = a;
  float fVal = fp6_val;
  fVal *= scale;

  return fVal;
}

// fp6x2 -> float2
template <>
__inline__ __device__ float2
scaled_vec_conversion<float2, uint16_t>(const uint16_t& a, float scale) {
  float2 res;
  res.x = scaled_vec_conversion<float, uint8_t>((uint8_t)a, scale);
  res.y = scaled_vec_conversion<float, uint8_t>((uint8_t)(a >> 8U), scale);
  return res;
}

// fp6x4 -> float4
template <>
__inline__ __device__ Float4_
scaled_vec_conversion<Float4_, uint32_t>(const uint32_t& a, const float scale) {
  Float4_ res;
  res.x = scaled_vec_conversion<float2, uint16_t>((uint16_t)a, scale);
  res.y = scaled_vec_conversion<float2, uint16_t>((uint16_t)(a >> 16U), scale);
  return res;
}

// fp8x4 -> float4
template <>
__inline__ __device__ float4
scaled_vec_conversion<float4, uint32_t>(const uint32_t& a, float scale) {
  Float4_ res = scaled_vec_conversion<Float4_, uint32_t>(a, scale);
  return {res.x.x, res.x.y, res.y.x, res.y.y};
}

//todo: fix
// bf16x8 -> fp6x8
template <>
__inline__ __device__ uint2
scaled_vec_conversion<uint2, bf16_8_t>(const bf16_8_t& a, float scale) {
  uint2 res;
  res.x = scaled_vec_conversion<uint32_t, bf16_4_t>({a.x, a.y}, scale);
  res.y = scaled_vec_conversion<uint32_t, bf16_4_t>({a.z, a.w}, scale);
  return res;
}

  #endif  // ENABLE_FP8

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout convert(const Tin& x) {
  #ifdef ENABLE_FP8
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    return vec_conversion<Tout, Tin>(x);
  }
  #endif
  assert(false);
  return {};  // Squash missing return statement warning
}

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout scaled_convert(const Tin& x, const float scale) {
  #ifdef ENABLE_FP8
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    // printf("llamado!\n");
    return scaled_vec_conversion<Tout, Tin>(x, scale);
  }
  #endif
  assert(false);
  return {};  // Squash missing return statement warning
}

  // The following macro is used to dispatch the conversion function based on
  // the data type of the key and value cache. The FN is a macro that calls a
  // function with template<typename scalar_t, typename cache_t,
  // Fp8KVCacheDataType kv_dt>.
  #define DISPATCH_BY_KV_CACHE_DTYPE_FP6(SRC_DTYPE, KV_DTYPE, FN)              \
    if (KV_DTYPE == "auto") {                                                  \
      if (SRC_DTYPE == at::ScalarType::Float) {                                \
        FN(float, float, vllm::Fp8KVCacheDataType::kAuto);                     \
      } else if (SRC_DTYPE == at::ScalarType::Half) {                          \
        FN(uint16_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto);               \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                      \
        FN(__nv_bfloat16, __nv_bfloat16, vllm::Fp8KVCacheDataType::kAuto);     \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE); \
      }                                                                        \
    } else {                                                                   \
      if (KV_DTYPE == "fp8" || KV_DTYPE == "fp8_e4m3" || KV_DTYPE == "fp6") {  \
        if (SRC_DTYPE == at::ScalarType::Float) {                              \
          FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);              \
        } else if (SRC_DTYPE == at::ScalarType::Half) {                        \
          FN(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);           \
        } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                    \
          FN(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);      \
        } else {                                                               \
          TORCH_CHECK(false,                                                   \
                      "Unsupported input type of kv cache: ", SRC_DTYPE);      \
        }                                                                      \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported data type of kv cache: ", KV_DTYPE);   \
      }                                                                        \
    }

}  // namespace fp6
#endif  // USE_ROCM
}  // namespace vllm
