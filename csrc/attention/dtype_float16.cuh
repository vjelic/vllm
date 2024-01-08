/*
 * Adapted from https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * and https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention_utils.h
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "attention_generic.cuh"
#include "dtype_float32.cuh"

#include <stdint.h>
#include <cuda_fp16.h>

namespace vllm {

// Define custom FP32 vector data types.
struct Half4_ {
  __half2 x;
  __half2 y;
};

struct Half8_ {
  __half2 x;
  __half2 y;
  __half2 z;
  __half2 w;
};

// FP16 vector types for Q, K, V.
template<>
struct Vec<__half, 1> {
  using Type = __half;
};
template<>
struct Vec<__half, 2> {
  using Type = __half2;
};
template<>
struct Vec<__half, 4> {
  using Type = Half4_;
};
template<>
struct Vec<__half, 8> {
  using Type = Half8_;
};

// FP32 accumulator vector types corresponding to Vec.
template<>
struct FloatVec<__half> {
  using Type = float;
};
template<>
struct FloatVec<__half2> {
  using Type = float2;
};
template<>
struct FloatVec<Half4_> {
  using Type = Float4_;
};
template<>
struct FloatVec<Half8_> {
  using Type = Float8_;
};

// Utility functions for type conversions.
inline __device__ __half2 h0_h0(__half a) {
  __half2 b;
  //asm volatile("mov.b32 %0, {%1, %1};" : "=r"(b) : "h"(a));
  b = __half2half2(a); 
  return b;
}

inline __device__ float half_to_float(__half h) {
  float f;
  //asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
  f = __half2float(h);
  return f;
}

inline __device__ float2 half2_to_float2(__half2 v) {
  //uint16_t lo, hi;
  //asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(v));
  //return make_float2(half_to_float(lo), half_to_float(hi));
  return __half22float2(v);
}

inline __device__ __half float_to_half(float f) {
  //union {
  //  uint32_t u32;
  //  uint16_t u16[2];
  //} tmp;
  //union {
  //  __half   h16;
  //  uint16_t u16;
  //} tmp;
  //asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f));
  //return tmp.u16[0];
  //return uint16_t(static_cast<__half>(f));
  //tmp.h16 = __float2half(f); 
  //return reinterpret_cast<uint16_t&>(tmp);
  //return tmp.u16;
    return __float2half(f);
}

inline __device__ __half2 float2_to_half2(float2 f) {
  //union {
  //  uint32_t u32;
  //  __half2  h16x2;
  //} tmp;

//#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
//  asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(tmp.u32) : "f"(f.y), "f"(f.x));
//#else
//  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f.x));
//  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[1]) : "f"(f.y));
//#endif
  //tmp.h16x2 = __float22half2_rn(f);
  //return tmp.u32;
    return __float22half2_rn(f);
}

// Vector addition.
inline __device__ __half add(__half a, __half b) {
  //uint16_t c;
  //asm volatile("add.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
  //return c;
    return __hadd(a,b);
}

inline __device__ __half2 add(__half2 a, __half2 b) {
  //uint32_t c;
  //asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
  //return c;
    return __hadd2(a,b);
}

inline __device__ Half4_ add(Half4_ a, Half4_ b) {
  Half4_ c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

inline __device__ Half8_ add(Half8_ a, Half8_ b) {
  Half8_ c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}

inline __device__ float2 add(__half2 a, float2 fb) {
  float2 fa = half2_to_float2(a);
  return add(fa, fb);
}

inline __device__ Float4_ add(Half4_ a, Float4_ fb) {
  Float4_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  return fc;
}

inline __device__ Float8_ add(Half8_ a, Float8_ fb) {
  Float8_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  fc.z = add(a.z, fb.z);
  fc.w = add(a.w, fb.w);
  return fc;
}

// Vector multiplication.
template<>
inline __device__ __half mul(__half a, __half b) {
  //uint16_t c;
  //asm volatile("mul.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
  //return c;
    return __hmul(a,b);
}

template<>
inline __device__ __half2 mul(__half2 a, __half2 b) {
  //uint32_t c;
  //asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
  //return c;
    return __hmul2(a,b);
}

template<>
inline __device__ __half2 mul(__half a, __half2 b) {
  return mul<__half2, __half2, __half2>(h0_h0(a), b);
}

template<>
inline __device__ Half4_ mul(Half4_ a, Half4_ b) {
  Half4_ c;
  c.x = mul<__half2, __half2, __half2>(a.x, b.x);
  c.y = mul<__half2, __half2, __half2>(a.y, b.y);
  return c;
}

template<>
inline __device__ Half4_ mul(__half a, Half4_ b) {
  __half2 s = h0_h0(a);
  Half4_ c;
  c.x = mul<__half2, __half2, __half2>(s, b.x);
  c.y = mul<__half2, __half2, __half2>(s, b.y);
  return c;
}

template<>
inline __device__ Half8_ mul(Half8_ a, Half8_ b) {
  Half8_ c;
  c.x = mul<__half2, __half2, __half2>(a.x, b.x);
  c.y = mul<__half2, __half2, __half2>(a.y, b.y);
  c.z = mul<__half2, __half2, __half2>(a.z, b.z);
  c.w = mul<__half2, __half2, __half2>(a.w, b.w);
  return c;
}

template<>
inline __device__ Half8_ mul(__half a, Half8_ b) {
  __half2 s = h0_h0(a);
  Half8_ c;
  c.x = mul<__half2, __half2, __half2>(s, b.x);
  c.y = mul<__half2, __half2, __half2>(s, b.y);
  c.z = mul<__half2, __half2, __half2>(s, b.z);
  c.w = mul<__half2, __half2, __half2>(s, b.w);
  return c;
}

template<>
inline __device__ float mul(__half a, __half b) {
  float fa = half_to_float(a);
  float fb = half_to_float(b);
  return fa * fb;
}

template<>
inline __device__ float2 mul(__half2 a, __half2 b) {
  float2 fa = half2_to_float2(a);
  float2 fb = half2_to_float2(b);
  return mul<float2, float2, float2>(fa, fb);
}

template<>
inline __device__ float2 mul(__half a, __half2 b) {
  return mul<float2, __half2, __half2>(h0_h0(a), b);
}

template<>
inline __device__ Float4_ mul(Half4_ a, Half4_ b) {
  Float4_ fc;
  fc.x = mul<float2, __half2, __half2>(a.x, b.x);
  fc.y = mul<float2, __half2, __half2>(a.y, b.y);
  return fc;
}

template<>
inline __device__ Float4_ mul(__half a, Half4_ b) {
  __half2 s = h0_h0(a);
  Float4_ fc;
  fc.x = mul<float2, __half2, __half2>(s, b.x);
  fc.y = mul<float2, __half2, __half2>(s, b.y);
  return fc;
}

template<>
inline __device__ Float8_ mul(Half8_ a, Half8_ b) {
  Float8_ fc;
  fc.x = mul<float2, __half2, __half2>(a.x, b.x);
  fc.y = mul<float2, __half2, __half2>(a.y, b.y);
  fc.z = mul<float2, __half2, __half2>(a.z, b.z);
  fc.w = mul<float2, __half2, __half2>(a.w, b.w);
  return fc;
}

template<>
inline __device__ Float8_ mul(__half a, Half8_ b) {
  __half2 s = h0_h0(a);
  Float8_ fc;
  fc.x = mul<float2, __half2, __half2>(s, b.x);
  fc.y = mul<float2, __half2, __half2>(s, b.y);
  fc.z = mul<float2, __half2, __half2>(s, b.z);
  fc.w = mul<float2, __half2, __half2>(s, b.w);
  return fc;
}

// Vector fused multiply-add.
inline __device__ __half2 fma(__half2 a, __half2 b, __half2 c) {
  //uint32_t d;
  //asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  //return d;
    return __hfma2(a,b,c);
}

inline __device__ __half2 fma(__half a, __half2 b, __half2 c) {
  return fma(h0_h0(a), b, c);
}

inline __device__ Half4_ fma(Half4_ a, Half4_ b, Half4_ c) {
  Half4_ d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}

inline __device__ Half4_ fma(__half a, Half4_ b, Half4_ c) {
  __half2 s = h0_h0(a);
  Half4_ d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  return d;
}

inline __device__ Half8_ fma(Half8_ a, Half8_ b, Half8_ c) {
  Half8_ d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}

inline __device__ Half8_ fma(__half a, Half8_ b, Half8_ c) {
  __half2 s = h0_h0(a);
  Half8_ d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  d.z = fma(s, b.z, c.z);
  d.w = fma(s, b.w, c.w);
  return d;
}

inline __device__ float fma(__half a, __half b, float fc) {
  float fa = half_to_float(a);
  float fb = half_to_float(b);
  return fa * fb + fc;
}

inline __device__ float2 fma(__half2 a, __half2 b, float2 fc) {
  float2 fa = half2_to_float2(a);
  float2 fb = half2_to_float2(b);
  return fma(fa, fb, fc);
}

inline __device__ float2 fma(__half a, __half2 b, float2 fc) {
  return fma(h0_h0(a), b, fc);
}

inline __device__ Float4_ fma(Half4_ a, Half4_ b, Float4_ fc) {
  Float4_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  return fd;
}

inline __device__ Float4_ fma(__half a, Half4_ b, Float4_ fc) {
  __half2 s = h0_h0(a);
  Float4_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  return fd;
}

inline __device__ Float8_ fma(Half8_ a, Half8_ b, Float8_ fc) {
  Float8_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  fd.z = fma(a.z, b.z, fc.z);
  fd.w = fma(a.w, b.w, fc.w);
  return fd;
}

inline __device__ Float8_ fma(__half a, Half8_ b, Float8_ fc) {
  __half2 s = h0_h0(a);
  Float8_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  fd.z = fma(s, b.z, fc.z);
  fd.w = fma(s, b.w, fc.w);
  return fd;
}

// Vector sum.
template<>
inline __device__ float sum(__half v) {
  return half_to_float(v);
}

template<>
inline __device__ float sum(__half2 v) {
  float2 tmp = half2_to_float2(v);
  return tmp.x + tmp.y;
}

template<>
inline __device__ float sum(Half4_ v) {
  __half2 c = add(v.x, v.y);
  return sum(c);
}

template<>
inline __device__ float sum(Half8_ v) {
  __half2 c = add(v.x, v.y);
  c = add(c, v.z);
  c = add(c, v.w);
  return sum(c);
}

// Zero-out a vector.
inline __device__ void zero(uint16_t& dst) {
  dst = uint16_t(0);
}

// From float32 to float16.
inline __device__ void from_float(__half& dst, float src) {
  dst = float_to_half(src);
}

inline __device__ void from_float(__half2& dst, float2 src) {
  dst = float2_to_half2(src);
}

inline __device__ void from_float(Half4_& dst, Float4_ src) {
  dst.x = float2_to_half2(src.x);
  dst.y = float2_to_half2(src.y);
}

inline __device__ void from_float(Half8_& dst, Float8_ src) {
  dst.x = float2_to_half2(src.x);
  dst.y = float2_to_half2(src.y);
  dst.z = float2_to_half2(src.z);
  dst.w = float2_to_half2(src.w);
}

// From float16 to float32.
inline __device__ float to_float(__half u) {
  return half_to_float(u);
}

inline __device__ float2 to_float(__half2 u) {
  return half2_to_float2(u);
}

inline __device__ Float4_ to_float(Half4_ u) {
  Float4_ tmp;
  tmp.x = half2_to_float2(u.x);
  tmp.y = half2_to_float2(u.y);
  return tmp;
}

inline __device__ Float8_ to_float(Half8_ u) {
  Float8_ tmp;
  tmp.x = half2_to_float2(u.x);
  tmp.y = half2_to_float2(u.y);
  tmp.z = half2_to_float2(u.z);
  tmp.w = half2_to_float2(u.w);
  return tmp;
}

} // namespace vllm
