// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "torch/all.h"
#include "fused_moe.hpp"

#define FOREACH_BUFFER_TORCH_TYPE_MAP(F) \
    F("fp32", torch::kFloat)             \
    F("fp16", torch::kHalf)              \
    F("bf16", torch::kBFloat16)          \
    F("int32", torch::kInt32)            \
    F("int8", torch::kInt8)              \
    F("fp8", c10::kFloat8_e4m3fnuz)

inline std::string torchDTypeToStr_(caffe2::TypeMeta dtype)
{
#define TYPE_CASE(type, torch_type) \
    case torch_type:                \
    {                               \
        return type;                \
    }

    switch (dtype.toScalarType())
    {
        FOREACH_BUFFER_TORCH_TYPE_MAP(TYPE_CASE);
    default:
        throw std::runtime_error("CKPyInterface: Unsupported data type " + std::to_string((int8_t)(dtype.toScalarType())));
    }

#undef TYPE_CASE
}

float fused_ck_moe_(fused_moe_traits t, fused_moe_args a, const ck_tile::stream_config& s)
{
    auto s_sub = ck_tile::stream_config{s.stream_id_, false, s.log_level_, 0, 1};

    auto o_data_bytes = [&]() {
        if(t.prec_o == "fp32")
            return 4;
        else if(t.prec_o == "fp16" || t.prec_o == "bf16")
            return 2;
        else if(t.prec_o == "int8" || t.prec_o == "fp8")
            return 1;
        return 1;
    }();

    auto t0 = fused_moesorting_trait{"int32", "fp32"};
    auto a0 = fused_moesorting_args{
        a.topk_ids_ptr,                              // const void* p_topk_ids;
        a.topk_weight_ptr,                           // const void* p_weights;
        a.sorted_token_ids_ptr,                      // void* p_sorted_token_ids;
        a.sorted_weight_ptr,                         // void* p_sorted_weights;
        a.sorted_expert_ids_ptr,                     // void* p_sorted_expert_ids;
        a.num_sorted_tiles_ptr,                      // void* p_total_tokens_post_pad;
        a.o_ptr,                                     // void* p_moe_buf;
        a.num_tokens,                                // index_t tokens;
        a.block_m,                                   // index_t unit_size;
        a.num_experts,                               // index_t num_experts;
        a.topk,                                      // index_t topk;
        a.num_tokens * a.stride_token * o_data_bytes // index_t moe_buf_bytes;
    };

    auto t1 = fused_moegemm_traits{t.prec_i,
                                   t.prec_w,
                                   t.prec_o,
                                   t.prec_st,
                                   t.prec_sw,
                                   t.prec_sq,
                                   t.prec_kw,
                                   t.block_m,
                                   t.activation,
                                   t.gate_only,
                                   t.fused_quant};
    auto a1 = fused_moegemm_args{
        a.a_ptr,                 // const void* a_ptr;
        a.a_scale_ptr,           // const void* a_scale_ptr;
        a.g_ptr,                 // const void* g_ptr;
        a.d_ptr,                 // const void* d_ptr;
        a.g_scale_ptr,           // const void* g_scale_ptr;
        a.d_scale_ptr,           // const void* d_scale_ptr;
        a.y_smooth_scale_ptr,    // const void* y_smooth_scale_ptr;
        a.o_ptr,                 // void* o_ptr;
        a.sorted_token_ids_ptr,  // const void* sorted_token_ids_ptr;
        a.sorted_weight_ptr,     // const void* sorted_weight_ptr;
        a.sorted_expert_ids_ptr, // const void* sorted_expert_ids_ptr;
        a.num_sorted_tiles_ptr,  // const void* num_sorted_tiles_ptr;
        a.hidden_size,           // index_t hidden_size;
        a.intermediate_size,     // index_t intermediate_size;
        a.num_tokens,            // index_t num_tokens;
        a.num_experts,           // index_t num_experts;
        a.topk,                  // index_t topk;
        a.stride_token           // index_t stride_token;
    };

    float r0 = -1;
    float r1 = -1;

    float r = ck_tile::launch_kernel(
        s,
        [=, &r0](const ck_tile::stream_config&) { r0 = fused_moesorting(t0, a0, s_sub); },
        [=, &r1](const ck_tile::stream_config&) { r1 = fused_moegemm(t1, a1, s_sub); });

    // keep unsupported case return negative
    if(r0 < 0 || r1 < 0)
        return -1;

    return r;
}

torch::Tensor fused_ck_moe(
    torch::Tensor &hidden_states,          // [m, k], input token
    torch::Tensor &w1,                     // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
    torch::Tensor &w2,                     // [e, n, k], pre-shuffle([e, nr, kr, w])
    torch::Tensor &topk_weights,           // [tokens, topk]
    torch::Tensor &topk_ids,               // [tokens, topk]
    std::optional<torch::Tensor> w1_scale, // [e, 1, n], gate(up) scale
    std::optional<torch::Tensor> w2_scale, // [e, 1, k], down scale
    std::optional<torch::Tensor> a1_scale, // [m, 1], token scale
    std::optional<torch::Tensor> a2_scale, // [e, 1, n], smooth-quant-scale for 2nd gemm input
    std::optional<int64_t> block_m = 32)
{
    auto device = hidden_states.device();
    int topk_ids_numel = topk_ids.numel();
    int experts = w1.size(0);
    int topk = topk_ids.size(1);
    int tokens = topk_ids.size(0);
    int hidden_size = w1.size(2);
    int shared_intermediate_size_0 = w1.size(1);
    int shared_intermediate_size = w2.size(-1);
    //int block_size = block_m.value();
    int block_size = 32;

    int max_num_tokens_padded = topk_ids_numel + experts * block_size - topk;
    int max_num_m_blocks = (max_num_tokens_padded + block_size - 1) / block_size;

    auto sorted_ids = torch::empty({max_num_tokens_padded}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto sorted_weights = torch::empty({max_num_tokens_padded}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto sorted_expert_ids = torch::empty({max_num_m_blocks}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto num_tokens_post_pad = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto out = torch::empty({tokens, hidden_size}, torch::TensorOptions().dtype(hidden_states.dtype()).device(device));

    auto prec_i = torchDTypeToStr_(hidden_states.dtype());
    auto prec_w = torchDTypeToStr_(w1.dtype());
    auto prec_o = torchDTypeToStr_(out.dtype());
    auto prec_kw = torchDTypeToStr_(topk_weights.dtype());

    int gate_only = 1;
    int activation = 0;
    int fused_quant = 0;
    if (shared_intermediate_size_0 == 2 * shared_intermediate_size)
    {
        gate_only = 0;
        activation = 1;
    }

    if (!w1_scale.has_value())
    {
        fused_quant = 0;
    }
    else if (a1_scale.has_value() && a2_scale.has_value())
    {
        fused_quant = 1;
    }
    else
    {
        fused_quant = 2;
    }

    int stride = hidden_size;
    std::string prec_st = !a1_scale ? "fp32" : torchDTypeToStr_(a1_scale->dtype());
    std::string prec_sw = !w1_scale ? "fp32" : torchDTypeToStr_(w1_scale->dtype());
    std::string prec_sq = !a2_scale ? "fp32" : torchDTypeToStr_(a2_scale->dtype());
    
    fused_moe_traits traits{prec_i,
                            prec_w,
                            prec_o,
                            prec_st,
                            prec_sw,
                            prec_sq,
                            prec_kw,
                            block_size,
                            activation,
                            gate_only,
                            fused_quant};

    fused_moe_args args{hidden_states.data_ptr(),
                        a1_scale.has_value() ? a1_scale.value().data_ptr() : nullptr,
                        w1.data_ptr(),
                        w2.data_ptr(),
                        w1_scale.has_value() ? w1_scale.value().data_ptr() : nullptr,
                        w2_scale.has_value() ? w2_scale.value().data_ptr() : nullptr,
                        a2_scale.has_value() ? a2_scale.value().data_ptr() : nullptr,
                        out.data_ptr(),

                        topk_ids.data_ptr(),
                        topk_weights.data_ptr(),
                        sorted_ids.data_ptr(),
                        sorted_weights.data_ptr(),
                        sorted_expert_ids.data_ptr(),
                        num_tokens_post_pad.data_ptr(),

                        block_size,
                        hidden_size,
                        shared_intermediate_size,
                        tokens,
                        experts,
                        topk,
                        stride};

    //here we use default stream (no time-kernel)
    fused_ck_moe_(traits, args, ck_tile::stream_config{});
    return out; 
}


