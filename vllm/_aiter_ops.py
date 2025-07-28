# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch

from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op


def rocm_aiter_tuned_gemm_impl(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
        scale_a: Optional[torch.Tensor] = None,
        scale_b: Optional[torch.Tensor] = None) -> torch.Tensor:

    # This AITER function can be used for
    # - BF16 and FP16 matmul
    #   e.g. vllm/model_executor/layers/linear.py
    # - per-tensor activations + per-tensor weights
    #   e.g. vllm/model_executor/layers/quantization/utils/w8a8_utils.py
    from aiter.tuned_gemm import tgemm as aiter_tgemm

    return aiter_tgemm.mm(input,
                          weight,
                          otype=out_dtype,
                          scale_a=scale_a,
                          scale_b=scale_b,
                          bias=bias)


def rocm_aiter_tuned_gemm_fake(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
        scale_a: Optional[torch.Tensor] = None,
        scale_b: Optional[torch.Tensor] = None) -> torch.Tensor:

    m = input.shape[0]
    n = weight.shape[0]
    if out_dtype is None:
        out_dtype = input.dtype
    return torch.empty((m, n), dtype=out_dtype, device=input.device)


def rocm_aiter_rmsnorm2d_fwd_with_add_quant_impl(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
    variance_epsilon: float, x_scale=None, y_scale_dtype=None, 
    q_dtype="fp8", model_sensitive=0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    import aiter as rocm_aiter
    from aiter import  dtypes 
    quant_dtype_map = {
        "i8": dtypes.i8 ,
        "fp8": dtypes.fp8
    }
    q_dtype = quant_dtype_map[q_dtype]
    assert y_scale_dtype is not None # TODO
    
    if x_scale is None: 
        output = torch.empty(input.shape, dtype=q_dtype, devive="cuda")
        y_scale = torch.empty(input.shape[0], 1, dtype=y_scale_dtype, device="cuda") #TODO: only per-token quant now
        if residual is None:
            residual_out = None 
            rocm_aiter.rmsnorm2d_fwd_with_dynamicquant(
                    output, input, y_scale, weight, variance_epsilon, model_sensitive
            )
        elif residual is not None:
            residual_out = torch.empty_like(input)
            rocm_aiter.rmsnorm2d_fwd_with_add_dynamicquant(
                output, input, residual, residual_out, y_scale, weight, variance_epsilon, model_sensitive 
            )
    else:
        output = torch.empty(input.shape, dtype=q_dtype, devive="cuda")
        y_scale = torch.empty(input.shape[0], 1, dtype=y_scale_dtype, device="cuda") #TODO: only per-token quant now
        if residual is None:
            residual_out = None 
            aiter.rmsnorm2d_fwd_with_smoothquant(
                output, input, x_scale, y_scale, weight, variance_epsilon, model_sensitive
            )
        elif residual is not None:
            residual_out = torch.empty_like(input)
            aiter.rmsnorm2d_fwd_with_add_smoothquant(
                output, input, residual, residual_out, x_scale, y_scale, weight, variance_epsilon
            )
    return output, residual_out, y_scale 

def rocm_aiter_rmsnorm2d_fwd_with_add_quant_fake(
        input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
        variance_epsilon: float, x_scale=None, y_scale_dtype=None, 
        q_dtype="fp8", model_sensitive=0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty(input.shape, dtype=q_dtype, device="cuda"), torch.empty_like(input), torch.empty(input.shape[0], 1, dtype=y_scale_dtype, device="cuda")


if current_platform.is_rocm():
    direct_register_custom_op(
        op_name="rocm_aiter_tuned_gemm",
        op_func=rocm_aiter_tuned_gemm_impl,
        mutates_args=[],
        fake_impl=rocm_aiter_tuned_gemm_fake,
        dispatch_key=current_platform.dispatch_key,
    )

    direct_register_custom_op(
        op_name="rocm_aiter_rmsnorm2d_fwd_with_add_quant",
        op_func=rocm_aiter_rmsnorm2d_fwd_with_add_quant_impl,
        mutates_args=[],
        fake_impl=rocm_aiter_rmsnorm2d_fwd_with_add_quant_fake,
        dispatch_key=current_platform.dispatch_key,
    )

class aiter_ops:

    @staticmethod
    def rocm_aiter_tuned_gemm(
            input: torch.Tensor,  # [M, K]
            weight: torch.Tensor,  # [N, K]
            bias: Optional[torch.Tensor] = None,
            out_dtype: Optional[torch.dtype] = None,
            scale_a: Optional[torch.Tensor] = None,
            scale_b: Optional[torch.Tensor] = None) -> torch.Tensor:

        return torch.ops.vllm.rocm_aiter_tuned_gemm(
            input,
            weight,
            bias=bias,
            out_dtype=out_dtype,
            scale_a=scale_a,
            scale_b=scale_b,
        )

    @staticmethod 
    def rocm_aiter_rmsnorm2d_fwd_with_add_quant(
            input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
            variance_epsilon: float, x_scale=None, y_scale_dtype=None, 
            q_dtype="fp8", model_sensitive=0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        return torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add_quant(
            input, residual, weight, variance_epsilon, x_scale, 
            y_scale_dtype, q_dtype, model_sensitive
            )
    