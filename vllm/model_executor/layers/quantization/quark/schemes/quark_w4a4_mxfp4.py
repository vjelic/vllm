# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

import vllm.envs as envs
from vllm.model_executor.layers.quantization.quark.schemes import QuarkScheme
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    OCP_MX_BLOCK_SIZE,
    quant_dequant_mxfp4,
    dequant_mxfp4,
)
from vllm.model_executor.parameter import (GroupQuantScaleParameter,
                                           PackedvLLMParameter)
from vllm.platforms import current_platform

try:
    from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
except ImportError:
    dynamic_mxfp4_quant = gemm_afp4wfp4 = None

__all__ = ["QuarkW4A4MXFP4"]


class QuarkW4A4MXFP4(QuarkScheme):

    def __init__(self, weight_quant_spec: Dict[str, Any],
                 input_quant_spec: Dict[str, Any]):
        self.out_dtype = torch.get_default_dtype()
        self.qscheme = "per_group"
        self.weight_quant_spec = weight_quant_spec
        self.input_quant_spec = input_quant_spec
        self.emulate = not current_platform.supports_mx()
        if not self.emulate and (dynamic_mxfp4_quant is None
                                 or gemm_afp4wfp4 is None):
            # Currently need these kernels if not emulating
            raise NotImplementedError(
                f"{self.__class__.__name__} requires AITER to be installed "
                "for non-emulation mode! Please refer to "
                "https://github.com/ROCm/aiter for installation details.")

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = torch.nn.Parameter(layer.weight.data,
                                          requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            layer.weight_scale.data.T.contiguous(), requires_grad=False)

        if self.emulate and not envs.VLLM_QUARK_EMU_MEM_OPT:
            layer.weight = torch.nn.Parameter(
                dequant_mxfp4(layer.weight.data, layer.weight_scale.data, self.out_dtype),
                requires_grad=False,
            )
            layer.weight_scale = None

            # This call is necessary to release the scales memory.
            torch.cuda.empty_cache()

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // 2,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=2,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // OCP_MX_BLOCK_SIZE,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        if self.emulate:
            if envs.VLLM_QUARK_EMU_MEM_OPT:
                dq_w = dequant_mxfp4(layer.weight, layer.weight_scale, x.dtype)
            else:
                dq_w = layer.weight

            x = quant_dequant_mxfp4(x)

            return F.linear(x, dq_w, bias)
        else:
            x_q, x_s = dynamic_mxfp4_quant(x)
            y = torch.empty(x_q.shape[0],
                            layer.weight.shape[0],
                            device=x_q.device,
                            dtype=self.out_dtype)
            gemm_afp4wfp4(x_q, layer.weight.T, y, x_s, layer.weight_scale.T,
                          self.out_dtype)
            return y
