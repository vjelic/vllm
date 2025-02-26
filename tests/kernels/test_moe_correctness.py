import pytest
import torch
from torch.nn import Parameter
from torch.nn import functional as F
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from vllm.model_executor.layers.fused_moe import override_config

import vllm.envs as envs
from tests.kernels.utils import torch_moe
from vllm.model_executor.layers.fused_moe import fused_moe

from vllm.model_executor.layers.fused_moe.moe_torch_iterative import (
    fused_moe as iterative_moe)


NUM_EXPERTS = [8]
TOP_KS = [2]

random_config = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "matrix_instr_nonkdim": 16
    # "num_warps": 1,
    # "num_stages": 2,
}

@pytest.mark.parametrize("m", [222])
@pytest.mark.parametrize("n", [2048])
@pytest.mark.parametrize("k", [1024])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("dtype", [torch.float16])
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    torch_output = torch_moe(a, w1, w2, score, topk)

    # triton_output_default_config = fused_moe(a, w1, w2, score, topk, renormalize=False)
    with override_config(random_config):
        triton_output = fused_moe(a, w1, w2, score, topk, renormalize=False)
        
    
    torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)
    # torch.testing.assert_close(triton_output, triton_output_default_config, atol=1e-3, rtol=0) # Passes
    # torch.testing.assert_close(triton_output, triton_output_default_config, atol=1e-4, rtol=1e-3) # 5 fails, 40 passed