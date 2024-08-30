import pytest
import torch

from vllm.model_executor.layers.layernorm import RMSNorm

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 4096]  # Arbitrary values for testing
HIDDEN_SIZES = [768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192,
                8199]  # Arbitrary values for testing
ADD_RESIDUAL = [False, True]
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]


# @pytest.mark.parametrize("num_tokens", NUM_TOKENS)
# @pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
# @pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
# @pytest.mark.parametrize("dtype", DTYPES)
# @pytest.mark.parametrize("seed", SEEDS)
# @pytest.mark.parametrize("device", CUDA_DEVICES)
# @torch.inference_mode()
# def test_rms_norm(
#     num_tokens: int,
#     hidden_size: int,
#     add_residual: bool,
#     dtype: torch.dtype,
#     seed: int,
#     device: str,
# ) -> None:
#     torch.random.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#     torch.set_default_device(device)
#     layer = RMSNorm(hidden_size).to(dtype=dtype)
#     layer.weight.data.normal_(mean=1.0, std=0.1)
#     scale = 1 / (2 * hidden_size)
#     x = torch.randn(num_tokens, hidden_size, dtype=dtype)
#     x *= scale
#     residual = torch.randn_like(x) * scale if add_residual else None

#     # NOTE(woosuk): The reference implementation should be executed first
#     # because the custom kernel is in-place.
#     ref_out = layer._forward(x, residual)
#     out = layer(x, residual)
#     # NOTE(woosuk): LayerNorm operators (including RMS) typically have larger
#     # numerical errors than other operators because they involve reductions.
#     # Therefore, we use a larger tolerance.
#     if add_residual:
#         assert torch.allclose(out[0], ref_out[0], atol=1e-2, rtol=1e-2)
#         assert torch.allclose(out[1], ref_out[1], atol=1e-2, rtol=1e-2)
#     else:
#         assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2)


import triton
import triton.language as tl
import functools

@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE": 1024,
                "waves_per_eu": 2
            },
            num_stages=0,
            num_warps=8,
        )
    ],
    key=["BLOCK_SIZE"],
)
@triton.jit()
def _triton_rms_add_norm(
    input_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    eps,
    num_tokens,
    hidden_size,
    BLOCK_SIZE: tl.constexpr
):
    row_start_idx = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_idx in tl.range(row_start_idx, num_tokens, row_step):
        row_start = row_idx * hidden_size

        varT = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for offsets in range(0, hidden_size, BLOCK_SIZE):
            col_offsets = offsets + tl.arange(0, BLOCK_SIZE)
            input_ptrs = input_ptr + row_start + col_offsets
    
            mask = col_offsets < hidden_size
            row = tl.load(input_ptrs, mask=mask, other=0.)
            varT += row * row

        var = tl.sum(varT, axis=0) / hidden_size
        rstd = 1 / tl.sqrt(var + eps)

        for offsets in range(0, hidden_size, BLOCK_SIZE):
            col_offsets = offsets + tl.arange(0, BLOCK_SIZE)
            input_ptrs = input_ptr + row_start + col_offsets

            mask = col_offsets < hidden_size
            weight = tl.load(weight_ptr + col_offsets,
                            mask=mask)
            residual = tl.load(residual_ptr + row_start + col_offsets,
                            mask=mask)
            input = tl.load(input_ptr + row_start + col_offsets, 
                            mask=mask, other=0.)
            input_rstd = input * rstd
            output = input_rstd * weight + residual

            # Write output
            tl.store(output_ptr + row_start + col_offsets, output, mask=mask)


class TritonRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, residual, weight, 
                eps, num_tokens, hidden_size):
        out = torch.zeros_like(input)

        grid = (triton.cdiv(num_tokens, 1),)
        _triton_rms_add_norm[grid](input, residual, weight,
                                out, eps, num_tokens, hidden_size)
        return out

triton_rms_add_norm = TritonRMSNorm.apply


warmup = 5
rep = 10

@pytest.mark.parametrize("num_tokens", [1024])
@pytest.mark.parametrize("hidden_size", [8 * 1024])
@pytest.mark.parametrize("dtype", [torch.float])
def test_benchmark_rms_norm(
    num_tokens, 
    hidden_size,
    dtype: torch.dtype
):
    seed = 42

    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = "cuda:0"
    torch.set_default_device(device)

    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    input = torch.randn(num_tokens, hidden_size, dtype=dtype)
    input = input.fill_(4)
    input *= scale
    # residual = torch.randn_like(input) * scale
    residual = torch.zeros_like(input) * scale

    ref_out = layer._forward(input, residual)
    out = triton_rms_add_norm(input, residual, layer.weight.data,
                layer.variance_epsilon, num_tokens, hidden_size)
    
    assert functools.reduce(lambda a, b: a and b,
            [ torch.allclose(out[idx], ref_out[0][idx], atol=1e-2, rtol=1e-2) 
                for idx in range(0, num_tokens) ])

    res_trit = triton.testing.do_bench(
        lambda: triton_rms_add_norm(input, residual, layer.weight,
                layer.variance_epsilon, num_tokens, hidden_size),
        warmup=warmup, rep=rep)
    
    print("")
    print(f'Triton RES: {res_trit=}')

    res_c = triton.testing.do_bench(
        lambda: layer(input, residual),
        warmup=warmup, rep=rep)
    print(f'C++ RES: {res_c=}')

    assert res_trit < res_c