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
import copy

# python -m pytest -v -s /root/workspace/vllm/tests/kernels/test_layernorm.py

@triton.autotune(
    configs=[
        # triton.Config(
        #     {
        #         "T_BLOCK_SIZE": tbs,
        #         "H_BLOCK_SIZE": hbs,
        #         "waves_per_eu": 4
        #     },
        #     num_stages=0,
        #     num_warps=nw,
        # ) 
        # for tbs in [ i for i in [ 2, 4, 8, 16, 32, 64 ] ]
        # for hbs in [ i for i in [ 32, 64, 128, 256, 512, 1024, 2048, 4096] ]
        # for nw in [ i for i in [ 1, 2, 4, 8, 16 ] ]
        triton.Config(
            {
                "T_BLOCK_SIZE": 2,
                "H_BLOCK_SIZE": 2 * 1024,
                "waves_per_eu": 4
            },
            num_stages=0,
            num_warps=16,
        ) 
    ],
    key=["num_tokens", "hidden_size"],
)
@triton.jit()
def _triton_rms_add_norm_block(
    input_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    eps,
    num_tokens,
    hidden_size,
    T_BLOCK_SIZE: tl.constexpr,
    H_BLOCK_SIZE: tl.constexpr
):
    row_start_idx = tl.program_id(0) * T_BLOCK_SIZE

    row_start = row_start_idx + tl.arange(0, T_BLOCK_SIZE)
    row_start = row_start[:, None] * hidden_size

    col_offsets = tl.arange(0, H_BLOCK_SIZE)
    acc = tl.zeros([T_BLOCK_SIZE], dtype=input_ptr.type.element_ty)
    for offsets in range(0, hidden_size, 2 * H_BLOCK_SIZE):
        col_offsets1 = offsets + col_offsets
        col_offsets2 = col_offsets1 + H_BLOCK_SIZE

        input_ptrs1 = input_ptr + row_start + col_offsets1
        input_ptrs2 = input_ptr + row_start + col_offsets2
        residual_ptrs1 = residual_ptr + row_start + col_offsets1
        residual_ptrs2 = residual_ptr + row_start + col_offsets2

        mask1 = (col_offsets1 < hidden_size)[None, :]
        mask2 = (col_offsets2 < hidden_size)[None, :]
        row1 = tl.load(input_ptrs1, mask=mask1, other=0.)
        row2 = tl.load(input_ptrs2, mask=mask2, other=0.)
        residual1 = tl.load(residual_ptrs1, mask=mask1, other=0.)
        residual2 = tl.load(residual_ptrs2, mask=mask2, other=0.)

        row1 += residual1
        row2 += residual2
        tl.store(residual_ptrs1, row1, mask=mask1)
        tl.store(residual_ptrs2, row2, mask=mask2)

        sq_row1 = row1 * row1
        sq_row2 = row2 * row2
        acc += tl.sum(sq_row1, axis=-1, keep_dims=False)
        acc += tl.sum(sq_row2, axis=-1, keep_dims=False)

    acc = acc / hidden_size
    rstd = 1 / tl.sqrt(acc + eps)
    rstd = rstd[:, None]

    for offsets in range(0, hidden_size, 2 * H_BLOCK_SIZE):
        col_offsets1 = offsets + col_offsets
        col_offsets2 = col_offsets1 + H_BLOCK_SIZE

        mask1 = (col_offsets1 < hidden_size)[None, :]
        mask2 = (col_offsets2 < hidden_size)[None, :]
        weight1 = tl.load(weight_ptr + col_offsets1, 
                          mask=col_offsets1 < hidden_size)
        weight2 = tl.load(weight_ptr + col_offsets2, 
                          mask=col_offsets2 < hidden_size)
        residual_ptrs1 = tl.load(residual_ptr + row_start + col_offsets1, 
                        mask=mask1, other=0.)
        residual_ptrs2 = tl.load(residual_ptr + row_start + col_offsets2, 
                        mask=mask2, other=0.)
        output1 = residual_ptrs1 * rstd * weight1
        output2 = residual_ptrs2 * rstd * weight2

        # Write output
        tl.store(output_ptr + row_start + col_offsets1, output1, mask=mask1)
        tl.store(output_ptr + row_start + col_offsets2, output2, mask=mask2)


class TritonRMSNormBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, residual, weight, 
                eps, num_tokens, hidden_size):
        out = torch.zeros_like(input)

        grid = lambda META: (  # noqa: E731
            triton.cdiv(num_tokens, META["T_BLOCK_SIZE"]), 
        )
        _triton_rms_add_norm_block[grid](input, residual, weight,
                                out, eps, num_tokens, hidden_size)
        return out

triton_rms_add_norm_block = TritonRMSNormBlock.apply


@triton.autotune(
    configs=[
        triton.Config(
            {
                "T_BLOCK_SIZE": 1,
                "H_BLOCK_SIZE": 128 * 64,
                "waves_per_eu": 4
            },
            num_stages=0,
            num_warps=8,
        )
    ],
    key=["num_tokens", "hidden_size"],
)
@triton.jit()
def _triton_rms_add_norm_iter(
    input_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    eps,
    num_tokens,
    hidden_size,
    T_BLOCK_SIZE: tl.constexpr,
    H_BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    pn = tl.num_programs(0)

    for row_start_idx in range(pid, num_tokens, pn):
        row_start = row_start_idx * hidden_size

        acc = 0.0
        for offsets in range(0, hidden_size, H_BLOCK_SIZE):
            col_offsets = tl.arange(0, H_BLOCK_SIZE)
            col_offsets += offsets
            input_ptrs = input_ptr + row_start + col_offsets
            residual_ptrs = residual_ptr + row_start + col_offsets

            mask = col_offsets < hidden_size
            row = tl.load(input_ptrs, mask=mask, other=0.)
            residual = tl.load(residual_ptrs, mask=mask, other=0.)
            row += residual

            tl.store(residual_ptrs, row, mask=mask)
            tl.store(input_ptrs, row, mask=mask)

            sq_row = row * row
            acc += tl.sum(sq_row, axis=-1, keep_dims=False)

        acc = acc / hidden_size
        rstd = 1 / tl.sqrt(acc + eps)

        for offsets in range(0, hidden_size, H_BLOCK_SIZE):
            col_offsets = tl.arange(0, H_BLOCK_SIZE)
            col_offsets += offsets

            mask = col_offsets < hidden_size
            weight = tl.load(weight_ptr + col_offsets,
                            mask=col_offsets < hidden_size)
            input = tl.load(input_ptr + row_start + col_offsets, 
                            mask=mask, other=0.)
            input_rstd = input * rstd
            output = input_rstd * weight

            # Write output
            tl.store(output_ptr + row_start + col_offsets, output, mask=mask)


class TritonRMSNormIter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, residual, weight, 
                eps, num_tokens, hidden_size):
        out = torch.zeros_like(input)

        grid = lambda META: (  # noqa: E731
            triton.cdiv(num_tokens, META["T_BLOCK_SIZE"]), 
        )
        _triton_rms_add_norm_iter[grid](input, residual, weight,
                                out, eps, num_tokens, hidden_size)
        return out

triton_rms_add_norm_iter = TritonRMSNormIter.apply

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
    input1 = torch.randn(num_tokens, hidden_size, dtype=dtype)
    input1 *= scale
    residual1 = torch.randn_like(input1) * scale

    # need this to run autotunning as kernel is not idempotent
    input2 = copy.deepcopy(input1)
    residual2 = copy.deepcopy(residual1)
    out = triton_rms_add_norm_block(input2, residual2, layer.weight.data,
                layer.variance_epsilon, num_tokens, hidden_size)
    
    ref_out, _ = layer(input1, residual1)

    assert functools.reduce(lambda a, b: a and b,
            [ torch.allclose(out[idx], ref_out[idx], atol=1e-2, rtol=1e-2) 
                for idx in range(0, num_tokens) ])

    res_trit = triton.testing.do_bench(
        lambda: triton_rms_add_norm_block(input2, residual2, layer.weight,
                layer.variance_epsilon, num_tokens, hidden_size),
        warmup=warmup, rep=rep)
    
    print("")
    print(f'Triton Block RES: {res_trit=}')

    # res_trit = triton.testing.do_bench(
    #     lambda: triton_rms_add_norm_iter(input2, residual2, layer.weight,
    #             layer.variance_epsilon, num_tokens, hidden_size),
    #     warmup=warmup, rep=rep)
    
    # print(f'Triton Iter RES: {res_trit=}')

    res_c = triton.testing.do_bench(
        lambda: layer(input1, residual1),
        warmup=warmup, rep=rep)
    print(f'C++ RES: {res_c=}')