import torch
import torch.nn.functional as F

from vllm import activation_ops


@torch.inference_mode()
def run_silu_and_mul(
    num_tokens: int,
    d: int,
    interm: int,
    dtype: torch.dtype,
) -> None:
    x = torch.randn(num_tokens, 2*interm, dtype=dtype, device='cuda')
    out = torch.empty(num_tokens, interm, dtype=dtype, device='cuda')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    inter = torch.cuda.Event(enable_timing=True)
    cold_iters=5000
    warm_iters=120
    inter_iters=10
    num_weight_buffers=40
    for i in range(cold_iters):
        activation_ops.silu_and_mul(out, x)
    start.record()
    for i in range(warm_iters):
        activation_ops.silu_and_mul(out, x)
    end.record()
    torch.cuda.synchronize()
    gtime_silu = start.elapsed_time(end)/warm_iters
    print('>>> Silu standalone ms',gtime_silu)

    inp = torch.randn((num_tokens, d), dtype=dtype, device='cuda')
    weights = torch.randn((num_weight_buffers,2*interm, d), dtype=dtype, device='cuda')

    for i in range(cold_iters):
        x = F.linear(inp,weights[i%num_weight_buffers])
    start.record()
    for i in range(warm_iters):
        x = F.linear(inp,weights[i%num_weight_buffers])
    end.record()
    torch.cuda.synchronize()
    gtime_gemm = start.elapsed_time(end)/warm_iters
    print('>>> Gemm standalone ms',gtime_gemm)

    for i in range(cold_iters):
        x = F.linear(inp,weights[i%num_weight_buffers])
        activation_ops.silu_and_mul(out, x)
    start.record()
    for i in range(warm_iters):
        if i == warm_iters-inter_iters: inter.record()
        x = F.linear(inp,weights[i%num_weight_buffers])
        activation_ops.silu_and_mul(out, x)
    end.record()
    torch.cuda.synchronize()
    gtime_gemm_silu = start.elapsed_time(end)/warm_iters
    print('>>> Gemm plus silu ms',gtime_gemm_silu)
    gtime_gemm_silu_inter = inter.elapsed_time(end)/inter_iters
    print('>>> Gemm plus silu inter ms',gtime_gemm_silu_inter)

    assert gtime_gemm_silu-gtime_gemm<1.05*gtime_silu
    assert gtime_gemm_silu_inter-gtime_gemm<1.05*gtime_silu

def test_silu_and_mul() -> None:
    for dtype in [torch.half]:
        for num_tokens in [3072]:
            for d in [5120]:
                for interm in [13824]:
                    print(f'Testing dtype={dtype}, num_tokens={num_tokens}, d={d}, interm={interm}')
                    run_silu_and_mul(num_tokens, d, interm, dtype)
