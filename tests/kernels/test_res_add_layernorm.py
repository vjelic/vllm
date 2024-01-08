import torch
import torch.nn as nn

from vllm import layernorm_ops


class RefRMSNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        weight = torch.empty(hidden_size)
        weight.uniform_(-1e-3, 1e-3)
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps

    def forward(self, residual, hidden_states):
        hidden_states = residual + hidden_states
        saved = hidden_states
        variance = hidden_states.to(torch.float32).pow(2).mean(-1,
                                                               keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    self.variance_epsilon)
        if self.weight.dtype in [torch.half, torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states, saved


@torch.inference_mode()
def run_rms_norm(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
) -> None:
    residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device='cuda')
    hidden = torch.randn(num_tokens, hidden_size, dtype=dtype, device='cuda')
    ref = RefRMSNorm(hidden_size).to(dtype).cuda()

    out = torch.empty_like(hidden)
    out_res = torch.empty_like(hidden)
    for i in range(1000):
        layernorm_ops.res_add_rms_norm(
            out,
            out_res,
            residual,
            hidden,
            ref.weight.data,
            ref.variance_epsilon
        )
    ref_out, ref_res = ref(residual,hidden)
    assert torch.allclose(out, ref_out, atol=1e-3, rtol=1e-5)
    assert torch.allclose(out_res, ref_res, atol=1e-3, rtol=1e-5)


def test_rms_norm() -> None:
    for dtype in [torch.half, torch.bfloat16, torch.float]:
        for num_tokens in [1,7,128,2048,16384]:
            #for hidden_size in [13, 64, 1024, 5120, 8192]:
            for hidden_size in [1024,2048,3072,4096,5120,8192]:
                print(f'Testing RMS kernel with dtype={dtype}, num_tokens='
                      f'{num_tokens}, hidden_size={hidden_size}')
                run_rms_norm(
                    num_tokens=num_tokens,
                    hidden_size=hidden_size,
                    dtype=dtype,
                )
