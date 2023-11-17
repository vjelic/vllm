from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm import pos_encoding_ops
from vllm import custom_ops

@torch.inference_mode()
def run_custom_llgemm(M,K,N=1,dtype=torch.float16,NB=37,iters=10000):
    inp = 0.25*torch.randn(N,K, dtype=dtype, device='cuda')
    weights = 0.25*torch.randn(NB,M,K, dtype=dtype, device='cuda')
    out = torch.empty(NB,N,M,dtype=dtype,device='cuda')

    for i in range(iters):
        inp1=inp+0.001
        custom_ops.LLMM1(weights[i%NB],inp1,out[i%NB])
    cref = F.linear(inp+0.001,weights[0])
    #print('>>> shapes',cref.shape,out[0].shape)
    assert torch.allclose(cref,out[0],atol=1,rtol=1e-5)

def test_custom() -> None:
    for dtype in [torch.float16]:
        LL70_mk = [(8192,3584)]
        for M,K in LL70_mk:
            print(f'Running tests for M={M} K={K} and dtype={dtype}')
            run_custom_llgemm(M,K,N=1,dtype=dtype)
