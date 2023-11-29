from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm import pos_encoding_ops
from vllm import custom_ops

@torch.inference_mode()
def run_custom_llgemm(M,K,N=1,dtype=torch.float16,NB=1,iters=2000,warmup=500,measure=1000):
    inp = 0.25*torch.randn(N,K, dtype=dtype, device='cuda')
    weights = 0.25*torch.randn(NB,M,K, dtype=dtype, device='cuda')
    out = torch.empty(NB,N,M,dtype=dtype,device='cuda')
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(iters):
        #inp1=inp #+0.001
        if i==warmup: start.record()
        if i==warmup+measure: end.record()
        #custom_ops.LLMM1(weights[i%NB],inp,out[i%NB])
        custom_ops.LLZZ(weights[i%NB],inp,out[i%NB],0)
    torch.cuda.synchronize()
    gtime = start.elapsed_time(end)/(measure)
    print('>>> GTime',gtime,'ms')
    cref = F.linear(inp,weights[0])
    #print('>>> shapes',cref.shape,out[0].shape)
    assert torch.allclose(cref,out[0],atol=1,rtol=1e-5)

def test_custom() -> None:
    for dtype in [torch.float16]:
        #LL70_mk = [(4000,8192),(1280,8192),(8192,1024),(7168,8192),(8192,3584)]
        #LL70_mk = [(1280,8192)]
        #LL13_mk = [(32000,5120),(15360,5120),(5120,5120),(27648,5120),(5120,13824)]
        LL13_mk = [(27648,5120)]
        for M,K in LL13_mk:
            print(f'Running tests for M={M} K={K} and dtype={dtype}')
            run_custom_llgemm(M,K,N=1,dtype=dtype)

