import torch

#from CustomPyTorchCUDAKernelBackend import AddGPU
from vllm.custom_ops import AddGPU
torch.set_printoptions(precision=10)

def add_gpu(a, b):
    assert isinstance(a, torch.cuda.FloatTensor) 
    assert isinstance(b, torch.cuda.FloatTensor)
    assert a.numel() == b.numel()

    c = a.new()
    AddGPU(a, b, c)
    return c

if __name__ == '__main__':
    a = torch.zeros(10,dtype=torch.float,device='cuda')
    b = torch.ones(10,dtype=torch.float,device='cuda')
    #b+=0.0000061
    b+=0.61
    c = add_gpu(a, b)

    print(c)
