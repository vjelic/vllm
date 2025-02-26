import triton
import torch
from vllm.model_executor.layers.quantization.utils.fp8_utils \
         import  w8a8_block_fp8_matmul
         
config1 = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 32,
    "num_warps": 4,
    "kpack": 1,
    "matrix_instr_nonkdim": 16,
}

config2 = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 16,
    "num_warps": 1,
    "kpack": 1,
    "matrix_instr_nonkdim": 16,
}

M, K, N = (32, 512, 2048)

group_n = 128
group_k = 128

a = torch.randn((M, K), dtype=torch.float16,
                device='cuda').to(torch.float8_e4m3fnuz)
b = torch.randn((N, K), dtype=torch.float16,
                device='cuda').to(torch.float8_e4m3fnuz)
a_per_token = torch.randn((M, triton.cdiv(K, group_k)),
                            dtype=torch.float16,
                            device='cuda')
b_per_block = torch.randn(
    (triton.cdiv(N, group_n), triton.cdiv(K, group_k)),
    dtype=torch.float16,
    device='cuda')

output_1 =  w8a8_block_fp8_matmul(a,
                                    b,
                                    a_per_token,
                                    b_per_block,
                                    [group_n, group_k],
                                    output_dtype=torch.float16,
                                    tune_config=config1)

output_2 =  w8a8_block_fp8_matmul(a,
                                    b,
                                    a_per_token,
                                    b_per_block,
                                    [group_n, group_k],
                                    output_dtype=torch.float16,
                                    tune_config=config2)
if not torch.allclose(output_1, output_2, rtol=1e-5, atol=1e-6):
    print("Test failed")
torch.testing.assert_close(output_1, output_2, atol=1e-5, rtol=0)
print("FINISHED!")