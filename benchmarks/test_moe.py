import torch
from vllm.model_executor.layers.fused_moe import fused_moe

def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
):
    dtype = torch.float16
    a = torch.randn((m, k), device='cuda', dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device='cuda', dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device='cuda', dtype=dtype) / 10

    score = torch.randn((m, e), device='cuda', dtype=dtype)
    triton_output = fused_moe(a, w1, w2, score, topk, renormalize=False)

if __name__ == "__main__":
    m_list =    [512,  1]
    n_list =    [2048, 18944]
    k_list =    [128,  3584]
    e_list =    [8,    64]
    topk_list = [2,    8]

    i = 1
    print("\nRunning fused_moe with m = {}, n = {}, k = {}, e = {}, topk = {}".format(m_list[i], n_list[i], k_list[i], e_list[i], topk_list[i]))
    test_fused_moe(m_list[i], n_list[i], k_list[i], e_list[i], topk_list[i])