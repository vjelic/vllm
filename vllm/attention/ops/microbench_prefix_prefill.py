import random
import sys
import torch
import triton
import triton.language as tl

from vllm.platforms import current_platform
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from prefix_prefill import _fwd_kernel


assert torch.cuda.is_available()

def generate_input(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    MAX_SEQ_LEN: int = 1024,
    MAX_CTX_LEN: int = 1024,
    batch: int = 10,
    cache_size: int = 640,
    block_size: int = 32,
    max_block_per_request: int = 64,
    dtype: torch.dtype = torch.float16,
    kv_cache_dtype: str = "auto",
    device: str = "None",
):
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    torch.set_default_device(device)
    torch.cuda.set_device(device)

    assert cache_size >= batch * max_block_per_request

    query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(batch)]
    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(batch)]
    seq_lens = [a + b for a, b in zip(query_lens, ctx_lens)]
    num_kv_heads = num_heads // num_queries_per_kv

    num_tokens = sum(query_lens)
    query = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)
    query.uniform_(-1e-3, 1e-3)
    output = torch.empty(num_tokens, num_heads, head_size, dtype=dtype)

    kv = torch.empty(sum(seq_lens), 2, num_kv_heads, head_size, dtype=dtype)
    kv.uniform_(-1e-3, 1e-3)
    key, value = kv.unbind(dim=1)

    if kv_cache_dtype == "auto":
        cache_dtype = dtype
    else:
        cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[kv_cache_dtype]
    k_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=cache_dtype)
    v_cache = torch.zeros(cache_size,
                          block_size,
                          num_kv_heads,
                          head_size,
                          dtype=cache_dtype)
    k = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    v = torch.zeros(sum(query_lens), num_kv_heads, head_size, dtype=dtype)
    values = torch.arange(0, cache_size, dtype=torch.long)
    values = values[torch.randperm(cache_size)]
    block_table = values[:batch * max_block_per_request].view(
        batch, max_block_per_request)
    b_seq_len = torch.tensor(seq_lens, dtype=torch.long)
    b_ctx_len = torch.tensor(ctx_lens, dtype=torch.long)
    b_start_loc = torch.cumsum(torch.tensor([0] + query_lens[:-1],
                                            dtype=torch.long),
                               dim=0)
    max_input_len = MAX_SEQ_LEN
    # copy kv to cache
    b_seq_start_loc = torch.cumsum(torch.tensor([0] + seq_lens[:-1],
                                                dtype=torch.long),
                                   dim=0)
    for i in range(batch):
        for j in range(query_lens[i]):
            k[b_start_loc[i] + j].copy_(key[b_seq_start_loc[i] + b_ctx_len[i] +
                                            j])
            v[b_start_loc[i] + j].copy_(value[b_seq_start_loc[i] +
                                              b_ctx_len[i] + j])
        cur_ctx = 0
        block_id = 0
        while cur_ctx < b_ctx_len[i]:
            start_loc = b_seq_start_loc[i] + cur_ctx
            if cur_ctx + block_size > b_ctx_len[i]:
                end_loc = b_seq_start_loc[i] + b_ctx_len[i]
            else:
                end_loc = start_loc + block_size
            start_slot = block_table[i, block_id] * block_size
            end_slot = start_slot + end_loc - start_loc
            k_cache.view(-1, num_kv_heads,
                         head_size)[start_slot:end_slot].copy_(
                             key[start_loc:end_loc])
            v_cache.view(-1, num_kv_heads,
                         head_size)[start_slot:end_slot].copy_(
                             value[start_loc:end_loc])
            cur_ctx += block_size
            block_id += 1
    # transpose K_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to K_cache[num_blocks, num_kv_heads, head_size/8, block_size, 8]
    k_cache = k_cache.view(-1, block_size, num_kv_heads, head_size // 8,
                           8).permute(0, 2, 3, 1, 4).contiguous()
    # transpose V_cache[num_blocks, block_size, num_kv_heads, head_size]
    # to V_cache[num_blocks, num_kv_heads, head_size, block_size]
    v_cache = v_cache.view(-1, block_size, num_kv_heads,
                           head_size).permute(0, 2, 3, 1).contiguous()

    return query, k, v, output, k_cache, v_cache, block_table, b_start_loc, \
        b_seq_len, b_ctx_len, max_input_len


arg_to_torch_dtype = {
    "fp16": torch.float16,
}

# self.max_num_blocks_per_seq = (self.model_config.max_model_len // self.block_size)
def generate_benchmark_configs():
    configs = []

    for dtype in ['fp16']:
        for kv_cache_dtype in ['auto']:  # 'fp8',
            for head_dim in [128]:
                for num_queries_per_kv in [1, 8, 64]:  # GQA ratio, i.e. how many Q each kv pair process, 1 == MHA
                    for sliding_window in [0, 16]:  # sliding window length only affects the mask applied on qk, so only zero and non-zero should be sufficient
                        configs.append(
                            triton.testing.Benchmark(
                                x_names=['batch', 'num_heads', 'max_seq_len', 'block_size', 'max_block_per_request'],
                                x_vals=[
                                    (10, 64, 1024, 32, 64),
                                    # (10, 64, 8192, 32, 64),
                                ],
                                line_arg='provider', line_vals=['triton'],
                                line_names=['us'],
                                styles=[('red', '-')], ylabel='us',
                                plot_name=f'prefix-prefill-{dtype}-kvDtype-{kv_cache_dtype}-d{head_dim}-gqaRatio-{num_queries_per_kv}-win{sliding_window}',
                                args={
                                    'dtype': arg_to_torch_dtype[dtype],
                                    'kv_cache_dtype': kv_cache_dtype,
                                    'head_dim': head_dim,
                                    'num_queries_per_kv': num_queries_per_kv,
                                    'sliding_window': sliding_window,
                                }))
    return configs


def run_benchmark():
    configs = generate_benchmark_configs()

    @triton.testing.perf_report(configs)
    def benchmark_prefix_prefill(
        batch, num_heads, max_seq_len, block_size, max_block_per_request,
        kv_cache_dtype, dtype, num_queries_per_kv,
        head_dim, sliding_window, provider, device='cuda'):
        warmup = 25
        rep = 100

        q, k, v, output, k_cache, v_cache, block_table, b_start_loc, \
            b_seq_len, b_ctx_len, max_input_len = \
        generate_input(
            num_heads,
            num_queries_per_kv,
            head_dim,
            MAX_SEQ_LEN=max_seq_len,
            MAX_CTX_LEN=max_seq_len,
            batch=batch,
            cache_size=640,
            block_size=block_size,
            max_block_per_request=max_block_per_request,
            dtype=dtype,
            kv_cache_dtype=kv_cache_dtype,
            device="cuda:0"
        )

        cap = current_platform.get_device_capability()
        BLOCK = 128 if cap[0] >= 8 else 64
        NUM_WARPS = 8

        # need to reduce num. blocks when using fp32
        # due to increased use of GPU shared memory
        if q.dtype is torch.float32:
            BLOCK = BLOCK // 2

        # Conversion of FP8 Tensor from uint8 storage to
        # appropriate torch.dtype for interpretation by Triton
        if "fp8" in kv_cache_dtype:
            assert (k_cache.dtype == torch.uint8)
            assert (v_cache.dtype == torch.uint8)

            if kv_cache_dtype in ("fp8", "fp8_e4m3"):
                target_dtype = torch.float8_e4m3fn
            elif kv_cache_dtype == "fp8_e5m2":
                target_dtype = torch.float8_e5m2
            else:
                raise ValueError("Unsupported FP8 dtype:", kv_cache_dtype)

            k_cache = k_cache.view(target_dtype)
            v_cache = v_cache.view(target_dtype)

        if (k_cache.dtype == torch.uint8
                or v_cache.dtype == torch.uint8 and kv_cache_dtype == "auto"):
            raise ValueError("kv_cache_dtype='auto' unsupported for\
                FP8 KV Cache prefill kernel")

        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        # round up Lk to a power of 2 - this is required for Triton block size
        Lk_padded = triton.next_power_of_2(Lk)

        sm_scale = 1.0 / (Lq**0.5)
        batch, head = b_seq_len.shape[0], q.shape[1]
        num_queries_per_kv = q.shape[1] // k.shape[1]

        grid = lambda meta : (batch, head, triton.cdiv(max_input_len, meta["BLOCK_M"]))  # batch, head,

        k_scale = 1.0
        v_scale = 1.0
        # print("q:", q.shape)
        # print("k:", k.shape)
        # print("v:", v.shape)
        # print("output:", output.shape)
        # print("k_cache:", k_cache.shape)
        # print("v_cache:", v_cache.shape)
        # print("block_table:", block_table.shape)
        # print("b_start_loc:", b_start_loc.shape)
        # print("b_seq_len:", b_seq_len.shape)
        # print("b_ctx_len:", b_ctx_len.shape)
        # print("max_input_len:", max_input_len)

        fn = lambda : _fwd_kernel[grid](
            q,
            k,
            v,
            k_cache,
            v_cache,
            block_table,
            sm_scale,
            k_scale,
            v_scale,
            b_start_loc,
            b_seq_len,
            b_ctx_len,
            v_cache.shape[3],
            k_cache.shape[4],
            output,
            block_table.stride(0),
            block_table.stride(1),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            k_cache.stride(4),  #[num_blocks, num_kv_heads, head_size/x, block_size, x]
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(
                3),  #[num_blocks, num_kv_heads, head_size, block_size]
            num_queries_per_kv=num_queries_per_kv,
            # BLOCK_M=BLOCK,
            BLOCK_DMODEL=Lk,
            BLOCK_DMODEL_PADDED=Lk_padded,
            # BLOCK_N=BLOCK,
            SLIDING_WINDOW=sliding_window,
            # num_warps=NUM_WARPS,
            # num_stages=1,
        )

        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

        return ms * 1e3


    benchmark_prefix_prefill.run(save_path=".", print_data=True)

def main():
    run_benchmark()


if __name__ == '__main__':
    sys.exit(main())