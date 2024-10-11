import random
import torch
import triton
import triton.language as tl


assert torch.cuda.is_available()

def generate_input(
    num_heads: int,
    num_queries_per_kv: int,
    head_size: int,
    sliding_window: int = 0,
    dtype: torch.dtype = torch.float16,
    kv_cache_dtype: str = "auto",
    device: str,
):
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    torch.set_default_device(device)

    # Need this, otherwise when we capture the graph the process
    # for GPU 1 would run on both GPU0 and GPU1 and things would hang
    #
    # see also similar issue: https://github.com/Dao-AILab/flash-attention/issues/523
    torch.cuda.set_device(device)

    MAX_SEQ_LEN = 1024
    MAX_CTX_LEN = 1024
    BS = 10
    cache_size = 640
    block_size = 32
    max_block_per_request = 64
    query_lens = [random.randint(16, MAX_SEQ_LEN) for _ in range(BS)]
    ctx_lens = [random.randint(16, MAX_CTX_LEN) for _ in range(BS)]
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
    block_table = values[:BS * max_block_per_request].view(
        BS, max_block_per_request)
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
    for i in range(BS):
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

grid = (batch, head, triton.cdiv(max_input_len, BLOCK))  # batch, head,

# 0 means "disable"
if sliding_window is None or sliding_window <= 0:
    sliding_window = 0

_fwd_kernel[grid](
    q,
    k,
    v,
    k_cache,
    v_cache,
    b_loc,
    sm_scale,
    k_scale,
    v_scale,
    b_start_loc,
    b_seq_len,
    b_ctx_len,
    v_cache.shape[3],
    k_cache.shape[4],
    o,
    b_loc.stride(0),
    b_loc.stride(1),
    q.stride(0),
    q.stride(1),
    q.stride(2),
    k.stride(0),
    k.stride(1),
    k.stride(2),
    v.stride(0),
    v.stride(1),
    v.stride(2),
    o.stride(0),
    o.stride(1),
    o.stride(2),
    k_cache.stride(0),
    k_cache.stride(1),
    k_cache.stride(2),
    k_cache.stride(3),
    k_cache.stride(
        4),  #[num_blocks, num_kv_heads, head_size/x, block_size, x]
    v_cache.stride(0),
    v_cache.stride(1),
    v_cache.stride(2),
    v_cache.stride(
        3),  #[num_blocks, num_kv_heads, head_size, block_size]
    num_queries_per_kv=num_queries_per_kv,
    BLOCK_M=BLOCK,
    BLOCK_DMODEL=Lk,
    BLOCK_DMODEL_PADDED=Lk_padded,
    BLOCK_N=BLOCK,
    SLIDING_WINDOW=sliding_window,
    num_warps=NUM_WARPS,
    num_stages=1,
)