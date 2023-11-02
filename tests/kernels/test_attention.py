import random
from typing import List, Optional, Tuple

import torch
#from xformers import ops as xops
#from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

from vllm import attention_ops
import flash_attn_cuda

import pytest

MAX_SEQ_LEN = 4096*8
TEST_SEED = 0


DTYPES = [torch.float16] # , torch.bfloat16
NUM_GEN_SEQS = [1]  # Arbitrary values for testing
NUM_PREFILL_SEQS = [3]  # Arbitrary values for testing
NUM_HEADS = [(8, 1), (64, 8)]  # Arbitrary values for testing
HEAD_SIZES = [128]
BLOCK_SIZES = [16]
USE_ALIBI = [False]
SEEDS = [0]

def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    query = query * scale
    attn = torch.einsum('qhd,khd->hqk', query, key)
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = torch.softmax(attn, dim=-1)
    out = torch.einsum('hqk,khd->qhd', attn, value)
    return out

def ref_single_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
) -> None:
    num_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]

    num_input_tokens = query.shape[0]
    for i in range(num_input_tokens):
        q = query[i].unsqueeze(0)
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)

        scale = 1.0 / (head_size**0.5)
        out = ref_masked_attention(q, keys, values, scale)
        out = out.view(num_heads, head_size)
        output[i].copy_(out, non_blocking=True)


def ref_multi_query_kv_attention(
    cu_seq_lens: List[int],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    head_size = query.shape[-1]
    scale = 1.0 / (head_size**0.5)

    num_seqs = len(cu_seq_lens) - 1
    ref_outputs = []
    for i in range(num_seqs):
        start_idx = cu_seq_lens[i]
        end_idx = cu_seq_lens[i + 1]
        seq_len = end_idx - start_idx

        # Create attention mask.
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=dtype),
                               diagonal=1)
        attn_mask = attn_mask * torch.finfo(dtype).min
        attn_mask = attn_mask.to(dtype=dtype, device='cuda')

        ref_output = ref_masked_attention(
            query[start_idx:end_idx],
            key[start_idx:end_idx],
            value[start_idx:end_idx],
            scale,
            attn_mask=attn_mask,
        )
        ref_outputs.append(ref_output)
    ref_output = torch.cat(ref_outputs, dim=0)
    return ref_output


def ref_multi_query_cached_kv_attention(
    cu_query_lens: List[int],
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    scale = 1.0 / (head_size**0.5)

    num_queries = len(cu_query_lens) - 1
    ref_outputs = []
    for i in range(num_queries):
        start_idx = cu_query_lens[i]
        end_idx = cu_query_lens[i + 1]
        query_len = end_idx - start_idx
        context_len = int(context_lens[i])
        block_table = block_tables[i]

        # Create attention mask
        attn_mask = torch.triu(torch.ones(query_len, context_len),
                               diagonal=context_len - query_len + 1) * -1e5
        attn_mask = attn_mask.to(dtype=dtype, device='cuda')

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)

        ref_output = ref_masked_attention(
            query[start_idx:end_idx],
            keys,
            values,
            scale,
            attn_mask=attn_mask,
        )
        ref_outputs.append(ref_output)
    ref_output = torch.cat(ref_outputs, dim=0)
    return ref_output


@torch.inference_mode()
def run_single_query_cached_kv_attention(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    num_kv_heads: int = None,
) -> None:
    qkv = torch.empty(num_tokens,
                      3,
                      num_heads,
                      head_size,
                      dtype=dtype,
                      device='cuda')
    qkv.uniform_(-1e-3, 1e-3)
    query, _, _ = qkv.unbind(dim=1)
    print('>>>q',query.shape)

    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_block_shape = (num_heads, head_size // x, block_size, x)
    key_cache = torch.empty(size=(num_blocks, *key_block_shape),
                            dtype=dtype,
                            device='cuda')
    key_cache.uniform_(-1e-3, 1e-3)
    value_block_shape = (num_heads, head_size, block_size)
    value_cache = torch.empty(size=(num_blocks, *value_block_shape),
                              dtype=dtype,
                              device='cuda')
    value_cache.uniform_(-1e-3, 1e-3)

    context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_tokens)]
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device='cuda')

    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(num_tokens):
        block_table = [
            random.randint(0, num_blocks - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device='cuda')
    head_mapping = torch.arange(num_heads, dtype=torch.int32, device="cuda")

    scale = float(1.0 / (head_size**0.5))

    num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    assert num_heads % num_kv_heads == 0
    num_queries_per_kv = num_heads // num_kv_heads
    head_mapping = torch.repeat_interleave(
        torch.arange(num_kv_heads, dtype=torch.int32, device="cuda"),
                     num_queries_per_kv)

    output = torch.empty(num_tokens,
                         num_heads,
                         head_size,
                         dtype=dtype,
                         device='cuda')
    attention_ops.single_query_cached_kv_attention(
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        None,  # ALiBi slopes.
    )

    ref_output = torch.empty_like(query)
    ref_single_query_cached_kv_attention(
        ref_output,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
    )
    # NOTE(woosuk): Due to the difference in the data types the two
    # implementations use for attention softmax logits and accumulation,
    # there is a small difference in the final outputs.
    # We should use a relaxed tolerance for the test.
    assert torch.allclose(output, ref_output, atol=1e-3, rtol=1e-5)


@torch.inference_mode()
def run_multi_query_kv_attention(
    num_seqs: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
) -> None:
    seq_lens = random.sample(range(1, MAX_SEQ_LEN), num_seqs)
    num_tokens = sum(seq_lens)

    scale = float(1.0 / (head_size**0.5))
    qkv = torch.empty(num_tokens,
                      3,
                      num_heads,
                      head_size,
                      dtype=dtype,
                      device='cuda')
    qkv.uniform_(-1e-3, 1e-3)
    query, key, value = qkv.unbind(dim=1)

    #attn_op = xops.fmha.cutlass.FwOp()
    #attn_bias = BlockDiagonalCausalMask.from_seqlens(seq_lens)
    #output = xops.memory_efficient_attention_forward(
    #    query.unsqueeze(0),
    #    key.unsqueeze(0),
    #    value.unsqueeze(0),
    #    attn_bias=attn_bias,
    #    p=0.0,
    #    scale=scale,
    #    op=attn_op,
    #)
    #output = output.squeeze(0)


    cu_seq_lens = [0]
    max_seqlen = 0
    for seq_len in seq_lens:
        cu_seq_lens.append(cu_seq_lens[-1] + seq_len)
        if seq_len > max_seqlen: max_seqlen = seq_len

    cu_seqlens_tensor = torch.tensor(cu_seq_lens,dtype=torch.int,device='cuda')
    print(seq_lens,cu_seqlens_tensor,max_seqlen)
    ref_output = ref_multi_query_kv_attention(
        cu_seq_lens,
        query,
        key,
        value,
        dtype,
    )
    output = torch.empty_like(query)
    softmax_lse,*rest = flash_attn_cuda.fwd( query, key, value, output, cu_seqlens_tensor, cu_seqlens_tensor, max_seqlen, max_seqlen, 0.0,
                                          query.shape[-1] ** (-0.5), False, True, False, True, False, 0, None)
    assert torch.allclose(output, ref_output, atol=1e-3, rtol=1e-5)

def test_single_query_cached_kv_attention() -> None:
    torch.random.manual_seed(TEST_SEED)
    torch.cuda.manual_seed(TEST_SEED)
    for dtype in [torch.half, torch.float]: #bf16, float commeted out            # FP16
        for block_size in [8, 16, 32]:                                           # 16
            for head_size in [64, 80, 96, 112, 128, 256]: #256 commented out     # 40
                print(f'Testing single_query_cached_kv_attention with '
                      f'dtype={dtype}, block_size={block_size}, '
                      f'head_size={head_size}')
                run_single_query_cached_kv_attention(
                    num_tokens=1, #37
                    num_heads=32, #3
                    head_size=head_size,
                    block_size=block_size,
                    num_blocks=1024,
                    dtype=dtype,
                )

def test_multi_query_kv_attention() -> None:
    torch.random.manual_seed(TEST_SEED)
    torch.cuda.manual_seed(TEST_SEED)
    for dtype in [torch.half, torch.bfloat16]: #torch.float commented out
        for head_size in [64, 80, 96, 112, 128]: #256 commented out
            print(f'Testing multi_query_kv_attention with dtype={dtype}, '
                  f'head_size={head_size}')
            run_multi_query_kv_attention(
                num_seqs=5,
                num_heads=3,
                head_size=head_size,
                dtype=dtype,
            )


def ref_masked_attention_V2(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out

def ref_single_query_cached_kv_attention_V2(
    output: torch.Tensor,
    query: torch.Tensor,
    num_queries_per_kv: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables = block_tables.cpu().tolist()
    context_lens = context_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables[i]
        context_len = int(context_lens[i])

        keys = []
        values = []
        for j in range(context_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_kv_heads, head_size)
            keys.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values.append(v)
        keys = torch.stack(keys, dim=0)
        values = torch.stack(values, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(context_len, device="cuda").int()
            alibi_bias = (position_ids - context_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(
                1, 1, -1)

        out = ref_masked_attention_V2(q, keys, values, scale, alibi_bias)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)

NUM_BLOCKS = 128  # Arbitrary values for testing
PARTITION_SIZE = 512

#@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("version", ["v2"])
@pytest.mark.parametrize("num_seqs", NUM_GEN_SEQS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("use_alibi", USE_ALIBI)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_paged_attention(
    kv_cache_factory,
    version: str,
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(num_seqs,
                        num_query_heads,
                        head_size,
                        dtype=dtype,
                        device="cuda")
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_mapping = torch.repeat_interleave(
        torch.arange(num_kv_heads, dtype=torch.int32, device="cuda"),
        num_queries_per_kv)
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads,
                                   dtype=torch.float,
                                   device="cuda")

    context_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
    context_lens[-1] = MAX_SEQ_LEN
    max_context_len = max(context_lens)
    context_lens = torch.tensor(context_lens, dtype=torch.int, device="cuda")

    # Create the block tables.
    max_num_blocks_per_seq = (max_context_len + block_size - 1) // block_size
    block_tables = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1)
            for _ in range(max_num_blocks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int, device="cuda")

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(NUM_BLOCKS, block_size, 1,
                                                num_kv_heads, head_size, dtype,
                                                seed)
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Call the paged attention kernel.
    output = torch.empty_like(query)
    if version == "v1":
        attention_ops.paged_attention_v1(
            output,
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
        )
    elif version == "v2":
        num_partitions = ((max_context_len + PARTITION_SIZE - 1) //
                          PARTITION_SIZE)
        assert PARTITION_SIZE % block_size == 0
        num_seqs, num_heads, head_size = output.shape
        tmp_output = torch.empty(
            size=(num_seqs, num_heads, num_partitions, head_size),
            dtype=output.dtype,
            device=output.device,
        )
        exp_sums = torch.empty(
            size=(num_seqs, num_heads, num_partitions),
            dtype=torch.float32,
            device=output.device,
        )
        max_logits = torch.empty_like(exp_sums)
        print('>>>',context_lens,max_context_len)
        attention_ops.paged_attention_v2(
            output,
            exp_sums,
            max_logits,
            tmp_output,
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
        )
    else:
        assert False, f"Unknown version: {version}"

    # Run the reference implementation.
    ref_output = torch.empty_like(query)
    ref_single_query_cached_kv_attention_V2(
        ref_output,
        query,
        num_queries_per_kv,
        key_cache,
        value_cache,
        block_tables,
        context_lens,
        scale,
        alibi_slopes,
    )

    # NOTE(woosuk): Due to the kernel-level differences in the two
    # implementations, there is a small numerical difference in the two
    # outputs. Thus, we use a relaxed tolerance for the test.
    assert torch.allclose(output, ref_output, atol=5e-3, rtol=1e-5)
