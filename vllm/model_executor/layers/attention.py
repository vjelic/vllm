"""Multi-head attention."""
from typing import List, Optional

import os
import torch
import torch.nn as nn
#from xformers import ops as xops
#from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
#                                         LowerTriangularMaskWithTensorBias)

from vllm import attention_ops
from vllm import cache_ops
from vllm import pos_encoding_ops
from vllm.model_executor.input_metadata import InputMetadata
import flash_attn_cuda
import triton
import triton.language as tl

_SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 128, 256]


class PagedAttention(nn.Module):
    # pylint: disable=line-too-long
    """GPT-style multi-head PagedAttention.

    This class takes flattened 1D query, key, and value tensors as input. The
    input 1D tensors can either contain prompt tokens or generation tokens, in
    addition to paddings.

    If the input tensors contain prompt tokens, the layout is as follows:

    |<---------------------- num_valid_tokens ---------------------->|
    |<--------------- num_prompt_tokens -------------->|
    |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->|<--padding-->|

    Otherwise, the layout is as follows:

    |<------------------ num_valid_tokens ------------------->|
    |<------- num_generation_tokens (M) ------->|
    |<--generation_0-->|...|<--generation_M-1-->|<--padding-->|

    The prompts might have different lengths, while the generation tokens always
    have length 1. The paddings are appended to make the input length a multiple
    of 8, which is desirable for Tensor Cores.

    The class does the following:
    1. Perform multi_query_kv_attention for the prompts. This operation does
        not use the KV cache.
    2. Wait for the cache operations (e.g., swap, copy) to finish. The cache
        operations are issued by the cache engine before executing the forward
        pass of the model, and they are executed asynchronously.
    3. Reshape and store the input key and value tensors in the KV cache.
    4. Perform single_query_cached_kv_attention for the generation tokens.
        This operation reads the previous key and value tensors from the KV
        cache.
    5. Output a flattened 1D tensor.
    """

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 scale: float,
                 num_kv_heads: Optional[int] = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        #self.attn_op = xops.fmha.cutlass.FwOp()
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_mapping = torch.repeat_interleave(
            torch.arange(self.num_kv_heads, dtype=torch.int32, device="cuda"),
            self.num_queries_per_kv)

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"head_size ({self.head_size}) is not supported. "
                             f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}.")

        self.singleq_cu_seqlens_tensor = torch.tensor([0,1],dtype=torch.int,device='cuda')

    def set_attn_bias(self, input_metadata: InputMetadata) -> None:
        #if input_metadata.attn_bias:
        #    # Already set by a previous layer.
        #    return
        if input_metadata.cu_seqlens_tensor is not None:
            #print('>>>already set')
            return
        cu_seq_lens = [0]
        max_seqlen = 0
        for seq_len in input_metadata.prompt_lens:
            cu_seq_lens.append(cu_seq_lens[-1] + seq_len)
            if seq_len > max_seqlen: max_seqlen = seq_len
        input_metadata.cu_seqlens_tensor = torch.tensor(cu_seq_lens,dtype=torch.int,device='cuda')
        input_metadata.max_seqlen = max_seqlen
        #prompt_lens = input_metadata.prompt_lens
        #attn_bias = BlockDiagonalCausalMask.from_seqlens(prompt_lens)
        #input_metadata.attn_bias.append(attn_bias)

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
        tokens, n_kv_heads, head_dim = x.shape
        return (
                x[:, :, None, :]
                .expand(tokens, n_kv_heads, n_rep, head_dim)
                .reshape(tokens, n_kv_heads * n_rep, head_dim)
        )

    @triton.jit
    def _triton_fwd_kernel(
        Q, K, V, sm_scale,
        Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        stride_oz, stride_oh, stride_om, stride_on,
        Z, H, N_CTX,
        BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        # Select batch
        off_z = tl.program_id(1) // H
        # Select head within that batch
        off_h = tl.program_id(1) % H
        # Set Q block start offset. Q has shape [batch, seqlen, nheads, d]
        q_offset = off_z * stride_qz + off_h * stride_qh
        kv_offset = off_z * stride_kz + off_h * stride_kh
        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0)
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + kv_offset,
            shape=(BLOCK_DMODEL, N_CTX),
            strides=(stride_kk, stride_kn),
            offsets=(0, 0),
            block_shape=(BLOCK_DMODEL, BLOCK_N),
            order=(0, 1)
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + kv_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_vk, stride_vn),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_DMODEL),
            order=(1, 0)
        )
        # initialize offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_qm = offs_m * stride_qm
        offs_d = tl.arange(0, BLOCK_DMODEL)
        # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # scale sm_scale by log_2(e) and use
        # 2^x instead of exp in the loop because CSE and LICM
        # don't work as expected with `exp` in the loop
        qk_scale = sm_scale * 1.4426950408889634
        # load q: it will stay in SRAM throughout
        q = tl.load(Q_block_ptr)
        q = (q * qk_scale).to(tl.float16)
        # loop over k, v and update accumulator
        lo = 0
        hi = N_CTX
        for start_n in range(lo, hi, BLOCK_N):
            # -- load k, v --
            k = tl.load(K_block_ptr)
            v = tl.load(V_block_ptr)
            # -- compute qk ---
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            # -- compute scaling constant ---
            m_i_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.math.exp2(m_i - m_i_new)
            p = tl.math.exp2(qk - m_i_new[:, None])
            # -- scale and update acc --
            acc_scale = l_i * 0 + alpha  # workaround some compiler bug
            acc *= acc_scale[:, None]
            acc += tl.dot(p.to(tl.float16), v)
            # -- update m_i and l_i --
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_i_new
            # update pointers
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        # write back l and m
        acc = acc / l_i[:, None]
        # write back O
        o_offset = off_z * stride_oz + off_h * stride_oh
        O_block_ptr = tl.make_block_ptr(
            base=Out + o_offset,
            shape=(N_CTX, BLOCK_DMODEL),
            strides=(stride_om, stride_on),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0)
        )
        tl.store(O_block_ptr, acc.to(tl.float16))

    def triton_fwd(self, q, k, v, o, sm_scale):
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        if q.dim() == 3:
            batch = 1
            seqlen_q, nheads, d = q.shape
            q = q.view(batch, seqlen_q, nheads, d)
            k = k.view(batch, seqlen_q, nheads, d)
            v = v.view(batch, seqlen_q, nheads, d)
            o = o.view(batch, seqlen_q, nheads, d)
        elif q.dim() == 4:
            batch, seqlen_q, nheads, d = q.shape
        else:
            raise ValueError(f"Unsupported q dimensions {q.dim()}")
        BLOCK_M = 128
        BLOCK_N = 64
        num_warps = 4
        num_stages = 1

        grid = (triton.cdiv(q.shape[1], BLOCK_M), q.shape[0] * q.shape[2], 1)

        self._triton_fwd_kernel[grid](
            q, k, v, sm_scale,
            o,
            q.stride(0), q.stride(2), q.stride(1), q.stride(3),
            k.stride(0), k.stride(2), k.stride(1), k.stride(3),
            v.stride(0), v.stride(2), v.stride(1), v.stride(3),
            o.stride(0), o.stride(2), o.stride(1), o.stride(3),
            batch, nheads, seqlen_q,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=d,
            num_warps=num_warps,
            num_stages=1,
        )

    def multi_query_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """Normal attention for the prompt tokens.

        Args:
            output: shape = [num_prompt_tokens, num_heads, head_size]
            query: shape = [num_prompt_tokens, num_heads, head_size]
            key: shape = [num_prompt_tokens, num_kv_heads, head_size]
            value: shape = [num_prompt_tokens, num_kv_heads, head_size]
            input_metadata: metadata for paged attention.
        """
        if self.num_kv_heads != self.num_heads:
            # Project the key and value tensors to the desired number of heads.
            #key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=1)
            #value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=1)
            key = self.repeat_kv(key, self.num_queries_per_kv)
            value = self.repeat_kv(value, self.num_queries_per_kv)

        # TODO(woosuk): The unsqueeze op may incur some CPU overhead. Optimize.
        #out = xops.memory_efficient_attention_forward(
        #    query.unsqueeze(0),
        #    key.unsqueeze(0),
        #    value.unsqueeze(0),
        #    attn_bias=input_metadata.attn_bias[0],
        #    p=0.0,
        #    scale=self.scale,
        #    op=self.attn_op,
        #)
        # TODO(woosuk): Unnecessary copy. Optimize.
        #output.copy_(out.squeeze(0))
        #cu_seq_lens = [0]
        #max_seqlen = 0
        #for seq_len in input_metadata.prompt_lens:
        #    cu_seq_lens.append(cu_seq_lens[-1] + seq_len)
        #    if seq_len > max_seqlen: max_seqlen = seq_len
        #cu_seqlens_tensor = torch.tensor(cu_seq_lens,dtype=torch.int,device=query.device)
        #print('>>prompt lens',input_metadata.prompt_lens,input_metadata.cu_seqlens_tensor,input_metadata.max_seqlen)
        if os.environ.get('VLLM_USE_TRITON'):
            self.triton_fwd(query, key, value, output, self.scale)
        else:
            flash_attn_cuda.fwd( query, key, value, output, input_metadata.cu_seqlens_tensor, input_metadata.cu_seqlens_tensor, 
                                 input_metadata.max_seqlen, input_metadata.max_seqlen, 0.0,
                                 self.scale, False, True, False, False, False, 0, None)
        #ref_out = self.ref_multi_query_kv_attention(
        #    input_metadata.prompt_lens,
        #    query,
        #    key,
        #    value,
        #    dtype=torch.float16,
        #)
        #output.copy_(ref_out)
        return output

    def ref_single_query_cached_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> None:
        #num_heads = value_cache.shape[1]
        #head_size = value_cache.shape[2]
        #block_size = value_cache.shape[3]
    
        num_input_tokens = query.shape[0]
        #print('>>>num inp',query.shape,output.shape,self.keyc.shape)
        for i in range(num_input_tokens):
            q = query[i].unsqueeze(0)
            '''block_table = block_tables[i]
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
            print('>>>kvblk',keys.shape,values.shape)'''
    
            #scale = 1.0 / (head_size**0.5)
            #out = self.ref_masked_attention(q, keys, values, scale)
            if self.num_kv_heads != self.num_heads:
                # Project the key and value tensors to the desired number of heads.
                key = torch.repeat_interleave(self.keyc, self.num_queries_per_kv, dim=1)
                value = torch.repeat_interleave(self.valc,
                                            self.num_queries_per_kv,
                                            dim=1)
            else:
                key=self.keyc
                value=self.valc
            out = self.ref_masked_attention(q, key, value, self.scale)
            out = out.view(self.num_heads, self.head_size)
            output[i].copy_(out, non_blocking=True)
            #batch_size = 1
            #seqlen_k = self.keyc.shape[0]
            #cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
            #                                device=query.device)
            #print('>>>',cu_seqlens_q,cu_seqlens_k)
            #softmax_lse,*rest = flash_attn_cuda.fwd( query, self.keyc, self.valc, output, self.singleq_cu_seqlens_tensor, cu_seqlens_k, 
            #                                         1, seqlen_k, 0.0,
            #                                         self.scale, False, False, False, True, False, 0, None)

    def ref_masked_attention(
        self,
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

    def single_query_cached_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> None:
        """PagedAttention for the generation tokens.

        Args:
            output: shape = [num_generation_tokens, num_heads, head_size]
            query: shape = [num_generation_tokens, num_heads, head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for paged attention.
        """
        block_size = value_cache.shape[3]
        attention_ops.single_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            self.head_mapping,
            self.scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len,
            None,  # alibi_slopes
        )
        #print('>>>single query',input_metadata.context_lens,query.shape,output.shape,self.keyc.shape,self.valc.shape)
        #self.ref_single_query_cached_kv_attention(
        #    output,
        #    query,
        #    key_cache,
        #    value_cache,
        #    input_metadata.block_tables,
        #    input_metadata.context_lens,
        #)

    def ref_multi_query_kv_attention(
        self,
        cu_seq_lens: List[int],
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        #print('>>>',cu_seq_lens)
        head_size = query.shape[-1]
        scale = 1.0 / (head_size**0.5)
    
        num_seqs = len(cu_seq_lens)
        ref_outputs = []
        for i in range(num_seqs):
            if i==0:
                start_idx = 0
            else:
                start_idx = cu_seq_lens[i-1]
            end_idx = cu_seq_lens[i]
            seq_len = end_idx - start_idx
    
            # Create attention mask.
            attn_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=dtype, device='cuda'),
                                   diagonal=1)
            attn_mask = attn_mask * torch.finfo(dtype).min
            attn_mask = attn_mask.to(dtype=dtype, device='cuda')
    
            ref_output = self.ref_masked_attention(
                query[start_idx:end_idx],
                key[start_idx:end_idx],
                value[start_idx:end_idx],
                scale,
                attn_mask=attn_mask,
            )
            ref_outputs.append(ref_output)
        ref_output = torch.cat(ref_outputs, dim=0)
        return ref_output

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        """PagedAttention forward pass.

        NOTE: The query, key, and value tensors must be sliced from a qkv
        tensor of shape [num_tokens, 3 * num_heads * head_size].

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for paged attention.
            cache_event: event to wait for the cache operations to finish.

        Returns:
            shape = [num_tokens, num_heads * head_size]
        """

        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        # Pre-allocate the output tensor.
        output = torch.empty_like(query)

        # Compute the attention op for prompts.
        num_prompt_tokens = input_metadata.num_prompt_tokens
        if num_prompt_tokens > 0:
            # Prompt run.
            assert input_metadata.num_generation_tokens == 0
            self.set_attn_bias(input_metadata)
            self.multi_query_kv_attention(
                output[:num_prompt_tokens],
                query[:num_prompt_tokens],
                key[:num_prompt_tokens],
                value[:num_prompt_tokens],
                input_metadata,
            )

        # Wait until the cache op is done.
        if cache_event is not None:
            cache_event.wait()

        # Reshape the keys and values and store them in the cache.
        # When key_cache and value_cache are not provided, the new key
        # and value vectors will not be cached.
        num_valid_tokens = input_metadata.num_valid_tokens
        #print('>>>num valid tokens',num_valid_tokens)
        if (num_valid_tokens > 0 and key_cache is not None
                and value_cache is not None):
            # The stride is 3 because the key and value are sliced from qkv.
            cache_ops.reshape_and_cache(
                key[:num_valid_tokens],
                value[:num_valid_tokens],
                key_cache,
                value_cache,
                input_metadata.slot_mapping,
            )
            #pass #do nothing as we're not using paged kvcache right now

        #if num_prompt_tokens > 0:
        #    self.keyc = key[:num_prompt_tokens]
        #    self.valc = value[:num_prompt_tokens]
        #    #print('>>>kvc',self.keyc.shape,self.valc.shape)
        #elif input_metadata.num_generation_tokens > 0 and num_valid_tokens>0:
        #    self.keyc = torch.cat((self.keyc,key[:num_valid_tokens]),dim=0)
        #    self.valc = torch.cat((self.valc,value[:num_valid_tokens]),dim=0)
        #    #print('>>>kvc',self.keyc.shape,self.valc.shape)
        #else:
        #    print('>>>should not be here')


        if input_metadata.num_generation_tokens > 0:
            # Decoding run.
            assert input_metadata.num_prompt_tokens == 0
            assert key_cache is not None and value_cache is not None, (
                "key_cache and value_cache must be provided when "
                "generating tokens.")
            # Compute the attention op for generation tokens.
            self.single_query_cached_kv_attention(
                output[num_prompt_tokens:num_valid_tokens],
                query[num_prompt_tokens:num_valid_tokens], key_cache,
                value_cache, input_metadata)

        # Reshape the output tensor.
        # NOTE(woosuk): The output tensor may include paddings.
        return output.view(-1, self.num_heads * self.head_size)


class PagedAttentionWithRoPE(PagedAttention):
    """PagedAttention with GPT-NeoX style rotary embedding."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        rotary_dim: int,
        max_position: int = 8192,
        base: int = 10000,
        num_kv_heads: Optional[int] = None,
    ) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads)

        # Create the cos and sin cache.
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2) / rotary_dim))
        t = torch.arange(max_position).float()
        freqs = torch.einsum("i,j -> ij", t, inv_freq.float())
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)

        # FIXME(woosuk): This assumes that we configure the default dtype when
        # initializing the model.
        # TODO(woosuk): Make it more robust.
        torch_dtype = torch.get_default_dtype()
        cache = cache.to(torch_dtype)
        # Embedding size: [max_position, rotary_dim]
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        """ PagedAttention forward pass with rotary embedding.

        Args:
            positions: shape = [num_tokens]
                        query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for paged attention.
            cache_event: event to wait for the cache operations to finish.

        Returns:
            shape = [num_tokens, num_heads * head_size]
        """

        # Apply rotary embedding to the query and key before passing them
        # to the attention op.
        pos_encoding_ops.rotary_embedding_neox(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
        )
        return super().forward(
            query,
            key,
            value,
            key_cache,
            value_cache,
            input_metadata,
            cache_event,
        )


class PagedAttentionWithALiBi(PagedAttention):
    """PagedAttention with ALiBi attention bias."""

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 scale: float,
                 slopes: List[float],
                 num_kv_heads: Optional[int] = None) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads)
        assert len(slopes) == num_heads

        slopes = torch.tensor(slopes, dtype=torch.float32)
        self.register_buffer("alibi_slopes", slopes, persistent=False)

    def set_attn_bias(self, input_metadata: InputMetadata) -> None:
        if input_metadata.attn_bias:
            # Already set by a previous layer.
            return
        # Generates ALiBi mask for each prompt.
        for prompt_len in input_metadata.prompt_lens:
            bias = torch.arange(prompt_len)
            # Note(zhuohan): HF uses
            #     `bias = bias[None, :].repeat(prompt_len, 1)`
            # here. We find that both biases give the same results, but
            # the bias below more accurately follows the original ALiBi
            # paper.
            bias = bias[None, :] - bias[:, None]
            bias = bias.to(self.alibi_slopes.device)

            # When using custom attention bias, xformers requires the bias to
            # be sliced from a tensor whose length is a multiple of 8.
            padded_len = (prompt_len + 7) // 8 * 8
            bias = torch.empty(
                self.num_heads,
                padded_len,
                padded_len,
                device=self.alibi_slopes.device,
            )[:, :prompt_len, :prompt_len].copy_(bias)
            bias.mul_(self.alibi_slopes[:, None, None])
            attn_bias = LowerTriangularMaskWithTensorBias(bias)
            input_metadata.attn_bias.append(attn_bias)

    def multi_query_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """Attention with ALiBi bias for the prompt tokens.

        Args:
            output: shape = [num_prompt_tokens, num_heads, head_size]
            query: shape = [num_prompt_tokens, num_heads, head_size]
            key: shape = [num_prompt_tokens, num_kv_heads, head_size]
            value: shape = [num_prompt_tokens, num_kv_heads, head_size]
            input_metadata: metadata for paged attention.
        """
        if self.num_kv_heads != self.num_heads:
            # Project the key and value tensors to the desired number of heads.
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=1)
            value = torch.repeat_interleave(value,
                                            self.num_queries_per_kv,
                                            dim=1)

        # FIXME(woosuk): Because xformers does not support dynamic sequence
        # lengths with custom attention bias, we process each prompt one by
        # one. This is inefficient, especially when we have many short prompts.
        start = 0
        for i, prompt_len in enumerate(input_metadata.prompt_lens):
            end = start + prompt_len
            out = xops.memory_efficient_attention_forward(
                query[None, start:end],
                key[None, start:end],
                value[None, start:end],
                attn_bias=input_metadata.attn_bias[i],
                p=0.0,
                scale=self.scale,
                op=self.attn_op,
            )
            # TODO(woosuk): Unnecessary copy. Optimize.
            output[start:end].copy_(out.squeeze(0))
            start += prompt_len
        return output

    def single_query_cached_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> None:
        """PagedAttention with ALiBi bias for the generation tokens.

        Args:
            output: shape = [num_generation_tokens, num_heads, head_size]
            query: shape = [num_generation_tokens, num_heads, head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for paged attention.
        """
        block_size = value_cache.shape[3]
        attention_ops.single_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            self.head_mapping,
            self.scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len,
            self.alibi_slopes,
        )
