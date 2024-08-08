import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from tqdm import tqdm

import vllm._moe_C as moe_kernels
from vllm._C import ops
from vllm.model_executor.layers.fused_moe import (get_config_file_name,
                                                  invoke_fused_moe_kernel,
                                                  moe_align_block_size)


def run_timing(
    num_calls: int,
    bs: int,
    d_model: int,
    num_total_experts: int,
    top_k: int,
    tp_size: int,
    model_intermediate_size: int,
    config,
) -> float:
    
    shard_intermediate_size = model_intermediate_size // tp_size

    hidden_states = torch.rand(
        (bs, d_model),
        device="cuda",
        dtype=torch.float16,
    )

    w1 = torch.rand(
        (num_total_experts, 2 * shard_intermediate_size, d_model),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    w2 = torch.rand(
        (num_total_experts, d_model, shard_intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    gating_output = F.softmax(
        torch.rand(
            # (num_calls, bs, num_total_experts), # THIS
            (bs, num_total_experts),
            device=hidden_states.device,
            dtype=torch.float32,
        ),
        dim=-1,
    )

    ###### Stuff from fused moe ######

    assert (hidden_states.shape[0] == gating_output.shape[0]
            ), "Number of tokens mismatch"
    assert hidden_states.shape[1] == w1.shape[2], "Hidden size mismatch"
    assert gating_output.shape[1] == w1.shape[0], "Number of experts mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]
    M, _ = hidden_states.shape
    E, N, _ = w1.shape
    topk_ = 2
    topk_weights = torch.empty(M,
                               topk_,
                               dtype=torch.float32,
                               device=hidden_states.device)
    topk_ids = torch.empty(M,
                           topk_,
                           dtype=torch.int32,
                           device=hidden_states.device)
    token_expert_indicies = torch.empty(M,
                                        topk_,
                                        dtype=torch.int32,
                                        device=hidden_states.device)
    moe_kernels.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output.float(),  # TODO(woosuk): Optimize this.
    )
    del token_expert_indicies  # Not used. Will be used in the future.

    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    intermediate_cache1 = torch.empty(
        (M, topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = torch.empty(
        (M, topk_ids.shape[1], w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E)

    ##################################

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)




    start_event.record()
    for i in range(num_calls):
        invoke_fused_moe_kernel(
            hidden_states,
            w1,
            intermediate_cache1,
            None,  # a1_scale
            None,  # w1_scale
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,
            topk_ids.shape[1],
            config,
            compute_type=(tl.bfloat16 if hidden_states.dtype == torch.bfloat16
                          else tl.float16),
            use_fp8=False,
        )

        ops.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, N))

        invoke_fused_moe_kernel(
            intermediate_cache2,
            w2,
            intermediate_cache3,
            None,  # a2_scale
            None,  # w2_scale
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            True,
            1,
            config,
            compute_type=(tl.bfloat16 if hidden_states.dtype == torch.bfloat16
                          else tl.float16),
            use_fp8=False,
        )

    end_event.record()
    end_event.synchronize()

    dur_ms = start_event.elapsed_time(end_event) / num_calls
    return dur_ms


def parse_args():
    parser = argparse.ArgumentParser(
        prog="tune_moe",
        description="Tune the fused_moe kernel")
    parser.add_argument(
        "-bs",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-d_model",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-num_expt",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-top_k",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-tp_size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-inter_size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-config_file",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    bs = args.bs
    d_model = args.d_model
    num_total_experts = args.num_expt
    top_k = args.top_k
    tp_size = args.tp_size
    model_intermediate_size = args.inter_size
    config_file_name = args.config_file
    with open(config_file_name, 'r') as f:
        configs = json.load(f)
    config = configs[bs]
    num_calls = 100
    kennel_time = run_timing(num_calls,
                      bs,
                      d_model,
                      num_total_experts,
                      top_k,
                      tp_size,
                      model_intermediate_size,
                      config)
    print(f"kernel_time = {kennel_time}")


if __name__ == "__main__":
    main()
