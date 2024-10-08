import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from tqdm import tqdm
import random

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import (get_config_file_name,
                                                  invoke_fused_moe_kernel,
                                                  moe_align_block_size)

def split_list_fixed(lst, num_chunks, split_id):
    # Determine the size of each chunk
    chunk_size = len(lst) // num_chunks
    remainder = len(lst) % num_chunks
    
    chunks = []
    start = 0
    for i in range(num_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    
    return chunks[split_id]

def main(args):
    os.environ["HIP_FORCE_DEV_KERNARG"] = "1"

    Max_M = 32768
    basic_M = [1, 2, 4] + list(range(8,264,8)) + [Max_M] + [1024, 1304, 744, 512]
    in_len = [200, 2000, 7000]
    BS = [1,2,4,8,16,32,64,128,256]
    extend_M_set = set()
    # The Mixtral 8x7B max M of MoE is 32768. When exceeding 32768, split M into several small M values.
    # when input_len is 6000 and bs 8 for example, we have to split 6000*8 by the 
    # {max multiple value of 1000 which is smaller than MaxM}, that is 30000 < 32768.
    # Split 48000 => {30000, 18000}
    for il in in_len:
        for bs in BS:
            bs_il = bs * il
            if bs_il <= Max_M:
                extend_M_set.add(bs_il)
            else:
                max_multiple = il * (Max_M//il)
                left = bs_il % max_multiple
                extend_M_set.add(max_multiple)
                if left>0:
                    extend_M_set.add(left)

    # Assign tasks as equaly as possible
    extend_M_set.update(basic_M)
    M_list = list(extend_M_set)
    M_list = sorted(M_list)
    print(f"All M value ={M_list}")
    M_list_split, idx, direction = [[] for i in range(args.TP)], 0, 1
    for M in M_list:
        M_list_split[idx].append(M)
        idx += direction
        if idx == args.TP:
            idx = args.TP-1
            direction = -1
        elif idx == -1:
            idx = 0
            direction = 1

    split_list = M_list_split[args.split_id]
    split_list = [400]
    print(f"GPU [{args.split_id}] The cases are: {split_list}")
    for M in tqdm(split_list, desc=f"GPU idx {args.split_id}"):
        run_grid(M, model=args.model, TP=args.TP)


## Utilize method from rocm/Triton tuning script
def get_full_tuning_space():
    configs = []

    block_m_range = [16, 32, 64, 128, 256]
    block_n_range = [16, 32, 64, 128, 256]
    block_k_range = [16, 32, 64, 128, 256]
    # split_k_range = [1] #, 2, 4, 5, 6, 8, 10, 12, 16, 18, 24]
    num_warps_range = [1, 4, 8, 16] # Good: 
    group_m_range = [1, 16]
    # For now we see better perf with num_stages=0 for all gemm configs we care
    # But keep this explicit so that we do not forget we may need to set it to
    # other values in the future
    num_stage_range = [0]
    waves_per_eu_range = [0]
    matrix_instr_nonkdim_range = [16]  # Triton crash when set 32
    kpack_range = [1, 2] 

    for block_m in block_m_range:
        for block_n in block_n_range:
            for block_k in block_k_range:
                for num_warps in num_warps_range:
                    for group_m in group_m_range:
                        # for split_k in split_k_range:
                        for num_stages in num_stage_range:
                            for waves_per_eu in waves_per_eu_range:
                                for (matrix_instr_nonkdim
                                     ) in matrix_instr_nonkdim_range:
                                    for kpack in kpack_range:
                                        configs.append({
                                            "BLOCK_SIZE_M": block_m,
                                            "BLOCK_SIZE_N": block_n,
                                            "BLOCK_SIZE_K": block_k,
                                            "GROUP_SIZE_M": group_m,
                                            "num_warps": num_warps,
                                            "num_stages": num_stages,
                                            "waves_per_eu": waves_per_eu,
                                            "matrix_instr_nonkdim":
                                            matrix_instr_nonkdim,
                                            "kpack": kpack,
                                        })

    return configs


## Utilize method from rocm/Triton tuning script
def prune_configs(M, N, K, configs):
    return configs


def union_of_list_of_dicts(l1, l2):
    result = []
    temp_list = l1.copy()
    temp_list.extend(l2)
    for myDict in temp_list:
        if myDict not in result:
            result.append(myDict)

    return result


def need_split_k(SIZE_M, SIZE_N, SIZE_K):
    return (SIZE_M < 64 or SIZE_N < 64) and SIZE_K > 1024


def run_grid(bs, model, TP):
    if model == '8x7B':
        d_model = 4096
        model_intermediate_size = 14336
    elif model == '8x22B':
        d_model = 6144
        model_intermediate_size = 16384
    else:
        raise ValueError(f'Unsupported Mixtral model {model}')

    num_total_experts = 8
    top_k = 2
    tp_size = TP
    num_warmup_calls =3
    num_calls = 10

    num_warmup_trials = 1
    num_trials = 1

    full_configs = get_full_tuning_space()
    M1 = bs * 2
    N1 = model_intermediate_size * 2 // tp_size
    K1 = d_model
    prune_configs_1 = prune_configs(M1, N1, K1, full_configs)

    M2 = bs * 2
    N2 = d_model
    K2 = model_intermediate_size // tp_size
    prune_configs_2 = prune_configs(M2, N2, K2, full_configs)

    configs = union_of_list_of_dicts(prune_configs_1, prune_configs_2)
    #print(f"{bs=} || {len(full_configs)=} | {len(prune_configs_1)=} | \
    #        {len(prune_configs_2)=} | {len(configs)=}")

    # my_config1 = {
    #     "BLOCK_SIZE_M": 64, # 256 SNOW
    #     "BLOCK_SIZE_N": 64,
    #     "BLOCK_SIZE_K": 64,
    #     "GROUP_SIZE_M": 1,
    #     "num_warps": 8,
    #     "num_stages": 0,
    #     "waves_per_eu": 0,
    #     "matrix_instr_nonkdim": 16,
    #     "kpack": 1
    # } 
    # my_config2 = {
    #     "BLOCK_SIZE_M": 128, # 400 Dell
    #     "BLOCK_SIZE_N": 128,
    #     "BLOCK_SIZE_K": 128,
    #     "GROUP_SIZE_M": 1,
    #     "num_warps": 16,
    #     "num_stages": 0,
    #     "waves_per_eu": 0,
    #     "matrix_instr_nonkdim": 16,
    #     "kpack": 2
    # } 
    # configs = [my_config1, my_config2]

    best_config = None
    best_time_us = 1e20

    for idx, config in enumerate(configs):
        # warmup
        try:
            for _ in range(num_warmup_trials):
                run_timing(
                    num_calls=num_warmup_calls,
                    bs=bs,
                    d_model=d_model,
                    num_total_experts=num_total_experts,
                    top_k=top_k,
                    tp_size=tp_size,
                    model_intermediate_size=model_intermediate_size,
                    config=config,
                )
        except triton.runtime.autotuner.OutOfResources:
            print("triton.runtime.autotuner.OutOfResources, skip.....")
            continue

        # benchmark
        for _ in range(num_trials):
            kernel_dur_ms = run_timing(
                num_calls=num_calls,
                bs=bs,
                d_model=d_model,
                num_total_experts=num_total_experts,
                top_k=top_k,
                tp_size=tp_size,
                model_intermediate_size=model_intermediate_size,
                config=config,
            )

            kernel_dur_us = 1000 * kernel_dur_ms
            # model_dur_ms = kernel_dur_ms * num_layers

            if kernel_dur_us < best_time_us:
                best_config = config
                best_time_us = kernel_dur_us

        short_config = {"M":config['BLOCK_SIZE_M'], "N":config['BLOCK_SIZE_N'], "K":config['BLOCK_SIZE_K'], 
                            'num_warps':config['num_warps'], 'GROUP_SIZE_M':config['GROUP_SIZE_M'],
                            'num_stages':config['num_stages'], 'matrix_instr_nonkdim':config['matrix_instr_nonkdim'],
                            'kpack': config['kpack']}
        print(f"M={bs}, [{idx}/{len(configs)}], config={short_config}, us={int(kernel_dur_us)}, best={int(best_time_us)}")

    print(f"bs={bs}, best_time_us={best_time_us}, best_config={best_config}")
    # print("best_config", best_config)

    # holds Dict[str, Dict[str, int]]
    filename = get_config_file_name(num_total_experts,
                                    model_intermediate_size // tp_size,
                                    dtype=None)
    print(f"writing config to file {filename}")
    existing_content = {}
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing_content = json.load(f)
    existing_content[str(bs)] = best_config
    with open(filename, "w") as f:
        json.dump(existing_content, f, indent=4)
        f.write("\n")

    best_content = {}
    if os.path.exists("best_perf.json"):
        with open("best_perf.json", "r") as f:
            best_content = json.load(f)
    best_content[str(bs)] = best_time_us
    with open("best_perf.json", "w") as f:
        json.dump(best_content, f, indent=4)
        f.write("\n")


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
        (num_total_experts, 2 * shard_intermediate_size, d_model+128),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    w2 = torch.rand(
        (num_total_experts, d_model, shard_intermediate_size+128),
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
    assert hidden_states.shape[1] == w1.shape[2] - 128, "Hidden size mismatch"
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
    # moe_kernels.topk_softmax(
    #     topk_weights,
    #     topk_ids,
    #     token_expert_indicies,
    #     gating_output.float(),  # TODO(woosuk): Optimize this.
    # )
    ops.topk_softmax(
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

    # print(f"w1.shape = {w1.shape}, w2.shape = {w2.shape}, "
    #       f"cache1.shape={intermediate_cache1.shape}, cache2.shape={intermediate_cache2.shape}, cache3.shape={intermediate_cache3.shape}")
    # exit(0)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E)

    ##################################

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    check_type=(tl.bfloat16 if hidden_states.dtype == torch.bfloat16
                          else tl.float16)

    error_flag = 0
    start_event.record()
    try:
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
                use_fp8_w8a8=False,
                use_int8_w8a16=False
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
               use_fp8_w8a8=False,
               use_int8_w8a16=False
            )
    except:
        #print("invoke_fused_moe_kernel error, skip this config...")
        error_flag = 1e9

    end_event.record()
    end_event.synchronize()
    # print(f"intermediate 0 shape = {intermediate_cache1.shape}")
    # print(f"intermediate 1 shape = {intermediate_cache2.shape}")
    # print(f"intermediate 2 shape = {intermediate_cache3.shape}")
    # print(f"config = {config}")
    # print(f"sorted token ids = {sorted_token_ids}")
    # print(f"sorted token ids shape = {sorted_token_ids.shape}")
    # print(f"expert ids = {expert_ids}")
    # print(f"num_tokens_post_padded = {num_tokens_post_padded}")

    dur_ms = start_event.elapsed_time(end_event) / num_calls + error_flag
    return dur_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="benchmark_mixtral_moe_rocm",
        description="Tune the fused_moe kernel for mixtral.")
    parser.add_argument(
        "--TP",
        type=int,
        choices=[16, 8, 4, 2, 1],
        help="Specify the TP value that the actual model will run on",
        required=True,
    )
    parser.add_argument(
        "--disable_split",
        default=False,
        action="store_true",
        help='If this option is specified, will not split the M_list into TP portions.'
    )
    parser.add_argument(
        "--split_id",
        type=int,
        help='The ith portion of M_list. The value should between [0, TP)'
    )
    parser.add_argument('--model',
                        type=str,
                        choices=['8x7B', '8x22B'],
                        help='The Mixtral model to benchmark')

    args = parser.parse_args()

    print(f"Running tuning for {args.model} model")
    print(f"TP is set to: {args.TP}")
    print(f"GPU-ID being used for tuning: {args.split_id}")
    sys.exit(main(args))
