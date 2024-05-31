import argparse
import os
import torch.profiler
from benchmarks.benchmark_latency import main

import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams


def profile_benchmark(args):
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            with_modules=True,
            ) as prof:

        main(args)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    chrome_trace_filename = f"mixtral_trace_h100_{timestamp}.json"
    stack_trace_file = f"mixtral_stack_trace_h100_{timestamp}.json"
    prof.export_chrome_trace(chrome_trace_filename)
    prof.export_stacks(stack_trace_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Profile Benchmark Latency")
    # Add the same arguments as your benchmark script
    # Example:
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--quantization',
                        '-q',
                        choices=['awq', 'gptq', 'squeezellm', None],
                        default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n',
                        type=int,
                        default=1,
                        help='Number of generated sequences per prompt.')
    parser.add_argument('--use-beam-search', action='store_true')
    parser.add_argument('--num-iters-warmup',
                        type=int,
                        default=10,
                        help='Number of iterations to run for warmup.')
    parser.add_argument('--num-iters',
                        type=int,
                        default=30,
                        help='Number of iterations to run.')
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--enforce-eager',
                        action='store_true',
                        help='enforce eager mode and disable CUDA graph')
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=['auto', 'fp8'],
        default='auto',
        help=
        'Data type for kv cache storage. If "auto", will use model data type. '
        'FP8_E5M2 (without scaling) is only supported on cuda version greater '
        'than 11.8. On ROCm (AMD GPU), FP8_E4M3 is instead supported for '
        'common inference criteria.')
    parser.add_argument(
        '--quantization-param-path',
        type=str,
        default=None,
        help='Path to the JSON file containing the KV cache scaling factors. '
        'This should generally be supplied, when KV cache dtype is FP8. '
        'Otherwise, KV cache scaling factors default to 1.0, which may cause '
        'accuracy issues. FP8_E5M2 (without scaling) is only supported on '
        'cuda version greater than 11.8. On ROCm (AMD GPU), FP8_E4M3 is '
        'instead supported for common inference criteria.')
    parser.add_argument(
        '--profile',
        action='store_true',
        help='profile the generation process of a single batch')
    parser.add_argument(
        '--profile-result-dir',
        type=str,
        default=None,
        help=('path to save the pytorch profiler output. Can be visualized '
              'with ui.perfetto.dev or Tensorboard.'))
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help='device type for vLLM execution, supporting CUDA and CPU.')
    parser.add_argument('--block-size',
                        type=int,
                        default=16,
                        help='block size of key/value cache')
    parser.add_argument(
        '--enable-chunked-prefill',
        type=bool,
        default=False,
        help='If True, the prefill requests can be chunked based on the '
        'max_num_batched_tokens')
    parser.add_argument(
        "--ray-workers-use-nsight",
        action='store_true',
        help="If specified, use nsight to profile ray workers",
    )
    parser.add_argument('--worker-use-ray',
                        action='store_true',
                        help='use Ray for distributed serving, will be '
                        'automatically set when using more than 1 GPU '
                        'unless on ROCm where the default is torchrun')
    parser.add_argument('--download-dir',
                        type=str,
                        default=None,
                        help='directory to download and load the weights, '
                        'default to the default cache dir of huggingface')


    args = parser.parse_args()
    profile_benchmark(args)

