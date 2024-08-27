"""Tests for the AWQ Triton kernel.

Run `pytest tests/kernels/test_awq_triton.py`.
"""
import argparse
import time

import torch

from vllm.model_executor.layers.quantization.awq import torch_awq_dequantize
from vllm.model_executor.layers.quantization.awq_triton import (
    awq_dequantize_triton, awq_gemm_triton)

import triton.profiler as proton

device = "cuda"


# qweights - [R     , C // 8], int32
# scales   - [R // G, C     ], float16
# zeros    - [R // G, C // 8], int32
def bench_dequantize(qweight_rows, qweight_cols):
    group_size = 128

    qweight_dtype = torch.int32
    scales_rows = qweight_rows // group_size
    scales_cols = qweight_cols * 8
    scales_dtype = torch.float16
    zeros_rows = scales_rows
    zeros_cols = qweight_cols
    zeros_dtype = torch.int32

    torch.manual_seed(0)

    qweight = torch.randint(0,
                            10000000, (qweight_rows, qweight_cols),
                            dtype=qweight_dtype,
                            device=device)
    scales = torch.rand(scales_rows,
                        scales_cols,
                        dtype=scales_dtype,
                        device=device)
    zeros = torch.randint(0,
                          10000000, (zeros_rows, zeros_cols),
                          dtype=zeros_dtype,
                          device=device)

    awq_dequantize_triton(qweight, scales, zeros)


def bench_gemm(N, K, M, splitK, reps = 10):
    print("="*10+" Benchmark GEMM "+"="*10)
    split_k_iters = splitK
    group_size = 128

    input_rows = N
    input_cols = K
    input_dtype = torch.float16
    qweight_rows = input_cols
    qweight_cols = M // 8
    scales_rows = qweight_rows // group_size
    scales_cols = M
    scales_dtype = torch.float16
    qzeros_rows = scales_rows
    qzeros_cols = qweight_cols

    torch.manual_seed(2)
    input = torch.rand((input_rows, input_cols),
                       dtype=input_dtype,
                       device=device)
    qweight = torch.randint(0,
                            torch.iinfo(torch.int32).max,
                            (qweight_rows, qweight_cols),
                            device=device)
    qzeros = torch.randint(0,
                           torch.iinfo(torch.int32).max,
                           (qzeros_rows, qzeros_cols),
                           device=device)
    scales = torch.rand((scales_rows, scales_cols),
                        dtype=scales_dtype,
                        device=device)

    print("Profiling with proton...")
    proton.activate(0)
    for _ in range(reps):
        awq_gemm_triton(input, qweight, scales, qzeros,
                                        split_k_iters)
        time.sleep(0.01)
    proton.deactivate(0)
    print("profiling done.")


def main():
    parser = argparse.ArgumentParser(
        description="awq_triton bench driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--bench")
    known_args, unknown_args = parser.parse_known_args()
    if known_args.bench is not None:
        if known_args.bench == "dequantize":
            # qweight_rows = 3584
            # qweight_cols = 576
            qweight_rows = 18944
            qweight_cols = 4736
            small_bench_size = False 
            if small_bench_size:
                qweight_rows = 256
                qweight_cols = 128
            bench_dequantize(qweight_rows, qweight_cols)
        elif known_args.bench == "gemm":
            small_bench_size = True
            N = 1
            K = 256 if small_bench_size else 3584
            M = 32 if small_bench_size else 448
            splitK = 1
            proton.start("awq_gemm", hook="triton")
            bench_gemm(N, K, M, splitK)
            proton.finalize()
        else:
            print(f"Unknown bench {known_args.bench}")
    else:
        print("No bench provided.")
        parser.print_help()


if __name__ == '__main__':
    main()
