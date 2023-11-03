#!/bin/bash

# parameter defalt values
tp=1
VLLM_DIR=$HOME/vllm
GRAD_DIR=$HOME/gradlib
MODEL=/data/llama-2-13b-chat-hf

# pring usage of the parameters
usage() {
    echo "Usage: $0 [--tp <n>] [--vllm_dir <path>] [--gradlib_dir <path>] [--model <path>]"
    exit 1
}

# parse parameters
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tp) tp="$2"; shift ;;
        --vllm_dir) VLLM_DIR="$2"; shift ;;
        --gradlib_dir) GRAD_DIR="$2"; shift ;;
        --model) MODEL="$2"; shift ;;
        *) usage ;; # Any other argument will show usage information.
    esac
    shift # Move to next argument
done

# print parameter settings
echo "tensor parallel: $tp"
echo "vllm_dir: $VLLM_DIR"
echo "gradlib_dir: $GRAD_DIR"
echo "model: $MODEL"

#enable to use triton flash attention
export VLLM_USE_TRITON=1
export VLLM_USE_HIPGRAPH=1
#set Tensor Parallelism

echo "tuned_gemm_csv: ./tuned_tp$tp.csv" > $VLLM_DIR/tuned_perf_tp$tp.yaml
if [ ! -f  $VLLM_DIR/tuned_tp$tp.csv ] ;
then
        echo "INFO: No Tuned configs detected. Generating now"
        cd $GRAD_DIR
        python gemm_tuner.py --model_dir $MODEL --output ../vllm/tuned_tp$tp.csv --tp $tp
fi
export VLLM_PERF_YAML=./tuned_perf_tp$tp.yaml

cd $VLLM_DIR
for gen_len in 1 32;
do
    for input_len in 512 1024 2048 3072;
    do
        echo "================================= RUNNING $MODEL tp$tp $input_len $gen_len ==============================================="
        python benchmarks/benchmark_latency.py --model $MODEL --input-len $input_len --output-len $gen_len --batch-size 1  --tensor-parallel-size $tp --num-iters 5
    done
done
