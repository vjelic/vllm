#!/bin/bash
SLOT_DIR=/var/lib/jenkins/
VLLM_DIR=$SLOT_DIR/vllm
GRAD_DIR=$SLOT_DIR/gradlib
MODEL=/data/llama2-70b-chat
#enable to use triton flash attention
export VLLM_USE_TRITON=1
export VLLM_USE_HIPGRAPH=1
#set Tensor Parallelism
for tp in 2;
do
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
            echo "================================= RUNNING $MODEL $input_len $gen_len ==============================================="
            python benchmarks/benchmark_latency.py --model $MODEL --input-len $input_len --output-len $gen_len --batch-size 1  --tensor-parallel-size $tp --num-iters 5
        done
    done
done
