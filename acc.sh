#!/bin/bash
BASE_DIR=/trees/
VLLM_DIR=$BASE_DIR/vllm
GRAD_DIR=$BASE_DIR/gradlib
RPD_DIR=/var/lib/jenkins/rocmProfileData
MODEL=/data/Llama-2-7b-chat-hf
#MODEL=/data/Llama-2-13B-Chat-fp16
#MODEL=/data/llama-2-13b-chat-hf
MODEL_SIZE=`echo $MODEL | sed 's/.*\(.[0-9][bB]\).*/\1/'`

GEN_LEN="128"
TP=4
INPUT_LEN=128
ITER=1
BS=1

export VLLM_WORKER_MULTIPROC_METHOD=spawn

for tp in $TP;
do
        for gen_len in $GEN_LEN;
        do
                for input_len in $INPUT_LEN;
                do

torchrun --standalone --nnodes=1 --nproc-per-node=$tp benchmarks/benchmark_acc.py --model $MODEL  --batch-size $BS  --input-len $input_len --output-len $gen_len --tensor-parallel-size $tp --num-iters $ITER --disable_custom_all_reduce --enforce-eager
    done
done
done

