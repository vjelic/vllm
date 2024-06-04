#!/bin/bash
BASE_DIR=/trees/
VLLM_DIR=$BASE_DIR/vllm
GRAD_DIR=$BASE_DIR/gradlib
RPD_DIR=/workspace/rocmProfileData
MODEL=/data/models/mistral_ai/Mixtral-8x7B-v0.1/snapshots/985aa055896a8f943d4a9f2572e6ea1341823841/
#MODEL=/data/Llama-2-13B-Chat-fp16
#MODEL=/data/llama-2-13b-chat-hf
MODEL_SIZE=`echo $MODEL | sed 's/.*\(.[0-9][bB]\).*/\1/'`

GEN_LEN="8"
TP=8
INPUT_LEN=2048
ITER=1

for tp in $TP;
do
        for gen_len in $GEN_LEN;
        do
                for input_len in $INPUT_LEN;
                do

torchrun --standalone --nnodes=$tp --nproc-per-node=1  benchmarks/benchmark_latency.py --model $MODEL  --batch-size 1    --input-len $input_len --output-len $gen_len \
                            --tensor-parallel-size $tp --num-iters $ITER
    done
done
done

