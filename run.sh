#!/bin/bash
BASE_DIR=/trees/
VLLM_DIR=$BASE_DIR/vllm
GRAD_DIR=$BASE_DIR/gradlib
RPD_DIR=/var/lib/jenkins/rocmProfileData
MODEL=/data/llama2-70b-chat
#MODEL=/data/Llama-2-13B-Chat-fp16
#MODEL=/data/llama-2-13b-chat-hf
MODEL_SIZE=`echo $MODEL | sed 's/.*\(.[0-9][bB]\).*/\1/'`

GEN_LEN="10"
TP=4
INPUT_LEN=1024
ITER=1
BS=2

export VLLM_WORKER_MULTIPROC_METHOD=spawn

for tp in $TP;
do
        for gen_len in $GEN_LEN;
        do
                for input_len in $INPUT_LEN;
                do

python benchmarks/benchmark_latency.py --model $MODEL  --batch-size $BS  --input-len $input_len --output-len $gen_len --tensor-parallel-size $tp --num-iters $ITER --distributed-executor-backend mp #ray

#torchrun --standalone --nnodes=1 --nproc-per-node=$tp benchmarks/benchmark_latency.py --model $MODEL  --batch-size $BS  --input-len $input_len --output-len $gen_len --tensor-parallel-size $tp --num-iters $ITER
    done
done
done

