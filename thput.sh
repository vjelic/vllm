#!/bin/bash
BASE_DIR=/trees/
VLLM_DIR=$BASE_DIR/vllm
GRAD_DIR=$BASE_DIR/gradlib
RPD_DIR=/workspace/rocmProfileData
MODEL=/data/llama2-70b-chat
MODEL=/data/Llama-2-7b-chat-hf
#MODEL=/data/Llama-2-13B-Chat-fp16
#MODEL=/data/llama-2-13b-chat-hf
MODEL_SIZE=`echo $MODEL | sed 's/.*\(.[0-9][bB]\).*/\1/'`
rocprof='rocprof --hip-trace --roctx-trace'
rpd=runTracer.sh

GEN_LEN="256"
TP=4
INPUT_LEN="256"
ITER=10
INP=2000

#rm ./trace.rpd
#python -m rocpd.schema --create ./trace.rpd

export VLLM_WORKER_MULTIPROC_METHOD=spawn

for tp in $TP;
do
        for gen_len in $GEN_LEN;
        do
                for input_len in $INPUT_LEN;
                do
#python benchmarks/benchmark_throughput.py --model $MODEL --input-len $input_len --output-len $gen_len --num-prompts $INP --tensor-parallel-size $tp


#python benchmarks/benchmark_throughput.py --model $MODEL --input-len $input_len --output-len $gen_len --num-prompts $INP --tensor-parallel-size $tp
torchrun --standalone --nnodes=1 --nproc-per-node=$tp benchmarks/benchmark_throughput.py --model $MODEL --input-len $input_len --output-len $gen_len --num-prompts $INP --tensor-parallel-size $tp

done
done
done

#--profile_rpd

#TRACE_FILE=./trace_${MODEL_SIZE}_${input_len}_${gen_len}.json
#echo "INFO: Creating Trace JSON file $TRACE_FILE"
#python /var/lib/jenkins/rocmProfileData/tools/rpd2tracing.py --format object ./trace.rpd $TRACE_FILE
