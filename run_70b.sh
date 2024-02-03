#!/bin/bash
BASE_DIR=/workspace
VLLM_DIR=$BASE_DIR/vllm-private
GRAD_DIR=$VLLM_DIR/gradlib
MODEL=/data/llama2-70b-chat
MODEL_SIZE=`echo $MODEL | sed 's/.*\(.[0-9][bB]\).*/\1/'`
#MODEL=/data/llama-2-13b-chat-hf

export VLLM_USE_TRITON=1

export VLLM_UNTUNE_FILE="/tmp/vllm_untuned.csv"
export VLLM_TUNE_FILE=$VLLM_DIR/"tuned.csv"

#Flag to use Triton Flash Attention vs CK
#export VLLM_USE_TRITON=1


#Delete tuned gemms before running.
#DELETE_TUNED_CSV=1

#Flag to disable MSCCL
#export RCCL_MSCCL_ENABLE=0

#HIPGraph performance flags
export HIP_FORCE_DEV_KERNARG=1
export DEBUG_CLR_GRAPH_PACKET_CAPTURE=1

#Enable full decoder graph mode
HIP_GRAPH=--use-cuda-graph

#Use top of tree build of RCCL
export LD_LIBRARY_PATH=/workspace/rccl/build/


#TP="1 2 4 8"
TP=8
GEN_LEN="1 32"
INPUT_LEN="512 1024"
ITER=10

for tp in $TP;
do
    echo "=============================== WARMING UP ======================================"
    export VLLM_TUNE_GEMM=1
    cd $VLLM_DIR
    for gen_len in $GEN_LEN;
        do
            for input_len in $INPUT_LEN;
            do
                echo "================================= RUNNING $MODEL $input_len $gen_len gpus=$tp ==============================================="
                torchrun --standalone --nnodes=1 --nproc-per-node=$tp benchmarks/benchmark_latency.py --model $MODEL  --batch-size 1 --input-len $input_len --output-len $gen_len \
                --tensor-parallel-size $tp --num-iters 1
                if [[ -v ROCPROF_PROFILE ]] ;
                then
                    TRACE_FILE=$BASE_DIR/trace_${MODEL_SIZE}_${input_len}_${gen_len}.json
                    echo "INFO: Creating Trace JSON file $TRACE_FILE"
                    mv $VLLM_DIR/results.json $TRACE_FILE
                fi
            done
        done

    echo "=============================== Tuning ======================================"
    python $GRAD_DIR/gemm_tuner.py --tuned_file $VLLM_TUNE_FILE --input_file $VLLM_UNTUNE_FILE

    echo "tuned_gemm_csv: ./tuned.csv" > $VLLM_DIR/tuned_perf.yaml
    tuned_file=$VLLM_DIR/tuned.csv
    export VLLM_PERF_YAML=./tuned_perf.yaml

    echo "================================= TUNED GEMMS  $VLLM_TUNE_FILE ==============================================="
    cat $VLLM_TUNE_FILE

    export VLLM_TUNE_GEMM=0
    cd $VLLM_DIR
    for gen_len in $GEN_LEN;
    do
        for input_len in $INPUT_LEN;
        do
            echo "================================= RUNNING $MODEL $input_len $gen_len gpus=$tp ==============================================="
            torchrun --standalone --nnodes=1 --nproc-per-node=$tp benchmarks/benchmark_latency.py --model $MODEL  --batch-size 1 --input-len $input_len --output-len $gen_len \
            --tensor-parallel-size $tp --num-iters $ITER $HIP_GRAPH
            if [[ -v ROCPROF_PROFILE ]] ;
            then
                TRACE_FILE=$BASE_DIR/trace_${MODEL_SIZE}_${input_len}_${gen_len}.json
                echo "INFO: Creating Trace JSON file $TRACE_FILE"
                mv $VLLM_DIR/results.json $TRACE_FILE
            fi
        done
    done
done
