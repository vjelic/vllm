# bash benchmark_latency.sh /home/zejchen/models/amd/Llama-3.1-8B-Instruct-FP8-KV 1 bfloat16 1 1024 1024 llama3.1-8B

MODEL=$1
TP=$2
DTYPE=$3
array_bs=$4
array_in=$5
array_out=$6
LOG_NAME=$7

export VLLM_USE_V1=1
export VLLM_ROCM_USE_AITER=1 
export VLLM_ROCM_USE_AITER_MHA=0 
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 
export VLLM_DISABLE_COMPILE_CACHE=1 

RES_PATH=./bench_results/${LOG_NAME}_TP${TP}_dtype${DTYPE}_wtype${WTYPE}_bs${bs}_input${IN}_output${OUT}
mkdir -p ${RES_PATH}
python3 ../../benchmark_latency.py \
    --distributed-executor-backend mp \
    --dtype $DTYPE \
    --disable-detokenize \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --model $MODEL \
    --batch-size $bs \
    --input-len $IN \
    --output-len $OUT \
    --tensor-parallel-size $TP \
    --num-iters-warmup 0 \
    --num-iters 1 \
    --compilation-config '{"full_cuda_graph": true}' \
    --output-json ${RES_PATH}/res.json \
    2>&1 | tee ${RES_PATH}/res.log

