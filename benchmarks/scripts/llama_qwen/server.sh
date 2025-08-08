ORI_PATH=./bench_results/llama
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_V1=1

#llama3.1 70B BF16 TP8
SIZE=llama3.1-70B
MODEL=/home/jun_chen2_qle/pretrained-models/Llama-3.1-70B-Instruct
DTYPE=bfloat16
WTYPE=bfloat16
TP=8
OUT=1024

#llama3.3 70B FP8 TP8
SIZE=llama3.3-70B
MODEL=/home/jun_chen2_qle/pretrained-models/amd/Llama-3.3-70B-Instruct-FP8-KV
DTYPE=bfloat16
WTYPE=fp8
TP=8
OUT=1024

#llama3.3 70B FP4 TP8
SIZE=llama3.3-70B
MODEL=/home/jun_chen2_qle/pretrained-models/amd/Llama-3.3-70B-Instruct-MXFP4-Preview
DTYPE=bfloat16
WTYPE=fp4
TP=8
OUT=1024

#llama3.3 8B FP8 TP1
SIZE=llama3.1-8B
MODEL=/home/jun_chen2_qle/pretrained-models/amd/Llama-3.1-8B-Instruct-FP8-KV
DTYPE=bfloat16
WTYPE=fp8
TP=1

#Qwen3 32B BF16 TP1
SIZE=Qwen3-32B
MODEL=/home/jun_chen2_qle/pretrained-models/Qwen/Qwen3-32B
DTYPE=bfloat16
WTYPE=bfloat16
TP=8

#Qwen3 32B FP8 TP1
SIZE=Qwen3-32B
MODEL=/home/jun_chen2_qle/pretrained-models/Qwen/Qwen3-32B-FP8
DTYPE=bfloat16
WTYPE=fp8
TP=8

#Qwen3 235B FP8 TP8
SIZE=Qwen3-235B
MODEL=/home/jun_chen2_qle/pretrained-models/Qwen/Qwen3-235B-A22B-FP8
DTYPE=bfloat16
WTYPE=fp8
TP=8


# BF16 server 
vllm serve ${MODEL} \
    --trust-remote-code \
    --swap-space 16 \
    --disable-log-requests \
    --dtype $DTYPE \
    --no-enable-chunked-prefill \
    --max-model-len 327680 \
    --max-num-batched-tokens 65536 \
    --gpu-memory-utilization 0.99 \
    --distributed-executor-backend mp \
    --tensor-parallel-size $TP \
    --enable-expert-parallel \
    --port 1115 \
    2>&1 | tee ${ORI_PATH}/${SIZE}.TP${TP}.dtype${DTYPE}.wtype${WTYPE}.log


# FP8 KV-Cache server
export VLLM_USE_TRITON_FLASH_ATTN=0
vllm serve ${MODEL} \
    --swap-space 16 \
    --disable-log-requests \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --dtype $DTYPE \
    --max-model-len 8192 \
    --tensor-parallel-size 8 \
    --max-num-batched-tokens 65536 \
    --gpu-memory-utilization 0.99 \
    --num_scheduler-steps 10 \
    --port 1115 \
    2>&1 | tee ${ORI_PATH}/${SIZE}.TP${TP}.dtype${DTYPE}.wtype${WTYPE}.log

