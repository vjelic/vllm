export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_V1=1

MODEL_PATH=/home/zejchen/models

#llama3.3 70B FP8 TP8
SIZE=llama3.3-70B
MODEL=${MODEL_PATH}/amd/Llama-3.3-70B-Instruct-FP8-KV/
DTYPE=bfloat16
WTYPE=fp8
TP=8

vllm serve $MODEL \
    --distributed-executor-backend mp \
    --tensor-parallel-size $TP \
    --block-size 16 \
    --trust-remote-code \
    --disable-log-requests \
    --max-seq-len-to-capture 32768 \
    --max-num-batched-tokens 32768 \
    --no-enable-prefix-caching \
    --dtype $DTYPE \
    --port 1119
