export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_V1=1

MODEL_PATH=/home/zejchen/models

#llama3.3 70B FP8 TP8
SIZE=llama3.3-70B
MODEL=${MODEL_PATH}/amd/Llama-3.3-70B-Instruct-FP8-KV/
DTYPE=bfloat16
WTYPE=fp8
TP=8

############# piecewise cuda graph ############
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

############# full cuda graph ############
# Currently only the prefill decode attention backend is supported in full graph
export VLLM_USE_V1=1 
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 
export VLLM_DISABLE_COMPILE_CACHE=1 
export VLLM_ROCM_USE_AITER=1 
export VLLM_ROCM_USE_AITER_MHA=0 
vllm serve meta-llama/Llama-3.1-405B \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2  \
    --compilation-config '{"full_cuda_graph": true}'
