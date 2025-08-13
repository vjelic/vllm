'''
bash server.sh /home/zejchen/models/amd/Llama-3.1-8B-Instruct-FP8-KV 1 bfloat16
'''
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_V1=1
export VLLM_ROCM_USE_AITER=1 
export VLLM_ROCM_USE_AITER_MHA=0 
export VLLM_DISABLE_COMPILE_CACHE=1 

MODEL=$1
TP=$2
DTYPE=$3

vllm serve $MODEL \
    --distributed-executor-backend mp \
    --tensor-parallel-size $TP \
    --block-size 16 \
    --trust-remote-code \
    --disable-log-requests \
    --max-seq-len-to-capture 131072 \
    --max-num-batched-tokens 131072 \
    --compilation-config '{"full_cuda_graph": true}' \
    --dtype $DTYPE \
    --port 1119

