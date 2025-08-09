vllm serve facebook/opt-125m --swap-space 16 \
    --disable-log-requests --dtype float16

curl http://0.0.0.0:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/data/models/Qwen3-30B-A3B-FP8", 
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'

export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_V1=1
MODEL_PATH=/home/zejchen/models

#llama3.3 70B FP8 TP8
SIZE=llama3.3-70B
MODEL=${MODEL_PATH}/amd/Llama-3.3-70B-Instruct-FP8-KV/
DTYPE=bfloat16
WTYPE=fp8
TP=8

python3 ../../benchmark_serving.py \
        --base-url http://0.0.0.0:1119 \
        --host 0.0.0.0 \
        --port 1119 \
        --model ${MODEL} \
        --backend vllm \
        --dataset-name random \
        --num-prompts 1 \
        --random-input-len 4096 \
        --random-output-len 2048 \
        --request-rate inf \
#        --max-concurrency None \