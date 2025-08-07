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
