export VLLM_TORCH_PROFILER_DIR=/home/hatwu/OOB/vllm_bench/profile_results
MODEL=/data/models/AMD-LLAMA3.1-70B-Instruct-FP8/

# Llama
# export VLLM_USE_TRITON_FLASH_ATTN=0
# vllm serve ${MODEL} \
#     --swap-space 16 \
#     --disable-log-requests \
#     --quantization fp8 \
#     --kv-cache-dtype fp8 \
#     --dtype float16 \
#     --max-model-len 8192 \
#     --tensor-parallel-size 8 \
#     --max-num-batched-tokens 65536 \
#     --gpu-memory-utilization 0.99 \
#     --num_scheduler-steps 10 \
#     --port 9000 \
#     2>&1 | tee ./log.server.log &

# Qwen
vllm serve /data/models/Qwen3-235B-A22B-Instruct-2507-FP8/ \
    --trust-remote-code \
    -tp 8 \
    --no-enable-chunked-prefill \
    --disable-log-requests \
    --swap-space 16 \
    --distributed-executor-backend mp \
    --port 1111 \
    --enable-expert-parallel \
    --max-model-len 260000