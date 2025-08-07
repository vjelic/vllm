export VLLM_TORCH_PROFILER_DIR=/opt/customer/llama3/profiler
BENCHMARK_PATH=/app/vllm/benchmarks
ORI_PATH=/opt/customer/llama3
cd ${BENCHMARK_PATH}
pwd

export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
MODEL=/data/models/AMD-LLAMA3.1-70B-Instruct-FP8/
BS=1
IN=128
OUT=3
TP=8

python3 -u /app/vllm/benchmarks/benchmark_latency.py \
    --distributed-executor-backend mp \
    --dtype float16 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --model $MODEL \
    --batch-size $BS \
    --input-len $IN \
    --output-len $OUT \
    --tensor-parallel-size $TP \
    --num-iters-warmup 3 \
    --num-iters 5 \
    --profile \
    2>&1 | tee ${ORI_PATH}/log.latency.log

cd ${ORI_PATH}
