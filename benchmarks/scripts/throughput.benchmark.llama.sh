#BENCHMARK_PATH=/app/vllm/benchmarks
ORI_PATH=./bench_results/llama/
#cd ${BENCHMARK_PATH}
#pwd
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_V1=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_ASMMOE=1
export SAFETENSORS_FAST_GPU=1



IN=1024
OUT=10
TP=8
PROMPTS=4
MAX_NUM_SEQS=8


export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
MODEL=/data/models/AMD-LLAMA3.1-70B-Instruct-FP8/
python3 /home/hatwu/OOB/rocm_vllm/benchmarks/benchmark_throughput.py \
    --distributed-executor-backend mp \
    --kv-cache-dtype fp8 \
    --dtype float16 \
    --disable-detokenize \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --model $MODEL \
    --max-model-len 8192 \
    --max-num-batched-tokens 131072 \
    --max-seq-len-to-capture 131072 \
    --input-len $IN \
    --output-len $OUT \
    --tensor-parallel-size $TP \
    --num-prompts $PROMPTS \
    --max-num-seqs $MAX_NUM_SEQS \
    --profile \
    2>&1 | tee ${ORI_PATH}/log.throughput.bs${bs}.input${IN}.output${OUT}.log
