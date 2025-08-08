export VLLM_TORCH_PROFILER_DIR=/home/hatwu/OOB/vllm_bench/profile_results
BENCHMARK_PATH=/home/hatwu/OOB/vllm/benchmarks
ORI_PATH=/home/hatwu/OOB/vllm_bench/profile_results
cd ${BENCHMARK_PATH}
pwd

MODEL=/data/models/AMD-LLAMA3.1-70B-Instruct-FP8/

# python3 -u ./benchmark_serving.py \
#     --port 9000 \
#     --model ${MODEL} \
#     --dataset-name random \
#     --random-input-len 1024 \
#     --random-output-len 512 \
#     --request-rate 1 \
#     --ignore-eos \
#     --num-prompts 1 \
#     --percentile-metrics ttft,tpot,itl,e2el \
#     --profile \
#     2>&1 | tee ${ORI_PATH}/log.client.log

# cd ${ORI_PATH}
# #     --profiled \

python3 vllm/benchmarks/benchmark_serving.py 
    --base-url http://0.0.0.0:1111 \
    --model /data/models/Qwen3-235B-A22B-Instruct-2507-FP8/ \
    --backend vllm \
    --dataset-name random \
    --num-prompts 1 \
    --random-input-len 4096 \
    --port 1111 \
    --random-output-len 2048 \
    --host 0.0.0.0 \
    --profile