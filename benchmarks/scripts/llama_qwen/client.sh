export VLLM_TORCH_PROFILER_DIR=/home/hatwu/OOB/vllm_bench/profile_results
BENCHMARK_PATH=/home/hatwu/OOB/vllm/benchmarks
ORI_PATH=/home/hatwu/OOB/vllm_bench/profile_results
cd ${BENCHMARK_PATH}
pwd

#llama3.1 70B BF16 TP8
SIZE=llama3.1-70B
MODEL=/home/jun_chen2_qle/pretrained-models/Llama-3.1-70B-Instruct
NUM=1
IN=4096
OUT=1024

python3 vllm/benchmarks/benchmark_serving.py 
    --base-url http://0.0.0.0:1111 \
    --model /data/models/Qwen3-235B-A22B-Instruct-2507-FP8/ \
    --backend vllm \
    --dataset-name random \
    --num-prompts ${NUM} \
    --random-input-len ${IN} \
    --random-output-len ${OUT} \
    --host 0.0.0.0 \
    --port 1115 \
    --profile
