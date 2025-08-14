# bash client.sh /home/zejchen/models/amd/Llama-3.1-8B-Instruct-FP8-KV 1000 1024 1024

MODEL=$1
CONCUR=$2
IN=$3
OUT=$4
NUM=$CONCUR * 10

result_filename=CON${CONCUR}_IN${IN}_OUT${OUT}.json
mkdir bench_results
python3 ../../benchmark_serving.py \
    --backend vllm \
    --host 0.0.0.0 \
    --port 1119 \
    --model $MODEL \
    --dataset-name random \
    --num-prompts ${NUM} \
    --max-concurrency ${CONCUR} \
    --random-input-len ${IN} \
    --random-output-len ${OUT} \
    --result-dir ./bench_results/ \
    --result-filename ${result_filename} \
    --percentile-metrics "ttft,tpot,itl,e2el"
