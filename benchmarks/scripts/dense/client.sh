# bash client.sh /home/zejchen/models/amd/Llama-3.1-8B-Instruct-FP8-KV 1000 1024 1024

MODEL=$1
NUM=$2
IN=$3
OUT=$4

python3 ../../benchmark_serving.py 
    --base-url http://0.0.0.0:1119 \
    --model $MODEL \
    --backend vllm \
    --dataset-name random \
    --num-prompts ${NUM} \
    --random-input-len ${IN} \
    --random-output-len ${OUT} \
    --host 0.0.0.0 \
    --port 1119 
