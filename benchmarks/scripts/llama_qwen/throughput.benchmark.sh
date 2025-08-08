ORI_PATH=./bench_results/llama
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_V1=1

#llama3.1 70B BF16 TP8
SIZE=llama3.1-70B
MODEL=/home/jun_chen2_qle/pretrained-models/Llama-3.1-70B-Instruct
DTYPE=bfloat16
WTYPE=bfloat16
TP=8
OUT=1024

#llama3.3 70B FP8 TP8
SIZE=llama3.3-70B
MODEL=/home/jun_chen2_qle/pretrained-models/amd/Llama-3.3-70B-Instruct-FP8-KV
DTYPE=bfloat16
WTYPE=fp8
TP=8
OUT=1024

#llama3.3 70B FP4 TP8
SIZE=llama3.3-70B
MODEL=/home/jun_chen2_qle/pretrained-models/amd/Llama-3.3-70B-Instruct-MXFP4-Preview
DTYPE=bfloat16
WTYPE=fp4
TP=8
OUT=1024

#llama3.3 8B FP8 TP1
SIZE=llama3.1-8B
MODEL=/home/jun_chen2_qle/pretrained-models/amd/Llama-3.1-8B-Instruct-FP8-KV
DTYPE=bfloat16
WTYPE=fp8
TP=1

#Qwen3 32B BF16 TP1
SIZE=Qwen3-32B
MODEL=/home/jun_chen2_qle/pretrained-models/Qwen/Qwen3-32B
DTYPE=bfloat16
WTYPE=bfloat16
TP=8

#Qwen3 32B FP8 TP1
SIZE=Qwen3-32B
MODEL=/home/jun_chen2_qle/pretrained-models/Qwen/Qwen3-32B-FP8
DTYPE=bfloat16
WTYPE=fp8
TP=8

#Qwen3 235B FP8 TP8
SIZE=Qwen3-235B
MODEL=/home/jun_chen2_qle/pretrained-models/Qwen/Qwen3-235B-A22B-FP8
DTYPE=bfloat16
WTYPE=fp8
TP=8


OUT=1024
#array_in=(1024 4096 10240)
#array_bs=(1 16 32 64 128 256)
array_in=(1024)
array_bs=(1)

for IN in ${array_in[@]} do
    for bs in ${array_bs[@]} do
        python3 ../benchmark_throughput.py \
            --distributed-executor-backend mp \
            --dtype $DTYPE \
            --disable-detokenize \
            --gpu-memory-utilization 0.9 \
            --trust-remote-code \
            --model $MODEL \
            --max-model-len 32768 \
            --max-num-batched-tokens 131072 \
            --max-seq-len-to-capture 131072 \
            --input-len $IN \
            --output-len $OUT \
            --tensor-parallel-size $TP \
            --num-prompts $bs \
            --max-num-seqs $bs \
            --output-json ${ORI_PATH}/${SIZE}.TP${TP}.dtype${DTYPE}.wtype${WTYPE}.bs${bs}.input${IN}.output${OUT}.json
        2>&1 | tee ${ORI_PATH}/${SIZE}.TP${TP}.dtype${DTYPE}.wtype${WTYPE}.bs${bs}.input${IN}.output${OUT}.log
    done
done




'''
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
for IN in 1024 4096 10240; do
    for bs in 1 16 32 64 128 256; do
        OUT=1024

        python3 ../benchmark_throughput.py \
            --distributed-executor-backend mp \
            --dtype $DTYPE \
            --disable-detokenize \
            --gpu-memory-utilization 0.9 \
            --trust-remote-code \
            --model $MODEL \
            --max-model-len 32768 \
            --max-num-batched-tokens 131072 \
            --max-seq-len-to-capture 131072 \
            --kv-cache-dtype fp8\
            --input-len $IN \
            --output-len $OUT \
            --tensor-parallel-size $TP \
            --num-prompts $bs \
            --max-num-seqs $bs \
            --output-json ${ORI_PATH}/llama70B.TP${TP}.dtype${DTYPE}.wtype${WTYPE}.bs${bs}.input${IN}.output${OUT}.json
        2>&1 | tee ${ORI_PATH}/llama70B.TP${TP}.dtype${DTYPE}.wtype${WTYPE}.bs${bs}.input${IN}.output${OUT}.log
    done
done
'''


