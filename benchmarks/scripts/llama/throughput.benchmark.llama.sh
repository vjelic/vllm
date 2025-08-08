ORI_PATH=./bench_results/llama
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_V1=1

#llama3.1 70B BF16
MODEL=/home/jun_chen2_qle/pretrained-models/Llama-3.1-70B-Instruct
DTYPE=bfloat16
WTYPE=bfloat16
TP=8
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
            --input-len $IN \
            --output-len $OUT \
            --tensor-parallel-size $TP \
            --num-prompts $bs \
            --max-num-seqs $bs \
            --output-json ${ORI_PATH}/llama70B.TP${TP}.dtype${DTYPE}.wtype${WTYPE}.bs${bs}.input${IN}.output${OUT}.json
        2>&1 | tee ${ORI_PATH}/llama70B.TP${TP}.dtype${DTYPE}.wtype${WTYPE}.bs${bs}.input${IN}.output${OUT}.log
    done
done

#llama3.3 70B FP8
MODEL=/home/jun_chen2_qle/pretrained-models/Llama-3.3-70B-Instruct-FP8-KV
DTYPE=bfloat16
WTYPE=fp8
TP=8

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
        --kv-cache-dtype \
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


