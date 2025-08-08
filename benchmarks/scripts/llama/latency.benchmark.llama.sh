ORI_PATH=./bench_results/llama
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_V1=1


for IN in 1024; do
    for bs in 1; do
        OUT=1024
        TP=8
        DTYPE=bfloat16

        MODEL=/home/jun_chen2_qle/pretrained-models/Llama-3.1-70B-Instruct
        python3 ../benchmark_latency.py \
            --distributed-executor-backend mp \
            --dtype $DTYPE \
            --disable-detokenize \
            --gpu-memory-utilization 0.9 \
            --trust-remote-code \
            --model $MODEL \
            --batch-size $bs \
            --input-len $IN \
            --output-len $OUT \
            --tensor-parallel-size $TP \
        --num-iters-warmup 0 \
        --num-iters 1 \
            --output-json ${ORI_PATH}/llama70B.TP${TP}.dtype${DTYPE}.bs${bs}.input${IN}.output${OUT}.json
        2>&1 | tee ${ORI_PATH}/llama70B.TP${TP}.dtype${DTYPE}.bs${PROMPTS}.input${IN}.output${OUT}.log
    done
done

