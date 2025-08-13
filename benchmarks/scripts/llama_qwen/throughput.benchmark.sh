'''
bash throughput.benchmark.sh /home/zejchen/models/amd/Llama-3.1-8B-Instruct-FP8-KV 1 bfloat16 [1] [1024] [1024] llama3.1-8B

'''
MODEL=$1
TP=$2
DTYPE=$3
array_bs=$4
array_in=$5
array_out=$6
LOG_NAME=$7

'''
# #llama3.1 70B BF16 TP8
SIZE=llama3.3-70B
MODEL=${MODEL_PATH}/meta-llama/Llama-3.3-70B-Instruct
DTYPE=bfloat16
WTYPE=bfloat16

#llama3.3 70B FP8 TP8
SIZE=llama3.3-70B
MODEL=${MODEL_PATH}/amd/Llama-3.3-70B-Instruct-FP8-KV/
DTYPE=bfloat16
WTYPE=fp8

# #llama3.3 70B FP4 TP8
# SIZE=llama3.3-70B
# MODEL=${MODEL_PATH}/amd/Llama-3.3-70B-Instruct-MXFP4-Preview
# DTYPE=bfloat16
# WTYPE=fp4

#llama3.3 8B FP8 TP1
SIZE=llama3.1-8B
MODEL=${MODEL_PATH}/amd/Llama-3.1-8B-Instruct-FP8-KV
DTYPE=bfloat16
WTYPE=fp8

#Qwen3 32B BF16 TP1
SIZE=Qwen3-32B
MODEL=${MODEL_PATH}/Qwen/Qwen3-32B
DTYPE=bfloat16
WTYPE=bfloat16
TP=8

#Qwen3 32B FP8 TP1
SIZE=Qwen3-32B
MODEL=${MODEL_PATH}/Qwen/Qwen3-32B-FP8
DTYPE=bfloat16
WTYPE=fp8
TP=1

#Qwen3 32B FP8 TP8
SIZE=Qwen3-32B
MODEL=${MODEL_PATH}/Qwen/Qwen3-32B-FP8
DTYPE=bfloat16
WTYPE=fp8
TP=8

#Qwen3 235B FP8 TP8
SIZE=Qwen3-235B
MODEL=${MODEL_PATH}/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
DTYPE=bfloat16
WTYPE=fp8
TP=8


OUT=1024
#array_in=(1024 4096 10240)
#array_bs=(1 16 32 64 128 256)
array_in=(1024)
array_bs=(1)
'''


############# piecewise cuda graph ############
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_V1=1
export VLLM_ROCM_USE_AITER=1
for bs in ${array_bs[@]}; do
    for IN in ${array_in[@]}; do
        for OUT in ${array_out[@]}; do
            RES_PATH=./bench_results/${LOG_NAME}_TP${TP}_dtype${DTYPE}_wtype${WTYPE}_bs${bs}_input${IN}_output${OUT}
            mkdir -p ${RES_PATH}
            export VLLM_TORCH_PROFILER_DIR=RES_PATH
            python3 ../../benchmark_throughput.py \
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
                --output-json ${RES_PATH}/res.json
            2>&1 | tee ${RES_PATH}/res.log
        done
    done
done



# ############# full cuda graph ############
# # Currently only the prefill decode attention backend is supported in full graph
# export VLLM_USE_V1=1
# export VLLM_ROCM_USE_AITER=1 
# export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 
# export VLLM_ROCM_USE_AITER_MHA=0 
# export VLLM_DISABLE_COMPILE_CACHE=1 
# for IN in ${array_in[@]}; do
#     for bs in ${array_bs[@]}; do
#         RES_PATH=./bench_results/${SIZE}_TP${TP}_dtype${DTYPE}_wtype${WTYPE}_bs${bs}_input${IN}_output${OUT}
#         mkdir -p ${RES_PATH}
#         export VLLM_TORCH_PROFILER_DIR=${RES_PATH}
#         python3 ../../benchmark_throughput.py \
#             --distributed-executor-backend mp \
#             --dtype $DTYPE \
#             --disable-detokenize \
#             --gpu-memory-utilization 0.9 \
#             --trust-remote-code \
#             --model $MODEL \
#             --max-model-len 32768 \
#             --max-num-batched-tokens 131072 \
#             --max-seq-len-to-capture 131072 \
#             --input-len $IN \
#             --output-len $OUT \
#             --tensor-parallel-size $TP \
#             --num-prompts $bs \
#             --max-num-seqs $bs \
#             --compilation-config '{"full_cuda_graph": true}'
#             --output-json ${RES_PATH}/res.json
#         2>&1 | tee ${RES_PATH}/res.log
#     done
# done


# ############# fp8 KV Cache ############
# export VLLM_USE_V1=1
# export VLLM_ROCM_USE_AITER=1 
# export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 
# export VLLM_ROCM_USE_AITER_MHA=0 
# export VLLM_DISABLE_COMPILE_CACHE=1 
# for IN in ${array_in[@]}; do
#     for bs in ${array_in[@]}; do
#         RES_PATH=./bench_results/${SIZE}_TP${TP}_dtype${DTYPE}_wtype${WTYPE}_bs${bs}_input${IN}_output${OUT}
#         mkdir -p ${RES_PATH}
#         export VLLM_TORCH_PROFILER_DIR=RES_PATH
#         python3 ../benchmark_throughput.py \
#             --distributed-executor-backend mp \
#             --dtype $DTYPE \
#             --disable-detokenize \
#             --gpu-memory-utilization 0.9 \
#             --trust-remote-code \
#             --model $MODEL \
#             --max-model-len 32768 \
#             --max-num-batched-tokens 131072 \
#             --max-seq-len-to-capture 131072 \
#             --kv-cache-dtype fp8\
#             --input-len $IN \
#             --output-len $OUT \
#             --tensor-parallel-size $TP \
#             --num-prompts $bs \
#             --max-num-seqs $bs \
#             --output-json ${RES_PATH}/res.json
#         2>&1 | tee ${RES_PATH}/res.log
#     done
# done





