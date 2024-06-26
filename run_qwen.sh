#!/bin/bash
#CUR_DIR=/opt/vllm
 
export VLLM_USE_ROCM_CUSTOM_PAGED_ATTN=1
export HIP_FORCE_DEV_KERNARG=1
export VLLM_USE_TRITON_FLASH_ATTN=false

export VLLM_USE_ROCM_CUSTOM_PAGED_ATTN=0

export PYTORCH_TUNABLEOP_FILENAME=full_tuned%d.csv
export PYTORCH_TUNABLEOP_TUNING=0
export PYTORCH_TUNABLEOP_ENABLED=1

################################
# Configurations
MODEL=/data/Qwen1.5-110B-Chat
NUM_PROMPTS=1000
INPUT_LEN=2000
OUTPUT_LEN=500
TP=4

DTYPE=bfloat16

# RPD Tracing
# if [[ -f $CUR_DIR/trace.rpd ]];
# then
#   rm -f $CUR_DIR/trace.rpd
# fi
# python -m rocpd.schema --create $CUR_DIR/trace.rpd

# python /opt/vllm/benchmarks/benchmark_throughput.py --model $MODEL --num-prompts=$NUM_PROMPTS --input-len $INPUT_LEN --output-len $OUTPUT_LEN --tensor-parallel-size $TP --trust-remote-code --worker-use-ray
torchrun --standalone --nproc_per_node=$TP --nnodes=1 benchmarks/benchmark_throughput.py --model $MODEL --num-prompts=$NUM_PROMPTS --input-len $INPUT_LEN --output-len $OUTPUT_LEN --tensor-parallel-size $TP --dtype ${DTYPE}

# RPD Tracing
# echo "INFO: Creating Trace JSON file $CUR_DIR/trace.json"
# python $CUR_DIR/rocmProfileData/tools/rpd2tracing.py --format object $CUR_DIR/trace.rpd $CUR_DIR/trace.json
