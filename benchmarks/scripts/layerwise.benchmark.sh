MODEL=/data/models/AMD-LLAMA3.1-70B-Instruct-FP8/
python3 /opt/vllm/examples/offline_inference/profiling.py \
        --model ${MODEL} \
	--batch-size 4 \
        --prompt-len 512 \
	--max-num-batched-tokens 8196 \
        --enforce-eager run_num_steps -n 2

#--json Llama31-8b-FP8 \
#python ~/vllm/tools/profiler/print_layerwise_table.py \\
#        --json-trace Llama31-8b-FP8.json --phase prefill --table summary
