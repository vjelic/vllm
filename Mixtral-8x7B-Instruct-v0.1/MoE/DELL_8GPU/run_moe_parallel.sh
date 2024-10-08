#!/bin/bash

export HIP_FORCE_DEV_KERNARG=1 

# HIP_VISIBLE_DEVICES=0 python benchmark_mixtral_moe_rocm.py --TP 8 --model 8x7B --split_id 0

rm best_perf.json
rm 'E=8,N=1792,device_name=AMD_Instinct_MI300X_OAM.json'

for i in {0..7}; do
    HIP_VISIBLE_DEVICES=$i python benchmark_mixtral_moe_rocm.py --TP 8 --model 8x7B --split_id $i &
done

