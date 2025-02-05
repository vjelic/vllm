import os
import json
from amdsmi import (amdsmi_get_gpu_board_info, amdsmi_get_processor_handles,
                    amdsmi_init, amdsmi_shut_down, amdsmi_topo_get_link_type)


amdsmi_init()
handle = amdsmi_get_processor_handles()[0]
device_name = amdsmi_get_gpu_board_info(handle)["product_name"]
amdsmi_shut_down

print(f"device_name = {device_name}")

num_result_files = 4

result_objs = {}
unique_nk = {}
for i in range(num_result_files):
    result_file_name = f"results_{i}.txt"
    with open(result_file_name, 'r') as f:
        json_obj = json.loads(f.read())
        for k, v in json_obj.items():
            if k in result_objs:
                try:
                    if v["best"][0]["speedup"] > result_objs[k]["best"][0]["speedup"]:
                        result_objs[k] = v
                except TypeError:
                    print(f"k={k},v={v}")

            else:
                result_objs[k] = v

config_objs = {}
for k in result_objs:
    obj = result_objs[k]
    K = obj["K"]
    N = obj["N"]
    M = obj["M"]
    key = (N, K)
    if key not in config_objs:
        config_objs[key] = {}
    config_objs[key][M] = obj["best"][0]
    del config_objs[key][M]["speedup"]
    del config_objs[key][M]["Triton"]

def convert_values_to_int(d):
    obj = {k : int(d[k]) for k in d}
    return obj

block_n = 128
block_k = 128
for (N, K) in config_objs:
    # import pdb
    # breakpoint()
    values = config_objs[(N, K)]
    values = {k : convert_values_to_int(values[k]) for k in values}
    json_file_name = f"N={N},K={K},device_name={device_name},dtype=fp8_w8a8,block_shape=[{block_n}, {block_k}].json"  # noqa: E501
    with open(json_file_name, 'w') as f:
       f.write(json.dumps(values, sort_keys = True, indent=4))

print(f"num_keys={len(result_objs.keys())}")

