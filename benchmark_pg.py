import subprocess
import re
import csv


BENCH_RE = (
    r"Kernel running time: (?P<RUNNING_TIME>[+-]?([\w]*[.])?[(\w| )]+)"
)

#configs = [(1,16,16), (1,32,32), (1,32,4), (64,32,4), (1,52,4), '''(64,52,4),''' (1,16,2), (64,16,2), (1,26,2), (64,26,2), (1,8,1), (64,8,1), (1,13,1), (64,13,1)]
#configs = [(1,16,16), (1,32,32), (1,32,4), (64,32,4), (1,52,4), (1,16,2), (64,16,2), (1,26,2), (64,26,2), (1,8,1), (64,8,1), (1,13,1), (64,13,1)]
configs = [(1, 32, 4), (64, 32, 4), (1, 16, 2), (64, 16, 2), (1, 8, 1), (64, 8, 1)]

outfile = "./paged_attention.csv"
rows = [["block size", "bs", "q length", "kv length", "headdim", "q head", "kv head", "run time"]]
for bs,q_head,kv_head in configs:
    outputs = subprocess.run(f"python ./benchmarks/kernels/benchmark_paged_attention.py --context-len 8192 --block-size 16 --batch-size {bs} --head-size 128 --num-query-heads {q_head} --num-kv-heads {kv_head}", shell=True, capture_output=True)
    print (outputs)
    match = re.search(BENCH_RE, outputs.stdout.decode('utf-8'))
    time = match.group("RUNNING_TIME")
    rows.append([16, bs, 1, 8192, 128, q_head, kv_head, time])

with open(outfile, "w", newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerows(rows)
