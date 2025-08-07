export VLLM_USE_FLASHINFER_SAMPLER=1
python -u bench.py --batch-size 1 --max-tokens 10 --tensor-parallel-size 8 --enable-profiling --warmup
#python -u bench.py --batch-size 128 --max-tokens 10 --tensor-parallel-size 8 --warmup 2>&1 | tee log_bs128_e2e
