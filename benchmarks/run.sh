./copy_kernel.sh
# python run_oneconfig_moe.py -bs 4 -d_model 4096 -num_expt 8 -top_k 2 -tp_size 1 -inter_size 14336 -config_file tune_config1.json
rocprof --stats python run_oneconfig_moe.py -bs 4 -d_model 4096 -num_expt 8 -top_k 2 -tp_size 1 -inter_size 14336 -config_file tune_config1.json
python get_kernel_time.py -rocprof_file results.csv
