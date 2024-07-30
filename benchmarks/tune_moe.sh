echo "tune BS=50"
time python tune_moe.py -bs 50 -d_model 4096 -num_expt 8 -top_k 2 -tp_size 1 -inter_size 14336
echo "tune BS=100"
time python tune_moe.py -bs 100 -d_model 4096 -num_expt 8 -top_k 2 -tp_size 1 -inter_size 14336