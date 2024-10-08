#!/bin/bash

# Sample cmd:
# ./benchmark_scan.sh --tp 4 --benchmark l --json benchmark_SNOW.json

ENABLE_LOGGING=1
READ_TUNABLEOP=0

NUM_GPUS=(8)
TP_SIZE=(8)

VLLM_NUM_PROMPT=400 # Throughput benchmark
VLLM_NUM_ITER_WARMUP=3 # Latency benchmark
VLLM_NUM_ITER=3 # Latency benchmark
VLLM_DTYPE=float16

CUR_DIR=`pwd`
vLLM_SCRIPT_DIR=$CUR_DIR/scripts
GPT_SCRIPT_DIR=$CUR_DIR/gpt-fast/mixtral-moe/
TUNING_DIR=$CUR_DIR/Tune

# Env variable
export HIP_FORCE_DEV_KERNARG=1
export ENABLE_INTRA_NODE_COMM=1

# Arguments
benchmark=$1
tp=8 
json=benchmark.json
TunableOpFolder=False

function show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help                  Show this help message and exit"
    echo "  --tp nGPU                   Set the number of GPUs to use"
    echo "  --benchmark BenchmarkType   vLLM has 't' and 'l' for throughput and latency; GPT-Fast has only 'l'"
    echo "  --json      JsonFile        The in/out benchmark json file"
    echo "  --gemm-tuning SaveFolder    Enable TunableOp tunning and given the stored folder"
    echo "E.g., "
    echo "  ./benchmark_scan.sh --tp 8 --benchmark l"
    echo
}

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --tp)
            tp=$2
            shift 2
            ;;
        --benchmark)
            benchmark=$2
            shift 2
            ;;
        --json)
            json=$2
            shift 2
            ;;
        --gemm-tuning)
            TunableOpFolder=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done



function check_gpu_platform()
{
        if command -v rocm-smi &>/dev/null; then
                echo "ROCM"
        elif command -v nvidia-smi &>/dev/null; then
                echo "CUDA"
        else
                echo "Neither ROCm nor CUDA could be detected. Exiting."
                exit 1
        fi
}

function replicate_file()
{
        filename=$2
        suffix="${filename##*.}"
        name="${filename%.*}"
        for i in $(seq 0 7)
        do
                cp $1/$2 "$1/$name$i.$suffix"
        done
}

# Note: if $TunableOpFolder already have TunableOp csv files, TunableOp will not retune those gemm problems in the csv.

if [ "${TunableOpFolder}" != "False" ]; then
        mkdir -p $TunableOpFolder
        VLLM_NUM_ITER_WARMUP=0 
        VLLM_NUM_ITER=2 # Somehow the 1st iter and 2nd iter has diff GEMM size
        export PYTORCH_TUNABLEOP_TUNING=1
        export PYTORCH_TUNABLEOP_ENABLED=1
        export PYTORCH_TUNABLEOP_VERBOSE=3
        export PYTORCH_TUNABLEOP_MAX_WARMUP_ITERATIONS=10 
        export PYTORCH_TUNABLEOP_MAX_WARMUP_DURATION_MS=100 
        export PYTORCH_TUNABLEOP_MAX_TUNING_ITERATIONS=10  
        export PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=300
        export PYTORCH_TUNABLEOP_FILENAME=$TunableOpFolder/TunableOp.csv 

fi



platform=$(check_gpu_platform)
if [ "${platform}" == "ROCM" ]; then
        rocm-smi
elif [ "${platform}" == "CUDA" ]; then
        nvidia-smi
fi


cmd_logging=""
if [ "${ENABLE_LOGGING}" == "1" ]; then
        logging_file=benchmark.log
        if [ "${benchmark}" == "-g" ]; then
                logging_file=GEMM_SIZE/GEMM_SIZE.log
        fi
        cmd_logging="2>&1 | tee -a ${logging_file}"
fi


SCRIPT_NAME="benchmark_throughput.py"

COMMON_ARG="--dtype $VLLM_DTYPE --enable-prefix-caching --num-scheduler-steps 16 --distributed-executor-backend ray"

if [ $benchmark == "t" ]; then
        SCRIPT_NAME=$vLLM_SCRIPT_DIR/benchmark_throughput.py
        ARG_PROMPT=`echo "--num-prompts ${VLLM_NUM_PROMPT} $COMMON_ARG"`
elif [ $benchmark == "l" ]; then
        SCRIPT_NAME=$vLLM_SCRIPT_DIR/benchmark_latency.py
        ARG_PROMPT=`echo "--num-iters-warmup ${VLLM_NUM_ITER_WARMUP} --num-iters ${VLLM_NUM_ITER} $COMMON_ARG"`
else
        echo "Invalid benchamrk argument for vLLM: $benchmark"
        exit 1
fi


for n_gpu in ${NUM_GPUS[@]}
do
        HAVE_NEXT_EXEC=true
        while [ "$HAVE_NEXT_EXEC" = true ]
        do
                # 1. Get the input_token_len and output_token_len to be tested
                #    JsonHelper.py find the data in json file is stored as 0 as it was not benchmarked yet.
                echo "python JsonHelper.py --benchmark ${benchmark} --json $json --tp ${tp}"
                output=$(python JsonHelper.py --benchmark "${benchmark}" --json $json --tp ${tp})
                echo "bs, in and out =${output}"

                # Split the output into ilen and olen
                read bs ilen olen <<< "$output"

                # Check if ilen is "Finish." which means all benchmarks are done
                if [ "$bs" = "Finish." ]; then
                        echo "All benchmarks are done."
                        HAVE_NEXT_EXEC=false
                        break
                fi

                # 2. Compose the benchmark cmd
                if [ "$n_gpu" -gt "1" ]; then
                        if [ "${platform}" == "ROCM" ]; then
                                cmd=" python $SCRIPT_NAME --model=$HF_HUB/$MODEL_REPO --batch-size=$bs --input-len=${ilen} --output-len=${olen} --tensor-parallel-size=${tp}  ${ARG_PROMPT}"
                        else
                                cmd=" python $SCRIPT_NAME --model=$HF_HUB/$MODEL_REPO --batch-size=$bs --input-len=${ilen} --output-len=${olen} --tensor-parallel-size=${tp} ${ARG_PROMPT}"
                        fi
                        long_cmd="${cmd_profiling}${cmd} ${cmd_tuning}"
                else
                        if [ "${platform}" == "ROCM" ]; then
                                cmd="python $SCRIPT_NAME --model=$HF_HUB/$MODEL_REPO --batch-size=$bs --input-len=${ilen} --output-len=${olen} --tensor-parallel-size=${tp} ${ARG_PROMPT}"
                        else
                                cmd="torchrun --standalone --nnode 1 --nproc_per_node=${n} python $SCRIPT_NAME --model=$HF_HUB/$MODEL_REPO --batch-size=$bs --input-len=${ilen} --output-len=${olen} --tensor-parallel-size=$tp ${ARG_PROMPT}"
                        fi
                        long_cmd="${cmd_profiling}${cmd} ${cmd_tuning}"
                fi


                temp_output=$(mktemp)
                # Execute the command, log the output, and capture it in a variable
                # echo "cmd: ${long_cmd}" 2>&1 | tee -a $logging_file
                echo "cmd: ${long_cmd} 2>&1 | tee -a ${logging_file} | tee ${temp_output}"  | tee -a ${logging_file}
                eval "${long_cmd} 2>&1 | tee -a ${logging_file} | tee ${temp_output}"


                # Read the content of the temporary file into a variable
                captured_output=$(<"${temp_output}")

                # Clean up the temporary file
                rm "${temp_output}"

                # 3. Fill the benchmark result into json
                # Filter the result to get the benchmark value
                if [ $benchmark == "t" ]; then
                        filtered_result=$(echo "$captured_output" | grep "Throughput")
                elif [ $benchmark == "l" ]; then
                        filtered_result=$(echo "$captured_output" | grep "Avg latency")
                fi
                if [[ $captured_output =~ "OOM" ]]; then
                        filtered_result="OOM"
                fi

		#filtered_result="Avg latency: 99999 seconds"

                echo "Benchmark = ${filtered_result}"
                cmd_save_json="python JsonHelper.py --benchmark ${benchmark} --json $json --tp ${tp} --result \"${filtered_result}\" --bs=$bs --input-len=${ilen} --output-len=${olen}"
                echo $cmd_save_json
                eval $cmd_save_json
                echo "------------------------------------------------------------------------" 2>&1 | tee -a $logging_file
                
                if [ "${TunableOpFolder}" != "False" ]; then
                        for i in {1..7}; do cp $TunableOpFolder/TunableOp0.csv $TunableOpFolder/TunableOp$i.csv; done
                fi

        done
done





