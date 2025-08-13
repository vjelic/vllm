pip install "huggingface_hub[cli]"
export PATH="$HOME/.local/bin:$PATH"

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_OFFLINE=0

# llama - bf16
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir ./meta-llama/Llama-3.3-70B-Instruct

# llama - fp8
huggingface-cli download amd/Llama-3.1-8B-Instruct-FP8-KV --local-dir ./amd/Llama-3.1-8B-Instruct-FP8-KV
huggingface-cli download amd/Llama-3.3-70B-Instruct-FP8-KV --local-dir ./amd/Llama-3.3-70B-Instruct-FP8-KV
huggingface-cli download amd/Llama-3.1-405B-Instruct-FP8-KV --local-dir ./amd/Llama-3.1-405B-Instruct-FP8-KV

# llama fp4
huggingface-cli download amd/Llama-3.3-70B-Instruct-MXFP4-Preview --local-dir ./amd/Llama-3.3-70B-Instruct-MXFP4-Preview
huggingface-cli download amd/Llama-3.1-405B-Instruct-MXFP4-Preview --local-dir ./amd/Llama-3.1-405B-Instruct-MXFP4-Preview

# qwen
huggingface-cli download Qwen/Qwen3-32B --local-dir Qwen/Qwen3-32B
huggingface-cli download Qwen/Qwen3-32B-FP8 --local-dir Qwen/Qwen3-32B-FP8
huggingface-cli download Qwen/Qwen3-235B-A22B-FP8 --local-dir Qwen/Qwen3-235B-A22B-FP8
huggingface-cli download Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 --local-dir Qwen/Qwen3-235B-A22B-Instruct-2507-FP8
