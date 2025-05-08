FROM rocm/vllm:rocm6.3.1_instinct_vllm0.8.3_20250410
ENV GO111MODULE on
ENV GOPROXY https://goproxy.cn
RUN pip uninstall -y vllm aiter
RUN git clone -b aiter https://github.com/ROCm/vllm.git vllm-develop
RUN cd vllm-develop && MAX_JOBS=128 python setup.py develop
RUN git clone -b attention_block_table https://github.com/ROCm/aiter.git
RUN cd aiter && git submodule update --init && cd 3rdparty/composable_kernel/ && git checkout origin/fix_fa_codegen
RUN cd aiter && MAX_JOBS=128 python setup.py develop
