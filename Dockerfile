FROM rocm/vllm-dev:nightly_0527_rc2_main_20250521
# ENV GO111MODULE on
# ENV GOPROXY https://goproxy.cn
RUN pip uninstall -y vllm aiter
RUN git clone -b fa_upstream3 https://github.com/ROCm/vllm.git vllm-develop
RUN cd vllm-develop && MAX_JOBS=128 python setup.py develop
RUN git clone -b vllm_v1_pa https://github.com/ROCm/aiter.git
RUN cd aiter && git submodule update --init
RUN cd aiter && python setup.py develop && python op_tests/test_pa_v1.py && python op_tests/test_mha_varlen.py