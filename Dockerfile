FROM rocm/pytorch:rocm6.0_ubuntu20.04_py3.9_pytorch_2.1.1
ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR
# Limit arch's so composable kernel doesn't take days to finish
ENV PYTORCH_ROCM_ARCH=gfx90a;gfx942

COPY fa_arch.patch $WORKSPACE_DIR
RUN apt update && apt install -y sqlite3 libsqlite3-dev libfmt-dev
RUN cd ${WORKSPACE_DIR} &&\
    git clone --recursive -b junhzhan/fa-mi300 https://github.com/ROCmSoftwarePlatform/flash-attention &&\
    cd flash-attention && git apply ../fa_arch.patch &&\
    python setup.py install

RUN cd ${WORKSPACE_DIR} && \
    git clone -b develop https://github.com/ROCmSoftwarePlatform/hipBLASLt && \
    export GTest_DIR="/usr/local/lib/cmake/GTest/" && \
    cd hipBLASLt && \
    ./install.sh -idc &&\
    ./install.sh -idc --architecture 'gfx90a;gfx942' &&\
    cd ../ && rm -rf hipBLASLt

RUN pip uninstall -y triton
RUN git clone https://github.com/ROCmSoftwarePlatform/triton.git && \
    cd triton/python && pip3 install -e .

RUN cd ${WORKSPACE_DIR} && \
    git clone -b exp_bandaid https://github.com/ROCmSoftwarePlatform/rccl && \
    cd rccl && mkdir build && cd build && \
    CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_PREFIX_PATH=/opt/rocm/ .. && make -j


RUN python3 -m pip install --upgrade pip
RUN pip install xformers==0.0.23 --no-deps

COPY ./ /workspace/vllm-private
RUN cd ${WORKSPACE_DIR} \
    && cd vllm-private \
    && cd gradlib \
    && python setup.py develop \
    && cd ../ \
    && pip install -r requirements.txt \
    && pip install typing-extensions==4.8.0 \
    && python3 setup.py install \
    && cd ..

RUN pip install pyarrow Ray pandas==2.0 numpy==1.20.3
COPY fa_arch.patch $WORKSPACE_DIR

COPY run_13b.sh $WORKSPACE_DIR/
COPY run_70b.sh $WORKSPACE_DIR/
