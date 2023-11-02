FROM compute-artifactory.amd.com:5000/rocm-plus-docker/framework/compute-rocm-dkms-no-npi-hipclang:12969_ubuntu20.04_py3.9_pytorch_rocm6.0_internal_testing_4b126e1
ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR
# Limit arch's so composable kernel doesn't take days to finish
ENV PYTORCH_ROCM_ARCH=gfx90a;gfx940;gfx941;gfx942
COPY fa_arch.patch $WORKSPACE_DIR
RUN cd ${WORKSPACE_DIR} &&\
    git clone --recursive -b junhzhan/fa-mi300 https://github.com/ROCmSoftwarePlatform/flash-attention &&\
    cd flash-attention && git apply ../fa_arch.patch &&\
    python setup.py install
RUN cd ${WORKSPACE_DIR} && \
    git clone -b develop https://github.com/ROCmSoftwarePlatform/hipBLASLt && \
    cd hipBLASLt && \
    ./install.sh -idc --architecture 'gfx90a;gfx940;gfx941;gfx942' &&\
    cd ../ && rm -rf hipBLASLt
RUN sed -i 's/, hipblaslt-dev \(.*\), hipcub-dev/, hipcub-dev/g' /var/lib/dpkg/status
RUN sed -i 's/, hipblaslt \(.*\), hipfft/, hipfft/g' /var/lib/dpkg/status
RUN cd ${WORKSPACE_DIR} && \
    git clone -b N1 https://streamhsa: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@github.com/amgddm/rocBLAS-internal.git && \
    cd rocBLAS-internal && ./install.sh -idc -a 'gfx90a;gfx940;gfx941;gfx942' &&\
    cd ../ && rm -rf rocBLAS-internal
RUN pip uninstall -y triton
RUN git clone https://github.com/ROCmSoftwarePlatform/triton.git && \
    cd triton/python && pip3 install -e .
RUN git clone https://streamhsa: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@github.com/ROCmSoftwarePlatform/gradlib.git &&\
    cd gradlib &&\
    python setup.py develop
RUN git clone https://streamhsa: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@github.com/ROCmSoftwarePlatform/vllm-private.git &&\
    cd vllm-private &&\
    pip install -r requirements.txt &&\
    pip install typing-extensions==4.8.0 &&\
    python setup.py build && python setup.py develop; exit 0
RUN pip install pyarrow Ray
RUN pip install pandas

# RUN git clone -b N1 https://streamhsa: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@github.com/amgddm/rocBLAS-internal.git && \
#     cd rocBLAS-internal && ./install.sh -idc -a gfx942 &&\
#     cd ../ && rm -rf rocBLAS-internal

COPY run_vllm.sh $WORKSPACE_DIR/
COPY run_vllm_tp1.sh $WORKSPACE_DIR/
COPY run_vllm_tp2.sh $WORKSPACE_DIR/
COPY run_vllm_tp4.sh $WORKSPACE_DIR/
COPY run_vllm_tp8.sh $WORKSPACE_DIR/
