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
    sed -i 's/gpu_memory_utilization: float = 0.9/gpu_memory_utilization: float = 0.4/g' vllm/engine/arg_utils.py &&\
    python setup.py build && python setup.py develop; exit 0
RUN pip install pyarrow Ray pandas==2.0 numpy==1.20.3

# RUN git clone -b N1 https://streamhsa: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@github.com/amgddm/rocBLAS-internal.git && \
#     cd rocBLAS-internal && ./install.sh -idc -a gfx942 &&\
#     cd ../ && rm -rf rocBLAS-internal
RUN cd ~ && wget --no-check-certificate \
    https://compute-artifactory.amd.com/artifactory/list/rocm-osdb-20.04-deb/compute-rocm-dkms-no-npi-hipclang-12845/hsa-rocr_1.12.0.60000-crdnnh.12845%7E20.04_amd64.deb &&\
    dpkg-deb -x hsa-rocr_1.12.0.60000-crdnnh.12845~20.04_amd64.deb . && \
    cp opt/rocm-6.0.0-12845/lib/* /opt/rocm-6.0.0-12969/lib/ && rm -rf opt/rocm-6.0.0-12845

ENV HSA_OVERRIDE_GFX_VERSION=9.4.1

RUN rm /opt/rocm/share/rccl/msccl-algorithms/allreduce-allpairs-8n-simple.xml

COPY run_vllm.sh $WORKSPACE_DIR/
COPY run_vllm_tp1.sh $WORKSPACE_DIR/
COPY run_vllm_tp2.sh $WORKSPACE_DIR/
COPY run_vllm_tp4.sh $WORKSPACE_DIR/
COPY run_vllm_tp8.sh $WORKSPACE_DIR/
