FROM compute-artifactory.amd.com:5000/rocm-plus-docker/framework/compute-rocm-dkms-no-npi-hipclang:12969_ubuntu20.04_py3.9_pytorch_rocm6.0_internal_testing_4b126e1
ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR
RUN apt install -y rocblas-benchmarks hipblaslt-benchmarks
# Limit arch's so composable kernel doesn't take days to finish
ENV PYTORCH_ROCM_ARCH=gfx90a;gfx940;gfx941;gfx942
COPY fa_arch.patch $WORKSPACE_DIR
RUN cd $WORKSPACE_DIR &&\
    git clone --recursive -b junhzhan/fa-mi300 https://github.com/ROCmSoftwarePlatform/flash-attention &&\
    cd flash-attention && git apply ../fa_arch.patch &&\
    python setup.py install 
RUN git clone -b vllm_mi300 https://streamhsa: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@github.com/ROCmSoftwarePlatform/SLOT.git
RUN mv $WORKSPACE_DIR/SLOT/projects/vllm $WORKSPACE_DIR
RUN mv $WORKSPACE_DIR/SLOT/projects/gradlib $WORKSPACE_DIR
RUN rm -rf $WORKSPACE_DIR/SLOT
RUN cd $WORKSPACE_DIR/gradlib &&\
    python setup.py develop
RUN cd $WORKSPACE_DIR/vllm/ &&\
    pip install -r requirements.txt &&\
    pip install typing-extensions==4.8.0 &&\
    python setup.py build && python setup.py develop; exit 0
ENV PYTHONPATH=$WORKSPACE_DIR/gradlib
RUN cd ${WORKSPACE_DIR} && \
    git clone -b develop https://github.com/ROCmSoftwarePlatform/hipBLASLt && \
    cd hipBLASLt && \
    ./install.sh -idc &&\
    cd ../ && rm -rf hipBLASLt

RUN pip install pandas Ray pyarrow
COPY run_vllm.sh $WORKSPACE_DIR/
COPY run_vllm_tp1.sh $WORKSPACE_DIR/
COPY run_vllm_tp2.sh $WORKSPACE_DIR/
COPY run_vllm_tp4.sh $WORKSPACE_DIR/
COPY run_vllm_tp8.sh $WORKSPACE_DIR/
