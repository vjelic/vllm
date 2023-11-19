FROM compute-artifactory.amd.com:5000/rocm-plus-docker/framework/compute-rocm-dkms-no-npi-hipclang:12969_ubuntu20.04_py3.9_pytorch_rocm6.0_internal_testing_4b126e1
ENV WORKSPACE_DIR=/workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR
# Limit arch's so composable kernel doesn't take days to finish
ENV PYTORCH_ROCM_ARCH=gfx90a;gfx940;gfx941;gfx942
COPY fa_arch.patch $WORKSPACE_DIR
RUN apt update && apt install -y sqlite3 libsqlite3-dev libfmt-dev
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
ENV MAX_JOBS=32
RUN cd ${WORKSPACE_DIR} && \
    git clone -b exp_bandaid https://github.com/ROCmSoftwarePlatform/rccl && \
    cd rccl && mkdir build && cd build && \
    CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_PREFIX_PATH=/opt/rocm/ .. && make -j
RUN git clone https://streamhsa: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@github.com/ROCmSoftwarePlatform/gradlib.git &&\
    cd gradlib &&\
    python setup.py develop
RUN git clone -b v5_msccl https://streamhsa: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@github.com/ROCmSoftwarePlatform/vllm-private.git &&\
    cd vllm-private &&\
    pip install -r requirements.txt &&\
    pip install typing-extensions==4.8.0 &&\
    python setup.py build && python setup.py develop; exit 0
RUN pip install pyarrow Ray pandas==2.0 numpy==1.20.3

RUN cd ~ && wget --no-check-certificate \
    https://compute-artifactory.amd.com/artifactory/list/rocm-osdb-20.04-deb/compute-rocm-dkms-no-npi-hipclang-12845/hsa-rocr_1.12.0.60000-crdnnh.12845%7E20.04_amd64.deb &&\
    dpkg-deb -x hsa-rocr_1.12.0.60000-crdnnh.12845~20.04_amd64.deb . && \
    cp opt/rocm-6.0.0-12845/lib/* /opt/rocm-6.0.0-12969/lib/ && rm -rf opt/rocm-6.0.0-12845

RUN git clone https://github.com/ROCmSoftwarePlatform/rocmProfileData.git &&\
    cd rocmProfileData && make; make install


COPY libamdhip64.so.6 /opt/rocm/lib/
COPY libfile_plugin.so /opt/rocm/lib/roctracer
COPY run_13b.sh $WORKSPACE_DIR/
COPY run_70b.sh $WORKSPACE_DIR/
