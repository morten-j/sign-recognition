# Largely inspired from https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/833#issuecomment-893304985
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub

RUN apt update && \
    apt install -y wget curl

# Make apt not ask for geographic area during install
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt update && \
    apt-get install -y git libopencv-dev python3-pip libprotobuf-dev protobuf-compiler libboost-all-dev libhdf5-dev libgoogle-glog-dev libatlas-base-dev
RUN pip3 install --upgrade pip
RUN pip3 install numpy opencv-python

# Clone Openpose 1.6.0
RUN git clone --branch v1.6.0 https://github.com/CMU-Perceptual-Computing-Lab/openpose
WORKDIR /openpose
RUN git submodule update --init --recursive --remote
RUN apt-get install -y sudo

# install openpose dependencies
RUN chmod +x scripts/ubuntu/install_deps.sh
RUN scripts/ubuntu/install_deps.sh

# Install newer version of CMAKE since the one pre-installed does not work with CAFFE 
RUN apt remove --purge --auto-remove cmake
RUN apt update && \
    apt install -y software-properties-common lsb-release && \
    apt clean all
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
RUN apt update && \
    apt install kitware-archive-keyring && \
    rm /etc/apt/trusted.gpg.d/kitware.gpg
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6AF7F09730B3F0A4
RUN apt update
RUN apt install -y cmake

# Install openpose with CMAKE
RUN mkdir build
WORKDIR /openpose/build
RUN cmake -DPYBIND11_INSTALL=ON -DUSE_PYTHON_INCLUDE_DIR=ON -DGPU_MODE=CUDA -DUSE_CUDNN=ON -DBUILD_PYTHON=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 ..
RUN make -j`nproc`

# Install openpose to python
WORKDIR /openpose/build/python
RUN make install
RUN export PYTHONPATH=/usr/local/python/openpose:$PYTHONPATH

# Install old bachelor project
WORKDIR /
RUN git clone https://github.com/martinloenne/sign-language-recognition-service
WORKDIR /sign-language-recognition-service
COPY ./overrides/ .
RUN pip3 install -r requirements.txt

# Run the container forever
CMD python3 app.py