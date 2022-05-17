# Largely inspired from https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/833#issuecomment-893304985
FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu16.04

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub

RUN apt update && \
    apt install -y wget

# Make apt not ask for geographic area during install
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt update && \
    apt-get install -y git cmake-qt-gui libopencv-dev python3-pip libprotobuf-dev protobuf-compiler libboost-all-dev libhdf5-dev libgoogle-glog-dev libatlas-base-dev

RUN pip3 install --upgrade pip
RUN pip3 install numpy opencv-python

RUN apt-get install -y sudo

# Clone Openpose
RUN git clone --branch v1.6.0 https://github.com/CMU-Perceptual-Computing-Lab/openpose
WORKDIR /openpose
RUN git submodule update --init --recursive --remote

# install openpose dependencies
RUN chmod +x scripts/ubuntu/install_deps.sh
RUN scripts/ubuntu/install_deps.sh

# # Install openpose with CMAKE
RUN mkdir build
WORKDIR /openpose/build

#RUN cmake -DPYBIND11_INSTALL=ON -DUSE_PYTHON_INCLUDE_DIR=ON -DGPU_MODE=CUDA -DUSE_CUDNN=ON -DBUILD_PYTHON=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 ..
RUN cmake -DBUILD_PYTHON=ON ..
RUN make -j`nproc`

# Install openpose to python
WORKDIR /openpose/build/python
RUN make install
RUN export PYTHONPATH=/usr/local/python/openpose:$PYTHONPATH

# Install old bachelor project
WORKDIR /
RUN git clone https://github.com/martinloenne/sign-language-recognition-service
WORKDIR /sign-language-recognition-service
RUN pip3 install -r requirements.txt

# Run the container forever
ENV FLASK_APP=app
CMD flask run --host=0.0.0.0