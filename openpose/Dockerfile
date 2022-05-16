# Largely inspired from https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/833#issuecomment-893304985
FROM nvidia/cuda:11.4.0-devel-ubuntu20.04

RUN apt update && \
    apt install -y wget

# install cuDNN
#RUN wget -P /tmp http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8_8.2.2.26-1+cuda11.4_amd64.deb
#RUN dpkg -i /tmp/libcudnn8_8.2.2.26-1+cuda11.4_amd64.deb
#RUN wget -P /tmp http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcudnn8-dev_8.2.2.26-1+cuda11.4_amd64.deb
#RUN dpkg -i /tmp/libcudnn8-dev_8.2.2.26-1+cuda11.4_amd64.deb

# Make apt not ask for geographic area during install
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt update && \
    apt-get install -y git cmake-qt-gui libopencv-dev python3-pip libprotobuf-dev protobuf-compiler libboost-all-dev libhdf5-dev libgoogle-glog-dev libatlas-base-dev

RUN pip install numpy opencv-python

# Clone Openpose
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
WORKDIR /openpose
RUN git submodule update --init --recursive --remote

# install openpose dependencies
RUN chmod +x scripts/ubuntu/install_deps.sh
RUN scripts/ubuntu/install_deps.sh

# Install openpose with CMAKE
RUN mkdir build
WORKDIR /openpose/build
RUN cmake -DPYBIND11_INSTALL=ON -DUSE_PYTHON_INCLUDE_DIR=ON -DGPU_MODE=CUDA -DUSE_CUDNN=ON -DBUILD_PYTHON=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 ..
RUN make -j`nproc`

# Install openpose to python
WORKDIR /openpose/build/python
RUN make install
RUN export PYTHONPATH=/usr/local/python/openpose:$PYTHONPATH

# Run the container forever
CMD tail -f /dev/null