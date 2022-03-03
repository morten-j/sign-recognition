FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04

RUN apt-get update && \ 
DEBIAN_FRONTEND=noninteractive apt-get install -y git cmake g++ protobuf-compiler libgoogle-glog-dev libopencv-dev libboost-program-options-dev libboost-test-dev libboost-python-dev libboost-all-dev libhdf5-dev libatlas-base-dev

RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose
WORKDIR /openpose/
RUN git submodule update --init --recursive --remote

RUN cmake --clean-first .
RUN cmake --build . --config Release


ENTRYPOINT [ "tail", "-f", "/dev/null" ]