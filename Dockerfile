FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu18.04

# Install tools and dependencies.
RUN apt-get -y update --fix-missing
RUN apt-get update -y
RUN apt-get install -y \
  emacs \
  git \
  wget \
  libgoogle-glog-dev \
  nsight-compute-2020.3.1

ENV PATH="/opt/nvidia/nsight-compute/2020.3.1/:${PATH}"

# Install the latest version of cmake.
RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.3/cmake-3.21.3-linux-x86_64.sh
RUN mkdir /usr/local/cmake
RUN sh cmake-3.21.3-linux-x86_64.sh --skip-license --prefix=/usr/local/cmake
ENV PATH="/usr/local/cmake/bin:${PATH}"

# Set the working directory.
WORKDIR /mount/sputnik
