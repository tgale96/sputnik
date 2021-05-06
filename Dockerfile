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

# Setup to install the latest version of cmake.
RUN apt-get install -y software-properties-common && \
    apt-get update && \
    wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null && \
    apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main' && \
    apt-get update && apt-get install -y cmake

# Set the working directory.
WORKDIR /mount/sputnik
