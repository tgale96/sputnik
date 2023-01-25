# NOTE

This repository is a hard-fork of [Sputnik](https://github.com/google-research/sputnik) that has been extended with block-sparse matrix multiplication kernels. It is in the process of being simplified to remove the majority of Sputnik code. The following is copied from Sputnik, with an updated citation for the new block-sparse kernels.

## Build

Sputnik uses the CMake build system. Sputnik depends on the CUDA toolkit (v10.1+) and supports SM70+. To build the library, enter the project directory and run the following commands:

`mkdir build && cd build`

`cmake .. -DCMAKE_BUILD_TYPE=Release`

`make -j12`

The test and benchmark suites additionally depend on [abseil/abseil-cpp](https://github.com/abseil/abseil-cpp), [google/googltest](https://github.com/google/googletest), and [google/benchmark](https://github.com/google/benchmark). These dependencies are includes as submodules in [third_party](https://github.com/google-research/sputnik/tree/os-build/third_party). To build the test suite and/or benchmark suite, set `-DBUILD_TEST=ON` and/or `-DBUILD_BENCHMARK=ON` in your `cmake` command.

`cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON -DCUDA_ARCHS="70;75"`

## Docker

Sputnik provides a [Dockerfile](https://github.com/google-research/sputnik/blob/os-build/Dockerfile) that builds the proper environment with all dependencies. Note that [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) must be installed to run on GPU. To build the image, run the following command:

`docker build . -t sputnik-dev`

To launch the container with the sputnik source mounted under `/mount` (assuming you're working out of $HOME), run the following:

`sudo docker run --runtime=nvidia -v ~/:/mount/ -it sputnik-dev:latest`

## Citation

If you make use of this library, please cite:

```
@article{megablocks-arxiv,
  author    = {Trevor Gale and
               Deepak Narayanan and
               Cliff Young and
               Matei Zaharia},
  title     = {MegaBlocks: Efficient Sparse Training with Mixture-of-Experts},
  journal   = {CoRR},
  volume    = {abs/2211.15841},
  year      = {2022},
}
```
