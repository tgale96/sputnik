include(cmake/Cuda.cmake)

# TODO(tgale): Move cuSPARSE, cuBLAS deps to test & benchmark only.
cuda_find_library(CUDART_LIBRARY cudart)
cuda_find_library(CUBLAS_LIBRARY cublas)
cuda_find_library(CUSPARSE_LIBRARY cusparse)
list(APPEND SPUTNIK_LIBS "cudart;cublas;cusparse;culibos;cublasLt")

# CUTLASS.
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/cutlass/include)

if (BUILD_TEST)
  # Google Abseil.
  add_subdirectory(third_party/abseil-cpp)

  # Google Test and Google Mock.
  add_subdirectory(third_party/googletest)
  set(BUILD_GTEST ON CACHE INTERNAL "Build gtest submodule.")
  set(BUILD_GMOCK ON CACHE INTERNAL "Build gmock submodule.")
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/googletest/googletest/include)
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/googletest/googlemock/include)

  list(APPEND SPUTNIK_TEST_LIBS "gtest_main;gmock;absl::random_random")
endif()

if (BUILD_BENCHMARK)
  # Google Benchmark.
  add_subdirectory(third_party/benchmark)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "Build benchmark test suite.")
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/benchmark/include)

  list(APPEND SPUTNIK_BENCHMARK_LIBS "gtest;absl::random_random;benchmark::benchmark_main")
endif()
