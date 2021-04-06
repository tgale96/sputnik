// Copyright 2020 The Sputnik Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sputnik/cuda_utils.h"
#include "sputnik/block/bdsd/cuda_bdsd.h"
#include "sputnik/block/matrix_utils.h"

#include "absl/random/random.h"
#include "benchmark/benchmark.h"

namespace sputnik {
namespace block {
    
int RoundUp(int x, int b) {
  return (x + b - 1) / b * b;
}

void BenchmarkArgs(benchmark::internal::Benchmark* b) {
    std::vector<int> dims;
  for (int i = 0; i < 31; ++i) {
    dims.push_back(512 + i * 256);
  }
  std::vector<float> sparsities = {1.0f};

  for (const auto& d : dims) {
    for (const auto& s : sparsities) {
      b->Args({d, d, d, RoundUp(static_cast<int>(d * d * s), 32*32)});
    }
  }
}

void BM_CudaBdsd(benchmark::State& state) {
  const int kDimM = state.range(0);
  const int kDimK = state.range(1);
  const int kDimN = state.range(2);
  const int kNonZeros = state.range(3);
  const int kBlockDim = 32;

  // Create the sparse matrix on cpu & gpu.
  absl::BitGen generator;
  CudaBlockSparseMatrix<half> sparse_matrix(
      kDimM, kDimK, kNonZeros, kBlockDim,
      RANDOM_UNIFORM, &generator, /*pad_rows_to=*/1);

  // Create the dense matrix on cpu & gpu
  CudaMatrix<half> matrix(kDimK, kDimN, &generator);

  // Create the output matrix on gpu & gpu.
  CudaMatrix<half> output_matrix(kDimM, kDimN, &generator);

  int batch_size = 10;
  while (state.KeepRunningBatch(batch_size)) {
    for (int i = 0; i < batch_size; ++i) {
      CUDA_CALL(CudaBdsd(
          kDimM, kDimK, kDimN,
          sparse_matrix.NumElementsWithPadding(),
          kBlockDim,
          sparse_matrix.Values(),
          sparse_matrix.RowOffsets(),
          sparse_matrix.ColumnIndices(),
          matrix.Values(),
          output_matrix.Values(), 0));
    }
    CUDA_CALL(cudaStreamSynchronize(0));
  }

  // Report throughput.
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) *
      sparse_matrix.NumElementsWithPadding() *
      kBlockDim * kBlockDim * kDimN * 2);
}

BENCHMARK(BM_CudaBdsd)->Apply(BenchmarkArgs)->UseRealTime();

}  // namespace block
}  // namespace sputnik
