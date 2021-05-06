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
#include "sputnik/block/dsd/dsd.h"
#include "sputnik/block/matrix_utils.h"
#include "sputnik/timer.h"

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
      b->Args({d, d, d, RoundUp((int)d * d * s, 128*128)});
    }
  }
}

void BM_Dsd(benchmark::State& state) {
  const int kDimM = state.range(0);
  const int kDimK = state.range(1);
  const int kDimN = state.range(2);
  const int kNonZeros = state.range(3);
  const int kBlockDim = 128;
  const bool kTransposeB = true;

  // Create the sparse matrix on cpu & gpu.
  absl::BitGen generator;
  CudaBlockSparseMatrix<half> lhs(
      kDimM, kDimK, kNonZeros, kBlockDim,
      RANDOM_UNIFORM, &generator, /*pad_rows_to=*/1);

  // Create the dense matrix on cpu & gpu
  CudaMatrix<half> rhs(kDimN, kDimK, &generator);

  // Create the output matrix on gpu & gpu.
  CudaMatrix<half> out(kDimM, kDimN, &generator);

  int iterations = 10;
  while (state.KeepRunningBatch(iterations)) {
    Timer timer;

    timer.start(0);
    for (int i = 0; i < iterations; ++i) {
      CUDA_CALL(Dsd(
          kDimM, kDimK, kDimN,
          lhs.NumElementsWithPadding(),
          kBlockDim,
          lhs.Values(),
          lhs.RowOffsets(),
          lhs.ColumnIndices(),
          rhs.Values(), kTransposeB,
          out.Values(), 0));
    }
    timer.stop(0);
    state.SetIterationTime(timer.duration() / 1e3);
  }

  // Report throughput.
  int64_t flops = (int64_t)state.iterations() *
                  lhs.NumElementsWithPadding() *
                  kBlockDim * kBlockDim * kDimN * 2;
  state.counters["FLOPS"] = benchmark::Counter(
      flops, benchmark::Counter::kIsRate,
      benchmark::Counter::OneK::kIs1000);
}

BENCHMARK(BM_Dsd)->Apply(BenchmarkArgs)->UseManualTime();

}  // namespace block
}  // namespace sputnik
