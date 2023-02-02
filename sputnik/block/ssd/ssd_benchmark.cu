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

#include <unistd.h>

#include "sputnik/cuda_utils.h"
#include "sputnik/block/ssd/ssd.h"
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
  std::vector<int> dims = {512, 1024, 2048, 4096, 8192, 16384};
  std::vector<float> sparsities = {1.0f, .5f, .1f, .01f};
  std::vector<int> transposes = {0, 1};

  for (const auto& d : dims) {
    for (const auto& s : sparsities) {
      for (const auto& ta : transposes) {
        for (const auto&tb : transposes) {
          b->Args({d, d, d, RoundUp((int)d * d * s, 128*128), ta, tb});
        }
      }
    }
  }
}

void BM_Ssd(benchmark::State& state) {
  const int kDimM = state.range(0);
  const int kDimK = state.range(1);
  const int kDimN = state.range(2);
  const int kNonZeros = state.range(3);
  const bool kTransposeA = state.range(4);
  const bool kTransposeB = state.range(5);
  const int kBlockDim = 128;

  // Need to set nonzeros differently if
  // we want to generalize.
  SPUTNIK_CHECK_EQ(kDimM, kDimK);
  SPUTNIK_CHECK_EQ(kDimN, kDimK);

  // Create the lhs matrix on gpu.
  absl::BitGen generator;
  int oda = kTransposeA ? kDimK : kDimM;
  int lda = kTransposeA ? kDimM : kDimK;
  CudaBlockSparseMatrix<half> lhs_matrix(
      oda, lda, kNonZeros, kBlockDim,
      RANDOM_UNIFORM, &generator,
      /*pad_rows_to=*/1);

  // Create the rhs matrix on gpu
  int odb = kTransposeB ? kDimN : kDimK;
  int ldb = kTransposeB ? kDimK : kDimN;
  CudaMatrix<half> rhs_matrix(odb, ldb, &generator);

  // Create the output matrix on gpu.
  CudaBlockSparseMatrix<half> out_matrix(
      kDimM, kDimN, kNonZeros, kBlockDim,
      RANDOM_UNIFORM, &generator,
      /*pad_rows_to=*/1);

  // Argument form.
  BlockMatrix lhs = Arg(lhs_matrix);
  Matrix rhs = Arg(rhs_matrix);
  BlockMatrix out = Arg(out_matrix);

  // Allocate the transpose workspace if necessary.
  if (kTransposeA) AllocateTransposeBuffers(lhs);

  int kIterations = 100;
  int kWarmupIterations = 10;
  int kSleepDuration = 50;
  while (state.KeepRunning()) {
    Timer timer;

    // Cool down.
    usleep(kSleepDuration * 1000);

    // Warmup.
    for (int i = 0; i < kWarmupIterations; ++i) {
      CUDA_CALL(Matmul(lhs, kTransposeA,
                       rhs, kTransposeB,
                       out, /*stream=*/0));
    }

    // Timed iterations.
    timer.start(0);
    for (int i = 0; i < kIterations; ++i) {
      CUDA_CALL(Matmul(lhs, kTransposeA,
                       rhs, kTransposeB,
                       out, /*stream=*/0));
    }
    timer.stop(0);
    state.SetIterationTime(timer.duration() / 1e3);
  }

  // Free the transpose workspace.
  if (kTransposeA) FreeTransposeBuffers(lhs);

  // Report throughput.
  double density = (double)kNonZeros / (kDimM * kDimN);
  int64_t flops = (int64_t)state.iterations() * kIterations *
                  kDimM * kDimN * kDimK * 2 *
                  density * density;
  state.counters["FLOPS"] = benchmark::Counter(
      flops, benchmark::Counter::kIsRate,
      benchmark::Counter::OneK::kIs1000);
}

BENCHMARK(BM_Ssd)->Apply(BenchmarkArgs)->UseManualTime()->MinTime(1e-2);

}  // namespace block
}  // namespace sputnik
