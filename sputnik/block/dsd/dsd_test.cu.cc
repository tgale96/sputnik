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

#include "absl/random/random.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace sputnik {
namespace block {

using ::testing::NanSensitiveFloatNear;
using ::testing::Pointwise;

template <int kDimM_, int kDimK_, int kDimN_, int kNonZeros_, int kBlockDim_>
struct Problem {
  static_assert(kNonZeros_ <= kDimM_ * kDimK_,
                "Number of non-zero must fit in the matrix.");

  static constexpr int kDimM = kDimM_;
  static constexpr int kDimK = kDimK_;
  static constexpr int kDimN = kDimN_;
  static constexpr int kNonZeros = kNonZeros_;
  static constexpr int kBlockDim = kBlockDim_;
};

template <typename Problem>
class DsdTest : public ::testing::Test {
 public:
  const int kDimM = Problem::kDimM;
  const int kDimK = Problem::kDimK;
  const int kDimN = Problem::kDimN;
  const int kNonZeros = Problem::kNonZeros;
  const int kBlockDim = Problem::kBlockDim;

  // Random number generator for creating matrices.
  absl::BitGen generator_;
};

typedef ::testing::Types<
  Problem<32, 32, 64, 32*32, 32>,   // Minimum problem size.
  Problem<32, 64, 64, 32*64, 32>,   // Two inner loops.
  Problem<64, 32, 64, 64*32, 32>,   // Two rows of blocks.
  Problem<32, 32, 128, 32*32, 32>,  // Two tile columns.
  Problem<32, 64, 64, 32*32, 32>,   // 50% sparse.
  Problem<64, 64, 64, 32*64, 32>,   // 50% sparse, multi-row.
  // Larger problems.
  Problem<128, 128, 128, 32*128, 32>,
  Problem<512, 512, 1024, 512*512, 32>,
  Problem<512, 512, 1024, 256*512, 32>,
  Problem<512, 512, 1024, 128*512, 32>,
  Problem<1024, 1024, 1024, 1024*1024, 32>,
  Problem<1024, 1024, 1024, 512*1024, 32>,
  Problem<1024, 1024, 1024, 256*1024, 32>
  > TestProblems;

TYPED_TEST_SUITE(DsdTest, TestProblems);

TYPED_TEST(DsdTest, Dsd) {
  // Create the sparse matrix on cpu & gpu.
  BlockSparseMatrix lhs_(
      this->kDimM, this->kDimK,
      this->kNonZeros, this->kBlockDim,
      RANDOM_UNIFORM, &this->generator_,
      /*pad_rows_to=*/1);  
  Matrix lhs = ToMatrix(lhs_);
  CudaBlockSparseMatrix<half> lhs_gpu(lhs_);
  
  // Create the dense matrix on cpu & gpu
  Matrix rhs(this->kDimK, this->kDimN, &this->generator_);
  CudaMatrix<half> rhs_gpu(rhs);

  // Create the output matrix on gpu & gpu.
  CudaMatrix<half> out_gpu(this->kDimM, this->kDimN, &this->generator_);
  
  // Run the gpu kernel.
  CUDA_CALL(Dsd(this->kDimM, this->kDimK, this->kDimN,
		lhs_gpu.NumElementsWithPadding(),
		this->kBlockDim, lhs_gpu.Values(),
		lhs_gpu.RowOffsets(),
		lhs_gpu.ColumnIndices(),
		rhs_gpu.Values(), false,
		out_gpu.Values(), 0));
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  // Verify the results.
  Matrix out = lhs * rhs;  
  Matrix results(out_gpu);  
  auto comparator = Pointwise(NanSensitiveFloatNear(5e-02), ToVector(out));
  ASSERT_THAT(ToVector(results), comparator);
}

}  // namespace block
}  // namespace sputnik
