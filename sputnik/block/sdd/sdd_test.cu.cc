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
#include "sputnik/block/sdd/sdd.h"
#include "sputnik/block/matrix_utils.h"
#include "sputnik/block/row_indices/row_indices.h"

#include "absl/random/random.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace sputnik {
namespace block {

using ::testing::NanSensitiveFloatNear;
using ::testing::Pointwise;

template <
    int kDimM_,
    int kDimK_,
    int kDimN_,
    int kNonZeros_,
    int kBlockDim_,
    bool kTransposeA_ = false,
    bool kTransposeB_ = false,
    bool kUnorderedIndices_ = false>
struct Problem {
  static_assert(kNonZeros_ <= kDimM_ * kDimN_,
                "Number of non-zero must fit in the matrix.");

  static constexpr int kDimM = kDimM_;
  static constexpr int kDimK = kDimK_;
  static constexpr int kDimN = kDimN_;
  static constexpr int kNonZeros = kNonZeros_;
  static constexpr int kBlockDim = kBlockDim_;
  static constexpr int kTransposeA = kTransposeA_;
  static constexpr int kTransposeB = kTransposeB_;
  static constexpr bool kUnorderedIndices = kUnorderedIndices_;
};

template <typename Problem>
class SddTest : public ::testing::Test {
 public:
  const int kDimM = Problem::kDimM;
  const int kDimK = Problem::kDimK;
  const int kDimN = Problem::kDimN;
  const int kNonZeros = Problem::kNonZeros;
  const int kBlockDim = Problem::kBlockDim;
  const int kTransposeA = Problem::kTransposeA;
  const int kTransposeB = Problem::kTransposeB;
  const bool kUnorderedIndices = Problem::kUnorderedIndices;

  // Random number generator for creating matrices.
  absl::BitGen generator_;
};

typedef ::testing::Types<
    // Block 128 problems NN.
    Problem<128, 8, 128, 128*128, 128>,  // Minimum problem size.
    Problem<256, 8, 128, 256*128, 128>,  // Two tile rows.
    Problem<128, 8, 256, 256*128, 128>,  // Two tile columns.
    Problem<256, 8, 256, 256*128, 128>,  // 50% sparse, multi-row.
    // Block 128 problems NT.
    Problem<128, 8, 128, 128*128, 128, false, true>,
    Problem<256, 8, 128, 256*128, 128, false, true>,
    Problem<128, 8, 256, 256*128, 128, false, true>,
    Problem<256, 8, 256, 256*128, 128, false, true>,
    // Block 128 problems TN.
    Problem<128, 8, 128, 128*128, 128, true>,
    Problem<256, 8, 128, 256*128, 128, true>,
    Problem<128, 8, 256, 256*128, 128, true>,
    Problem<256, 8, 256, 256*128, 128, true>,
    // Block 128 problems TT.
    Problem<128, 8, 128, 128*128, 128, true, true>,
    Problem<256, 8, 128, 256*128, 128, true, true>,
    Problem<128, 8, 256, 256*128, 128, true, true>,
    Problem<256, 8, 256, 256*128, 128, true, true>,
    // Larger problems NN.
    Problem<512, 512, 1024, 512*1024, 128>,
    Problem<512, 512, 1024, 256*1024, 128>,
    Problem<512, 512, 1024, 128*1024, 128>,
    Problem<1024, 1024, 1024, 1024*1024, 128>,
    Problem<1024, 1024, 1024, 512*1024, 128>,
    Problem<1024, 1024, 1024, 256*1024, 128>,
    // Larger problems NT.
    Problem<512, 512, 1024, 512*1024, 128, false, true>,
    Problem<512, 512, 1024, 256*1024, 128, false, true>,
    Problem<512, 512, 1024, 128*1024, 128, false, true>,
    Problem<1024, 1024, 1024, 1024*1024, 128, false, true>,
    Problem<1024, 1024, 1024, 512*1024, 128, false, true>,
    Problem<1024, 1024, 1024, 256*1024, 128, false, true>,
    // Larger problems TN.
    Problem<512, 512, 1024, 512*1024, 128, true>,
    Problem<512, 512, 1024, 256*1024, 128, true>,
    Problem<512, 512, 1024, 128*1024, 128, true>,
    Problem<1024, 1024, 1024, 1024*1024, 128, true>,
    Problem<1024, 1024, 1024, 512*1024, 128, true>,
    Problem<1024, 1024, 1024, 256*1024, 128, true>,
    // Larger problems TT.
    Problem<512, 512, 1024, 512*1024, 128, true, true>,
    Problem<512, 512, 1024, 256*1024, 128, true, true>,
    Problem<512, 512, 1024, 128*1024, 128, true, true>,
    Problem<1024, 1024, 1024, 1024*1024, 128, true, true>,
    Problem<1024, 1024, 1024, 512*1024, 128, true, true>,
    Problem<1024, 1024, 1024, 256*1024, 128, true, true>,
    // Unordered problems NN.
    Problem<512, 512, 1024, 512*512, 128, false, false, true>,
    Problem<512, 512, 1024, 256*512, 128, false, false, true>,
    Problem<512, 512, 1024, 128*512, 128, false, false, true>,
    Problem<1024, 1024, 1024, 1024*1024, 128, false, false, true>,
    Problem<1024, 1024, 1024, 512*1024, 128, false, false, true>,
    Problem<1024, 1024, 1024, 256*1024, 128, false, false, true>,
    // Unordered problems TN.
    Problem<512, 512, 1024, 512*512, 128, true, false, true>,
    Problem<512, 512, 1024, 256*512, 128, true, false, true>,
    Problem<512, 512, 1024, 128*512, 128, true, false, true>,
    Problem<1024, 1024, 1024, 1024*1024, 128, true, false, true>,
    Problem<1024, 1024, 1024, 512*1024, 128, true, false, true>,
    Problem<1024, 1024, 1024, 256*1024, 128, true, false, true>,
    // Unordered problems NT.
    Problem<512, 512, 1024, 512*512, 128, false, true, true>,
    Problem<512, 512, 1024, 256*512, 128, false, true, true>,
    Problem<512, 512, 1024, 128*512, 128, false, true, true>,
    Problem<1024, 1024, 1024, 1024*1024, 128, false, true, true>,
    Problem<1024, 1024, 1024, 512*1024, 128, false, true, true>,
    Problem<1024, 1024, 1024, 256*1024, 128, false, true, true>,
    // Unordered problems TT.
    Problem<512, 512, 1024, 512*512, 128, true, true, true>,
    Problem<512, 512, 1024, 256*512, 128, true, true, true>,
    Problem<512, 512, 1024, 128*512, 128, true, true, true>,
    Problem<1024, 1024, 1024, 1024*1024, 128, true, true, true>,
    Problem<1024, 1024, 1024, 512*1024, 128, true, true, true>,
    Problem<1024, 1024, 1024, 256*1024, 128, true, true, true>,
  > TestProblems;

TYPED_TEST_SUITE(SddTest, TestProblems);

TYPED_TEST(SddTest, Sdd) {
  // Create the lhs matrix on cpu & gpu.
  int oda = this->kTransposeA ? this->kDimK : this->kDimM;
  int lda = this->kTransposeA ? this->kDimM : this->kDimK;
  sputnik::Matrix lhs(oda, lda, &this->generator_);
  CudaMatrix<half> lhs_gpu(lhs);

  // Create the rhs matrix on cpu & gpu
  int odb = this->kTransposeB ? this->kDimN : this->kDimK;
  int ldb = this->kTransposeB ? this->kDimK : this->kDimN;
  sputnik::Matrix rhs(odb, ldb, &this->generator_);
  CudaMatrix<half> rhs_gpu(rhs);

  // Create the output matrix on gpu & gpu.
  BlockSparseMatrix out_(
      this->kDimM, this->kDimN,
      this->kNonZeros, this->kBlockDim,
      RANDOM_UNIFORM, &this->generator_,
      /*pad_rows_to=*/1,
      /*unordered_indices=*/this->kUnorderedIndices);
  out_.Fill(1);
  sputnik::Matrix out = ToMatrix(out_);
  CudaBlockSparseMatrix<half> out_gpu(out_);

  // Allocate space for the row indices and set them up.
  BlockMatrix out_args = Arg(out_gpu);
  AllocateRowIndicesBuffer(out_args);
  CUDA_CALL(RowIndices(out_args, (short*)out_args.row_indices, /*stream=*/0));
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  // Run the gpu kernel.
  CUDA_CALL(Matmul(Arg(lhs_gpu), this->kTransposeA,
                   Arg(rhs_gpu), this->kTransposeB,
                   out_args, /*stream=*/0));
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  // Free the row indices buffer.
  FreeRowIndicesBuffer(out_args);

  // Verify the results.
  sputnik::Matrix expected =
      (this->kTransposeA ? lhs.T() : lhs) *
      (this->kTransposeB ? rhs.T() : rhs);
  expected.Mul(out);
  sputnik::Matrix results = ToMatrix(BlockSparseMatrix(out_gpu));
  auto comparator = Pointwise(NanSensitiveFloatNear(5e-02), ToVector(expected));
  ASSERT_THAT(ToVector(results), comparator);
}

}  // namespace block
}  // namespace sputnik
