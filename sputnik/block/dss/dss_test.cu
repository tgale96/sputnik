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
#include <iostream>

#include "sputnik/cuda_utils.h"
#include "sputnik/block/dss/dss.h"
#include "sputnik/block/bitmask/bitmask.h"
#include "sputnik/block/matrix_utils.h"

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
    int kNonZerosA_,
    int kNonZerosB_,
    int kBlockDim_,
    bool kTransposeA_ = false,
    bool kTransposeB_ = false,
    bool kUnorderedIndices_ = false>
struct Problem {
  static_assert(kNonZerosA_ <= kDimM_ * kDimK_,
                "Number of non-zero must fit in the lhs matrix.");
  static_assert(kNonZerosB_ <= kDimK_ * kDimN_,
                "Number of non-zero must fit in the rhs matrix.");

  static constexpr int kDimM = kDimM_;
  static constexpr int kDimK = kDimK_;
  static constexpr int kDimN = kDimN_;
  static constexpr int kNonZerosA = kNonZerosA_;
  static constexpr int kNonZerosB = kNonZerosB_;
  static constexpr int kBlockDim = kBlockDim_;
  static constexpr int kTransposeA = kTransposeA_;
  static constexpr int kTransposeB = kTransposeB_;
  static constexpr bool kUnorderedIndices = kUnorderedIndices_;
};

template <typename Problem>
class DssTest : public ::testing::Test {
 public:
  const int kDimM = Problem::kDimM;
  const int kDimK = Problem::kDimK;
  const int kDimN = Problem::kDimN;
  const int kNonZerosA = Problem::kNonZerosA;
  const int kNonZerosB = Problem::kNonZerosB;
  const int kBlockDim = Problem::kBlockDim;
  const int kTransposeA = Problem::kTransposeA;
  const int kTransposeB = Problem::kTransposeB;
  const bool kUnorderedIndices = Problem::kUnorderedIndices;

  // Random number generator for creating matrices.
  absl::BitGen generator_;
};

// TODO(tgale): Make our test harness more efficient so that we
// can test larger problems (up to 32k).
typedef ::testing::Types<
    // Block 128 problems NN.
    Problem<128, 128, 128, 128*128, 128*128, 128>,  // Minimum problem size.
    Problem<128, 256, 128, 128*256, 128*256, 128>,  // Two inner loops.
    Problem<256, 128, 128, 128*256, 128*128, 128>,  // Two rows of blocks.
    Problem<128, 128, 256, 128*128, 128*256, 128>,  // Two columns of blocks.
    Problem<128, 256, 128, 128*128, 128*256, 128>,  // 50% sparse lhs.
    Problem<128, 256, 128, 128*256, 128*128, 128>,  // 50% sparse rhs.
    Problem<128, 256, 128, 128*128, 128*128, 128>,  // 50% sparse both.
    Problem<256, 128, 128, 128*128, 128*128, 128>,  // 50% lhs, multi-row.
    Problem<128, 128, 256, 128*128, 128*128, 128>,  // 50% rhs, multi-col.
    Problem<256, 128, 128, 128*128, 128*128, 128>,  // 50% both, multi-both.
    Problem<256, 256, 256, 128*256, 256*128, 128>,  // 50% both, two loops.
    // Block 128 problems NT.
    Problem<128, 128, 128, 128*128, 128*128, 128, false, true>,
    Problem<128, 256, 128, 128*256, 128*256, 128, false, true>,
    Problem<256, 128, 128, 128*256, 128*128, 128, false, true>,
    Problem<128, 128, 256, 128*128, 128*256, 128, false, true>,
    Problem<128, 256, 128, 128*128, 128*256, 128, false, true>,
    Problem<128, 256, 128, 128*256, 128*128, 128, false, true>,
    Problem<128, 256, 128, 128*128, 128*128, 128, false, true>,
    Problem<256, 128, 128, 128*128, 128*128, 128, false, true>,
    Problem<128, 128, 256, 128*128, 128*128, 128, false, true>,
    Problem<256, 128, 128, 128*128, 128*128, 128, false, true>,
    Problem<256, 256, 256, 128*256, 256*128, 128, false, true>,
    // Block 128 problems TN.
    Problem<128, 128, 128, 128*128, 128*128, 128, true>,
    Problem<128, 256, 128, 128*256, 128*256, 128, true>,
    Problem<256, 128, 128, 128*256, 128*128, 128, true>,
    Problem<128, 128, 256, 128*128, 128*256, 128, true>,
    Problem<128, 256, 128, 128*128, 128*256, 128, true>,
    Problem<128, 256, 128, 128*256, 128*128, 128, true>,
    Problem<128, 256, 128, 128*128, 128*128, 128, true>,
    Problem<256, 128, 128, 128*128, 128*128, 128, true>,
    Problem<128, 128, 256, 128*128, 128*128, 128, true>,
    Problem<256, 128, 128, 128*128, 128*128, 128, true>,
    Problem<256, 256, 256, 128*256, 256*128, 128, true>,
    // Block 128 problems TT.
    Problem<128, 128, 128, 128*128, 128*128, 128, true, true>,
    Problem<128, 256, 128, 128*256, 128*256, 128, true, true>,
    Problem<256, 128, 128, 128*256, 128*128, 128, true, true>,
    Problem<128, 128, 256, 128*128, 128*256, 128, true, true>,
    Problem<128, 256, 128, 128*128, 128*256, 128, true, true>,
    Problem<128, 256, 128, 128*256, 128*128, 128, true, true>,
    Problem<128, 256, 128, 128*128, 128*128, 128, true, true>,
    Problem<256, 128, 128, 128*128, 128*128, 128, true, true>,
    Problem<128, 128, 256, 128*128, 128*128, 128, true, true>,
    Problem<256, 128, 128, 128*128, 128*128, 128, true, true>,
    Problem<256, 256, 256, 128*256, 256*128, 128, true, true>,
    // Larger problems NN.
    Problem<512, 512, 512, 512*512, 512*512, 128>,
    Problem<512, 512, 512, 256*512, 256*512, 128>,
    Problem<512, 512, 512, 128*512, 128*512, 128>,
    Problem<1024, 1024, 1024, 1024*1024, 1024*1024, 128>,
    Problem<1024, 1024, 1024, 512*1024, 512*1024, 128>,
    Problem<1024, 1024, 1024, 256*1024, 256*1024, 128>,
    // Larger problems NT.
    Problem<512, 512, 512, 512*512, 512*512, 128, false, true>,
    Problem<512, 512, 512, 256*512, 256*512, 128, false, true>,
    Problem<512, 512, 512, 128*512, 128*512, 128, false, true>,
    Problem<1024, 1024, 1024, 1024*1024, 1024*1024, 128, false, true>,
    Problem<1024, 1024, 1024, 512*1024, 512*1024, 128, false, true>,
    Problem<1024, 1024, 1024, 256*1024, 256*1024, 128, false, true>,
    // Larger problems TN.
    Problem<512, 512, 512, 512*512, 512*512, 128, true>,
    Problem<512, 512, 512, 256*512, 256*512, 128, true>,
    Problem<512, 512, 512, 128*512, 128*512, 128, true>,
    Problem<1024, 1024, 1024, 1024*1024, 1024*1024, 128, true>,
    Problem<1024, 1024, 1024, 512*1024, 512*1024, 128, true>,
    Problem<1024, 1024, 1024, 256*1024, 256*1024, 128, true>,
    // Larger problems TT.
    Problem<512, 512, 512, 512*512, 512*512, 128, true, true>,
    Problem<512, 512, 512, 256*512, 256*512, 128, true, true>,
    Problem<512, 512, 512, 128*512, 128*512, 128, true, true>,
    Problem<1024, 1024, 1024, 1024*1024, 1024*1024, 128, true, true>,
    Problem<1024, 1024, 1024, 512*1024, 512*1024, 128, true, true>,
    Problem<1024, 1024, 1024, 256*1024, 256*1024, 128, true, true>,
    // Unordered problems NN.
    Problem<512, 512, 512, 512*512, 512*512, 128, false, false, true>,
    Problem<512, 512, 512, 256*512, 256*512, 128, false, false, true>,
    Problem<512, 512, 512, 128*512, 128*512, 128, false, false, true>,
    Problem<1024, 1024, 1024, 1024*1024, 1024*1024, 128, false, false, true>,
    Problem<1024, 1024, 1024, 512*1024, 512*1024, 128, false, false, true>,
    Problem<1024, 1024, 1024, 256*1024, 256*1024, 128, false, false, true>,
    // Unordered problems TN.
    Problem<512, 512, 512, 512*512, 512*512, 128, true, false, true>,
    Problem<512, 512, 512, 256*512, 256*512, 128, true, false, true>,
    Problem<512, 512, 512, 128*512, 128*512, 128, true, false, true>,
    Problem<1024, 1024, 1024, 1024*1024, 1024*1024, 128, true, false, true>,
    Problem<1024, 1024, 1024, 512*1024, 512*1024, 128, true, false, true>,
    Problem<1024, 1024, 1024, 256*1024, 256*1024, 128, true, false, true>,
    // Unordered problems NT.
    Problem<512, 512, 512, 512*512, 512*512, 128, false, true, true>,
    Problem<512, 512, 512, 256*512, 256*512, 128, false, true, true>,
    Problem<512, 512, 512, 128*512, 128*512, 128, false, true, true>,
    Problem<1024, 1024, 1024, 1024*1024, 1024*1024, 128, false, true, true>,
    Problem<1024, 1024, 1024, 512*1024, 512*1024, 128, false, true, true>,
    Problem<1024, 1024, 1024, 256*1024, 256*1024, 128, false, true, true>,
    // Unordered problems TT.
    Problem<512, 512, 512, 512*512, 512*512, 128, true, true, true>,
    Problem<512, 512, 512, 256*512, 256*512, 128, true, true, true>,
    Problem<512, 512, 512, 128*512, 128*512, 128, true, true, true>,
    Problem<1024, 1024, 1024, 1024*1024, 1024*1024, 128, true, true, true>,
    Problem<1024, 1024, 1024, 512*1024, 512*1024, 128, true, true, true>,
    Problem<1024, 1024, 1024, 256*1024, 256*1024, 128, true, true, true>,
    > TestProblems;

TYPED_TEST_SUITE(DssTest, TestProblems);

TYPED_TEST(DssTest, Dss) {
  // Create the lhs matrix on cpu & gpu.
  int oda = this->kTransposeA ? this->kDimK : this->kDimM;
  int lda = this->kTransposeA ? this->kDimM : this->kDimK;
  BlockSparseMatrix lhs_(
      oda, lda, this->kNonZerosA, this->kBlockDim,
      RANDOM_UNIFORM, &this->generator_,
      /*pad_rows_to=*/1,
      /*unordered_indices=*/this->kUnorderedIndices);
  sputnik::Matrix lhs = ToMatrix(lhs_);
  CudaBlockSparseMatrix<half> lhs_gpu(lhs_);

  // Create the rhs matrix on cpu & gpu
  int odb = this->kTransposeB ? this->kDimN : this->kDimK;
  int ldb = this->kTransposeB ? this->kDimK : this->kDimN;
  BlockSparseMatrix rhs_(
      odb, ldb, this->kNonZerosB, this->kBlockDim,
      RANDOM_UNIFORM, &this->generator_,
      /*pad_rows_to=*/1,
      /*unordered_indices=*/this->kUnorderedIndices);
  sputnik::Matrix rhs = ToMatrix(rhs_);
  CudaBlockSparseMatrix<half> rhs_gpu(rhs_);

  // Create the output matrix on gpu & gpu.
  CudaMatrix<half> out_gpu(this->kDimM, this->kDimN, &this->generator_);

  // Run the gpu kernel.
  BlockMatrix lhs_args = Arg(lhs_gpu);
  BlockMatrix rhs_args = Arg(rhs_gpu);
  if (this->kTransposeA) AllocateTransposeBuffers(lhs_args);
  if (!this->kTransposeB) AllocateTransposeBuffers(rhs_args);
  AllocateBitmaskBuffers(lhs_args);
  AllocateBitmaskBuffers(rhs_args);
  CUDA_CALL(Matmul(lhs_args, this->kTransposeA,
                   rhs_args, this->kTransposeB,
                   Arg(out_gpu), /*stream=*/0));
  CUDA_CALL(cudaStreamSynchronize(nullptr));
  FreeBitmaskBuffers(lhs_args);
  FreeBitmaskBuffers(rhs_args);
  if (this->kTransposeA) FreeTransposeBuffers(lhs_args);
  if (!this->kTransposeB) FreeTransposeBuffers(rhs_args);

  // Verify the results.
  sputnik::Matrix expected =
      (this->kTransposeA ? lhs.T() : lhs) *
      (this->kTransposeB ? rhs.T() : rhs);
  sputnik::Matrix results(out_gpu);
  auto comparator = Pointwise(NanSensitiveFloatNear(5e-02), ToVector(expected));
  ASSERT_THAT(ToVector(results), comparator);
}

}  // namespace block
}  // namespace sputnik
