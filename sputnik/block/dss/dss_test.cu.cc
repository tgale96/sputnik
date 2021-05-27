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
#include "sputnik/block/sds/sds.h"
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
    bool kTransposeB_ = false>
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

  // Random number generator for creating matrices.
  absl::BitGen generator_;
};

typedef ::testing::Types<
    // Block 128 problems NT.
    Problem<128, 128, 128, 128*128, 128*128, 128, false, true>,  // Minimum problem size.
    Problem<128, 256, 128, 128*256, 128*256, 128, false, true>,  // Two inner loops.
    Problem<256, 128, 128, 128*256, 128*128, 128, false, true>,  // Two rows of blocks.
    Problem<128, 128, 256, 128*128, 128*256, 128, false, true>,  // Two columns of blocks.
    Problem<128, 256, 128, 128*128, 128*256, 128, false, true>,  // 50% sparse lhs.
    Problem<128, 256, 128, 128*256, 128*128, 128, false, true>,  // 50% sparse rhs.
    Problem<128, 256, 128, 128*128, 128*128, 128, false, true>,  // 50% sparse both.
    Problem<256, 128, 128, 128*128, 128*128, 128, false, true>,  // 50% lhs, multi-row.
    Problem<128, 128, 256, 128*128, 128*128, 128, false, true>,  // 50% rhs, multi-col.
    Problem<256, 128, 128, 128*128, 128*128, 128, false, true>,  // 50% both, multi-both.
    Problem<256, 256, 256, 128*256, 256*128, 128, false, true>,  // 50% both, two loops.
    // Larger problems NT.
    Problem<512, 512, 512, 512*512, 512*512, 128, false, true>,
    Problem<512, 512, 512, 256*512, 256*512, 128, false, true>,
    Problem<512, 512, 512, 128*512, 128*512, 128, false, true>,
    Problem<1024, 1024, 1024, 1024*1024, 1024*1024, 128, false, true>,
    Problem<1024, 1024, 1024, 512*1024, 512*1024, 128, false, true>,
    Problem<1024, 1024, 1024, 256*1024, 256*1024, 128, false, true>,
    > TestProblems;

TYPED_TEST_SUITE(DssTest, TestProblems);

TYPED_TEST(DssTest, Dss) {
  // Create the lhs matrix on cpu & gpu.
  int oda = this->kTransposeA ? this->kDimK : this->kDimM;
  int lda = this->kTransposeA ? this->kDimM : this->kDimK;
  BlockSparseMatrix lhs_(
      oda, lda, this->kNonZerosA, this->kBlockDim,
      RANDOM_UNIFORM, &this->generator_,
      /*pad_rows_to=*/1);
  sputnik::Matrix lhs = ToMatrix(lhs_);
  CudaBlockSparseMatrix<half> lhs_gpu(lhs_);

  // Create the rhs matrix on cpu & gpu
  int odb = this->kTransposeB ? this->kDimN : this->kDimK;
  int ldb = this->kTransposeB ? this->kDimK : this->kDimN;
  BlockSparseMatrix rhs_(
      odb, ldb, this->kNonZerosB, this->kBlockDim,
      RANDOM_UNIFORM, &this->generator_,
      /*pad_rows_to=*/1);
  sputnik::Matrix rhs = ToMatrix(rhs_);
  CudaBlockSparseMatrix<half> rhs_gpu(rhs_);

  // Create the output matrix on gpu & gpu.
  CudaMatrix<half> out_gpu(this->kDimM, this->kDimN, &this->generator_);

  // Run the gpu kernel.
  BlockMatrix lhs_args = Arg(lhs_gpu);
  BlockMatrix rhs_args = Arg(rhs_gpu);
  AllocateBitmaskBuffers(lhs_args);
  AllocateBitmaskBuffers(rhs_args);
  if (this->kTransposeA) AllocateTransposeBuffers(lhs_args);
  if (!this->kTransposeB) AllocateTransposeBuffers(rhs_args);
  CUDA_CALL(Matmul(lhs_args, this->kTransposeA,
                   rhs_args, this->kTransposeB,
                   Arg(out_gpu), /*stream=*/0));
  CUDA_CALL(cudaStreamSynchronize(nullptr));
  if (this->kTransposeA) FreeTransposeBuffers(lhs_args);
  if (!this->kTransposeB) FreeTransposeBuffers(rhs_args);

  // DEBUG
  // std::cout << "Running SDS!" << std::endl;
  // CudaMatrix<half> tmp_rhs_gpu(rhs);
  // CudaBlockSparseMatrix<half> tmp_out_gpu(
  //     this->kDimM, this->kDimN, this->kDimM * this->kDimN,
  //     this->kBlockDim, RANDOM_UNIFORM, &this->generator_,
  //     /*pad_rows_to=*/1);
  // CUDA_CALL(Matmul(lhs_args, false,
  //                  Arg(tmp_rhs_gpu), true,
  //                  Arg(tmp_out_gpu), 0));
  // CUDA_CALL(cudaStreamSynchronize(nullptr));

  // double out = 0;
  // for (int i = 0; i < 128; ++i) {
  //   out += lhs.Values()[i] * rhs.Values()[i];
  // }
  // std::cout << "output = " << out << std::endl;

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
