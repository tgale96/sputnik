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
class BdsdTest : public ::testing::Test {
 public:
  const int kDimM = Problem::kDimM;
  const int kDimK = Problem::kDimK;
  const int kDimN = Problem::kDimN;
  const int kNonZeros = Problem::kNonZeros;
  const int kBlockDim = Problem::kBlockDim;

  // Random number generator for creating matrices.
  absl::BitGen generator_;

  /**
   * @brief Basic matrix-multiplication routine for testing.
   */
  void Bdsd(int m, int k, int n, int block_dim, const float *a_values,
            const int *a_row_offsets, const int *a_column_indices,
            const float *b, float *c) {
    for (int i = 0; i < m / block_dim; ++i) {
      for (int j = 0; j < n; ++j) {
        double accum[Problem::kBlockDim] = {};
        for (int l = a_row_offsets[i];
             l < a_row_offsets[i + 1];
             l += block_dim * block_dim) {
          int idx_offset = l / (block_dim * block_dim);
          int a_col = a_column_indices[idx_offset];

          for (int block_y = 0; block_y < block_dim; ++block_y) {
            for (int block_x = 0; block_x < block_dim; ++block_x) {
              float a_val = a_values[l + block_y * block_dim + block_x];
              float b_val = b[(a_col + block_x) * n + j];

              accum[block_y] += static_cast<double>(a_val) *
                                static_cast<double>(b_val);
            }
          }
        }

        // Write the results.
        for (int block_y = 0; block_y < block_dim; ++block_y) {
          c[(i * block_dim + block_y) * n + j] = accum[block_y];
        }
      }
    }
  }
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

TYPED_TEST_SUITE(BdsdTest, TestProblems);

TYPED_TEST(BdsdTest, Bdsd) {
  // Create the sparse matrix on cpu & gpu.
  BlockSparseMatrix sparse_matrix(
      this->kDimM, this->kDimK,
      this->kNonZeros, this->kBlockDim,
      RANDOM_UNIFORM,
      &this->generator_,
      /*pad_rows_to=*/1);
  CudaBlockSparseMatrix<half> sparse_matrix_gpu(sparse_matrix);
  
  // Create the dense matrix on cpu & gpu
  Matrix matrix(this->kDimK, this->kDimN, &this->generator_);
  CudaMatrix<half> matrix_gpu(matrix);

  // Create the output matrix on gpu & gpu.
  Matrix output_matrix(this->kDimM, this->kDimN, &this->generator_);
  CudaMatrix<half> output_matrix_gpu(output_matrix);

  // std::cout << "lhs = " << std::endl;
  // for (int i = 0; i < 32; ++i) {
  //   std::cout << sparse_matrix.Values()[i] << std::endl;
  // }
  // std::cout << "rhs = " << std::endl;
  // for (int i = 0; i < 32; ++i) {
  //   std::cout << matrix.Values()[i * 64] << std::endl;
  // }
  
  // Run the gpu kernel.
  CUDA_CALL(CudaBdsd(this->kDimM, this->kDimK, this->kDimN,
		     sparse_matrix_gpu.NumElementsWithPadding(),
		     this->kBlockDim, sparse_matrix_gpu.Values(),
		     sparse_matrix_gpu.RowOffsets(),
		     sparse_matrix_gpu.ColumnIndices(),
		     matrix_gpu.Values(),
		     output_matrix_gpu.Values(), 0));
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  this->Bdsd(this->kDimM, this->kDimK, this->kDimN, this->kBlockDim,
             sparse_matrix.Values(), sparse_matrix.RowOffsets(),
             sparse_matrix.ColumnIndices(), matrix.Values(),
             output_matrix.Values());

  // std::cout << "Expected: " << std::endl;
  // for (int i = 0; i < this->kDimM; ++i) {
  //   for (int j = 0; j < this->kDimN; ++j) {
  //     std::cout << output_matrix.Values()[i * this->kDimN + j] << ", ";
  //   }
  //   std::cout << std::endl;
  // }

  // Verify the results.
  Matrix results(output_matrix_gpu);
  
  // std::cout << "Got: " << std::endl;
  // for (int i = 0; i < this->kDimM; ++i) {
  //   for (int j = 0; j < this->kDimN; ++j) {
  //     std::cout << results.Values()[i * this->kDimN + j] << ", ";
  //   }
  //   std::cout << std::endl;
  // }  
  
  auto comparator = Pointwise(
      NanSensitiveFloatNear(5e-02),
      ToVector(output_matrix));
  ASSERT_THAT(ToVector(results), comparator);
}

}  // namespace block
}  // namespace sputnik
