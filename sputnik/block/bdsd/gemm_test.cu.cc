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
#include "sputnik/block/bdsd/cutlass/cuda_bdsd.h"
#include "sputnik/block/matrix_utils.h"

#include "absl/random/random.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace sputnik {
namespace block {

using ::testing::NanSensitiveFloatNear;
using ::testing::Pointwise;

template <int kDimM_, int kDimK_, int kDimN_>
struct Problem {
  static constexpr int kDimM = kDimM_;
  static constexpr int kDimK = kDimK_;
  static constexpr int kDimN = kDimN_;
};

template <typename Problem>
class GemmTest : public ::testing::Test {
 public:
  const int kDimM = Problem::kDimM;
  const int kDimK = Problem::kDimK;
  const int kDimN = Problem::kDimN;

  // Random number generator for creating matrices.
  absl::BitGen generator_;

  void Gemm(int m, int n, int k, const float *a, const float *b, float *c) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
	double acc = 0.0;
	for (int l = 0; l < k; ++l) {
	  acc += a[i * k + l] * b[l * n + j];
	}
	c[i * n + j] = (float)acc;
      }
    }
  }
};

typedef ::testing::Types<
  Problem<8, 8, 8>,
  Problem<16, 8, 8>,
  Problem<8, 16, 8>,
  Problem<32, 64, 128>,
  Problem<128, 512, 64>,
  Problem<512, 512, 1024>
  > TestProblems;

TYPED_TEST_SUITE(GemmTest, TestProblems);

TYPED_TEST(GemmTest, GemmTN) {
  // Create the sparse matrix on cpu & gpu.
  Matrix lhs(this->kDimK, this->kDimM, &this->generator_);
  CudaMatrix<half> lhs_gpu(lhs);
  
  // Create the dense matrix on cpu & gpu
  Matrix rhs(this->kDimK, this->kDimN, &this->generator_);
  CudaMatrix<half> rhs_gpu(rhs);

  // Create the output matrix on gpu & gpu.
  Matrix out(this->kDimM, this->kDimN, &this->generator_);
  CudaMatrix<half> out_gpu(out);
  
  // Run the gpu kernel.
  CUDA_CALL(cutlass::hgemm_tn(this->kDimM,
			      this->kDimN,
			      this->kDimK,
			      lhs_gpu.Values(),
			      rhs_gpu.Values(),
			      out_gpu.Values()));
  CUDA_CALL(cudaStreamSynchronize(nullptr));
  CUDA_CALL(cudaDeviceSynchronize());

  // Note the transposed LHS matrix.
  this->Gemm(this->kDimM,
	     this->kDimN,
	     this->kDimK,	     
	     lhs.T().Values(),
	     rhs.Values(),
	     out.Values());
  
  // Verify the results.
  Matrix results(out_gpu);  
  auto comparator = Pointwise(NanSensitiveFloatNear(5e-02), ToVector(out));
  ASSERT_THAT(ToVector(results), comparator);
}

TYPED_TEST(GemmTest, GemmNT) {
  // Create the sparse matrix on cpu & gpu.
  Matrix lhs(this->kDimM, this->kDimK, &this->generator_);
  CudaMatrix<half> lhs_gpu(lhs);
  
  // Create the dense matrix on cpu & gpu
  Matrix rhs(this->kDimN, this->kDimK, &this->generator_);
  CudaMatrix<half> rhs_gpu(rhs);

  // Create the output matrix on gpu & gpu.
  Matrix out(this->kDimM, this->kDimN, &this->generator_);
  CudaMatrix<half> out_gpu(out);
  
  // Run the gpu kernel.
  CUDA_CALL(cutlass::hgemm_nt(this->kDimM,
			      this->kDimN,
			      this->kDimK,
			      lhs_gpu.Values(),
			      rhs_gpu.Values(),
			      out_gpu.Values()));
  CUDA_CALL(cudaStreamSynchronize(nullptr));
  CUDA_CALL(cudaDeviceSynchronize());

  // Note the transposed LHS matrix.
  this->Gemm(this->kDimM,
	     this->kDimN,
	     this->kDimK,	     
	     lhs.Values(),
	     rhs.T().Values(),
	     out.Values());
  
  // Verify the results.
  Matrix results(out_gpu);  
  auto comparator = Pointwise(NanSensitiveFloatNear(5e-02), ToVector(out));
  ASSERT_THAT(ToVector(results), comparator);
}
  
}  // namespace block
}  // namespace sputnik
