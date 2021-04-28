#include "sputnik/block/bdsd/cutlass/cuda_bdsd.h"
#include "sputnik/block/cutlass/default_block_gemm.h"
#include "sputnik/block/cutlass/kernel.h"

namespace sputnik {
namespace block {
namespace cutlass {
    
namespace {

using gemm_mixed_256x128_32x3_nt_align8_base = 
  typename DefaultBlockGemm<
  BlockSize::kNone,
  // Non-transposed B operand.
  ::cutlass::half_t,
  ::cutlass::layout::ColumnMajor,
  8,
  // Transposed A operand.
  ::cutlass::half_t,
  ::cutlass::layout::RowMajor,
  8,
  // C operand.
  ::cutlass::half_t,
  ::cutlass::layout::RowMajor,
  float,
  ::cutlass::arch::OpClassTensorOp,
  ::cutlass::arch::Sm80,
  ::cutlass::gemm::GemmShape<256, 128, 32>,
  ::cutlass::gemm::GemmShape<64, 64, 32>,
  ::cutlass::gemm::GemmShape<16, 8, 16>,
  ::cutlass::epilogue::thread::LinearCombination<::cutlass::half_t, 8, float, float>,
  ::cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
  3,
  ::cutlass::arch::OpMultiplyAdd>::GemmKernel;

// Define named type
struct gemm_mixed_256x128_32x3_nt_align8 : 
  public gemm_mixed_256x128_32x3_nt_align8_base { };

using gemm_mixed_128x256_32x3_tn_align8_base = 
  typename DefaultBlockGemm<
  BlockSize::kNone,  
  // Transposed B operand.
  ::cutlass::half_t,
  ::cutlass::layout::RowMajor,
  8,
  // Non-transposed A operand.
  ::cutlass::half_t,
  ::cutlass::layout::ColumnMajor,
  8,
  // C operand.
  ::cutlass::half_t,
  ::cutlass::layout::RowMajor,
  float,
  ::cutlass::arch::OpClassTensorOp,
  ::cutlass::arch::Sm80,
  ::cutlass::gemm::GemmShape<128, 256, 32>,
  ::cutlass::gemm::GemmShape<64, 64, 32>,
  ::cutlass::gemm::GemmShape<16, 8, 16>,
  ::cutlass::epilogue::thread::LinearCombination<::cutlass::half_t, 8, float, float>,
  ::cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
  3,
  ::cutlass::arch::OpMultiplyAdd
>::GemmKernel;

// Define named type
struct gemm_mixed_128x256_32x3_tn_align8 : 
  public gemm_mixed_128x256_32x3_tn_align8_base { };
  
}  // namespace

cudaError_t hgemm_tn(
  int M,
  int N,
  int K,
  half const *A,
  half const *B,
  half *C) {
  using Gemm = Kernel<gemm_mixed_256x128_32x3_nt_align8>;

  Gemm::Arguments args({M, N, K},
		       {1.0f, 0.0f},
		       {A, /*lda=*/M},
		       {B, /*ldb=*/N},
		       {C, /*ldC=*/N},
		       {C, /*ldc=*/N});
  
  // Verify that we can implement the given problem.
  ::cutlass::Status status = Gemm::KernelFn::can_implement(args);
  if (status != ::cutlass::Status::kSuccess) {
    return cudaErrorNotSupported;
  }
  
  Gemm gemm_operator;
  return gemm_operator(args);
}

cudaError_t hgemm_nt(
  int M,
  int N,
  int K,
  half const *A,
  half const *B,
  half *C) {
  using Gemm = Kernel<gemm_mixed_128x256_32x3_tn_align8>;

  Gemm::Arguments args({M, N, K},
		       {1.0f, 0.0f},
		       {A, /*lda=*/K},
		       {B, /*ldb=*/K},
		       {C, /*ldc=*/N},
		       {C, /*ldc=*/N});

  // Verify that we can implement the given problem.
  ::cutlass::Status status = Gemm::KernelFn::can_implement(args);
  if (status != ::cutlass::Status::kSuccess) {
    return cudaErrorNotSupported;
  }
  
  Gemm gemm_operator;  
  return gemm_operator(args);
}  

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

