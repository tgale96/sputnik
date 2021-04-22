#include "sputnik/block/bdsd/cutlass/cuda_bdsd.h"
#include "sputnik/block/cutlass/default_block_gemm.h"
#include "sputnik/block/cutlass/kernel.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

namespace sputnik {
namespace block {
namespace cutlass {
    
namespace {

using gemm_mixed_256x128_32x3_nt_align8_base = 
  typename ::cutlass::gemm::kernel::DefaultGemmUniversal<
  // Non-transposed B operand.
  ::cutlass::half_t,
  ::cutlass::layout::ColumnMajor,
  ::cutlass::ComplexTransform::kNone,
  8,
  // Transposed A operand.
  ::cutlass::half_t,
  ::cutlass::layout::RowMajor,
  ::cutlass::ComplexTransform::kNone,
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
  using Gemm = ::cutlass::gemm::device::GemmUniversalAdapter<
    gemm_mixed_256x128_32x3_nt_align8>;

  Gemm::Arguments args(::cutlass::gemm::GemmUniversalMode::kGemm,
		       {M, N, K},
		       /*batch_count=*/1,
		       {(::cutlass::half_t)1.0f, (::cutlass::half_t)0.0f},
		       A, B, C, C,
		       0, 0, 0, 0,
		       /*lda=*/M,
		       /*ldb=*/N,
		       /*ldc=*/N,
		       /*ldd=*/N);
  args = args.transposed_problem();
  
  //
  /// Launch the kernel.
  //
  
  Gemm gemm_operator;

  ::cutlass::Status status = gemm_operator.can_implement(args);
  if (status != ::cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  
  status = gemm_operator(args);

  // TODO(tgale): Can we return more informative errors here?
  if (status != ::cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

cudaError_t hgemm_nt(
  int M,
  int N,
  int K,
  half const *A,
  half const *B,
  half *C) {
  using Gemm = Kernel<gemm_mixed_128x256_32x3_tn_align8>;

  Gemm::Arguments args(::cutlass::gemm::GemmUniversalMode::kGemm,
		       {M, N, K},
		       /*batch_count=*/1,
		       {(::cutlass::half_t)1.0f, (::cutlass::half_t)0.0f},
		       A, B, C, C,
		       0, 0, 0, 0,
		       /*lda=*/K,
		       /*ldb=*/K,
		       /*ldc=*/N,
		       /*ldd=*/N);
  // args = args.transposed_problem();

  // TODO(tgale): Verify that we can implement the given problem
  // with this kernel before launching.
  Gemm gemm_operator;  
  return gemm_operator(args);
}  

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

