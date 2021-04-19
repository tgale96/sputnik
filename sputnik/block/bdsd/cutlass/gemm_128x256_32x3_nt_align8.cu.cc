#include "sputnik/block/bdsd/cutlass/cuda_bdsd.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

namespace sputnik {
namespace block {
namespace cutlass {
    
namespace {

using gemm_f16_128x256_32x3_nt_align8_base = 
  typename ::cutlass::gemm::kernel::DefaultGemmUniversal<
  // transposed B operand.
  ::cutlass::half_t,
  ::cutlass::layout::ColumnMajor,
  ::cutlass::ComplexTransform::kNone,
  8,
  // transposed A operand.
  ::cutlass::half_t,
  ::cutlass::layout::RowMajor,
  ::cutlass::ComplexTransform::kNone,
  8,
  // C operand.
  float,
  ::cutlass::layout::RowMajor,
  float,
  ::cutlass::arch::OpClassTensorOp,
  ::cutlass::arch::Sm80,
  ::cutlass::gemm::GemmShape<128, 256, 32>,
  ::cutlass::gemm::GemmShape<64, 64, 32>,
  ::cutlass::gemm::GemmShape<16, 8, 16>,
  ::cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
  ::cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
  3,
  ::cutlass::arch::OpMultiplyAdd>::GemmKernel;

// Define named type
struct gemm_f16_128x256_32x3_nt_align8 : 
  public gemm_f16_128x256_32x3_nt_align8_base { };
}  // namespace

  
cudaError_t hgemm_nt(
  int M,
  int N,
  int K,
  half const *A,
  half const *B,
  half *C) {
  using Gemm = ::cutlass::gemm::device::GemmUniversalAdapter<
    gemm_f16_128x256_32x3_nt_align8>;

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

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

