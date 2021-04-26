#include "sputnik/block/bdsd/cutlass/cuda_bdsd.h"
#include "sputnik/block/cutlass/block_pitch_linear.h"
#include "sputnik/block/cutlass/block_size.h"
#include "sputnik/block/cutlass/default_block_gemm.h"
#include "sputnik/block/cutlass/kernel.h"

namespace sputnik {
namespace block {
namespace cutlass {
    
namespace {

using dsd_mixed_128x256_32x3_tn_align8_base = 
  typename DefaultBlockGemm<
  BlockSize::k128,
  // Non-transposed A operand.
  ::cutlass::half_t,
  BlockRowMajor,
  8,
  // Transposed B operand.
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
struct dsd_mixed_128x256_32x3_tn_align8 : 
  public dsd_mixed_128x256_32x3_tn_align8_base { };

  
}  // namespace

cudaError_t dsd_nt(
  int M,
  int N,
  int K,
  half const *A,
  half const *B,
  half *C) {
  using Dsd = Kernel<dsd_mixed_128x256_32x3_tn_align8>;

  Dsd::Arguments args({M, N, K},
		      {1.0f, 0.0f},
		      A, B, C, C,
		      /*lda=*/K,
		      /*ldb=*/K,
		      /*ldc=*/N,
		      /*ldd=*/N);

  // Verify that we can implement the given problem.
  ::cutlass::Status status = Dsd::KernelFn::can_implement(args);
  if (status != ::cutlass::Status::kSuccess) {
    return cudaErrorNotSupported;
  }
  
  Dsd dsd_operator;  
  return dsd_operator(args);
}  

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik


