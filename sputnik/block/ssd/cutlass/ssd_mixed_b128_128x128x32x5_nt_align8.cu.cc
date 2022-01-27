#include "sputnik/block/arguments.h"
#include "sputnik/block/ssd/cutlass/ssd.h"
#include "sputnik/block/cutlass/block_pitch_linear.h"
#include "sputnik/block/cutlass/default_block_gemm.h"
#include "sputnik/block/cutlass/kernel.h"
#include "sputnik/block/cutlass/threadblock_swizzle.h"
#include "sputnik/block/transpose/transpose.h"

namespace sputnik {
namespace block {
namespace cutlass {

namespace {

using ssd_mixed_b128_128x128x32x5_nt_align8_base =
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
  BlockRowMajor,
  float,
  ::cutlass::arch::OpClassTensorOp,
  ::cutlass::arch::Sm80,
  ::cutlass::gemm::GemmShape<128, 128, 32>,
  ::cutlass::gemm::GemmShape<64, 64, 32>,
  ::cutlass::gemm::GemmShape<16, 8, 16>,
  ::cutlass::epilogue::thread::LinearCombination<::cutlass::half_t, 8, float, float>,
  SparseOutputThreadblockSwizzle,
  5,
  ::cutlass::arch::OpMultiplyAdd
>::GemmKernel;

// Define named type
struct ssd_mixed_b128_128x128x32x5_nt_align8 :
  public ssd_mixed_b128_128x128x32x5_nt_align8_base { };

}  // namespace


bool can_launch_ssd_mixed_b128_128x128x32x5_nt_align8(
    const BlockMatrix a, bool transpose_a,
    const Matrix b, bool transpose_b, BlockMatrix c) {
  using Ssd = Kernel<ssd_mixed_b128_128x128x32x5_nt_align8>;

  MatmulShape shape(a, transpose_a, b, transpose_b);
  Ssd::Arguments args({shape.m, shape.n, shape.k},
                      {1.0f, 0.0f},
                      {nullptr, 0},
                      {nullptr, 0},
                      {nullptr, 0},
                      {nullptr, 0});

  // Verify that we can implement the given problem.
  ::cutlass::Status status = Ssd::KernelFn::can_implement(args);
  bool can_implement = status == ::cutlass::Status::kSuccess;
  can_implement &= c.block_size == BlockSize::k128;
  can_implement &= a.block_size == BlockSize::k128;
  can_implement &= !transpose_a && transpose_b;
  can_implement &= ValidMatmul(a, transpose_a, b, transpose_b, c);
  return can_implement;
}

cudaError_t launch_ssd_mixed_b128_128x128x32x5_nt_align8(
    const BlockMatrix a, bool transpose_a,
    const Matrix b, bool transpose_b,
    BlockMatrix c, cudaStream_t stream) {
  using Ssd = SparseOutputKernel<ssd_mixed_b128_128x128x32x5_nt_align8>;
  CHECK(c.row_indices);

  MatmulShape shape(a, transpose_a, b, transpose_b);
  Ssd::Arguments args({shape.m, shape.n, shape.k},
                      {1.0f, 0.0f},
                      {a.data, a.offsets, a.indices, shape.lda},
                      {b.data, shape.ldb},
                      {c.data,
		       c.offsets,
		       c.indices,
		       c.row_indices,
		       shape.ldc,
		       c.nonzeros},
                      {c.data,
		       c.offsets,
		       c.indices,
		       c.row_indices,
		       shape.ldc,
		       c.nonzeros});

  // Verify that we can implement the given problem.
  ::cutlass::Status status = Ssd::KernelFn::can_implement(args);
  if (status != ::cutlass::Status::kSuccess) {
    return cudaErrorNotSupported;
  }

  Ssd ssd_operator;
  return ssd_operator(args, stream);
}

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik
