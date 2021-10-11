#include "sputnik/block/arguments.h"
#include "sputnik/block/sds/cutlass/sds.h"
#include "sputnik/block/cutlass/block_pitch_linear.h"
#include "sputnik/block/cutlass/default_block_gemm.h"
#include "sputnik/block/cutlass/kernel.h"
#include "sputnik/block/transpose/transpose.h"

namespace sputnik {
namespace block {
namespace cutlass {

namespace {

using sds_mixed_b128_128x128x32x5_tt_align8_base =
  typename DefaultBlockGemm<
  BlockSize::k128,
  // Transposed A operand.
  ::cutlass::half_t,
  ::cutlass::layout::ColumnMajor,
  8,
  // Transposed B operand.
  ::cutlass::half_t,
  BlockColumnMajor,
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
  ::cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
  5,
  ::cutlass::arch::OpMultiplyAdd
>::GemmKernel;

// Define named type
struct sds_mixed_b128_128x128x32x5_tt_align8 :
  public sds_mixed_b128_128x128x32x5_tt_align8_base { };

}  // namespace


bool can_launch_sds_mixed_b128_128x128x32x5_tt_align8(
    const Matrix a, bool transpose_a,
    const BlockMatrix b, bool transpose_b, BlockMatrix c) {
  using Sds = Kernel<sds_mixed_b128_128x128x32x5_tt_align8>;

  MatmulShape shape(a, transpose_a, b, transpose_b);
  Sds::Arguments args({shape.m, shape.n, shape.k},
                      {1.0f, 0.0f},
                      {nullptr, 0},
                      {nullptr, 0},
                      {nullptr, 0},
                      {nullptr, 0});

  // Verify that we can implement the given problem.
  ::cutlass::Status status = Sds::KernelFn::can_implement(args);
  bool can_implement = status == ::cutlass::Status::kSuccess;
  can_implement &= c.block_size == BlockSize::k128;
  can_implement &= b.block_size == BlockSize::k128;
  can_implement &= transpose_a && transpose_b;
  can_implement &= ValidMatmul(a, transpose_a, b, transpose_b, c);
  return can_implement;
}

cudaError_t launch_sds_mixed_b128_128x128x32x5_tt_align8(
    const Matrix a, bool transpose_a,
    const BlockMatrix b, bool transpose_b,
    BlockMatrix c, cudaStream_t stream) {
  using Sds = Kernel<sds_mixed_b128_128x128x32x5_tt_align8>;

  MatmulShape shape(a, transpose_a, b, transpose_b);
  Sds::Arguments args({shape.m, shape.n, shape.k},
                      {1.0f, 0.0f},
                      {a.data, shape.lda},
                      {b.data, b.offsets, b.indices, shape.ldb},
                      {c.data, c.offsets, c.indices, shape.ldc},
                      {c.data, c.offsets, c.indices, shape.ldc});

  // Verify that we can implement the given problem.
  ::cutlass::Status status = Sds::KernelFn::can_implement(args);
  if (status != ::cutlass::Status::kSuccess) {
    return cudaErrorNotSupported;
  }

  Sds sds_operator;
  return sds_operator(args, stream);
}

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik