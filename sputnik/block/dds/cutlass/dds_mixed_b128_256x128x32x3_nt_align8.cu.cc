#include "sputnik/block/arguments.h"
#include "sputnik/block/dds/cutlass/dds.h"
#include "sputnik/block/cutlass/block_pitch_linear.h"
#include "sputnik/block/cutlass/default_block_gemm.h"
#include "sputnik/block/cutlass/kernel.h"

namespace sputnik {
namespace block {
namespace cutlass {

namespace {

using dds_mixed_b128_128x128x32x5_nt_align8_base =
  typename DefaultBlockGemm<
  BlockSize::k128,
  // Non-transposed A operand.
  ::cutlass::half_t,
  ::cutlass::layout::RowMajor,
  8,
  // Transposed B operand.
  ::cutlass::half_t,
  BlockColumnMajor,
  8,
  // C operand.
  ::cutlass::half_t,
  ::cutlass::layout::RowMajor,
  float,
  ::cutlass::arch::OpClassTensorOp,
  ::cutlass::arch::Sm80,
  ::cutlass::gemm::GemmShape<128, 128, 32>,
  ::cutlass::gemm::GemmShape<64, 64, 32>,
  ::cutlass::gemm::GemmShape<16, 8, 16>,
  ::cutlass::epilogue::thread::LinearCombination<::cutlass::half_t, 8, float, float>,
  ::cutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle,
  5,
  ::cutlass::arch::OpMultiplyAdd
>::GemmKernel;

// Define named type
struct dds_mixed_b128_128x128x32x5_nt_align8 :
  public dds_mixed_b128_128x128x32x5_nt_align8_base { };

}  // namespace


bool can_launch_dds_mixed_b128_128x128x32x5_nt_align8(
    const Matrix a, bool transpose_a,
    const BlockMatrix b, bool transpose_b, Matrix c) {
  using Dds = Kernel<dds_mixed_b128_128x128x32x5_nt_align8>;

  MatmulShape shape(a, transpose_a, b, transpose_b);
  Dds::Arguments args({shape.m, shape.n, shape.k},
                      {1.0f, 0.0f},
                      {nullptr, 0},
                      {nullptr, 0},
                      {nullptr, 0},
                      {nullptr, 0});

  // Verify that we can implement the given problem.
  ::cutlass::Status status = Dds::KernelFn::can_implement(args);
  bool can_implement = status == ::cutlass::Status::kSuccess;
  can_implement &= b.block_size == BlockSize::k128;
  can_implement &= !transpose_a && transpose_b;
  can_implement &= ValidMatmul(a, transpose_a, b, transpose_b, c);
  return can_implement;
}

cudaError_t launch_dds_mixed_b128_128x128x32x5_nt_align8(
    const Matrix a, bool transpose_a,
    const BlockMatrix b, bool transpose_b,
    Matrix c, cudaStream_t stream) {
  using Dds = Kernel<dds_mixed_b128_128x128x32x5_nt_align8>;

  MatmulShape shape(a, transpose_a, b, transpose_b);
  Dds::Arguments args({shape.m, shape.n, shape.k},
                      {1.0f, 0.0f},
                      {a.data, shape.lda},
                      {b.data, b.offsets, b.indices, shape.ldb},
                      {c.data, shape.ldc},
                      {c.data, shape.ldc});

  // Verify that we can implement the given problem.
  ::cutlass::Status status = Dds::KernelFn::can_implement(args);
  if (status != ::cutlass::Status::kSuccess) {
    return cudaErrorNotSupported;
  }

  Dds dds_operator;
  return dds_operator(args, stream);
}

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik
