#include "sputnik/block/arguments.h"
#include "sputnik/block/dss/cutlass/dss.h"
#include "sputnik/block/cutlass/block_pitch_linear.h"
#include "sputnik/block/cutlass/default_block_gemm.h"
#include "sputnik/block/cutlass/kernel.h"
#include "sputnik/block/bitmask/bitmask.h"
#include "sputnik/block/transpose/transpose.h"

namespace sputnik {
namespace block {
namespace cutlass {

namespace {

using dss_mixed_b128_128x128x32x5_tn_align8_base =
  typename DefaultBlockGemm<
  BlockSize::k128,
  // Transposed A operand.
  ::cutlass::half_t,
  BlockColumnMajor,
  8,
  // Non-transposed B operand.
  ::cutlass::half_t,
  BlockRowMajor,
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
  ::cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
  5,
  ::cutlass::arch::OpMultiplyAdd
>::GemmKernel;

// Define named type
struct dss_mixed_b128_128x128x32x5_tn_align8 :
  public dss_mixed_b128_128x128x32x5_tn_align8_base { };

}  // namespace


bool can_launch_dss_mixed_b128_128x128x32x5_tn_align8(
    const BlockMatrix a, bool transpose_a,
    const BlockMatrix b, bool transpose_b, Matrix c) {
  using Dss = Kernel<dss_mixed_b128_128x128x32x5_tn_align8>;

  MatmulShape shape(a, transpose_a, b, transpose_b);
  Dss::Arguments args({shape.m, shape.n, shape.k},
                      {1.0f, 0.0f},
                      {nullptr, 0},
                      {nullptr, 0},
                      {nullptr, 0},
                      {nullptr, 0});

  // Verify that we can implement the given problem.
  ::cutlass::Status status = Dss::KernelFn::can_implement(args);
  bool can_implement = status == ::cutlass::Status::kSuccess;
  can_implement &= a.block_size == BlockSize::k128;
  can_implement &= b.block_size == BlockSize::k128;
  can_implement &= transpose_a && !transpose_b;
  can_implement &= shape.k <= 32 * 1024;
  can_implement &= ValidMatmul(a, transpose_a, b, transpose_b, c);
  return can_implement;
}

cudaError_t launch_dss_mixed_b128_128x128x32x5_tn_align8(
    const BlockMatrix a, bool transpose_a,
    const BlockMatrix b, bool transpose_b,
    Matrix c, cudaStream_t stream) {
  SPUTNIK_CHECK(a.bitmask);
  SPUTNIK_CHECK(a.offsets_t);
  SPUTNIK_CHECK(a.indices_t);
  SPUTNIK_CHECK(a.block_offsets);
  SPUTNIK_CHECK(b.bitmask);
  SPUTNIK_CHECK(b.offsets_t);
  SPUTNIK_CHECK(b.indices_t);
  SPUTNIK_CHECK(b.block_offsets);

  // Produce the transpose meta-data.
  if (a.create_metadata) {
    cudaError_t custatus = Transpose(a, stream);
    if (custatus != cudaSuccess) {
      return custatus;
    }
  }
  if (b.create_metadata) {
    cudaError_t custatus = Transpose(b, stream);
    if (custatus != cudaSuccess) {
      return custatus;
    }
  }

  // Produce the bitmasks for both matrices.
  //
  // TODO(tgale): Add the ability to cache this data
  // for cases where it can be reused across calls.
  cudaError_t custatus = Bitmask(a, stream);
  if (custatus != cudaSuccess) {
    return custatus;
  }
  custatus = Bitmask(b, stream);
  if (custatus != cudaSuccess) {
    return custatus;
  }

  using Dss = Kernel<dss_mixed_b128_128x128x32x5_tn_align8>;

  MatmulShape shape(a, transpose_a, b, transpose_b);
  Dss::Arguments args({shape.m, shape.n, shape.k},
                      {1.0f, 0.0f},
                      {a.data,
                       a.offsets_t,
                       a.indices_t,
                       a.block_offsets,
                       a.bitmask,
                       shape.lda},
                      {b.data,
                       b.offsets_t,
                       b.indices_t,
                       b.block_offsets,
                       b.bitmask,
                       shape.ldb},
                      {c.data, shape.ldc},
                      {c.data, shape.ldc});

  // Verify that we can implement the given problem.
  ::cutlass::Status status = Dss::KernelFn::can_implement(args);
  if (status != ::cutlass::Status::kSuccess) {
    return cudaErrorNotSupported;
  }

  Dss dss_operator;
  return dss_operator(args, stream);
}

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik
