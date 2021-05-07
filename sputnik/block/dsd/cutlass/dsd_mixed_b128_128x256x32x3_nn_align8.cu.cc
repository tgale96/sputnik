#include "sputnik/block/dsd/cutlass/dsd.h"
#include "sputnik/block/cutlass/block_pitch_linear.h"
#include "sputnik/block/cutlass/default_block_gemm.h"
#include "sputnik/block/cutlass/kernel.h"

namespace sputnik {
namespace block {
namespace cutlass {

namespace {

using dsd_mixed_b128_128x256x32x3_nn_align8_base =
  typename DefaultBlockGemm<
  BlockSize::k128,
  // Non-transposed A operand.
  ::cutlass::half_t,
  BlockRowMajor,
  8,
  // Non-transposed B operand.
  ::cutlass::half_t,
  ::cutlass::layout::RowMajor,
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
struct dsd_mixed_b128_128x256x32x3_nn_align8 :
  public dsd_mixed_b128_128x256x32x3_nn_align8_base { };

}  // namespace


bool can_launch_dsd_mixed_b128_128x256x32x3_nn_align8(
    int m, int k, int n, int nonzeros, int block_dim,
    bool transpose_a, bool transpose_b) {
  using Dsd = Kernel<dsd_mixed_b128_128x256x32x3_nn_align8>;

  Dsd::Arguments args({m, n, k},
                      {1.0f, 0.0f},
                      {nullptr, 0},
                      {nullptr, 0},
                      {nullptr, 0},
                      {nullptr, 0});

  // Verify that we can implement the given problem.
  ::cutlass::Status status = Dsd::KernelFn::can_implement(args);
  bool can_implement = block_dim == 128 && !transpose_a && !transpose_b;
  return status == ::cutlass::Status::kSuccess && can_implement;
}

cudaError_t launch_dsd_mixed_b128_128x256x32x3_nn_align8(
    int m, int k, int n,
    int nonzeros, int block_dim,
    const half* a,
    const int* offsets_a,
    const short* indices_a,
    const half* b, bool transpose_b,
    half* c, cudaStream_t stream) {
  using Dsd = Kernel<dsd_mixed_b128_128x256x32x3_nn_align8>;

  Dsd::Arguments args({m, n, k},
                      {1.0f, 0.0f},
                      {a, offsets_a, indices_a, /*lda=*/k},
                      {b, /*ldb=*/n},
                      {c, /*ldc=*/n},
                      {c, /*ldc=*/n});

  // Verify that we can implement the given problem.
  ::cutlass::Status status = Dsd::KernelFn::can_implement(args);
  if (status != ::cutlass::Status::kSuccess) {
    return cudaErrorNotSupported;
  }

  Dsd dsd_operator;
  return dsd_operator(args, stream);
}

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik
