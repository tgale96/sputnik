#ifndef THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_DEFAULT_BLOCK_GEMM_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_DEFAULT_BLOCK_GEMM_H_

#include "sputnik/block/arguments.h"
#include "sputnik/block/cutlass/block_gemm.h"
#include "sputnik/block/cutlass/block_mma.h"
#include "sputnik/block/cutlass/default_block_epilogue.h"

#include "cutlass/gemm/kernel/default_gemm.h"

namespace sputnik {
namespace block {
namespace cutlass {

template <
  // Block size for sparse operands.
  BlockSize kBlockSize,
  // Element type for A matrix operand
  typename ElementA,
  // Layout type for A matrix operand
  typename LayoutA,
  // Access granularity of A matrix in units of elements
  int kAlignmentA,
  // Element type for B matrix operand
  typename ElementB,
  // Layout type for B matrix operand
  typename LayoutB,
  // Access granularity of B matrix in units of elements
  int kAlignmentB,
  // Element type for C and D matrix operands
  typename ElementC,
  // Layout type for C and D matrix operands
  typename LayoutC,
  // Element type for internal accumulation
  typename ElementAccumulator,
  // Operator class tag
  typename OperatorClass,
  // Tag indicating architecture to tune for
  typename ArchTag,
  // Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape,
  // Warp-level tile size (concept: GemmShape)
  typename WarpShape,
  // Warp-level tile size (concept: GemmShape)
  typename InstructionShape,
  // Epilogue output operator
  typename EpilogueOutputOp,
  // Threadblock-level swizzling operator
  typename ThreadblockSwizzle,
  // Number of stages used in the pipelined mainloop
  int Stages,
  // Operation performed by GEMM
  typename Operator>
struct DefaultBlockGemm {
  // TODO(tgale): These constraints are added because of the simple
  // Mma/Epilogue definitions below. Fix this to generalize
  // beyond Ampere and FP16.
  static_assert(::cutlass::platform::is_same<
		ArchTag, ::cutlass::arch::Sm80>::value);
  static_assert(::cutlass::platform::is_same<
		OperatorClass, ::cutlass::arch::OpClassTensorOp>::value);
  static_assert(::cutlass::platform::is_same<
		LayoutC, ::cutlass::layout::RowMajor>::value ||
                ::cutlass::platform::is_same<
                LayoutC, BlockRowMajor>::value);

  using Mma = typename BlockMma<
    kBlockSize, ElementA, LayoutA, kAlignmentA, ElementB,
    LayoutB, kAlignmentB, ElementAccumulator, ::cutlass::layout::RowMajor,
    ::cutlass::arch::OpClassTensorOp, ::cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape, Stages,
    Operator>::ThreadblockMma;

  static_assert(WarpShape::kK == ThreadblockShape::kK, "Split-k not supported.");

  using Epilogue = typename DefaultBlockEpilogue<
      kBlockSize, LayoutC, ThreadblockShape, typename Mma::Operator,
      /*kPartitionsK=*/1, EpilogueOutputOp, EpilogueOutputOp::kCount>::Epilogue;

  using GemmKernel = BlockGemm<Mma, Epilogue, ThreadblockSwizzle>;
};

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_DEFAULT_BLOCK_GEMM_H_
