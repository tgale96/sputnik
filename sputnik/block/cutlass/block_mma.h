#ifndef THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_MMA_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_MMA_H_

#include "sputnik/block/cutlass/block_pitch_linear.h"
#include "sputnik/block/cutlass/block_size.h"
#include "sputnik/block/cutlass/block_tile_access_iterator.h"
#include "sputnik/block/cutlass/dependent_tile_access_iterator.h"

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
  // Element type for internal accumulation
  typename ElementAccumulator,
  // Layout type for C and D matrix operands
  typename LayoutC,
  // Operator class tag
  typename OperatorClass,
  // Tag indicating architecture to tune for
  typename ArchTag,
  // Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape,
  // Warp-level tile size (concept: GemmShape)
  typename WarpShape,
  // Instruction-level tile size (concept: GemmShape)
  typename InstructionShape,
  // Number of stages used in the pipelined mainloop
  int Stages,
  // Operation perfomed by GEMM
  typename Operator,
  // Store the accumulators in row major or column major.  Row major is used
  // when output layout is interleaved.
  bool AccumulatorsInRowMajor = false>
struct BlockMma;

// Specialization for no block sparse operands. Hand off to standard mma.
template <
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
  // Element type for internal accumulation
  typename ElementAccumulator,
  // Layout type for C and D matrix operands
  typename LayoutC,
  // Operator class tag
  typename OperatorClass,
  // Tag indicating architecture to tune for
  typename ArchTag,
  // Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape,
  // Warp-level tile size (concept: GemmShape)
  typename WarpShape,
  // Instruction-level tile size (concept: GemmShape)
  typename InstructionShape,
  // Number of stages used in the pipelined mainloop
  int Stages,
  // Operation perfomed by GEMM
  typename Operator,
  // Store the accumulators in row major or column major.  Row major is used
  // when output layout is interleaved.
  bool AccumulatorsInRowMajor>
struct BlockMma<BlockSize::kNone, ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
		kAlignmentB, ElementAccumulator, LayoutC, OperatorClass,
		ArchTag, ThreadblockShape, WarpShape, InstructionShape,
		Stages, Operator, AccumulatorsInRowMajor> {
  using Mma = ::cutlass::gemm::threadblock::DefaultMma<
    ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
    kAlignmentB, ElementAccumulator, LayoutC, OperatorClass,
    ArchTag, ThreadblockShape, WarpShape, InstructionShape,
    Stages, Operator, AccumulatorsInRowMajor>;

  // Define the threadblock-level mma type.
  using ThreadblockMma = typename Mma::ThreadblockMma;
};

// Specialization for Dense = Sparse x Dense (DSD).
//
// Also specialized on row-major output and TensorOp.
template <
  // Block size for sparse operands.
  BlockSize kBlockSize,  
  // Element type for A matrix operand
  typename ElementA,
  // Access granularity of A matrix in units of elements
  int kAlignmentA,
  // Element type for B matrix operand
  typename ElementB,
  // Layout type for B matrix operand
  typename LayoutB,
  // Access granularity of B matrix in units of elements
  int kAlignmentB,
  // Element type for internal accumulation
  typename ElementAccumulator,
  // Tag indicating architecture to tune for
  typename ArchTag,
  // Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape,
  // Warp-level tile size (concept: GemmShape)
  typename WarpShape,
  // Instruction-level tile size (concept: GemmShape)
  typename InstructionShape,
  // Number of stages used in the pipelined mainloop
  int Stages,
  // Operation perfomed by GEMM
  typename Operator>
struct BlockMma<kBlockSize, ElementA, BlockRowMajor, kAlignmentA,
  ElementB, LayoutB, kAlignmentB, ElementAccumulator,
  ::cutlass::layout::RowMajor, ::cutlass::arch::OpClassTensorOp,
  ArchTag, ThreadblockShape, WarpShape, InstructionShape, Stages,
  Operator, false> {
  // TODO(tgale): Move these first three definitions into a DefaultMma.
  static ::cutlass::arch::CacheOperation::Kind const CacheOpA =
    ((::cutlass::sizeof_bits<ElementA>::value * kAlignmentA) == 128)
    ? ::cutlass::arch::CacheOperation::Global
    : ::cutlass::arch::CacheOperation::Always;

  static ::cutlass::arch::CacheOperation::Kind const CacheOpB =
    ((::cutlass::sizeof_bits<ElementB>::value * kAlignmentB) == 128)
    ? ::cutlass::arch::CacheOperation::Global
    : ::cutlass::arch::CacheOperation::Always;

  // Define the MmaCore components
  using MmaCore = typename ::cutlass::gemm::threadblock::DefaultMmaCore<
    ThreadblockShape, WarpShape, InstructionShape, ElementA,
    ::cutlass::layout::RowMajor, ElementB, LayoutB, ElementAccumulator,
    ::cutlass::layout::RowMajor, ::cutlass::arch::OpClassTensorOp,
    Stages, Operator, false, CacheOpA, CacheOpB>;

  // Define iterators over tiles from the A operand
  //
  // NOTE: We convert from BlockRowMajor to BlockPitchLinear here.
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = ::cutlass::Array<ElementA, kAlignmentA>;

  using ShapeA = BlockPitchLinearShape<
    ThreadblockShape::kK,
    ThreadblockShape::kM,
    Block2Int<kBlockSize>::value>;
  using IteratorA = BlockTileAccessIterator<
    ShapeA, ElementA, 0, ThreadMapA, AccessTypeA>;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = ::cutlass::Array<ElementB, kAlignmentB>;

  using ShapeB = BlockMatrixShape<
    ThreadblockShape::kK,
    ThreadblockShape::kN,
    Block2Int<kBlockSize>::value>;
  using IteratorB = DependentTileAccessIterator<
    ShapeB, ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;
  
  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma = ::cutlass::gemm::threadblock::MmaMultistage<
    typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
    MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
    MmaCore::kCacheOpB, ElementAccumulator, ::cutlass::layout::RowMajor,
      typename MmaCore::MmaPolicy, Stages>;
};

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_MMA_H_
  
