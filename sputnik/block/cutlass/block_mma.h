#include "sputnik/block/cutlass/block_size.h"

template <
  // Block size for sparse operands.
  int kBlockSize,
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
  bool AccumulatorsInRowMajor = false>
struct BlockMma<BlockSize::kNone, ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
		kAlignmentB, ElementAccumulator, LayoutC, OperatorClass,
		ArchTag, ThreadblockShape, WarpShape, InstructionShape,
		Stages, Operator, AccumumulatorsInRowMajor> {
  using Mma = ::cutlass::gemm::threadblock::DefaultMma<
    ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
    kAlignmentB, ElementAccumulator, LayoutC, OperatorClass,
    ArchTag, ThreadblockShape, WarpShape, InstructionShape,
    Stages, Operator, AccumumulatorsInRowMajor>;

  // Define the threadblock-level mma type.
  using ThreadblockMma = Mma::ThreadblockMma;
}
