#ifndef THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_MMA_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_MMA_H_

#include "sputnik/block/arguments.h"
#include "sputnik/block/cutlass/block_pitch_linear.h"
#include "sputnik/block/cutlass/block_tile_access_iterator.h"
#include "sputnik/block/cutlass/block_tile_union_iterator.h"
#include "sputnik/block/cutlass/dependent_tile_access_iterator.h"

#include "cutlass/gemm/threadblock/default_mma.h"

namespace sputnik {
namespace block {
namespace cutlass {

// Default specialization for no block sparse inputs. Hand off to standard mma.
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
struct BlockMma {
  using Mma = ::cutlass::gemm::threadblock::DefaultMma<
    ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
    kAlignmentB, ElementAccumulator, LayoutC, OperatorClass,
    ArchTag, ThreadblockShape, WarpShape, InstructionShape,
    Stages, Operator, AccumulatorsInRowMajor>;

  // Define the threadblock-level mma type.
  using ThreadblockMma = typename Mma::ThreadblockMma;
};

// Specialization for Dense = N(Sparse) x Dense (DSD).
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
    AsInt<kBlockSize>::value>;
  using IteratorA = BlockTileAccessIterator<
    ShapeA, ElementA, 0, ThreadMapA, AccessTypeA>;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = ::cutlass::Array<ElementB, kAlignmentB>;

  using ShapeB = BlockMatrixShape<
    ThreadblockShape::kK,
    ThreadblockShape::kN,
    AsInt<kBlockSize>::value>;
  using IteratorB = DependentTileAccessIterator<
    ShapeB, ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;

  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma = ::cutlass::gemm::threadblock::MmaMultistage<
    typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
    MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
    MmaCore::kCacheOpB, ElementAccumulator, ::cutlass::layout::RowMajor,
      typename MmaCore::MmaPolicy, Stages>;
};

// Specialization for Dense = T(Sparse) x Dense (DSD).
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
struct BlockMma<kBlockSize, ElementA, BlockColumnMajor, kAlignmentA,
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
    ::cutlass::layout::ColumnMajor, ElementB, LayoutB, ElementAccumulator,
    ::cutlass::layout::RowMajor, ::cutlass::arch::OpClassTensorOp,
    Stages, Operator, false, CacheOpA, CacheOpB>;

  // Define iterators over tiles from the A operand
  //
  // NOTE: We convert from BlockRowMajor to BlockPitchLinear here.
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = ::cutlass::Array<ElementA, kAlignmentA>;

  // TODO(tgale): Move the conversion from row/col major to pitch
  // linear in simple switch structs in the iteration rather than
  // here. Then we can combine this specialization with the above.
  // We also need to translate the above layouts for block->non-block
  // in MmaCore.
  using ShapeA = BlockPitchLinearShape<
    ThreadblockShape::kM,
    ThreadblockShape::kK,
    AsInt<kBlockSize>::value>;
  using IteratorA = BlockTileAccessIterator<
    ShapeA, ElementA, 1, ThreadMapA, AccessTypeA>;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = ::cutlass::Array<ElementB, kAlignmentB>;

  using ShapeB = BlockMatrixShape<
    ThreadblockShape::kK,
    ThreadblockShape::kN,
    AsInt<kBlockSize>::value>;
  using IteratorB = DependentTileAccessIterator<
    ShapeB, ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;

  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma = ::cutlass::gemm::threadblock::MmaMultistage<
    typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
    MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
    MmaCore::kCacheOpB, ElementAccumulator, ::cutlass::layout::RowMajor,
      typename MmaCore::MmaPolicy, Stages>;
};

// Specialization for Dense = Dense x N(Sparse) (DDS).
//
// Also specialized on row-major output and TensorOp.
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
struct BlockMma<kBlockSize, ElementA, LayoutA, kAlignmentA,
  ElementB, BlockRowMajor, kAlignmentB, ElementAccumulator,
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
    LayoutA, ElementB, ::cutlass::layout::RowMajor, ElementAccumulator,
    ::cutlass::layout::RowMajor, ::cutlass::arch::OpClassTensorOp,
    Stages, Operator, false, CacheOpA, CacheOpB>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = ::cutlass::Array<ElementA, kAlignmentA>;

  using ShapeA = BlockMatrixShape<
    ThreadblockShape::kM,
    ThreadblockShape::kK,
    AsInt<kBlockSize>::value>;
  using IteratorA = DependentTileAccessIterator<
    ShapeA, ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

  // Define iterators over tiles from the B operand
  //
  // NOTE: We convert from BlockRowMajor to BlockPitchLinear here.
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = ::cutlass::Array<ElementB, kAlignmentB>;

  using ShapeB = BlockPitchLinearShape<
    ThreadblockShape::kN,
    ThreadblockShape::kK,
    AsInt<kBlockSize>::value>;
  using IteratorB = BlockTileAccessIterator<
    ShapeB, ElementB, 1, ThreadMapB, AccessTypeB>;

  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma = ::cutlass::gemm::threadblock::MmaMultistage<
    typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
    MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
    MmaCore::kCacheOpB, ElementAccumulator, ::cutlass::layout::RowMajor,
      typename MmaCore::MmaPolicy, Stages>;
};

// Specialization for Dense = Dense x N(Sparse) (DDS).
//
// Also specialized on row-major output and TensorOp.
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
struct BlockMma<kBlockSize, ElementA, LayoutA, kAlignmentA,
  ElementB, BlockColumnMajor, kAlignmentB, ElementAccumulator,
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
    LayoutA, ElementB, ::cutlass::layout::ColumnMajor, ElementAccumulator,
    ::cutlass::layout::RowMajor, ::cutlass::arch::OpClassTensorOp,
    Stages, Operator, false, CacheOpA, CacheOpB>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = ::cutlass::Array<ElementA, kAlignmentA>;

  using ShapeA = BlockMatrixShape<
    ThreadblockShape::kM,
    ThreadblockShape::kK,
    AsInt<kBlockSize>::value>;
  using IteratorA = DependentTileAccessIterator<
    ShapeA, ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

  // Define iterators over tiles from the B operand
  //
  // NOTE: We convert from BlockColumnMajor to BlockPitchLinear here.
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = ::cutlass::Array<ElementB, kAlignmentB>;

  using ShapeB = BlockPitchLinearShape<
    ThreadblockShape::kK,
    ThreadblockShape::kN,
    AsInt<kBlockSize>::value>;
  using IteratorB = BlockTileAccessIterator<
    ShapeB, ElementB, 0, ThreadMapB, AccessTypeB>;

  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma = ::cutlass::gemm::threadblock::MmaMultistage<
    typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
    MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
    MmaCore::kCacheOpB, ElementAccumulator, ::cutlass::layout::RowMajor,
      typename MmaCore::MmaPolicy, Stages>;
};

// Specialization for Dense = N(Sparse) x T(Sparse) (DSS).
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
  ElementB, BlockColumnMajor, kAlignmentB, ElementAccumulator,
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
      ThreadblockShape, WarpShape, InstructionShape,
      ElementA, ::cutlass::layout::RowMajor,
      ElementB, ::cutlass::layout::ColumnMajor,
      ElementAccumulator, ::cutlass::layout::RowMajor,
      ::cutlass::arch::OpClassTensorOp, Stages,
      Operator, false, CacheOpA, CacheOpB>;

  // Define iterators over tiles from the A operand
  //
  // NOTE: We convert from BlockRowMajor to BlockPitchLinear here.
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = ::cutlass::Array<ElementA, kAlignmentA>;

  using ShapeA = BlockPitchLinearShape<
    ThreadblockShape::kK,
    ThreadblockShape::kM,
    AsInt<kBlockSize>::value>;
  using IteratorA = BlockTileUnionIterator<
    ShapeA, ElementA, 0, ThreadMapA, AccessTypeA>;

  // Define iterators over tiles from the B operand
  //
  // NOTE: We convert from BlockColumnMajor to BlockPitchLinear here.
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = ::cutlass::Array<ElementB, kAlignmentB>;

  using ShapeB = BlockPitchLinearShape<
    ThreadblockShape::kK,
    ThreadblockShape::kN,
    AsInt<kBlockSize>::value>;
  using IteratorB = BlockTileUnionIterator<
    ShapeB, ElementB, 0, ThreadMapB, AccessTypeB>;

  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma = ::cutlass::gemm::threadblock::MmaMultistage<
    typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
    MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
    MmaCore::kCacheOpB, ElementAccumulator, ::cutlass::layout::RowMajor,
      typename MmaCore::MmaPolicy, Stages>;
};

// Specialization for Dense = N(Sparse) x N(Sparse) (DSS).
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
  ElementB, BlockRowMajor, kAlignmentB, ElementAccumulator,
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
      ThreadblockShape, WarpShape, InstructionShape,
      ElementA, ::cutlass::layout::RowMajor,
      ElementB, ::cutlass::layout::RowMajor,
      ElementAccumulator, ::cutlass::layout::RowMajor,
      ::cutlass::arch::OpClassTensorOp, Stages,
      Operator, false, CacheOpA, CacheOpB>;

  // Define iterators over tiles from the A operand
  //
  // NOTE: We convert from BlockRowMajor to BlockPitchLinear here.
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = ::cutlass::Array<ElementA, kAlignmentA>;

  using ShapeA = BlockPitchLinearShape<
    ThreadblockShape::kK,
    ThreadblockShape::kM,
    AsInt<kBlockSize>::value>;
  using IteratorA = BlockTileUnionIterator<
    ShapeA, ElementA, 0, ThreadMapA, AccessTypeA>;

  // Define iterators over tiles from the B operand
  //
  // NOTE: We convert from BlockRowMajor to BlockPitchLinear here.
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = ::cutlass::Array<ElementB, kAlignmentB>;

  using ShapeB = BlockPitchLinearShape<
    ThreadblockShape::kN,
    ThreadblockShape::kK,
    AsInt<kBlockSize>::value>;
  using IteratorB = BlockTileUnionIterator<
    ShapeB, ElementB, 1, ThreadMapB, AccessTypeB>;

  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma = ::cutlass::gemm::threadblock::MmaMultistage<
    typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
    MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
    MmaCore::kCacheOpB, ElementAccumulator, ::cutlass::layout::RowMajor,
      typename MmaCore::MmaPolicy, Stages>;
};

// Specialization for Dense = T(Sparse) x N(Sparse) (DSS).
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
struct BlockMma<kBlockSize, ElementA, BlockColumnMajor, kAlignmentA,
  ElementB, BlockRowMajor, kAlignmentB, ElementAccumulator,
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
      ThreadblockShape, WarpShape, InstructionShape,
      ElementA, ::cutlass::layout::ColumnMajor,
      ElementB, ::cutlass::layout::RowMajor,
      ElementAccumulator, ::cutlass::layout::RowMajor,
      ::cutlass::arch::OpClassTensorOp, Stages,
      Operator, false, CacheOpA, CacheOpB>;

  // Define iterators over tiles from the A operand
  //
  // NOTE: We convert from BlockColumnMajor to BlockPitchLinear here.
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = ::cutlass::Array<ElementA, kAlignmentA>;

  using ShapeA = BlockPitchLinearShape<
    ThreadblockShape::kM,
    ThreadblockShape::kK,
    AsInt<kBlockSize>::value>;
  using IteratorA = BlockTileUnionIterator<
    ShapeA, ElementA, 1, ThreadMapA, AccessTypeA>;

  // Define iterators over tiles from the B operand
  //
  // NOTE: We convert from BlockRowMajor to BlockPitchLinear here.
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = ::cutlass::Array<ElementB, kAlignmentB>;

  using ShapeB = BlockPitchLinearShape<
    ThreadblockShape::kN,
    ThreadblockShape::kK,
    AsInt<kBlockSize>::value>;
  using IteratorB = BlockTileUnionIterator<
    ShapeB, ElementB, 1, ThreadMapB, AccessTypeB>;

  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma = ::cutlass::gemm::threadblock::MmaMultistage<
    typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
    MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
    MmaCore::kCacheOpB, ElementAccumulator, ::cutlass::layout::RowMajor,
      typename MmaCore::MmaPolicy, Stages>;
};

// Specialization for Dense = T(Sparse) x T(Sparse) (DSS).
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
struct BlockMma<kBlockSize, ElementA, BlockColumnMajor, kAlignmentA,
  ElementB, BlockColumnMajor, kAlignmentB, ElementAccumulator,
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
      ThreadblockShape, WarpShape, InstructionShape,
      ElementA, ::cutlass::layout::ColumnMajor,
      ElementB, ::cutlass::layout::ColumnMajor,
      ElementAccumulator, ::cutlass::layout::RowMajor,
      ::cutlass::arch::OpClassTensorOp, Stages,
      Operator, false, CacheOpA, CacheOpB>;

  // Define iterators over tiles from the A operand
  //
  // NOTE: We convert from BlockColumnMajor to BlockPitchLinear here.
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = ::cutlass::Array<ElementA, kAlignmentA>;

  using ShapeA = BlockPitchLinearShape<
    ThreadblockShape::kM,
    ThreadblockShape::kK,
    AsInt<kBlockSize>::value>;
  using IteratorA = BlockTileUnionIterator<
    ShapeA, ElementA, 1, ThreadMapA, AccessTypeA>;

  // Define iterators over tiles from the B operand
  //
  // NOTE: We convert from BlockColumnMajor to BlockPitchLinear here.
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = ::cutlass::Array<ElementB, kAlignmentB>;

  using ShapeB = BlockPitchLinearShape<
    ThreadblockShape::kK,
    ThreadblockShape::kN,
    AsInt<kBlockSize>::value>;
  using IteratorB = BlockTileUnionIterator<
    ShapeB, ElementB, 0, ThreadMapB, AccessTypeB>;

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
