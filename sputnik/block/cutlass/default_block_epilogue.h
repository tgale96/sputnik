#ifndef SPUTNIK_BLOCK_CUTLASS_DEFAULT_BLOCK_EPILOGUE_H_
#define SPUTNIK_BLOCK_CUTLASS_DEFAULT_BLOCK_EPILOGUE_H_

#include "sputnik/block/cutlass/block_tile_output_iterator.h"

#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/epilogue.h"

namespace sputnik {
namespace block {
namespace cutlass {

// Default epilogue for dense outputs.
template <
    BlockSize kBlockSize,
    typename Layout,
    typename Shape,
    typename WarpMmaTensorOp,
    int PartitionsK,
    typename OutputOp,
    int ElementsPerAccess>
struct DefaultBlockEpilogue {
  using Epilogue = typename ::cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
      Shape, WarpMmaTensorOp, PartitionsK, OutputOp, ElementsPerAccess>::Epilogue;
};

// Specialization for blocksparse outputs.
template <
    BlockSize kBlockSize,
    typename Shape,
    typename WarpMmaTensorOp,
    int PartitionsK,
    typename OutputOp,
    int ElementsPerAccess>
struct DefaultBlockEpilogue<
    kBlockSize,
    BlockRowMajor,
    Shape,
    WarpMmaTensorOp,
    PartitionsK,
    OutputOp,
    ElementsPerAccess> {
  using Default = typename ::cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
      Shape, WarpMmaTensorOp, PartitionsK, OutputOp, ElementsPerAccess>;

  // Special iterator for blocksparse outputs.
  using OutputTileIterator = BlockTileOutputIterator<
      kBlockSize,
      typename Default::OutputTileThreadMap,
      typename Default::ElementOutput>;

  using Epilogue = ::cutlass::epilogue::threadblock::Epilogue<
      typename Default::Shape,
      typename Default::WarpMmaTensorOp,
      Default::kPartitionsK,
      OutputTileIterator,
      typename Default::AccumulatorFragmentIterator,
      typename Default::WarpTileIterator,
      typename Default::SharedLoadIterator,
      typename Default::OutputOp,
      typename Default::Padding,
      Default::kFragmentsPerIteration>;
};

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // SPUTNIK_BLOCK_CUTLASS_DEFAULT_BLOCK_EPILOGUE_H_
