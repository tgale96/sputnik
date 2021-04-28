#ifndef THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_DEPENDENT_TILE_ACCESS_ITERATOR_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_DEPENDENT_TILE_ACCESS_ITERATOR_H_

#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"

namespace sputnik {
namespace block {
namespace cutlass {

// Everything operates the same as a standard CUTLASS iterator,
// but tile offsets are now data dependent.
template <
  typename Shape_,  // BlockMatrixShape
  typename Element_,
  typename Layout_,
  int AdvanceRank_,
  typename ThreadMap_,
  typename AccessType_>
class DependentTileAccessIterator {
 public:
  using LongIndex = typename Layout_::LongIndex;
  using TensorCoord = typename Layout_::TensorCoord;

  // Underlying iterator type.
  using Iterator =
    ::cutlass::transform::threadblock::PredicatedTileAccessIterator<
    ::cutlass::MatrixShape<Shape_::kRow, Shape_::kColumn>,
    Element_, Layout_, AdvanceRank_, ThreadMap_, AccessType_>;

  using Shape = Shape_;
  using Element = typename Iterator::Element;
  using Layout = typename Iterator::Layout;
  
  // The advance rank as pitch-linear.
  static const int kAdvanceRank = Iterator::UnderlyingIterator::kAdvanceRank;
  
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;
  
  static const int kAccessesPerVector = Iterator::kAccessesPerVector;
  
  // The number of tiles in each sparse block.
  static const int kIterationsBlock = Shape::kBlock /
    Iterator::UnderlyingIterator::Shape::kContiguous;

  // Pointer type.
  using Pointer = typename Iterator::Pointer;
  
  // Matrix meta-data type.
  using Meta = typename Type<Element>::Meta;
  
  struct Params {
    typename Iterator::Params iterator_params;
    Meta *offsets;
    int stride;

    // Default ctor
    CUTLASS_HOST_DEVICE
    Params() {}

    // Construct from operand.
    CUTLASS_HOST_DEVICE
    Params(Op op)
      : iterator_params(op.ld),
	offsets((Meta*)op.offsets),
	stride(op.ld) {}
  };

  // Underyling iterator.
  Iterator iterator_;

  // Iterator parameters.
  Params params_;

  // Iteration inside a block.
  int iteration_block_;

  // Current absolute offset in blocks.
  int current_offset_;
  
  CUTLASS_HOST_DEVICE
  DependentTileAccessIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset)
    : iterator_(params.iterator_params,
		pointer, extent, thread_id,
		threadblock_offset),
      params_(params),
      iteration_block_(0),
      current_offset_(0) {
    // NOTE: This is pre-offset to the correct place by Config.
    //
    // TODO(tgale): Figure out how to handle this for the
    // transposed case. We likely need to sort the column
    // indices to get the block offsets anyways, so we could
    // probably still pass them in and have the correct
    // indices be contiguous.
  }

  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iterator_.set_iteration_index(index);
  }

  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset_) {
    // TODO(tgale): This only supports advancing by a single
    // tile. Generalize this.
    iterator_.add_tile_offset(tile_offset_);

    // TODO(tgale): Pipeline the loading of offsets to avoid
    // the added latency.
    if (iteration_block_ == 0) {
      int absolute_offset = (int)__ldg(params_.offsets);
      int relative_offset = absolute_offset - current_offset_ - 1;

      LongIndex offset = Shape::kBlock * relative_offset;
      offset = kAdvanceRank ? offset * params_.stride : offset;
      iterator_.add_pointer_offset(offset);
      
      // Update our current offset and pointer for next iteration.
      current_offset_ = absolute_offset;
      ++params_.offsets;
    }

    // TODO(tgale): We could express this more succinctly
    // with add and mod (as bitwise-and).
    ++iteration_block_;
    if (iteration_block_ >= kIterationsBlock) {
      iteration_block_ = 0;
    }
  }

  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    return iterator_.get();
  }

  CUTLASS_HOST_DEVICE
  DependentTileAccessIterator &operator++() {     
    ++iterator_;
    return *this;
  }

  CUTLASS_HOST_DEVICE
  DependentTileAccessIterator operator++(int) {     
    DependentTileAccessIterator self(*this);
    operator++();
    return self;
  }  

  CUTLASS_HOST_DEVICE
  void clear_mask() { iterator_.clear_mask(); }

  CUTLASS_HOST_DEVICE
  bool valid() { return iterator_.valid(); }
};
  
}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_DEPENDENT_TILE_ACCESS_ITERATOR_H_
