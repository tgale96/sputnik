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
  static const int kIterationsBlock =
      kAdvanceRank ?
      Shape::kBlock / Iterator::UnderlyingIterator::Shape::kStrided :
      Shape::kBlock / Iterator::UnderlyingIterator::Shape::kContiguous;

  // Pointer type.
  using Pointer = typename Iterator::Pointer;

  // Matrix meta-data type.
  using Meta = typename Type<Element>::Meta;

  struct Params {
    typename Iterator::Params iterator_params;
    Meta *indices;
    int stride;

    // NOTE: We rely on our config to update this parameter inside
    // the kernel. This is a hack.
    int steps_k;

    // Default ctor
    CUTLASS_HOST_DEVICE Params() {}

    // Construct from operand.
    CUTLASS_HOST_DEVICE
    Params(Op op)
      : iterator_params(op.ld),
	indices((Meta*)op.indices),
	stride(op.ld),
        steps_k(0) {}
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
      current_offset_(-Shape::kBlock) {
    // Offset to the first block.
    add_block_offset();
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
  void add_block_offset() {
    if (params_.steps_k <= 0) return;
    params_.steps_k -= kIterationsBlock;

    int absolute_offset = (int)__ldg(params_.indices) * Shape::kBlock;
    int relative_offset = absolute_offset - current_offset_ - Shape::kBlock;

    if (kAdvanceRank) relative_offset *= params_.stride;
    iterator_.add_pointer_offset(relative_offset);

    // Update our current offset and pointer for next iteration.
    current_offset_ = absolute_offset;
    ++params_.indices;
  }

  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset_) {
    // TODO(tgale): This only supports advancing by a single
    // tile. Generalize this.
    iterator_.add_tile_offset(tile_offset_);

    // TODO(tgale): This iterator adds 26 registers to our
    // kernel. If we see negative impact, we can try keeping
    // the data-dependent offset as a separate value and
    // only offsetting the pointer in get().
    //
    // TODO(tgale): Pipeline the loading of indices to avoid
    // the added latency.
    ++iteration_block_;
    if (iteration_block_ >= kIterationsBlock) {
      iteration_block_ = 0;
      add_block_offset();
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
