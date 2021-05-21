#ifndef SPUTNIK_BLOCK_CUTLASS_BLOCK_TILE_UNION_ITERATOR_H_
#define SPUTNIK_BLOCK_CUTLASS_BLOCK_TILE_UNION_ITERATOR_H_

#include "sputnik/block/cutlass/block_tile_access_iterator.h"

namespace sputnik {
namespace block {
namespace cutlass {

template <
    typename Shape_,  // BlockPitchLinearShape
    typename Element_,
    int AdvanceRank_,
    typename ThreadMap_,
    typename AccessType_>
class BlockTileUnionIterator : BlockTileAccessIterator<
    Shape_, Element_, AdvanceRank_, ThreadMap_, AccessType_> {
 public:

  using Base = BlockTileAccessIterator<
      Shape_, Element_, AdvanceRank_, ThreadMap_, AccessType_>;

  using Shape = typename Base::Shape;
  using Element = typename Base::Element;
  using Layout = typename Base::Layout;
  static constexpr int kAdvanceRank = Base::kAdvanceRank;
  using ThreadMap = typename Base::ThreadMap;
  using AccessType = typename Base::AccessType;
  using Index = typename Base::Index;
  using LongIndex = typename Base::LongIndex;
  using TensorRef = typename Base::TensorRef;
  using TensorView = typename Base::TensorView;
  using TensorCoord = typename Base::TensorCoord;
  using Pointer = typename Base::Pointer;
  using NonConstPointer = typename Base::NonConstPointer;

  struct Params {
    typename Base::Params base_params;

    // NOTE: We rely on our config to update this
    // parameter inside the kernel. This is a hack.
    uint8_t *offsets;

    CUTLASS_HOST_DEVICE Params() {}
    CUTLASS_HOST_DEVICE Params(Op op)
        : base_params(op) {}
  };

 private:
  Base iterator;
  uint8_t *offsets;

 public:

  CUTLASS_HOST_DEVICE
  BlockTileUnionIterator(
      Params const &params,
      Pointer pointer,
      TensorCoord extent,
      int thread_id,
      int block_row_offset) :
      iterator(params.base_params,
               pointer, extent, thread_id,
               block_row_offset),
      offsets(params.offsets) {
    if (!kAdvanceRank) {
      iterator.current_offset_ = -1;
      add_block_offset();
    }
  }

  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    return iterator.set_iteration_index(index);
  }

  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator.add_pointer_offset(pointer_offset);
  }

  CUTLASS_DEVICE
  void add_block_offset() {
    // Do nothing if we're out of work to do.
    if (iterator.params_.steps_k == 0) return;
    ++iterator.params_.steps_k;

    // Load the next offset from shared memory.
    int offset_to_block = (int)*offsets;
    ++offsets;

    if (kAdvanceRank) {
      int absolute_offset = __ldg(
          iterator.params_.block_offsets + offset_to_block);
      int relative_offset = absolute_offset -
                            iterator.current_offset_ -
                            Base::kBytesPerBlock;

      iterator.pointer_ += relative_offset;

      // Update our current offset and pointer for next iteration.
      iterator.current_offset_ = absolute_offset;
    } else {
      // Advance to the start of the next block.
      iterator.pointer_ += Base::kIncBlock;

      // Offset to the next block in the union.
      int relative_offset = offset_to_block - iterator.current_offset_ - 1;
      iterator.pointer_ += Base::kBytesPerBlock * relative_offset;
      iterator.current_offset_ = offset_to_block;
    }
  }

  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset_) {
    iterator.add_tile_offset(tile_offset_);
  }

  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    return iterator.get();
  }

  CUTLASS_HOST_DEVICE
  BlockTileUnionIterator &operator++() {
    ++iterator;
    return *this;
  }

    CUTLASS_HOST_DEVICE
  BlockTileUnionIterator operator++(int) {
    BlockTileUnionIterator self(*this);
    operator++();
    return self;
  }

  CUTLASS_HOST_DEVICE
  bool valid() const {
    return iterator.valid();
  }

  CUTLASS_HOST_DEVICE
  void clear_mask() {
    iterator.clear_mask();
  }
};

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // SPUTNIK_BLOCK_CUTLASS_BLOCK_TILE_UNION_ITERATOR_H_
