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
class BlockTileUnionIterator : public BlockTileAccessIterator<
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
  using BytePointer = typename Base::BytePointer;
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

  CUTLASS_DEVICE
  BlockTileUnionIterator(
      Params const &params,
      Pointer pointer,
      TensorCoord extent,
      int thread_id,
      int block_row_offset) :
      offsets(params.offsets) {
    iterator.pointer_ = reinterpret_cast<BytePointer>(
        const_cast<NonConstPointer>(pointer));
    iterator.params_ = params.base_params;

    // Add thread offset to pointer.
    Layout layout(Base::kStride);
    const TensorCoord thread_offset = ThreadMap::initial_offset(thread_id);
    add_pointer_offset(layout(thread_offset));

    iterator.predicate_ = true;
    iterator.iteration_block_ = 0;
    set_iteration_index(0);

    if (kAdvanceRank) {
      iterator.current_offset_ = -Base::kBytesPerBlock;
    } else {
      add_pointer_offset(block_row_offset);
      iterator.current_offset_ = -1;
    }
    add_block_offset();
  }

  CUTLASS_DEVICE
  void set_iteration_index(int index) {
    return iterator.set_iteration_index(index);
  }

  CUTLASS_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator.add_pointer_offset(pointer_offset);
  }

  CUTLASS_DEVICE
  void add_block_offset() {
    if (iterator.params_.steps_k <= 0) return;
    iterator.params_.steps_k -= Base::kIterationsBlock;

    // Load the next offset from shared memory.
    int offset_to_block = (int)*offsets;
    ++offsets;

    if (kAdvanceRank) {
      int absolute_offset = __ldg(
	iterator.params_.block_offsets + offset_to_block) *
	Base::kBytesPerBlock;
      int relative_offset = absolute_offset -
                            iterator.current_offset_ -
                            Base::kBytesPerBlock;

      iterator.pointer_ += relative_offset;

      // Update our current offset and pointer for next iteration.
      iterator.current_offset_ = absolute_offset;
    } else {
      // Offset to the next block in the union.
      int relative_offset = offset_to_block - iterator.current_offset_ - 1;

      iterator.pointer_ += Base::kBytesPerBlock * relative_offset;
      iterator.current_offset_ = offset_to_block;
    }
  }

  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset_) {
    // TODO(tgale): This function only supports increments by one
    // tile in the 'advance' direction.
    iterator.pointer_ += Base::kIncAdvance;

    ++iterator.iteration_block_;
    if (iterator.iteration_block_ >= Base::kIterationsBlock) {
      iterator.iteration_block_ = 0;

      // Increment to the start of the next block.
      if (!kAdvanceRank) iterator.pointer_ += Base::kIncBlock;

      // Offset to the next sparse block.
      add_block_offset();
    }
  }

  CUTLASS_DEVICE
  AccessType *get() const {
    return iterator.get();
  }

  CUTLASS_DEVICE
  BlockTileUnionIterator &operator++() {
    ++iterator;
    return *this;
  }

    CUTLASS_DEVICE
  BlockTileUnionIterator operator++(int) {
    BlockTileUnionIterator self(*this);
    operator++();
    return self;
  }

  CUTLASS_DEVICE
  bool valid() const {
    return iterator.valid();
  }

  CUTLASS_DEVICE
  void clear_mask() {
    iterator.clear_mask();
  }
};

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // SPUTNIK_BLOCK_CUTLASS_BLOCK_TILE_UNION_ITERATOR_H_
