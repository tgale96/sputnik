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

  using Pointer = typename Base::Pointer;
  using TensorCoord = typename Base::TensorCoord;
  static constexpr int kAdvanceRank = Base::kAdvanceRank;

  struct Params {
    typename Base::Params base_params;

    // NOTE: We rely on our config to update this
    // parameter inside the kernel. This is a hack.
    uint8_t *offsets;

    CUTLASS_HOST_DEVICE Params() {}
    CUTLASS_HOST_DEVICE Params(Op op)
        : base_params(op) {}
  };

 protected:
  uint8_t *offsets;

 public:

  CUTLASS_HOST_DEVICE
  BlockTileUnionIterator(
      Params const &params,
      Pointer pointer,
      TensorCoord extent,
      int thread_id,
      int block_row_offset) :
      Base(params.base_params,
           pointer, extent, thread_id,
           block_row_offset),
      offsets(params.offsets) {
    if (!kAdvanceRank) {
      this->current_offset_ = -1;
      add_block_offset();
    }
  }

  CUTLASS_DEVICE
  void add_block_offset() override {
    // Do nothing if we're out of work to do.
    if (this->params_.steps_k == 0) return;
    --this->params_.steps_k;

    // Load the next offset from shared memory.
    int offset_to_block = (int)*offsets;
    ++offsets;

    if (kAdvanceRank) {
      int absolute_offset = __ldg(
          this->params_.block_offsets + offset_to_block);
      int relative_offset = absolute_offset -
                            this->current_offset_ -
                            Base::kBytesPerBlock;

      this->pointer_ += relative_offset;

      // Update our current offset and pointer for next iteration.
      this->current_offset_ = absolute_offset;
    } else {
      // Advance to the start of the next block.
      this->pointer_ += Base::kIncBlock;

      // Offset to the next block in the union.
      int relative_offset = offset_to_block - this->current_offset_ - 1;
      this->pointer_ += Base::kBytesPerBlock * relative_offset;
      this->current_offset_ = offset_to_block;
    }
  }
};

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // SPUTNIK_BLOCK_CUTLASS_BLOCK_TILE_UNION_ITERATOR_H_
