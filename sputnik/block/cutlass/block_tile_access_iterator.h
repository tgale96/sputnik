#ifndef THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_TILE_ACCESS_ITERATOR_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_TILE_ACCESS_ITERATOR_H_

#include "sputnik/block/cutlass/block_pitch_linear.h"
#include "sputnik/block/cutlass/op.h"
#include "sputnik/block/cutlass/type_utils.h"

#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"

namespace sputnik {
namespace block {
namespace cutlass {

// Iterator for block-sparse matrices.
//
// We have two key differences from a PredicatedTileAccessIterator
// 1. We never have a residue tile.
// 2. Matrix dimensions are both dividible by block size.
//
// Thus, we don't need any predicates and don't need to worry
// about residue handling.
template <
    typename Shape_,  // BlockPitchLinearShape
    typename Element_,
    int AdvanceRank_,
    typename ThreadMap_,
    typename AccessType_>
class BlockTileAccessIterator {
 public:

  using Shape = Shape_;
  using Element = Element_;
  using Layout = BlockPitchLinear;
  static int const kAdvanceRank = AdvanceRank_;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  // TODO(tgale): Expose this as the layout argument and use
  // simple converters to get the shape right.
  using LayoutCvt = BlockRowMajor;

  // TODO(tgale): Relax this constraint. We'd like to be able to use
  // smaller tile dimensions for cases that are occupancy limited.
  static_assert(kAdvanceRank ?
                Shape::kContiguous == Shape::kBlock :
                Shape::kStrided == Shape::kBlock,
		"The outer tile dimension must equal the block size.");
  // TODO(tgale): Relax this constraint. For small/medium block sizes
  // we'd like to pack multiple blocks into one mma tile.
  static_assert(kAdvanceRank ?
                Shape::kStrided <= Shape::kBlock :
                Shape::kContiguous <= Shape::kBlock,
		"The advance rank tile dimension must be less-than or "
		"equal to the blocks size.");
  static_assert(Shape::kBlock % Shape::kContiguous == 0,
		"The block size must be divisible by the contiguous "
		"tile dimension.");
  static_assert(Shape::kBlock % Shape::kStrided == 0,
                "The block size must be divisible by the strided "
                "tile dimension.");

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = ::cutlass::TensorRef<Element, Layout>;
  using TensorView = ::cutlass::TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element *;
  using NonConstPointer = typename ::cutlass::platform::remove_const<Element>::type *;

  static int const kAccessesPerVector = ThreadMap::kElementsPerAccess /
    AccessType::kElements;

  static_assert(!(ThreadMap::kElementsPerAccess % AccessType::kElements),
    "Vectors implied by the thread map must be divisible by the access type.");

  // NOTE: All increments are statically computable for block-sparse
  // iterator with known block dimensions.
  struct Params {
    int *block_offsets;

    // NOTE: We rely on our config to update this parameter inside
    // the kernel. This is a hack.
    int steps_k;

    CUTLASS_HOST_DEVICE Params() {}
    CUTLASS_HOST_DEVICE Params(Op op)
        : block_offsets((int*)op.block_offsets), steps_k(0) {}
  };

  /// Internal pointer type permits fast address arithmetic
  using BytePointer = char *;

  static const int kElementBytes = ::cutlass::sizeof_bits<Element>::value / 8;

  static const LongIndex kStride = Shape::kBlock;

  // TODO(tgale): Do we need to alter this for differen advance rank?
  static const LongIndex kIncStrided = Shape::kBlock *
    ThreadMap::Delta::kStrided * kElementBytes;

  // Advance to the next tile in the block.
  static const LongIndex kIncAdvance =
      kAdvanceRank ?
      Shape::kStrided * kStride * kElementBytes :
      Shape::kContiguous * kElementBytes;

  static const LongIndex kIncNext = kIncAdvance -
    LongIndex(ThreadMap::Iterations::kStrided - 1) *
    ThreadMap::Delta::kStrided * LongIndex(kStride) *
    kElementBytes;

  // TODO(tgale): Use our Index/LongIndex types for these
  // values.
  //
  // The number of tiles in each sparse block.
  static const int kIterationsBlock =
      kAdvanceRank ?
      Shape::kBlock / Shape::kStrided :
      Shape::kBlock / Shape::kContiguous;

  // Increment to the next block.
  static const int kIncBlock = Shape::kBlock * Shape::kBlock *
    kElementBytes - (kIterationsBlock - 1) * kIncAdvance - kIncAdvance;

  // The number of bytes in each block.
  static const int kBytesPerBlock =
      Shape::kBlock * Shape::kBlock * kElementBytes;

  //
  /// Data members
  //

  // Single predicate used to stop after the last tile.
  bool predicate_;

  // Internal pointer to first access of tile
  BytePointer pointer_;

  // Size of tensor
  TensorCoord extent_;

  // Iteration along vectors implied by the thread map
  int iteration_vector_;

  // Iteration in the contiguous dimension
  int iteration_contiguous_;

  // Iteration in the strided dimension
  int iteration_strided_;

  // Iteration inside a block.
  int iteration_block_;

  // Current block offset.
  int current_offset_;

  // Iterator parameters.
  Params params_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_DEVICE
  BlockTileAccessIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      int block_row_offset)
      : pointer_(reinterpret_cast<BytePointer>(
            const_cast<NonConstPointer>(pointer))),
        extent_(extent), current_offset_(-kBytesPerBlock), params_(params) {
    // Add thread offset to pointer.
    Layout layout(kStride);
    const TensorCoord thread_offset = ThreadMap::initial_offset(thread_id);
    add_pointer_offset(layout(thread_offset));

    predicate_ = true;
    iteration_block_ = 0;
    set_iteration_index(0);

    if (kAdvanceRank) {
      // Offset to the first block.
      add_block_offset();
    } else {
      add_pointer_offset(block_row_offset);
    }
  }

  CUTLASS_DEVICE
  BlockTileAccessIterator() {}

  // Overrides the internal iteration index
  CUTLASS_DEVICE
  void set_iteration_index(int index) {
    iteration_vector_ = index % kAccessesPerVector;
    int residual_access = index / kAccessesPerVector;

    iteration_contiguous_ = residual_access % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = residual_access / ThreadMap::Iterations::kContiguous;
  }

  // Adds a pointer offset in units of Element
  CUTLASS_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    pointer_ += kElementBytes * pointer_offset;
  }

  CUTLASS_DEVICE
  void add_block_offset() {
    if (kAdvanceRank) {
      if (params_.steps_k <= 0) return;
      params_.steps_k -= kIterationsBlock;

      int absolute_offset = __ldg(params_.block_offsets) * kBytesPerBlock;
      int relative_offset = absolute_offset - current_offset_ - kBytesPerBlock;

      pointer_ += relative_offset;

      // Update our current offset and pointer for next iteration.
      current_offset_ = absolute_offset;
      ++params_.block_offsets;
    } else {
      // Offset to the next block in the compressed block-row.
      pointer_ += kIncBlock;
    }
  }

  // Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset_) {
    // TODO(tgale): This function only supports increments by one
    // tile in the 'advance' direction.
    pointer_ += kIncAdvance;

    ++iteration_block_;
    if (iteration_block_ >= kIterationsBlock) {
      iteration_block_ = 0;

      // Offset to the next sparse block.
      add_block_offset();
    }
  }

  // Returns a pointer
  CUTLASS_DEVICE
  AccessType *get() const {
    AccessType *out = reinterpret_cast<AccessType *>(
	pointer_ +
	iteration_contiguous_ *
	(ThreadMap::Delta::kContiguous * ::cutlass::sizeof_bits<Element>::value) / 8) +
        iteration_vector_;
    return out;
  }

  // Increment and return an instance to self.
  CUTLASS_DEVICE
  BlockTileAccessIterator &operator++() {
    ++iteration_vector_;
    if (iteration_vector_ < kAccessesPerVector) {
      return *this;
    }

    iteration_vector_ = 0;
    ++iteration_contiguous_;

    if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous) {
      return *this;
    }

    // Enter here only if (iteration_contiguous_ ==
    // ThreadMap::Iteration::kContiguous)
    iteration_contiguous_ = 0;
    ++iteration_strided_;

    if (iteration_strided_ < ThreadMap::Iterations::kStrided) {
      pointer_ += kIncStrided;
      return *this;
    }

    // Enter here only if (iteration_stride_ == ThreadMap::Iteration::kStrided)
    // which means we enter the next tile.
    iteration_strided_ = 0;

    // advance to next tile
    pointer_ += kIncNext;

    // now return to start tile - if the iterator is subsequently advanced, this
    // subtraction as well as the subsequent integer addition are both elided by
    // the compiler.
    pointer_ -= kIncAdvance;

    return *this;
  }

  // Increment and return an instance to self.
  CUTLASS_DEVICE
  BlockTileAccessIterator operator++(int) {
    BlockTileAccessIterator self(*this);
    operator++();
    return self;
  }

  CUTLASS_DEVICE
  bool valid() const {
    return predicate_;
  }

  CUTLASS_DEVICE
  void clear_mask() {
    predicate_ = false;
  }

};

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_TILE_ACCESS_ITERATOR_H_
