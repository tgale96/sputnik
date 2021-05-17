#ifndef SPUTNIK_BLOCK_CUTLASS_BLOCK_TILE_OUTPUT_ITERATOR_H_
#define SPUTNIK_BLOCK_CUTLASS_BLOCK_TILE_OUTPUT_ITERATOR_H_

#include "sputnik/block/arguments.h"
#include "sputnik/block/cutlass/block_pitch_linear.h"

#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"

namespace sputnik {
namespace block {
namespace cutlass {

using ::cutlass::epilogue::threadblock::make_OutputTileThreadMapDesc;

// Mirrors ::cutlass::epilogue::threadblock::PredicatedTileIterator.
template <BlockSize kBlockSize_, typename ThreadMap_, typename Element_>
class BlockTileOutputIterator {
 public:
  using ThreadMap = ThreadMap_;
  using Element = Element_;

  using Layout = BlockRowMajor;

  // The block size as an integer.
  static constexpr int kBlockSize = AsInt<kBlockSize_>::value;

  // The number of nonzeros in a block.
  static constexpr int kBlockElements = kBlockSize * kBlockSize;

  // TODO(tgale): To generalize to multiple sparse blocks per
  // threadblock, we need to add support for multiple column
  // indices to the dense rhs tile loader. Until we do this,
  // make sure that we don't use invalid tile sizes.
  static_assert(ThreadMap::Shape::kColumn == kBlockSize,
                "Column size must be equal to block size.");

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = typename Layout::TensorCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Count::kTile;

  using TensorRef = ::cutlass::TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  // Register fragment type.
  static constexpr LongIndex kFragmentSize =
      ThreadMap::Iterations::kColumn *
      ThreadMap::Iterations::kRow *
      ThreadMap::Iterations::kGroup *
      ThreadMap::Iterations::kCluster * kElementsPerAccess;
  using Fragment = ::cutlass::Array<Element, kFragmentSize>;

  // Memory access type.
  using AccessType = ::cutlass::AlignedArray<
      Element, kElementsPerAccess>;

  using BaseParams = ::cutlass::epilogue::threadblock::PredicatedTileIteratorParams;

  struct Params : BaseParams {

    CUTLASS_HOST_DEVICE Params() {}

    // NOTE: layout.stride is always the block size.
    CUTLASS_HOST_DEVICE
    Params(int unused) : BaseParams(
        kBlockSize * int(sizeof(AccessType)) / kElementsPerAccess,
        make_OutputTileThreadMapDesc<ThreadMap>()) {}
  };

  // Parameters with pre-computed offsets.
  Params params_;

  // Global memory pointer.
  char *byte_pointer_;

  // State counters.
  int state_[3];

  CUTLASS_DEVICE
  BlockTileOutputIterator(
      Params const &params,
      Element *pointer,
      TensorCoord extent,
      int thread_idx,
      int threadblock_offset) : params_(params) {
    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);

    // Add threadblock & thread offset.
    byte_pointer_ = reinterpret_cast<char*>(pointer) +
                    LongIndex(thread_offset.row()) *
                    LongIndex(params_.stride) +
                    LongIndex(thread_offset.column()) *
                    sizeof(AccessType) / kElementsPerAccess;
    add_pointer_offset(threadblock_offset);

    // Initialize internal state.
    state_[0] = state_[1] = state_[2] = 0;
  }

  CUTLASS_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_pointer_ += pointer_offset * ::cutlass::sizeof_bits<Element>::value / 8;
  }

  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, int64_t byte_offset) {
    char *byte_pointer = byte_pointer_;
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int frag_row_idx = row + ThreadMap::Iterations::kRow *
                             (group + ThreadMap::Iterations::kGroup * cluster);

          AccessType *memory_pointer = reinterpret_cast<AccessType *>(
              byte_pointer + byte_offset);

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            const int frag_idx = frag_row_idx * ThreadMap::Iterations::kColumn + column;
            const int memory_idx = column * ThreadMap::Delta::kColumn / kElementsPerAccess;
            ::cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                 frag_ptr[frag_idx], (void *)&memory_pointer[memory_idx], true);
          }

          if (row + 1 < ThreadMap::Iterations::kRow) {
            byte_pointer += params_.increment_row;
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) {
          byte_pointer += params_.increment_group;
        }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_byte_offset(frag, 0);
  }

  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, int64_t byte_offset) {
    char *byte_pointer = byte_pointer_;
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int frag_row_idx = row + ThreadMap::Iterations::kRow *
                             (group + ThreadMap::Iterations::kGroup * cluster);

          AccessType *memory_pointer = reinterpret_cast<AccessType *>(
              byte_pointer + byte_offset);

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            const int frag_idx = frag_row_idx * ThreadMap::Iterations::kColumn + column;
            const int memory_idx = column * ThreadMap::Delta::kColumn / kElementsPerAccess;
            ::cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                frag_ptr[frag_idx], (void *)&memory_pointer[memory_idx], true);
          }

          if (row + 1 < ThreadMap::Iterations::kRow) {
            byte_pointer += params_.increment_row;
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) {
          byte_pointer += params_.increment_group;
        }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_byte_offset(frag, 0);
  }

  CUTLASS_DEVICE
  BlockTileOutputIterator &operator++() {
    ++state_[0];
    byte_pointer_ += params_.advance_row;

    if (state_[0] == ThreadMap::Count::kRow) {
      state_[0] = 0;
      ++state_[1];
      byte_pointer_ += params_.advance_group;

      if (state_[1] == ThreadMap::Count::kGroup) {
        state_[1] = 0;
        ++state_[2];
        byte_pointer_ += params_.advance_cluster;

        if (state_[2] == ThreadMap::Count::kCluster) {
          state_[2] = 0;
          byte_pointer_ += params_.advance_tile;
        }
      }
    }
    return *this;
  }

};

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // SPUTNIK_BLOCK_CUTLASS_BLOCK_TILE_OUTPUT_ITERATOR_H_
