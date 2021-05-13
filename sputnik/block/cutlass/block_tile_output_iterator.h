#ifndef SPUTNIK_BLOCK_CUTLASS_BLOCK_TILE_OUTPUT_ITERATOR_H_
#define SPUTNIK_BLOCK_CUTLASS_BLOCK_TILE_OUTPUT_ITERATOR_H_

#include "sputnik/block/arguments.h"

#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"

namespace sputnik {
namespace block {
namespace cutlass {

// Mirrors ::cutlass::epilogue::threadblock::PredicatedTileIterator.
//
// TODO(tgale): We only need predicates for whole blocks. Add these.
template <BlockSize kBlockSize_, typename ThreadMap_, typename Element_>
class BlockTileOutputIterator {
 public:
  using ThreadMap = ThreadMap_;
  using Element = Element;
  using Layout = ::cutlass::layout::RowMajor;

  // The block size as an integer.
  static constexpr int kBlockSize = AsInt<kBlockSize_>::value;

  // TODO(tgale): We can relax this without adding any dynamic
  // code. This might be needed for different tile configs.
  static_assert(ThreadMap::Delta::kColumn == kBlockSize,
                "Column delta must equal block size.");

  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = typename Layout::TensorCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Count::kTile;

  using Fragment = ::cutlass::Array<
    Element,
    ThreadMap::Iterations::kColumn *
    ThreadMap::Iterations::kRow *
    ThreadMap::Iterations::kGroup *
    ThreadMap::Iterations::kCluster * kElementsPerAccess>;

  using ::cutlass::epilogue::threadblock::PredicatedTileIteratorParams;
  using ::cutlass::epilogue::threadblock::make_OutputTileThreadMapDesc;

  struct Params : PredicatedTileIteratorParams {

    CUTLASS_HOST_DEVICE Params() {}

    CUTLASS_HOST_DEVICE
    Params(Layout const &layout): PredicatedTileIteratorParams(
        layout.stride(0) * int(sizeof(AccessType)) / kElementsPerAccess,
        make_OutputTileThreadMapDesc<ThreadMap>()) {}
  };

  CUTLASS_DEVICE
  BlockTileOutputIterator(
      Params const &params,
      Element *pointer,
      TensorCoord extent,
      int thread_idx,
      int threadblock_offset) : params_(params) {
    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx);

    // Add threadblock & thread offset.
    byte_pointer_ = reinterpret_cast<uint8_t *>(pointer) +
      LongIndex(thread_offset.row()) * LongIndex(params_.stride) +
      LongIndex(thread_offset.column()) * sizeof(AccessType) / kElementsPerAccess;
    add_pointer_offset(threadblock_offset);

    state_[0] = state_[1] = state_[2] = 0;
  }

  CUTLASS_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, int64_t byte_offset) {
    uint8_t *byte_pointer = byte_pointer_;
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
    uint8_t *byte_pointer = byte_pointer_;
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
            cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
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
