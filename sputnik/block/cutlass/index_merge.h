#ifndef SPUTNIK_BLOCK_CUTLASS_INDEX_MERGE_H_
#define SPUTNIK_BLOCK_CUTLASS_INDEX_MERGE_H_

#include "sputnik/block/cutlass/op.h"
#include "cutlass/gemm/gemm.h"

namespace sputnik {
namespace block {
namespace cutlass {

template <int Elements>
struct BitVector {

  using Storage = uint64_t;

  static constexpr int kBitsPerEntry = sizeof(Storage) * 8;
  static constexpr int kEntries = Elements / kBitsPerEntry;

  Storage data[Elements / kBitsPerEntry];

  CUTLASS_DEVICE
  BitVector() {
    Clear();
  }

  CUTLASS_DEVICE
  void Clear() {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kEntries; ++i) {
      data[i] = 0;
    }
  }

  CUTLASS_DEVICE
  bool Get(int idx) {
    int entry_idx = idx / kBitsPerEntry;
    int bit_idx = idx % kBitsPerEntry;

    bool out = false;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kEntries; ++i) {
      if (i == entry_idx) {
        out |= (bool)(data[i] & (1ull << bit_idx));
      }
    }
    return out;
  }

  CUTLASS_DEVICE
  void SetLessThan(int idx) {
    int entry_idx = idx / kBitsPerEntry;
    int bit_idx = idx % kBitsPerEntry;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kEntries; ++i) {
      if (i < entry_idx) {
        data[i] = ~0ull;
      } else if (i == entry_idx) {
        data[i] = ((Storage)(1ull << bit_idx)) - 1;
      }
    }
  }

  CUTLASS_DEVICE
  int Sum() const {
    int out = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kEntries; ++i) {
      out += __popcll(data[i]);
    }
    return out;
  }

  CUTLASS_DEVICE
  int SumWithMask(const BitVector<Elements> &m) const {
    int out = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kEntries; ++i) {
      out += __popcll(data[i] & m.data[i]);
    }
    return out;
  }

  CUTLASS_DEVICE
  Storage* Data() {
    return data;
  }
};

template <typename Gemm_, int kBlockSize_>
struct IndexMerge {
  // NOTE: We only support block size 128 for now.
  static_assert(kBlockSize_ == 128);

  using Gemm = Gemm_;
  using GemmCoord = ::cutlass::gemm::GemmCoord;

  // The number of threads in the threadblock.
  static constexpr int kThreads = Gemm::kThreadCount;

  // The maximum contraction size supported.
  static constexpr int kMaxContraction = 32 * 1024;

  // Block size as an integer.
  static constexpr int kBlockSize = kBlockSize_;

  // The number of blocks in each contraction.
  static constexpr int kOffsets = kMaxContraction / kBlockSize;

  // The number of offsets processed by each thread.
  static constexpr int kOffsetsPerThread = kOffsets / kThreads;

  // NOTE: This needs to be to let us use 8-bit offsets.
  static_assert(kOffsets <= 256);

  using Mask = BitVector<kOffsets>;

  // The total number of shared memory bytes.
  static constexpr int kSmemBytes = kOffsets * 2;

  // Mask storage type.
  using Storage = typename Mask::Storage;

  // Offset type.
  using Offset = uint8_t;

  Offset *data;
  int steps_k;

  CUTLASS_DEVICE
  IndexMerge() {}

  CUTLASS_DEVICE
  IndexMerge(Op op_a, Op op_b,
             int problem_size_k,
             const GemmCoord &offset,
             Offset *smem) : data(smem) {
    // NOTE: Bitmasks are always stored contraction dimension contiguous.
    int row_offset_a = offset.m();
    int row_offset_b = offset.n();

    // if (threadIdx.x == 0) {
    //   printf("tid.x %d: Performing index merge.\n",
    //          threadIdx.x);
    //   printf("tid.x %d: row_offset_a %d, row_offset_b %d\n",
    //          threadIdx.x, row_offset_a, row_offset_b);
    // }

    Storage *bitmask_a = (Storage*)op_a.bitmask + row_offset_a;
    Storage *bitmask_b = (Storage*)op_b.bitmask + row_offset_b;

    // Load bitmasks for both operands.
    Mask mask_a, mask_b;
    int mask_loads_k = ((problem_size_k / kBlockSize) + Mask::kBitsPerEntry - 1) /
                       Mask::kBitsPerEntry;

    // if (threadIdx.x == 0) {
    //   printf("tid.x %d: mask_loads_k = %d, problem_size_k %d\n",
    //          threadIdx.x, mask_loads_k, problem_size_k);
    // }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Mask::kEntries; ++i) {
      if (i < mask_loads_k) {
        mask_a.data[i] = bitmask_a[i];
        mask_b.data[i] = bitmask_b[i];
      }
    }

    // Find the mask union.
    Mask op_union;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Mask::kEntries; ++i) {
      op_union.data[i] = mask_a.data[i] & mask_b.data[i];
    }
    steps_k = op_union.Sum() * (kBlockSize / Gemm::Mma::Shape::kK);

    // if (threadIdx.x == 0) {
    //   printf("tid.x %d: mask_a %lld, mask_b %lld, union %lld, steps_k %d\n",
    //          threadIdx.x, mask_a.Data()[0], mask_b.Data()[0], op_union.Data()[0],
    //          steps_k);
    //   printf("tid.x %d: kOffsetsPerThread %d\n",
    //          threadIdx.x, kOffsetsPerThread);
    // }

    // Set the block offsets for each operand.
    Mask prefix;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kOffsetsPerThread; ++i) {
      int bit_idx = threadIdx.x + kThreads * i;

      // Mask the prefix and sum to get the offset.
      prefix.SetLessThan(bit_idx);

      // if (threadIdx.x == 0) {
      //   printf("tid.x %d: bit_idx %d, prefix[0] %lld\n",
      //          threadIdx.x, bit_idx, prefix.Data()[0]);
      // }

      Offset offset_a = (Offset)mask_a.SumWithMask(prefix);
      Offset offset_b = (Offset)mask_b.SumWithMask(prefix);

      // if (threadIdx.x == 0) {
      //   printf("tid.x %d: offset_a %d, offset_b %d\n",
      //          threadIdx.x, (int)offset_a, (int)offset_b);
      // }

      // Prefix sum up to this thread's bit to get
      // offset into shared memory.
      int write_offset = op_union.SumWithMask(prefix);

      // Whether or not this thread has a valid value to
      // write to shared memory.
      bool should_write = op_union.Get(bit_idx);
      // printf("tid.x %d: write_offset %d, should_write %d\n",
      //        threadIdx.x, write_offset, should_write);
      if (should_write) {
        data[write_offset] = offset_a;
        data[write_offset + kOffsets] = offset_b;
      }
    }
  }

  CUTLASS_DEVICE
  int StepsK() const {
    return steps_k;
  }

  CUTLASS_DEVICE
  Offset* OffsetPtrA() const {
    return data;
  }

  CUTLASS_DEVICE
  Offset* OffsetPtrB() const {
    return data + kOffsets;
  }
};

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // SPUTNIK_BLOCK_CUTLASS_INDEX_MERGE_H_
