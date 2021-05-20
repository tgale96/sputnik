#ifndef SPUTNIK_BLOCK_CUTLASS_INDEX_MERGE_H_
#define SPUTNIK_BLOCK_CUTLASS_INDEX_MERGE_H_

namespace sputnik {
namespace block {
namespace cutlass {

template <int Elements>
struct BitVector {

  using Storage = uint64_t;

  static constexpr int kBitsPerEntry = ::cutlass::sizeof_bits<Storage>::value;
  static constexpr int kEntries = Elements / kBitsPerEntry;

  Storage data[Elements / kBitsPerEntry];

  BitVector() {}

  void Clear() {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kEntries; ++i) {
      data[i] = 0;
    }
  }

  bool Get(int idx) {
    int entry_idx = idx / kBitsPerEntry;
    int bit_idx = idx % kBitsPerEntry;

    bool out = false;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kEntries; ++i) {
      if (i == entry_idx) {
        out |= (bool)(data[i] & (1ull << idx));
      }
    }
    return out;
  }

  bool SetLessThan(int idx) {
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

  int SumWithMask(const BitVector<Elements> &m) {
    int out = 0;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kEntries; ++i) {
      out += __popcll(data[i] * m.data[i]);
    }
    return out;
  }
};

template <typename Gemm_, BlockSize kBlockSize_>
struct IndexMerge {
  // NOTE: We only support block size 128 for now.
  static_assert(kBlockSize_ == BlockSize::k128);

  using Gemm = Gemm_;

  // The number of threads in the threadblock.
  static constexpr int kThreads = Gemm::kThreadCount;

  // The maximum contraction size supported.
  static constexpr int kMaxContraction = 32 * 1024;

  // Block size as an integer.
  static constexpr int kBlockSize = AsInt<kBlockSize_>::value;

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
  using Storage = Mask::Storage;

  // Offset type.
  using Offset = uint8_t;

  __shared__ Offset data[kSmemBytes];

  CUTLASS_DEVICE
  IndexMerge(Op op_a, Op op_b, int problem_size_k, const GemmCoord &offset) {
    // NOTE: Bitmasks are always stored contraction dimension contiguous.
    int row_offset_a = offset.m() * Gemm::Shape::kM / kBlockSize / Mask::kBitsPerEntry;
    int row_offset_b = offset.n() * Gemm::Shape::kN / kBlockSize / Mask::kBitsPerEntry;
    Storage *bitmask_a = (Storage*)op_a.bitmask + row_offset_a;
    Storage *bitmask_b = (Storage*)op_b.bitmask + row_offset_b;

    // Load bitmasks for both operands.
    Mask mask_a, mask_b;
    int mask_loads_k = (problem_size_k + Mask::kBitsPerEntry - 1) /
                       Mask::kBitsPerEntry / kBlockSize;
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

    // Set the block offsets for each operand.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kOffsetsPerThread; ++i) {
      int bit_idx = threadIdx.x + kThreads * i;

      // Mask the prefix and sum to get the offset.
      Mask prefix;
      prefix.Clear();
      prefix.SetLessThan(bit_idx);

      Offset offset_a = (Offset)mask_a.SumWithMask(prefix);
      Offset offset_b = (Offset)mask_b.SumWithMask(prefix);

      // Prefix sum up to this thread's bit to get
      // offset into shared memory.
      int write_offset = op_union.SumWithMask(prefix);

      // Whether or not this thread has a valid value to
      // write to shared memory.
      bool should_write = op_union.Get(bit_idx);
      if (should_write) {
        data[write_offset] = offset_a;
        data[write_offset + kOffsets] = offset_b;
      }
    }
  }
}

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // SPUTNIK_BLOCK_CUTLASS_INDEX_MERGE_H_
