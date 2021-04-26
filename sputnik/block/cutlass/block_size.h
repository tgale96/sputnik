#ifndef THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_SIZE_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_SIZE_H_

namespace sputnik {
namespace block {
namespace cutlass {
  
enum class BlockSize {
  kNone = 0,
  k16 = 16,
  k32 = 32,
  k64 = 64,
  k128 = 128,
};

template <BlockSize kBlockSize>
struct Block2Int;

template <>
struct Block2Int<BlockSize::kNone> {
  static const int value = 0;
};
  
template <>
struct Block2Int<BlockSize::k16> {
  static const int value = 16;
};
  
template <>
struct Block2Int<BlockSize::k32> {
  static const int value = 32;
};

template <>
struct Block2Int<BlockSize::k64> {
  static const int value = 64;
};

template <>
struct Block2Int<BlockSize::k128> {
  static const int value = 128;
};       

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik 

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_SIZE_H_ 

