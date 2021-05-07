#ifndef SPUTNIK_BLOCK_CUTLASS_TYPE_UTILS_H_
#define SPUTNIK_BLOCK_CUTLASS_TYPE_UTILS_H_

#include "sputnik/block/arguments.h"
#include "cutlass/half.h"

namespace sputnik {
namespace block {
namespace cutlass {

template <typename T>
struct Type {
  using Data = T;
  using Meta = int;
};

template <>
struct Type<::cutlass::half_t> {
  using Data = ::cutlass::half_t;
  using Meta =  int16_t;
};

template <BlockSize kBlockSize>
struct AsInt;

template <>
struct AsInt<BlockSize::kNone> {
  static const int value = 0;
};

template <>
struct AsInt<BlockSize::k16> {
  static const int value = 16;
};

template <>
struct AsInt<BlockSize::k32> {
  static const int value = 32;
};

template <>
struct AsInt<BlockSize::k64> {
  static const int value = 64;
};

template <>
struct AsInt<BlockSize::k128> {
  static const int value = 128;
};

}  // namespace sputnik
}  // namespace block
}  // namespace cutlass

#endif  // SPUTNIK_BLOCK_CUTLASS_TYPE_UTILS_H_
