#ifndef SPUTNIK_BLOCK_CUTLASS_TYPE_UTILS_H_
#define SPUTNIK_BLOCK_CUTLASS_TYPE_UTILS_H_

#include "cutlass/cutlass.h"

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
 
}  // namespace sputnik
}  // namespace block
}  // namespace cutlass

#endif  // SPUTNIK_BLOCK_CUTLASS_TYPE_UTILS_H_
