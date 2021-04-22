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

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik 

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_SIZE_H_ 

