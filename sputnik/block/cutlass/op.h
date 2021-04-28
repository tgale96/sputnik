#ifndef THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_OP_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_OP_H_

namespace sputnik {
namespace block {
namespace cutlass {

// Operand structure.
struct Op {
  void * data;
  void * offsets;
  void * indices;
  
  int ld;
  
  CUTLASS_HOST_DEVICE
  Op(void const *data_, void const *offsets_, void const *indices_, int ld_)
    : data(const_cast<void*>(data_)),
    offsets(const_cast<void*>(offsets_)),
    indices(const_cast<void*>(indices_)),
    ld(ld_) {}
  
  CUTLASS_HOST_DEVICE
  Op(void const *data_, int ld_)
    : data(const_cast<void*>(data_)),
    offsets(nullptr),
    indices(nullptr),
    ld(ld_) {}
};
  
}  // namespace cutlass
}  // namespace block
}  // namespace sputnik 

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_OP_H_ 
