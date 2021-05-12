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

  // NOTE: Offset to blocks in memory. For non-transposed access, this
  // is arange(nblock) and we don't need to explicitly construct it.
  void * block_offsets;

  int ld;

  // TODO(tgale): Including this in our op structure is a hack.
  // find a better way to pass this to our iterators.
  int steps_k;

  // Constructor for non-transposed sparse matrix.
  CUTLASS_HOST_DEVICE
  Op(void const *data_,
     void const *offsets_,
     void const *indices_,
     int ld_)
      : data(const_cast<void*>(data_)),
        offsets(const_cast<void*>(offsets_)),
        indices(const_cast<void*>(indices_)),
        block_offsets(nullptr),
        ld(ld_), steps_k(0) {}

  // Constructor for transposed sparse matrix.
  CUTLASS_HOST_DEVICE
  Op(void const *data_,
     void const *offsets_,
     void const *indices_,
     void const *block_offsets_,
     int ld_)
      : data(const_cast<void*>(data_)),
        offsets(const_cast<void*>(offsets_)),
        indices(const_cast<void*>(indices_)),
        block_offsets(const_cast<void*>(block_offsets_)),
        ld(ld_), steps_k(0) {}

  // Constructor for dense matrix.
  CUTLASS_HOST_DEVICE
  Op(void const *data_, int ld_)
      : data(const_cast<void*>(data_)),
        offsets(nullptr),
        indices(nullptr),
        block_offsets(nullptr),
        ld(ld_), steps_k(0) {}
};

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_OP_H_
