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

  // Bitmask representation of the sparse matrix. Useful for
  // sparse * sparse products.
  void * bitmask;

  int ld;

  // The number of nonzeros in the matrix.
  int nnz;
  void *row_indices;

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
        bitmask(nullptr),
        ld(ld_),
	nnz(-1) {}

  // Constructor for non-transposed sparse matrix as
  // output of the kernel.
  CUTLASS_HOST_DEVICE
  Op(void const *data_,
     void const *offsets_,
     void const *indices_,
     void const *row_indices_,
     int ld_,
     int nnz_)
      : data(const_cast<void*>(data_)),
        offsets(const_cast<void*>(offsets_)),
        indices(const_cast<void*>(indices_)),
        block_offsets(nullptr),
        bitmask(nullptr),
        ld(ld_),
	nnz(nnz_),
	row_indices(const_cast<void*>(row_indices_)) {}

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
        bitmask(nullptr),
        ld(ld_),
	nnz(-1),
	row_indices(nullptr) {}

  // Constructor for dense matrix.
  CUTLASS_HOST_DEVICE
  Op(void const *data_, int ld_)
      : data(const_cast<void*>(data_)),
        offsets(nullptr),
        indices(nullptr),
        block_offsets(nullptr),
        bitmask(nullptr),
        ld(ld_),
	nnz(-1),
	row_indices(nullptr) {}

  // Constructor for sparse matrix with bitmask.
  CUTLASS_HOST_DEVICE
  Op(void const *data_,
     void const *offsets_,
     void const *indices_,
     void const *block_offsets_,
     void const *bitmask_,
     int ld_)
      : data(const_cast<void*>(data_)),
        offsets(const_cast<void*>(offsets_)),
        indices(const_cast<void*>(indices_)),
        block_offsets(const_cast<void*>(block_offsets_)),
        bitmask(const_cast<void*>(bitmask_)),
        ld(ld_),
	nnz(-1),
	row_indices(nullptr) {}
};

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_OP_H_
