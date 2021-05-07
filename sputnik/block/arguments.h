#ifndef SPUTNIK_BLOCK_ARGUMENTS_H_
#define SPUTNIK_BLOCK_ARGUMENTS_H_

#include "glog/logging.h"

namespace sputnik {
namespace block {

enum class BlockSize {
  kNone = 0,
  k16 = 16,
  k32 = 32,
  k64 = 64,
  k128 = 128,
};

inline int AsInt(BlockSize b) {
  if (b == BlockSize::k16) {
    return 16;
  } else if (b == BlockSize::k32) {
    return 32;
  } else if (b == BlockSize::k64) {
    return 64;
  } else if (b == BlockSize::k128) {
    return 128;
  }
  return 0;
}

inline BlockSize AsBlockSize(int b) {
  if (b == 128) {
    return BlockSize::k128;
  } else if (b == 64) {
    return BlockSize::k64;
  } else if (b == 32) {
    return BlockSize::k32;
  } else if (b == 16) {
    return BlockSize::k16;
  }
  LOG(FATAL) << "Invalid block size.";
  return BlockSize::kNone;
}

struct BlockMatrix {
  int rows, cols, nonzeros;
  BlockSize block_size;

  void *data;
  void *offsets;
  void *indices;

  void *offsets_t;
  void *indices_t;
  void *block_offsets;

  BlockMatrix(int rows_,
              int cols_,
              BlockSize block_size_,
              int nonzeros_,
              void const *data_,
              void const *offsets_,
              void const *indices_)
      : rows(rows_), cols(cols_),
        block_size(block_size_),
        nonzeros(nonzeros_),
        data(const_cast<void*>(data_)),
        offsets(const_cast<void*>(offsets_)),
        indices(const_cast<void*>(indices_)),
        offsets_t(nullptr),
        indices_t(nullptr),
        block_offsets(nullptr) {}

  BlockMatrix(int rows_,
              int cols_,
              BlockSize block_size_,
              int nonzeros_,
              void const *data_,
              void const *offsets_,
              void const *indices_,
              void const *offsets_t_,
              void const *indices_t_,
              void const *block_offsets_)
      : rows(rows_), cols(cols_),
        block_size(block_size_),
        nonzeros(nonzeros_),
        data(const_cast<void*>(data_)),
        offsets(const_cast<void*>(offsets_)),
        indices(const_cast<void*>(indices_)),
        offsets_t(const_cast<void*>(offsets_t_)),
        indices_t(const_cast<void*>(indices_t_)),
        block_offsets(const_cast<void*>(block_offsets_)) {}
};

struct Matrix {
  int rows, cols;
  void *data;

  Matrix(int rows_, int cols_, void const *data_)
      : rows(rows_), cols(cols_),
        data(const_cast<void*>(data_)) {}
};

}  // namespace block
}  // namespace sputnik

#endif  // SPUTNIK_BLOCK_ARGUMENTS_H_
