#ifndef SPUTNIK_BLOCK_ARGUMENTS_H_
#define SPUTNIK_BLOCK_ARGUMENTS_H_

#include <iostream>

#include "sputnik/cuda_utils.h"
#include "sputnik/test_utils.h"
#include "sputnik/logging.h"

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
  SPUTNIK_LOG(FATAL) << "Invalid block size.";
  return BlockSize::kNone;
}

struct BlockMatrix {
  int rows, cols, nonzeros;
  BlockSize block_size;

  void *data;
  void *offsets;
  void *indices;

  // Data for transposed iteration.
  void *offsets_t;
  void *indices_t;
  void *block_offsets;

  // Data for sparse output.
  void *row_indices;

  // Data for sparse * sparse products.
  void *bitmask;

  // TODO(tgale): Get rid of this and plumb the APIs.
  bool create_metadata = true;

  BlockMatrix(int rows_,
              int cols_,
              BlockSize block_size_,
              int nonzeros_,
              void const *data_,
              void const *offsets_,
              void const *indices_)
      : rows(rows_), cols(cols_),
        nonzeros(nonzeros_),
        block_size(block_size_),
        data(const_cast<void*>(data_)),
        offsets(const_cast<void*>(offsets_)),
        indices(const_cast<void*>(indices_)),
        offsets_t(nullptr),
        indices_t(nullptr),
        block_offsets(nullptr),
	row_indices(nullptr),
        bitmask(nullptr) {}

  BlockMatrix(int rows_,
              int cols_,
              BlockSize block_size_,
              int nonzeros_,
              void const *data_,
              void const *offsets_,
              void const *indices_,
	      void const *row_indices_)
      : rows(rows_), cols(cols_),
        nonzeros(nonzeros_),
        block_size(block_size_),
        data(const_cast<void*>(data_)),
        offsets(const_cast<void*>(offsets_)),
        indices(const_cast<void*>(indices_)),
        offsets_t(nullptr),
        indices_t(nullptr),
        block_offsets(nullptr),
	row_indices(const_cast<void*>(row_indices_)),
        bitmask(nullptr) {}

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
        nonzeros(nonzeros_),
        block_size(block_size_),
        data(const_cast<void*>(data_)),
        offsets(const_cast<void*>(offsets_)),
        indices(const_cast<void*>(indices_)),
        offsets_t(const_cast<void*>(offsets_t_)),
        indices_t(const_cast<void*>(indices_t_)),
        block_offsets(const_cast<void*>(block_offsets_)),
	row_indices(nullptr),
        bitmask(nullptr) {}

  BlockMatrix(int rows_,
              int cols_,
              BlockSize block_size_,
              int nonzeros_,
              void const *data_,
              void const *offsets_,
              void const *indices_,
              void const *offsets_t_,
              void const *indices_t_,
              void const *block_offsets_,
              void const *bitmask_)
      : rows(rows_), cols(cols_),
        nonzeros(nonzeros_),
        block_size(block_size_),
        data(const_cast<void*>(data_)),
        offsets(const_cast<void*>(offsets_)),
        indices(const_cast<void*>(indices_)),
        offsets_t(const_cast<void*>(offsets_t_)),
        indices_t(const_cast<void*>(indices_t_)),
        block_offsets(const_cast<void*>(block_offsets_)),
	row_indices(nullptr),
        bitmask(const_cast<void*>(bitmask_)) {}
};

struct Matrix {
  int rows, cols;
  void *data;

  Matrix(int rows_, int cols_, void const *data_)
      : rows(rows_), cols(cols_),
        data(const_cast<void*>(data_)) {}
};

struct MatmulShape {
  int m, n, k, lda, ldb, ldc;

  MatmulShape(
      int m_, int n_, int k_,
      bool transpose_a, bool transpose_b)
      : m(m_), n(n_), k(k_) {
    lda = transpose_a ? m : k;
    ldb = transpose_b ? k : n;
    ldc = n;
  }

  MatmulShape(const BlockMatrix a, bool transpose_a,
              const Matrix b, bool transpose_b) {
    m = transpose_a ? a.cols : a.rows;
    k = transpose_a ? a.rows : a.cols;
    n = transpose_b ? b.rows : b.cols;
    lda = transpose_a ? m : k;
    ldb = transpose_b ? k : n;
    ldc = n;
  }

  MatmulShape(const Matrix a, bool transpose_a,
              const BlockMatrix b, bool transpose_b) {
    m = transpose_a ? a.cols : a.rows;
    k = transpose_a ? a.rows : a.cols;
    n = transpose_b ? b.rows : b.cols;
    lda = transpose_a ? m : k;
    ldb = transpose_b ? k : n;
    ldc = n;
  }

  MatmulShape(const Matrix a, bool transpose_a,
              const Matrix b, bool transpose_b) {
    m = transpose_a ? a.cols : a.rows;
    k = transpose_a ? a.rows : a.cols;
    n = transpose_b ? b.rows : b.cols;
    lda = transpose_a ? m : k;
    ldb = transpose_b ? k : n;
    ldc = n;
  }

  MatmulShape(const BlockMatrix a, bool transpose_a,
              const BlockMatrix b, bool transpose_b) {
    m = transpose_a ? a.cols : a.rows;
    k = transpose_a ? a.rows : a.cols;
    n = transpose_b ? b.rows : b.cols;
    lda = transpose_a ? m : k;
    ldb = transpose_b ? k : n;
    ldc = n;
  }
};

template <typename TypeA, typename TypeB, typename TypeC>
inline bool ValidMatmul(
    const TypeA a, bool transpose_a,
    const TypeB b, bool transpose_b, TypeC c) {
  MatmulShape shape(a, transpose_a, b, transpose_b);

  bool valid = true;
  valid &= (transpose_a ? a.cols : a.rows) == shape.m;
  valid &= (transpose_a ? a.rows : a.cols) == shape.k;
  valid &= (transpose_b ? b.cols : b.rows) == shape.k;
  valid &= (transpose_b ? b.rows : b.cols) == shape.n;
  valid &= c.rows == shape.m;
  valid &= c.cols == shape.n;
  return valid;
}

inline void AllocateTransposeBuffers(BlockMatrix &a) {
  const int kBlockCols = a.cols / AsInt(a.block_size);
  size_t offset_bytes = (kBlockCols + 1) * sizeof(int);
  CUDA_CALL(cudaMalloc(&a.offsets_t, offset_bytes));

  const int kBlockSize = AsInt(a.block_size);
  const int kNonzeroBlocks = a.nonzeros / (kBlockSize * kBlockSize);
  size_t indices_bytes = kNonzeroBlocks * sizeof(short);
  CUDA_CALL(cudaMalloc(&a.indices_t, indices_bytes));

  size_t block_offsets_bytes = kNonzeroBlocks * sizeof(int);
  CUDA_CALL(cudaMalloc(&a.block_offsets, block_offsets_bytes));
}

inline void FreeTransposeBuffers(BlockMatrix &a) {
  if (a.offsets_t != nullptr) {
    CUDA_CALL(cudaFree(a.offsets_t));
  }
  if (a.indices_t != nullptr) {
    CUDA_CALL(cudaFree(a.indices_t));
  }
  if (a.block_offsets != nullptr) {
    CUDA_CALL(cudaFree(a.block_offsets));
  }
}

inline void AllocateRowIndicesBuffer(BlockMatrix &a) {
  const int kBlockSize = AsInt(a.block_size);
  size_t bytes = a.nonzeros / (kBlockSize * kBlockSize) * sizeof(short);
  CUDA_CALL(cudaMalloc(&a.row_indices, bytes));
}

inline void FreeRowIndicesBuffer(BlockMatrix &a) {
  if (a.row_indices != nullptr) {
    CUDA_CALL(cudaFree(a.row_indices));
  }
}

}  // namespace block
}  // namespace sputnik

#endif  // SPUTNIK_BLOCK_ARGUMENTS_H_
