#ifndef SPUTNIK_BLOCK_BITMASK_BITMASK_H_
#define SPUTNIK_BLOCK_BITMASK_BITMASK_H_

#include "sputnik/block/arguments.h"
#include "sputnik/block/bitmask/bit_matrix.h"
#include "sputnik/cuda_utils.h"

namespace sputnik {
namespace block {

cudaError_t Bitmask(BlockMatrix m, cudaStream_t stream);

// TODO(tgale): Break these (and the transpose) helpers
// out into functions that give us the allocation size.
// Then the user can do the allocation themselves.
inline void AllocateBitmaskBuffers(BlockMatrix &m) {
  bool trans = m.offsets_t != nullptr;
  int block_size = AsInt(m.block_size);
  int block_rows = (trans ? m.cols : m.rows) / block_size;
  int block_cols = (trans ? m.rows : m.cols) / block_size;
  size_t bytes = BitMatrix::SizeInBytes(block_rows, block_cols);
  CUDA_CALL(cudaMalloc(&m.bitmask, bytes));
}

inline void FreeBitmaskBuffers(BlockMatrix &m) {
  if (m.bitmask != nullptr) {
    CUDA_CALL(cudaFree(m.bitmask));
  }
}

}  // namespace block
}  // namespace sputnik

#endif  // SPUTNIK_BLOCK_BITMASK_BITMASK_H_
