#ifndef SPUTNIK_BLOCK_ROW_INDICES_ROW_INDICES_H_
#define SPUTNIK_BLOCK_ROW_INDICES_ROW_INDICES_H_

#include "sputnik/block/arguments.h"
#include "sputnik/cuda_utils.h"

namespace sputnik {
namespace block {

cudaError_t RowIndices(BlockMatrix a, short* row_indices, cudaStream_t stream);

}  // namespace block
}  // namespaces sputnik

#endif  // SPUTNIK_BLOCK_ROW_INDICES_ROW_INDICES_H_
