#ifndef SPUTNIK_BLOCK_TRANSPOSE_TRANSPOSE_H_
#define SPUTNIK_BLOCK_TRANSPOSE_TRANSPOSE_H_

#include "sputnik/block/arguments.h"
#include "sputnik/cuda_utils.h"

namespace sputnik {
namespace block {

cudaError_t Transpose(BlockMatrix a, cudaStream_t stream);

}  // namespace block
}  // namespaces sputnik

#endif  // SPUTNIK_BLOCK_TRANSPOSE_TRANSPOSE_H_
