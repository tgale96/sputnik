#ifndef SPUTNIK_BLOCK_BITMASK_BITMASK_H_
#define SPUTNIK_BLOCK_BITMASK_BITMASK_H_

#include "sputnik/block/arguments.h"
#include "sputnik/cuda_utils.h"

namespace sputnik {
namespace block {

cudaError_t Bitmask(BlockMatrix a, cudaStream_t stream);

}
}  // namespaces sputnik

#endif  // SPUTNIK_BLOCK_BITMASK_BITMASK_H_
