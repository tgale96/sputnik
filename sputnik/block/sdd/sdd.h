#ifndef THIRD_PARTY_SPUTNIK_BLOCK_SDD_SDD_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_SDD_SDD_H_

#include "sputnik/cuda_utils.h"
#include "sputnik/block/arguments.h"

namespace sputnik {
namespace block {

cudaError_t Matmul(const Matrix a,
                   bool transpose_a,
                   const Matrix b,
                   bool transpose_b,
                   BlockMatrix c,
                   cudaStream_t stream);

}  // namespace block
}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_SDD_SDD_H_
