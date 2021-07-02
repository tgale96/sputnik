#ifndef THIRD_PARTY_SPUTNIK_BLOCK_DSD_DSD_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_DSD_DSD_H_

#include "sputnik/cuda_utils.h"
#include "sputnik/block/arguments.h"

namespace sputnik {
namespace block {

cudaError_t Matmul(const BlockMatrix a,
                   bool transpose_a,
                   const Matrix b,
                   bool transpose_b,
                   Matrix c,
                   cudaStream_t stream);

cudaError_t MatmulEx(const BlockMatrix a,
		     bool transpose_a,
		     const Matrix b,
		     bool transpose_b,
		     Matrix c,
		     cudaStream_t stream); 

}  // namespace block
}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_DSD_DSD_H_
