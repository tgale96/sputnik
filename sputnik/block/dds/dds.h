#ifndef SPUTNIK_BLOCK_DDS_DDS_H_
#define SPUTNIK_BLOCK_DDS_DDS_H_

#include "sputnik/cuda_utils.h"
#include "sputnik/block/arguments.h"

namespace sputnik {
namespace block {

cudaError_t Matmul(const Matrix a,
                   bool transpose_a,
                   const BlockMatrix b,
                   bool transpose_b,
                   Matrix c,
                   cudaStream_t stream);

cudaError_t MatmulEx(const Matrix a,
		     bool transpose_a,
		     const BlockMatrix b,
		     bool transpose_b,
		     Matrix c,
		     cudaStream_t stream); 

}  // namespace block
}  // namespace sputnik

#endif  // SPUTNIK_BLOCK_DDS_DDS_H_
