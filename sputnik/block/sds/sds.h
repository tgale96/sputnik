#ifndef THIRD_PARTY_SPUTNIK_BLOCK_SDS_SDS_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_SDS_SDS_H_

#include "sputnik/cuda_utils.h"
#include "sputnik/block/arguments.h"

namespace sputnik {
namespace block {

cudaError_t Matmul(const Matrix a,
                   bool transpose_a,
                   const BlockMatrix b,
                   bool transpose_b,
                   BlockMatrix c,
                   cudaStream_t stream);

cudaError_t MatmulEx(const Matrix a,
		     bool transpose_a,
		     const BlockMatrix b,
		     bool transpose_b,
		     BlockMatrix c,
		     cudaStream_t stream); 

}  // namespace block
}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_SDS_SDS_H_
