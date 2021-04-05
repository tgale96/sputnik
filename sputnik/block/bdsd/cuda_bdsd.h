#ifndef THIRD_PARTY_SPUTNIK_BLOCK_CUDA_BDSD_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_CUDA_BDSD_H_

#include "sputnik/cuda_utils.h"

namespace sputnik {
namespace block {
  
cudaError_t CudaBdsd(int m, int k, int n,
		     int nonzeros, int block_dim,
		     const half* __restrict__ values,
		     const int* __restrict__ row_offsets,
		     const short* __restrict__ column_indices,
		     const half* __restrict__ dense_matrix,
		     half* __restrict__ output_matrix,
		     cudaStream_t stream);

}  // namespace block
}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_CUDA_BDSD_H_
