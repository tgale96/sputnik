#include <iostream>

#include "sputnik/block/bdsd/cuda_bdsd.h"
#include "sputnik/block/bdsd/sm80/cuda_bdsd.h"

namespace sputnik {
namespace block {
  
cudaError_t CudaBdsd(int m, int k, int n,
		     int nonzeros, int block_size,
		     const half* __restrict__ values,
		     const int* __restrict__ offsets,
		     const short* __restrict__ indices,
		     const half* __restrict__ dense_matrix,
		     half* __restrict__ output_matrix,
		     cudaStream_t stream) {
  // TODO(tgale): Figure out a way to identify what platform
  // we're going to issue to and launch the appropriate kernel.
  return sm80::CudaBdsd(m, k, n, nonzeros, block_size,
			values, offsets, indices,
			dense_matrix, output_matrix,
			stream);
  // return cudaErrorNotYetImplemented;
}

}  // namespace block
}  // namespace sputnik
