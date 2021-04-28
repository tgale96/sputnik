#include <iostream>

#include "sputnik/block/dsd/dsd.h"
#include "sputnik/block/dsd/cutlass/dsd.h"
#include "sputnik/block/dsd/sm80/dsd.h"

namespace sputnik {
namespace block {
  
cudaError_t Dsd(int m, int k, int n,
		int nonzeros, int block_dim,
		const half* a,
		const int* offsets_a,
		const short* indices_a,
		const half* b, bool transpose_b,
		half* c, cudaStream_t stream) {  
  // TODO(tgale): Figure out a way to identify what platform
  // we're going to issue to and launch the appropriate kernel.
  //
  // TODO(tgale): Figure out a way to tune across implementations
  // and constraints.
  //
  // We can achieve both of these goals by registering all
  // kernels available, filtering based on problem properties
  // and then auto-tuning and caching the results.
  if (block_dim == 32) {
    return sm80::Dsd(m, k, n, nonzeros, block_dim,
		     a, offsets_a, indices_a,
		     b, transpose_b, c, stream);
  } else if (block_dim == 128) {
    // TODO(tgale): Expose the transpose operation for rhs.
    return cutlass::Dsd(m, k, n, nonzeros, block_dim,
			a, offsets_a, indices_a,
			b, transpose_b, c, stream);
  }
  return cudaErrorNotYetImplemented;
}

}  // namespace block
}  // namespace sputnik
