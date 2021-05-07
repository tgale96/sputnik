#include <iostream>

#include "sputnik/block/dsd/dsd.h"
#include "sputnik/block/dsd/cutlass/dsd.h"
#include "sputnik/block/dsd/sm80/dsd.h"

namespace sputnik {
namespace block {

cudaError_t Matmul(const BlockMatrix a,
                   bool transpose_a,
                   const Matrix b,
                   bool transpose_b,
                   Matrix c,
                   cudaStream_t stream) {
  // TODO(tgale): Figure out a way to identify what platform
  // we're going to issue to and launch the appropriate kernel.
  //
  // TODO(tgale): Figure out a way to tune across implementations
  // and constraints.
  //
  // We can achieve both of these goals by registering all
  // kernels available, filtering based on problem properties
  // and then auto-tuning and caching the results.
  CHECK_EQ(transpose_a, false) << "Not yet supported";
  int m = transpose_a ? a.cols : a.rows;
  int k = transpose_a ? a.rows : a.cols;
  int n = transpose_b ? b.rows : b.cols;
  if (a.block_size == BlockSize::k32) {
    return sm80::Dsd(m, k, n, a.nonzeros, AsInt(a.block_size),
		     (half*)a.data, (int*)a.offsets, (short*)a.indices,
                     (half*)b.data, transpose_b, (half*)c.data, stream);
  } else if (a.block_size == BlockSize::k128) {
    return cutlass::Dsd(m, k, n, a.nonzeros, AsInt(a.block_size),
                        (half*)a.data, (int*)a.offsets, (short*)a.indices,
                        (half*)b.data, transpose_b, (half*)c.data, stream);
  }
  return cudaErrorNotYetImplemented;
}

}  // namespace block
}  // namespace sputnik
