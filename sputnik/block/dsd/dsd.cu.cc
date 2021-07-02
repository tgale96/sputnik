#include <iostream>

#include "sputnik/block/dsd/dsd.h"
#include "sputnik/block/dsd/cutlass/dsd.h"
#include "sputnik/block/dsd/sm80/dsd.h"

namespace sputnik {
namespace block {

cudaError_t Matmul(
    const BlockMatrix a, bool transpose_a,
    const Matrix b, bool transpose_b,
    Matrix c, cudaStream_t stream) {
  // TODO(tgale): Figure out a way to identify what platform
  // we're going to issue to and launch the appropriate kernel.
  //
  // TODO(tgale): Figure out a way to tune across implementations
  // and constraints.
  //
  // We can achieve both of these goals by registering all
  // kernels available, filtering based on problem properties
  // and then auto-tuning and caching the results.
  MatmulShape shape(a, transpose_a, b, transpose_b);
  if (a.block_size == BlockSize::k32) {
    CHECK_EQ(transpose_a, false) << "Not yet supported for block_size == 32.";
    return sm80::Dsd(shape.m, shape.k, shape.n, a.nonzeros, AsInt(a.block_size),
		     (half*)a.data, (int*)a.offsets, (short*)a.indices,
                     (half*)b.data, transpose_b, (half*)c.data, stream);
  } else if (a.block_size == BlockSize::k128) {
    return cutlass::Matmul(a, transpose_a, b, transpose_b, c, stream);
  }
  return cudaErrorNotSupported;
}


cudaError_t MatmulEx(
    const BlockMatrix a, bool transpose_a,
    const Matrix b, bool transpose_b,
    Matrix c, cudaStream_t stream) {
  BlockMatrix acp = a;
  acp.create_metadata = false;
  return Matmul(acp, transpose_a, b, transpose_b, c, stream);
}
  
}  // namespace block
}  // namespace sputnik
