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
  if (a.block_size == BlockSize::k128) {
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
