#include "sputnik/block/sdd/sdd.h"
#include "sputnik/block/sdd/cutlass/sdd.h"

namespace sputnik {
namespace block {

cudaError_t Matmul(
    const Matrix a, bool transpose_a,
    const Matrix b, bool transpose_b,
    BlockMatrix c, cudaStream_t stream) {
  if (c.block_size == BlockSize::k128) {
    return cutlass::Matmul(a, transpose_a, b, transpose_b, c, stream);
  }
  return cudaErrorNotSupported;
}

}  // namespace block
}  // namespace sputnik
