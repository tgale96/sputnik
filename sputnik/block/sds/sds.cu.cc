#include "sputnik/block/sds/sds.h"
#include "sputnik/block/sds/cutlass/sds.h"

namespace sputnik {
namespace block {

cudaError_t Matmul(
    const BlockMatrix a, bool transpose_a,
    const Matrix b, bool transpose_b,
    BlockMatrix c, cudaStream_t stream) {
  if (c.block_size == BlockSize::k128) {
    return cutlass::Matmul(a, transpose_a, b, transpose_b, c, stream);
  }
  return cudaErrorNotSupported;
}

cudaError_t MatmulEx(
    const BlockMatrix a, bool transpose_a,
    const Matrix b, bool transpose_b,
    BlockMatrix c, cudaStream_t stream) {
  BlockMatrix acp = a;
  acp.create_metadata = false;
  return Matmul(acp, transpose_a, b, transpose_b, c, stream);
}  

}  // namespace block
}  // namespace sputnik
