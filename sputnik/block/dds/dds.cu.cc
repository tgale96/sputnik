#include "sputnik/block/dds/dds.h"
#include "sputnik/block/dds/cutlass/dds.h"

namespace sputnik {
namespace block {

cudaError_t Matmul(
    const Matrix a, bool transpose_a,
    const BlockMatrix b, bool transpose_b,
    Matrix c, cudaStream_t stream) {
  if (b.block_size == BlockSize::k128) {
    return cutlass::Matmul(a, transpose_a, b, transpose_b, c, stream);
  }
  return cudaErrorNotSupported;
}

cudaError_t MatmulEx(
    const Matrix a, bool transpose_a,
    const BlockMatrix b, bool transpose_b,
    Matrix c, cudaStream_t stream) {
  BlockMatrix bcp = b;
  bcp.create_metadata = false;
  return Matmul(a, transpose_a, bcp, transpose_b, c, stream);
}  

}  // namespace block
}  // namespace sputnik
