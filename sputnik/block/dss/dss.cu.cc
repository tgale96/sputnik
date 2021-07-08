#include <iostream>

#include "sputnik/block/dss/dss.h"
#include "sputnik/block/dss/cutlass/dss.h"

namespace sputnik {
namespace block {

cudaError_t Matmul(
    const BlockMatrix a, bool transpose_a,
    const BlockMatrix b, bool transpose_b,
    Matrix c, cudaStream_t stream) {
  return cutlass::Matmul(a, transpose_a, b, transpose_b, c, stream);
}

cudaError_t MatmulEx(
    const BlockMatrix a, bool transpose_a,
    const BlockMatrix b, bool transpose_b,
    Matrix c, cudaStream_t stream) {
  BlockMatrix acp = a, bcp = b;
  acp.create_metadata = false;
  bcp.create_metadata = false;
  return cutlass::Matmul(acp, transpose_a, bcp, transpose_b, c, stream);
}

}  // namespace block
}  // namespace sputnik
