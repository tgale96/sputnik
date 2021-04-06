#include "sputnik/block/bdsd/sm80/cuda_bdsd.h"

namespace sputnik {
namespace block {
namespace sm80 {
      
cudaError_t CudaBdsd(
    int m, int k, int n,
    int nonzeros, int block_dim,
    const half* __restrict__ values,
    const int* __restrict__ row_offsets,
    const short* __restrict__ column_indices,
    const half* __restrict__ dense_matrix,
    half* __restrict__ output_matrix,
    cudaStream_t stream) {
  // TODO(tgale): Add a proper kernel selector.
  if (can_launch_bdsd_b32_m32n128k32_h8_h8(m, k, n, nonzeros, block_dim) && false) {
    return launch_bdsd_b32_m32n128k32_h8_h8(
        m, k, n, nonzeros, block_dim,
	values, row_offsets, column_indices,
	dense_matrix, output_matrix, stream);
  } else if (can_launch_bdsd_b32_m32n64k32_h8_h8(m, k, n, nonzeros, block_dim)) {
    return launch_bdsd_b32_m32n64k32_h8_h8(
        m, k, n, nonzeros, block_dim,
	values, row_offsets, column_indices,
	dense_matrix, output_matrix, stream);    
  } else {
    LOG(FATAL) << "No compatible kernel for problem. m/n/k/b = " <<
      m << "/" << n << "/" << k << "/" << block_dim << ".";
  }
  return cudaGetLastError();
}

}  // namespace sm80
}  // namespace block
}  // namespace sputnik
