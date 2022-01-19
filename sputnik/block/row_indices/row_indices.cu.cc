#include "sputnik/block/row_indices/row_indices.h"

namespace sputnik {
namespace block {
namespace {

__global__ void RowIndicesKernel(int m, int block_size,
				 const int * __restrict__ offsets,
				 short * __restrict__ row_indices) {
  int m_index = blockIdx.x;
  if (m_index >= m) return;
  int row_offset = __ldg(offsets + m_index);
  int nonzeros = __ldg(offsets + m_index + 1) - row_offset;

  // Write the row index out for every sparse block.
  for (int idx = threadIdx.x; idx < nonzeros; idx += blockDim.x) {
    row_indices[row_offset + idx] = m_index;
  }
}
  
}  // namespace


cudaError_t RowIndices(BlockMatrix a, short* row_indices, cudaStream_t stream) {
  // TODO(tgale): Tune this.
  constexpr int kBlockWidth = 32;
  const int kBlockRows = a.rows / AsInt(a.block_size);
  dim3 grid_dim(kBlockRows);
  dim3 block_dim(kBlockWidth);

  RowIndicesKernel<<<grid_dim, block_dim, 0, stream>>>(kBlockRows,
						       AsInt(a.block_size),
						       (int*)a.offsets,
						       row_indices);
  return cudaGetLastError();
}

}  // namespace block
}  // namespaces sputnik
