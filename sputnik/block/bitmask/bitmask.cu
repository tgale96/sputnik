#include "sputnik/block/bitmask/bitmask.h"
#include "sputnik/block/bitmask/bit_matrix.h"

namespace sputnik {
namespace block {

cudaError_t Bitmask(BlockMatrix m, cudaStream_t stream) {
  bool trans = m.offsets_t != nullptr;
  int *offsets_d = (int*)(trans ? m.offsets_t : m.offsets);
  short *indices_d = (short*)(trans ? m.indices_t : m.indices);

  // Block domain constants.
  int block_size = AsInt(m.block_size);
  int nonzero_blocks = m.nonzeros / (block_size * block_size);
  int block_rows = (trans ? m.cols : m.rows) / block_size;
  int block_cols = (trans ? m.rows : m.cols) / block_size;

  // Copy the meta-data from the device.
  std::vector<int> offsets(block_rows + 1);
  std::vector<short> indices(nonzero_blocks);

  CUDA_CALL(cudaMemcpyAsync(
      offsets.data(), offsets_d,
      offsets.size() * sizeof(int),
      cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaMemcpyAsync(
      indices.data(), indices_d,
      indices.size() * sizeof(short),
      cudaMemcpyDeviceToHost, stream));

  BitMatrix bmat(block_rows, block_cols);
  for (int i = 0; i < block_rows; ++i) {
    int start = offsets[i];
    int end = offsets[i + 1];
    for (int offset = start; offset < end; ++offset) {
      int j = indices[offset];
      bmat.Set(i, j);
    }
  }

  CUDA_CALL(cudaMemcpyAsync(
      m.bitmask, bmat.Data(), bmat.Bytes(),
      cudaMemcpyHostToDevice, stream));
  return cudaGetLastError();
}

}  // namespace block
}  // namespace sputnik
