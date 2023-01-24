#include <algorithm>
#include <numeric>
#include <vector>

#include "sputnik/block/transpose/transpose.h"
#include "sputnik/logging.h"

namespace sputnik {
namespace block {

template <typename T>
std::vector<int> Argsort(const std::vector<T> &x) {
  std::vector<int> out(x.size());
  std::iota(out.begin(), out.end(), 0);

  std::stable_sort(out.begin(), out.end(),
                   [&x](int a, int b) { return x[a] < x[b]; });
  return out;
}

std::vector<short> RowIndices(
    const std::vector<int> &offsets,
    int nnz, int block_size) {
  std::vector<short> out(nnz);
  for (int i = 0; i < offsets.size() - 1; ++i) {
    int start = offsets[i];
    int end = offsets[i + 1];
    for (int off = start; off < end; ++off) {
      out[off] = i;
    }
  }
  return out;
}

template <typename T>
std::vector<T> Gather(const std::vector<T> &data, const std::vector<int> &idxs) {
  SPUTNIK_CHECK_EQ(data.size(), idxs.size());
  std::vector<T> out(data.size());
  for (int i = 0; i < data.size(); ++i) {
    out[i] = data[idxs[i]];
  }
  return out;
}

std::vector<int> Iota(int start, int end) {
  std::vector<int> out(end - start);
  std::iota(out.begin(), out.end(), start);
  return out;
}

std::vector<int> Histogram(const std::vector<short> &x, int n) {
  std::vector<int> out(n, 0);
  for (const short &i : x) {
    ++out[(int)i];
  }
  return out;
}

std::vector<int> Cumsum(const std::vector<int> &x) {
  std::vector<int> out(x.size() + 1);
  out[0] = 0;
  for (int i = 0; i < x.size(); ++i) {
    out[i + 1] = out[i] + x[i];
  }
  return out;
}

// TODO(tgale): Replace this PoC with device kernels.
cudaError_t Transpose(BlockMatrix a, cudaStream_t stream) {
  // Copy the meta-data to the host.
  const int kBlockSize = AsInt(a.block_size);
  const int kBlockRows = a.rows / kBlockSize;
  const int kBlockCols = a.cols / kBlockSize;
  const int kBlocks = a.nonzeros / (kBlockSize * kBlockSize);
  std::vector<int> offsets(kBlockRows + 1);
  std::vector<short> indices(kBlocks);

  CUDA_CALL(cudaMemcpyAsync(
      offsets.data(), a.offsets,
      offsets.size() * sizeof(int),
      cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaMemcpyAsync(
      indices.data(), a.indices,
      indices.size() * sizeof(short),
      cudaMemcpyDeviceToHost, stream));

  // Sort row indices by column indices to get the transposed
  // matrix's column indices.
  std::vector<int> gather_indices = Argsort(indices);
  std::vector<short> row_indices = RowIndices(
      offsets, kBlocks, kBlockSize);

  std::vector<short> indices_t = Gather(row_indices, gather_indices);

  // Sort block offsets by column indices to get the transposed
  // matrix's block locations for each block row.
  //
  // NOTE: We need to use 32-bit precision for these offsets.
  std::vector<int> block_offsets = Iota(0, kBlocks);
  std::vector<int> block_offsets_t = Gather(block_offsets, gather_indices);

  // Calculate the transposed matrix's offsets.
  std::vector<int> nnz_per_column = Histogram(indices, kBlockCols);
  std::vector<int> offsets_t = Cumsum(nnz_per_column);

  // Make sure the outputs are allocated.
  SPUTNIK_CHECK(a.offsets_t);
  SPUTNIK_CHECK(a.indices_t);
  SPUTNIK_CHECK(a.block_offsets);

  // Copy the results to the device.
  CUDA_CALL(cudaMemcpyAsync(
      a.offsets_t, offsets_t.data(),
      offsets_t.size() * sizeof(int),
      cudaMemcpyHostToDevice, stream));
  CUDA_CALL(cudaMemcpyAsync(
      a.indices_t, indices_t.data(),
      indices_t.size() * sizeof(short),
      cudaMemcpyHostToDevice, stream));
  CUDA_CALL(cudaMemcpyAsync(
      a.block_offsets, block_offsets_t.data(),
      block_offsets_t.size() * sizeof(int),
      cudaMemcpyHostToDevice, stream));
  return cudaGetLastError();
}

}  // namespace block
}  // namespaces sputnik
