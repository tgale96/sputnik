#ifndef THIRD_PARTY_SPUTNIK_BLOCK_MATRIX_UTILS_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_MATRIX_UTILS_H_

#include "sputnik/matrix_utils.h"
#include "sputnik/block/arguments.h"

namespace sputnik {

// Prototype for CudaBlockSparseMatrix.
template <typename Value>
class CudaBlockSparseMatrix;

class BlockSparseMatrix : public SparseMatrix {
 public:
  // NOTE: All matrix properties are currently in terms of
  // individual nonzero values, not blocks. This is also
  // true for all matrix meta-data.
  //
  // TODO(tgale): Consider altering this depending on how
  // we want to expose the meta-data. It would be most
  // useful to match cuSPARSE.
  BlockSparseMatrix(int rows, int columns, int nonzeros, int block_dim,
                    ElementDistribution weight_distribution,
                    absl::BitGen* generator, int pad_rows_to=4);

  template <typename T>
  explicit BlockSparseMatrix(const CudaBlockSparseMatrix<T>& sparse_matrix);

  ~BlockSparseMatrix() = default;

  BlockSparseMatrix(const BlockSparseMatrix&) = delete;
  BlockSparseMatrix& operator=(const BlockSparseMatrix&) = delete;
  BlockSparseMatrix(BlockSparseMatrix&&) = delete;
  BlockSparseMatrix& operator=(BlockSparseMatrix&&) = delete;

  int BlockDim() const { return block_dim_; }

 private:
  int block_dim_;
};

template <typename Value>
class CudaBlockSparseMatrix : public CudaSparseMatrix<Value> {
 public:
  CudaBlockSparseMatrix(int rows, int columns, int nonzeros, int block_dim,
                        ElementDistribution weight_distribution,
                        absl::BitGen* generator, int pad_rows_to=4);

  explicit CudaBlockSparseMatrix(const BlockSparseMatrix& sparse_matrix);

  ~CudaBlockSparseMatrix() = default;

  CudaBlockSparseMatrix(const CudaBlockSparseMatrix&) = delete;
  CudaBlockSparseMatrix& operator=(const CudaBlockSparseMatrix&) = delete;
  CudaBlockSparseMatrix(CudaBlockSparseMatrix&&) = delete;
  CudaBlockSparseMatrix& operator=(CudaBlockSparseMatrix&&) = delete;

  typedef typename Value2Index<Value>::Index Index;

  int BlockDim() const { return block_dim_; }
 private:
  int block_dim_;

  void InitFromBlockSparseMatrix(const BlockSparseMatrix& sparse_matrix);
};

/**
 * @brief Helper to load sparse matrix values into a std::vector.
 */
inline std::vector<float> ToVector(const BlockSparseMatrix& sparse_matrix) {
  int num = sparse_matrix.NumElementsWithPadding() *
            sparse_matrix.BlockDim() * sparse_matrix.BlockDim();
  std::vector<float> out(sparse_matrix.Values(), sparse_matrix.Values() + num);
  return out;
}

inline Matrix ToMatrix(const BlockSparseMatrix &x) {
  size_t n = x.Rows() * x.Columns();
  std::vector<float> out(n, 0);
  
  const int* ro = x.RowOffsets();
  const int* co = x.ColumnIndices();
  
  int bd = x.BlockDim();
  int rows = x.Rows();
  int cols = x.Columns();
  int brows = rows / bd;
  for (int i = 0; i < brows; ++i) {
    for (int l = ro[i]; l < ro[i + 1]; l += bd * bd) {
      int idx_offset = l / (bd * bd);
      int j = co[idx_offset];

      for (int br = 0; br < bd; ++br) {
	for (int bc = 0; bc < bd; ++bc) {
	  float v = x.Values()[l + br * bd + bc];
	  int row_idx = i * bd + br;
	  int col_idx = j + bc;
	  out[row_idx * cols + col_idx] = v;
	}
      }

    }
  }

  Matrix ret(rows, cols);
  std::memcpy(ret.Values(), out.data(), n*sizeof(float));
  return ret;
}

template <typename T>
inline block::BlockMatrix Arg(const CudaBlockSparseMatrix<T> &m) {
  block::BlockMatrix out(m.Rows(), m.Columns(),
                         block::AsBlockSize(m.BlockDim()),
                         m.NumElementsWithPadding(),
                         (const void*)m.Values(),
                         (const void*)m.RowOffsets(),
                         (const void*)m.ColumnIndices());
  return out;
}

template <typename T>
inline block::Matrix Arg(const CudaMatrix<T> &m) {
  block::Matrix out(m.Rows(), m.Columns(), (const void*)m.Values());
  return out;
}

}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_MATRIX_UTILS_H_
