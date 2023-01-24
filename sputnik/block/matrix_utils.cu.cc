#include <algorithm>

#include "sputnik/block/matrix_utils.h"

namespace sputnik {

BlockSparseMatrix::BlockSparseMatrix(
    int rows, int columns, int nonzeros, int block_dim,
    ElementDistribution weight_distribution,
    absl::BitGen* generator, int pad_rows_to,
    bool unordered_indices) {
  SPUTNIK_CHECK_EQ(rows % block_dim, 0);
  SPUTNIK_CHECK_EQ(columns % block_dim, 0);
  SPUTNIK_CHECK_EQ(nonzeros % (block_dim * block_dim), 0);

  // Save the matrix meta-data.
  block_dim_ = block_dim;
  rows_ = rows;
  columns_ = columns;
  nonzeros_ = nonzeros;
  weight_distribution_ = weight_distribution;
  row_swizzle_ = IDENTITY;
  pad_rows_to_ = pad_rows_to;

  SPUTNIK_CHECK_LE(pad_rows_to_, columns)
      << "Rows cannot be padded to more values than there are columns.";

  // Block domain matrix properties.
  int block_rows = rows / block_dim;
  int block_columns = columns / block_dim;
  int nonzero_blocks = nonzeros / (block_dim * block_dim);

  // Create some temporary host-side buffers to build the matrix in.
  // Note that we have to pad these buffers to account for potential
  // extra storage requirements for row padding.
  int padding_elements = std::max((pad_rows_to_ - 1) * block_rows, 0);
  std::vector<float> null_staging(nonzero_blocks + padding_elements);
  std::vector<int> row_offsets_staging(block_rows + 1);
  std::vector<int> column_indices_staging(nonzero_blocks + padding_elements);

  if (weight_distribution == RANDOM_UNIFORM) {
    MakeSparseMatrixRandomUniform(
        block_rows, block_columns, nonzero_blocks, null_staging.data(),
        row_offsets_staging.data(), column_indices_staging.data(),
        generator, pad_rows_to_);
  } else {
    // Verify that the number of nonzeros divides evenly into the
    // number of rows.
    SPUTNIK_CHECK_EQ(nonzero_blocks % block_rows, 0)
        << "The number of nonzeros must divide "
        << "evenly by the number of rows to "
        << "construct a PERFECT_UNIFORM matrix.";

    MakeSparseMatrixPerfectUniform(
        block_rows, block_columns, nonzero_blocks / block_rows,
        null_staging.data(), row_offsets_staging.data(),
        column_indices_staging.data(), generator);
  }

  // Figure out exactly how much storage we need for the padded matrices,
  // allocate the storage, and copy the matrices into our storage.
  num_elements_with_padding_ = row_offsets_staging[block_rows];

  column_indices_ = new int[num_elements_with_padding_];
  row_offsets_ = new int[block_rows + 1];

  // Copy the data into our allocated buffers.
  std::memcpy(column_indices_, column_indices_staging.data(),
              num_elements_with_padding_ * sizeof(int));
  std::memcpy(row_offsets_, row_offsets_staging.data(),
              (block_rows + 1) * sizeof(int));

  // Allocate storage for our swizzled row indices and set the values.
  row_indices_ = new int[block_rows];
  IdentityRowSwizzle(block_rows, row_offsets_, row_indices_);

  // Create the sparse block values.
  //
  // NOTE: Scale the nnz counter to be in terms of individual nonzeros.
  num_elements_with_padding_ *= block_dim * block_dim;
  values_ = new float[num_elements_with_padding_];
  for (int64_t i = 0; i < num_elements_with_padding_; ++i) {
    values_[i] = absl::Uniform<float>(*generator, -1, 1);
  }

  if (unordered_indices) {
    // If we want unordered indices, randomly shuffle each
    // block row of indices.
    for (int i = 0; i < block_rows; ++i) {
      int start = row_offsets_[i];
      int end = row_offsets_[i + 1];
      std::shuffle(column_indices_ + start, column_indices_ + end, *generator);
    }
  }
}

template <typename T>
BlockSparseMatrix::BlockSparseMatrix(
    const CudaBlockSparseMatrix<T> &sparse_matrix) {
  // Save the matrix meta-data.
  this->rows_ = sparse_matrix.Rows();
  this->columns_ = sparse_matrix.Columns();
  this->nonzeros_ = sparse_matrix.Nonzeros();
  this->pad_rows_to_ = sparse_matrix.PadRowsTo();
  this->num_elements_with_padding_ = sparse_matrix.NumElementsWithPadding();
  this->weight_distribution_ = sparse_matrix.WeightDistribution();
  this->row_swizzle_ = sparse_matrix.RowSwizzle();
  this->block_dim_ = sparse_matrix.BlockDim();

  // Block domain matrix properties.
  int block_rows = rows_ / block_dim_;
  int num_blocks = num_elements_with_padding_ / (block_dim_ * block_dim_);

  // Allocate temporary single-precision data on the GPU and
  // convert the matrix data to int/float.
  int *column_indices_gpu = nullptr;
  CUDA_CALL(cudaMalloc(&column_indices_gpu,
		       num_blocks * sizeof(int)));
  Convert(sparse_matrix.ColumnIndices(),
	  column_indices_gpu,
	  num_blocks);

  float *values_gpu = nullptr;
  CUDA_CALL(cudaMalloc(
      &values_gpu,
      num_elements_with_padding_ * sizeof(float)));
  Convert(sparse_matrix.Values(), values_gpu, num_elements_with_padding_);

  // Allocate storage for the matrix.
  column_indices_ = new int[num_blocks];
  row_offsets_ = new int[block_rows + 1];
  row_indices_ = new int[block_rows];
  values_ = new float[num_elements_with_padding_];

  // Copy the data from the device.
  CUDA_CALL(cudaMemcpy(
      column_indices_,
      column_indices_gpu,
      sizeof(int) * num_blocks,
      cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(
      row_offsets_,
      sparse_matrix.RowOffsets(),
      (block_rows + 1) * sizeof(int),
      cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(
      row_indices_,
      sparse_matrix.RowIndices(),
      block_rows * sizeof(int),
      cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(
      values_,
      values_gpu,
      num_elements_with_padding_ * sizeof(float),
      cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaStreamSynchronize(0));

  // Free the temporary storage.
  CUDA_CALL(cudaFree(column_indices_gpu));
  CUDA_CALL(cudaFree(values_gpu));
}

template <typename Value>
CudaBlockSparseMatrix<Value>::CudaBlockSparseMatrix(
    int rows, int columns, int nonzeros, int block_dim,
    ElementDistribution weight_distribution,
    absl::BitGen* generator, int pad_rows_to,
    bool unordered_indices) {
  BlockSparseMatrix sparse_matrix(
      rows, columns, nonzeros, block_dim,
      weight_distribution, generator, pad_rows_to,
      unordered_indices);
  InitFromBlockSparseMatrix(sparse_matrix);
}

template <typename Value>
CudaBlockSparseMatrix<Value>::CudaBlockSparseMatrix(
    const BlockSparseMatrix& sparse_matrix) {
  InitFromBlockSparseMatrix(sparse_matrix);
}

template <typename Value>
void CudaBlockSparseMatrix<Value>::InitFromBlockSparseMatrix(
    const BlockSparseMatrix& sparse_matrix) {
  // Copy the sparse matrix meta-data.
  this->rows_ = sparse_matrix.Rows();
  this->columns_ = sparse_matrix.Columns();
  this->nonzeros_ = sparse_matrix.Nonzeros();
  this->pad_rows_to_ = sparse_matrix.PadRowsTo();
  this->num_elements_with_padding_ = sparse_matrix.NumElementsWithPadding();
  this->weight_distribution_ = sparse_matrix.WeightDistribution();
  this->row_swizzle_ = sparse_matrix.RowSwizzle();
  this->block_dim_ = sparse_matrix.BlockDim();

  // Block domain properties.
  int block_rows = this->rows_ / block_dim_;
  int num_blocks = this->num_elements_with_padding_ / (block_dim_ * block_dim_);

  // Allocate memory on the GPU for our matrix.
  const int kNumValues = this->num_elements_with_padding_;
  float *values_float = nullptr;
  const int kNumIndices = num_blocks;
  int *column_indices_int = nullptr;
  CUDA_CALL(cudaMalloc(
      &values_float, sizeof(float) * kNumValues));
  CUDA_CALL(cudaMalloc(
      &column_indices_int, sizeof(int) * kNumIndices));
  CUDA_CALL(cudaMalloc(&this->row_offsets_, sizeof(int) * (block_rows + 1)));
  CUDA_CALL(cudaMalloc(&this->row_indices_, sizeof(int) * block_rows));

  // Copy the results to the GPU.
  CUDA_CALL(cudaMemcpy(values_float, sparse_matrix.Values(),
                       sizeof(float) * kNumValues,
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(column_indices_int, sparse_matrix.ColumnIndices(),
                       sizeof(int) * kNumIndices,
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(this->row_offsets_, sparse_matrix.RowOffsets(),
                       sizeof(int) * (block_rows + 1), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(this->row_indices_, sparse_matrix.RowIndices(),
                       sizeof(int) * block_rows, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  // Allocate memory for the values and indices in the target datatype.
  int vwidth = TypeUtils<Value>::kElementsPerScalar;
  CUDA_CALL(cudaMalloc(
      &this->values_, sizeof(Value) * kNumValues / vwidth));
  CUDA_CALL(cudaMalloc(
      &this->column_indices_, sizeof(Index) * kNumIndices / vwidth));

  // Convert to the target datatype.
  CUDA_CALL(Convert(
      values_float, this->values_, kNumValues));
  CUDA_CALL(Convert(
      column_indices_int, this->column_indices_, kNumIndices));
  CUDA_CALL(cudaStreamSynchronize(nullptr));

  // Free the temporary memory.
  CUDA_CALL(cudaFree(values_float));
  CUDA_CALL(cudaFree(column_indices_int));
}

template class CudaBlockSparseMatrix<float>;
template class CudaBlockSparseMatrix<half2>;
template class CudaBlockSparseMatrix<half>;
template BlockSparseMatrix::BlockSparseMatrix(
    const CudaBlockSparseMatrix<float> &);
template BlockSparseMatrix::BlockSparseMatrix(
    const CudaBlockSparseMatrix<half2> &);
template BlockSparseMatrix::BlockSparseMatrix(
    const CudaBlockSparseMatrix<half> &);

}  // namespace sputnik
