#ifndef THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_PITCH_LINEAR_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_PITCH_LINEAR_H_

#include "cutlass/coord.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/layout/pitch_linear.h"

namespace sputnik {
namespace block {
namespace cutlass {

template <int Row, int Column, int Block>
struct BlockMatrixShape {
  static int const kRow = Row;
  static int const kColumn = Column;
  static int const kBlock = Block;
  static int const kCount = Row * Column;
};

// Template defining shape used by block-sparse
// pitch-linear matrices. Matrices are represented
// in 4d: [rows // block, cols // block, block, block].
template <
  int Contiguous,
  int Strided,
  int Block
>
struct BlockPitchLinearShape {
  static int const kContiguous = Contiguous;
  static int const kStrided = Strided;
  static int const kBlock = Block;
  static int const kCount = Contiguous * Strided;
};

// Layout for block-sparse pitch-linear memory.
class BlockPitchLinear {
public:
  // Logical rank of tensor.
  static constexpr int kRank = 2;

  // Rank of stride vector.
  //
  // NOTE: Despite the 4d matrix representation,
  // we only store the stride of the inner block.
  // Because the matrix is sparse, this is the only
  // stride that is statically known.
  static int const kStrideRank = 1;

  // Index type used for coordinates.
  using Index = int32_t;

  // Long index type used for offsets.
  using LongIndex = int64_t;

  // Stride vector.
  using Stride = ::cutlass::Coord<kStrideRank, Index>;

  // Coordinate into a block.
  using TensorCoord = ::cutlass::layout::PitchLinearCoord;

  //
  /// Methods
  //

  // Constructor.
  CUTLASS_HOST_DEVICE
  BlockPitchLinear(Index ldm = 0): stride_(ldm) { }

  // Constructor.
  CUTLASS_HOST_DEVICE
  BlockPitchLinear(Stride _stride): stride_(_stride) { }

  // Returns the offset to a coordinate (inside a block) in
  // linear memory. Assume coordinate has convention
  // (contiguous, strided).
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    // Contigous dimension is 0, strided dimension is 1.
    return LongIndex(coord.contiguous()) + LongIndex(coord.strided()) * LongIndex(stride_[0]);
  }

  // Getter for layout stride.
  CUTLASS_HOST_DEVICE
  Index stride(int rank) const {
    return stride_[rank];
  }

  // Getter for layout stride.
  CUTLASS_HOST_DEVICE
  Index & stride(int rank) {
    return stride_[rank];
  }

private:
  //
  /// Data members
  //

  /// Stride data member
  Stride stride_;
};

// Useful to switch on in specializations.
struct BlockRowMajor {
  using TensorCoord = ::cutlass::MatrixCoord;

  static const int kRank = 2;
  static const int kStrideRank = 1;
  using Index = int32_t;
  using LongIndex = int64_t;
  using Stride = ::cutlass::Coord<kStrideRank, Index>;

  CUTLASS_HOST_DEVICE
  static TensorCoord to_pitch_linear(const ::cutlass::MatrixCoord &coord) {
    return {coord.column(), coord.row()};
  }
};


struct BlockColumnMajor {

  using TensorCoord = ::cutlass::MatrixCoord;

  CUTLASS_HOST_DEVICE
  static TensorCoord to_pitch_linear(const ::cutlass::MatrixCoord &coord) {
    return {coord.row(), coord.column()};
  }
};

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_PITCH_LINEAR_H_
