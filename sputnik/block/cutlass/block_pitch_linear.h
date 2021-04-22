#ifndef THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_PITCH_LINEAR_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_PITCH_LINEAR_H_

#include "cutlass/coord.h"

namespace sputnik {
namespace block {
namespace cutlass {

// Template defining shape used by block-sparse
// pitch-linear matrices. Matrices are represented
// in 4d: [rows // block, cols // block, block, block].
/* template < */
/*   int Contiguous, */
/*   int Strided, */
/*   int Block */
/* > */
/* struct BlockPitchLinearShape { */
/*   static int const kContiguous = Contiguous; */
/*   static int const kStrided = Strided; */
/*   static int const kBlock = Block; */
/*   static int const kCount = Contiguous * Strided; */
/* }; */
  
/* // Layout for block-sparse pitch-linear memory. */
/* template */
/* class BlockPitchLinear { */
/* public: */
/*   // Logical rank of tensor. */
/*   static int const kRank = 2; */

/*   // Rank of stride vector. */
/*   // */
/*   // NOTE: Despite the 4d matrix representation, */
/*   // we only store the stride of the inner block. */
/*   // Because the matrix is sparse, this is the only */
/*   // stride that is statically known. */
/*   static int const kStrideRank = 1; */

/*   // Index type used for coordinates. */
/*   using Index = int32_t; */

/*   // Long index type used for offsets. */
/*   using LongIndex = int64_t; */

/*   // Stride vector. */
/*   using Stride = ::cutlass::Coord<kStrideRank, Index>; */

/*   // Coordinate into a block. */
/*   using BlockCoord = Coord<2, int>; */
  
/*   // */
/*   /// Methods */
/*   // */
  
/*   // Constructor. */
/*   CUTLASS_HOST_DEVICE */
/*   BlockPitchLinear(Index ldm = 0): stride_(ldm) { } */

/*   // Constructor. */
/*   CUTLASS_HOST_DEVICE */
/*   BlockPitchLinear(Stride _stride): stride_(_stride) { } */

/*   // Returns the offset to a coordinate (inside a block) in */
/*   // linear memory. Assume coordinate has convention */
/*   // (contiguous, strided). */
/*   CUTLASS_HOST_DEVICE */
/*   LongIndex operator()(BlockCoord const &coord) const { */
/*     return LongIndex(coord.contiguous()) + LongIndex(coord.strided()) * LongIndex(stride_[0]); */
/*   } */

/*   // Getter for layout stride. */
/*   CUTLASS_HOST_DEVICE */
/*   Index stride(int rank) const { */
/*     return stride_[rank]; */
/*   } */

/*   // Getter for layout stride. */
/*   CUTLASS_HOST_DEVICE */
/*   Index & stride(int rank) { */
/*     return stride_[rank]; */
/*   } */
  
/* private: */
/*   // */
/*   /// Data members */
/*   // */

/*   /// Stride data member */
/*   Stride stride_;   */
/* }; */

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_PITCH_LINEAR_H_ 
