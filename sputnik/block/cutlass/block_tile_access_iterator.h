#ifndef THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_TILE_ACCESS_ITERATOR_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_TILE_ACCESS_ITERATOR_H_

#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"

namespace sputnik {
namespace block {
namespace cutlass {

// Specialization of PredicatedTileAccessIterator for
// block-sparse pitch-linear data and AdvanceRank == 0
// (i.e., movement through the compressed dimension).
//
// NOTE: Two key differences from a normal PredicatedTileAccessIterator
// 1. We never have a residue tile
// 2. Matrix dimensions are both dividible by block size
//
// Thus, we don't need any predicates, and we can remove any code
// that handles residual tiles.
/* template < */
/*   typename Shape_,  // BlockPitchLinearShape */
/*   typename Element_, */
/*   typename ThreadMap_, */
/*   typename AccessType_> */
/* class PredicatedTileAccessIterator< */
/*   Shape_, */
/*   Element_, */
/*   BlockPitchLinear, */
/*   /\*AdvanceRank=*\/0, */
/*   ThreadMap_, */
/*   AccessType_> { */
/*  public: */

/*   // TODO(tgale): Relax this constraint. We'd like to be able to use */
/*   // smaller tile dimensions for cases that are occupancy limited. */
/*   static_assert(Shape::kStrided == Shape::kBlock, */
/* 		"The strided tile dimension must equal the block size."); */
/*   // TODO(tgale): Relax this constraint. For small/medium block sizes */
/*   // we'd like to pack multiple blocks into one mma tile. */
/*   static_assert(Shape::kContiguous <= Shape::kBlock, */
/* 		"The contiguous tile dimension must be less-than or " */
/* 		"equal to the blocks size."); */
/*   static_assert(Shape::kBlock % Shape::kContiguous == 0, */
/* 		"The block size must be divisible by the contiguous " */
/* 		"tile dimension."); */
  
/*   using Shape = Shape_; */
/*   using Element = Element_; */
/*   using Layout = BlockPitchLinear; */
/*   static int const kAdvanceRank = 0;  // Contiguous dimension. */
/*   using ThreadMap = ThreadMap_; */
/*   using AccessType = AccessType_; */

/*   using Index = typename Layout::Index; */
/*   using LongIndex = typename Layout::LongIndex; */

/*   using TensorRef = TensorRef<Element, Layout>; */
/*   using TensorView = TensorView<Element, Layout>; */
/*   using TensorCoord = typename Layout::TensorCoord; */

/*   using Pointer = Element *; */
/*   using NonConstPointer = typename platform::remove_const<Element>::type *; */

/*   static int const kAccessesPerVector = ThreadMap::kElementsPerAccess / */
/*     AccessType::kElements; */
  
/*   static_assert(!(ThreadMap::kElementsPerAccess % AccessType::kElements),  */
/*     "Vectors implied by the thread map must be divisible by the access type."); */

/*   // NOTE: All increments are statically computable for block-sparse */
/*   // iterator with known block dimensions. */
/*   class Params { */
/*    public: */
/*     CUTLASS_HOST_DEVICE Params() {} */
/*     CUTLASS_HOST_DEVICE Params(Layout const &layout) {} */
/*   }; */

/*  private: */
/*   /// Internal pointer type permits fast address arithmetic */
/*   using BytePointer = char *; */

/*   // TODO(tgale): Update these names to use constant naming. */
/*   // Make Params empty and update code to use these static */
/*   // values. */
/*   static const LongIndex kStride = Shape::kBlock; */
  
/*   static const LongIndex kIncStrided = Shape::kBlock * */
/*     ThreadMap::Delta::kStrided * */
/*     sizeof_bits<Element>::value / 8; */

/*   // Advance to the next tile in the block. */
/*   static const LongIndex kIncAdvance = Shape::kContiguous * */
/*     Shape::kBlock * sizeof_bits<Element>::value / 8; */

/*   static const LongIndex kIncNext = kIncAdvance - */
/*     LongIndex(ThreadMap::Iterations::kStrided - 1) * */
/*     ThreadMap::Delta::kStrided * LongIndex(stride_) * */
/*     sizeof_bits<Element>::value / 8; */
      
/*   // */
/*   /// Data members */
/*   // */

/*   // Internal pointer to first access of tile */
/*   BytePointer pointer_; */

/*   // Size of tensor */
/*   TensorCoord extent_; */

/*   // Initial offset for each thread */
/*   TensorCoord thread_offset_; */

/*   // Iteration along vectors implied by the thread map */
/*   int iteration_vector_; */

/*   // Iteration in the contiguous dimension */
/*   int iteration_contiguous_; */

/*   // Iteration in the strided dimension */
/*   int iteration_strided_; */

/*  public: */
/*   /// Constructs a TileIterator from its precomputed state, threadblock offset, */
/*   /// and thread ID */
/*   CUTLASS_HOST_DEVICE */
/*   PredicatedTileAccessIterator( */
/*       /// Precomputed parameters object */
/*       Params const &params, */
/*       /// Pointer to start of tensor */
/*       Pointer pointer, */
/*       /// Extent of tensor */
/*       TensorCoord extent, */
/*       /// ID of each participating thread */
/*       int thread_id, */
/*       /// Initial offset of threadblock */
/*       TensorCoord const &threadblock_offset) */
/*       : pointer_(reinterpret_cast<BytePointer>( */
/*             const_cast<NonConstPointer>(pointer))), */
/*         extent_(extent) { */
/*     // Per-thread offset in logical coordinates of tensor */
/*     thread_offset_ = threadblock_offset + ThreadMap::initial_offset(thread_id); */

/*     // Update the pointer. */
/*     Layout layout(kStride); */
/*     add_pointer_offset(layout(thread_offset_)); */

/*     set_iteration_index(0); */
/*   } */

/*   // Overrides the internal iteration index */
/*   CUTLASS_HOST_DEVICE */
/*   void set_iteration_index(int index) { */
/*     iteration_vector_ = index % kAccessesPerVector; */
/*     int residual_access = index / kAccessesPerVector; */

/*     iteration_contiguous_ = residual_access % ThreadMap::Iterations::kContiguous; */
/*     iteration_strided_ = residual_access / ThreadMap::Iterations::kContiguous; */

/*   } */

/*   // Adds a pointer offset in units of Element */
/*   CUTLASS_HOST_DEVICE */
/*   void add_pointer_offset(LongIndex pointer_offset) { */
/*     pointer_ += sizeof_bits<Element>::value * pointer_offset / 8; */
/*   } */

/*   // Advances an iterator along logical dimensions of matrix in units of whole tiles */
/*   CUTLASS_DEVICE */
/*   void add_tile_offset(TensorCoord const &tile_offset) { */
/*     pointer_ += kIncAdvance * LongIndex(tile_offset.contiguous()); */

/*     // TODO(tgale): Is this correct? Seems like we need an extra */
/*     // factor to get the full block size moving in the strided */
/*     // dimension. */
/*     pointer_ += Shape::kStrided * tile_offset.strided(); */
/*   } */

/*   // Returns a pointer */
/*   CUTLASS_HOST_DEVICE */
/*   AccessType *get() const { */
/*     return reinterpret_cast<AccessType *>( */
/*         pointer_ +  */
/*         iteration_contiguous_ * (ThreadMap::Delta::kContiguous * sizeof_bits<Element>::value) / 8) + iteration_vector_; */
/*   } */

/*   // Increment and return an instance to self. */
/*   CUTLASS_HOST_DEVICE */
/*   PredicatedTileAccessIterator &operator++() { */
/*     ++iteration_vector_; */
/*     if (iteration_vector_ < kAccessesPerVector) { */
/*       return *this; */
/*     } */

/*     iteration_vector_ = 0; */
/*     ++iteration_contiguous_; */

/*     if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous) { */
/*       return *this; */
/*     } */

/*     // Enter here only if (iteration_contiguous_ == */
/*     // ThreadMap::Iteration::kContiguous) */
/*     iteration_contiguous_ = 0; */
/*     ++iteration_strided_; */

/*     if (iteration_strided_ < ThreadMap::Iterations::kStrided) { */
/*       pointer_ += kIncStrided; */
/*       return *this; */
/*     } */

/*     // Enter here only if (iteration_stride_ == ThreadMap::Iteration::kStrided) */
/*     // which means we enter the next tile. */
/*     iteration_strided_ = 0; */

/*     // advance to next tile */
/*     pointer_ += kIncNext; */

/*     // now return to start tile - if the iterator is subsequently advanced, this */
/*     // subtraction as well as the subsequent integer addition are both elided by */
/*     // the compiler. */
/*     pointer_ -= kIncAdvance; */

/*     return *this; */
/*   } */

/*   // Increment and return an instance to self. */
/*   CUTLASS_HOST_DEVICE */
/*   PredicatedTileAccessIterator operator++(int) { */
/*     PredicatedTileAccessIterator self(*this); */
/*     operator++(); */
/*     return self; */
/*   } */

/*   // No residue and perfect tiles - all accesses are valid. */
/*   CUTLASS_HOST_DEVICE */
/*   bool valid() { return true; } */

/* }; */

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_TILE_ACCESS_ITERATOR_H_ 
