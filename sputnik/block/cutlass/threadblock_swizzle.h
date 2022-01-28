#ifndef THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_THREADBLOCK_SWIZZLE_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_THREADBLOCK_SWIZZLE_H_

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

struct SparseOutputThreadblockSwizzle {

  using GemmCoord = typename ::cutlass::gemm::GemmCoord;

  CUTLASS_HOST_DEVICE
  SparseOutputThreadblockSwizzle() {}

  CUTLASS_HOST_DEVICE
  GemmCoord get_tiled_shape(int nnz, int block_size) const {
    // Launch one threadblock per nonzero block. We'll sort
    // out who maps to which tiles inside the kernel based
    // on row/column indices.
    return GemmCoord(nnz / (block_size * block_size), 1, 1);
  }

  CUTLASS_HOST_DEVICE
  dim3 get_grid_shape(GemmCoord tiled_shape) const {
    return dim3(tiled_shape.m(), tiled_shape.n(), tiled_shape.k());
  }

  CUTLASS_DEVICE
  GemmCoord get_tile_offset(GemmCoord tiled_shape) const {
    return GemmCoord{
      ::cutlass::gemm::threadblock::RematerializeBlockIdxX(),
      ::cutlass::gemm::threadblock::RematerializeBlockIdxY(),
      ::cutlass::gemm::threadblock::RematerializeBlockIdxZ()
    };
  }
};

struct GemmVerticalThreadblockSwizzle {

  using GemmCoord = typename ::cutlass::gemm::GemmCoord;

  CUTLASS_HOST_DEVICE
  GemmVerticalThreadblockSwizzle() { }

  CUTLASS_HOST_DEVICE
  GemmCoord get_tiled_shape(
    GemmCoord problem_size,
    GemmCoord tile_size,
    int split_k_slices) const {

    return GemmCoord(
      (problem_size.m() + tile_size.m() - 1) / tile_size.m(),
      (problem_size.n() + tile_size.n() - 1) / tile_size.n(),
      split_k_slices);
  }

  CUTLASS_HOST_DEVICE
  dim3 get_grid_shape(GemmCoord tiled_shape) const {
    return dim3(tiled_shape.m(), tiled_shape.n(), tiled_shape.k());
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  GemmCoord get_tile_offset(GemmCoord tiled_shape) const {
    return GemmCoord{
      ::cutlass::gemm::threadblock::RematerializeBlockIdxX(),
      ::cutlass::gemm::threadblock::RematerializeBlockIdxY(),
      ::cutlass::gemm::threadblock::RematerializeBlockIdxZ()
    };
  }
};

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_THREADBLOCK_SWIZZLE_H_
