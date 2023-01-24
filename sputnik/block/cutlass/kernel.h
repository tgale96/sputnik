#ifndef THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_KERNEL_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_KERNEL_H_

#include "cutlass/device_kernel.h"

namespace sputnik {
namespace block {
namespace cutlass {

template <typename KernelFn_>
class Kernel {
 public:

  using KernelFn = KernelFn_;
  using ThreadblockSwizzle = typename KernelFn::ThreadblockSwizzle;
  using ThreadblockShape = typename KernelFn::Mma::Shape;
  using Arguments = typename KernelFn::Arguments;

  static cudaError_t smem_config() {
    // Get shared memory requirements for kernel and adjust
    // devices settings if necessary.
    int smem_size = int(sizeof(typename KernelFn::SharedStorage));

    if (smem_size >= (48 << 10)) {
      cudaError_t result = cudaFuncSetAttribute(
	::cutlass::Kernel<KernelFn>,
	cudaFuncAttributeMaxDynamicSharedMemorySize,
	smem_size);
      if (result != cudaSuccess) return result;

      result = cudaFuncSetAttribute(
	::cutlass::Kernel<KernelFn>,
	cudaFuncAttributePreferredSharedMemoryCarveout,
	100);
      if (result != cudaSuccess) return result;
    }
    return cudaSuccess;
  }

  cudaError_t initialize(Arguments const &args) {
    // Get kernel grid dimensions.
    ThreadblockSwizzle threadblock_swizzle;

    ::cutlass::gemm::GemmCoord grid_tiled_shape =
      threadblock_swizzle.get_tiled_shape(
	args.problem_size,
	{ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
	/*batch_count=*/1);

    // Initialize the kernel parameters.
    params_ = typename KernelFn::Params(args, grid_tiled_shape);
    return smem_config();
  }

  cudaError_t operator()(Arguments const &args, cudaStream_t stream = nullptr) {
    cudaError_t status = initialize(args);
    if (status != cudaSuccess) return status;

    ThreadblockSwizzle threadblock_swizzle;
    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(KernelFn::kThreadCount, 1, 1);
    int smem_size = int(sizeof(typename KernelFn::SharedStorage));

    // Launch the kernel.
    ::cutlass::Kernel<KernelFn><<<grid, block, smem_size, stream>>>(params_);
    return cudaGetLastError();
  }

protected:
  typename KernelFn::Params params_;
};

template <typename KernelFn_>
class SparseOutputKernel {
 public:
  using KernelFn = KernelFn_;
  using ThreadblockSwizzle = typename KernelFn::ThreadblockSwizzle;
  using ThreadblockShape = typename KernelFn::Mma::Shape;
  using Arguments = typename KernelFn::Arguments;

  cudaError_t initialize(Arguments const &args) {
    // Get kernel grid dimensions.
    ThreadblockSwizzle threadblock_swizzle;

    // TODO(tgale): Add nnz to op_c/op_d and then pass it in here.
    constexpr int kBlockSize =
      KernelFn::Epilogue::OutputTileIterator::kBlockSize;
    SPUTNIK_CHECK_EQ(args.op_C.nnz, args.op_D.nnz);
    ::cutlass::gemm::GemmCoord grid_tiled_shape =
	threadblock_swizzle.get_tiled_shape(args.op_C.nnz, kBlockSize);

    // Initialize the kernel parameters.
    params_ = typename KernelFn::Params(args, grid_tiled_shape);
    return Kernel<KernelFn>::smem_config();
  }

  cudaError_t operator()(Arguments const &args, cudaStream_t stream = nullptr) {
    cudaError_t status = initialize(args);
    if (status != cudaSuccess) return status;

    ThreadblockSwizzle threadblock_swizzle;
    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(KernelFn::kThreadCount, 1, 1);
    int smem_size = int(sizeof(typename KernelFn::SharedStorage));

    // Launch the kernel.
    ::cutlass::Kernel<KernelFn><<<grid, block, smem_size, stream>>>(params_);
    return cudaGetLastError();
  }

protected:
  typename KernelFn::Params params_;
};

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_KERNEL_H_
