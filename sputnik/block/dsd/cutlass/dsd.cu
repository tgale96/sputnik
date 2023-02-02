#include <functional>
#include <utility>
#include <vector>

#include "sputnik/block/dsd/cutlass/dsd.h"
#include "sputnik/block/cutlass/block_pitch_linear.h"
#include "sputnik/block/dsd/cutlass/dsd.h"
#include "sputnik/logging.h"

namespace sputnik {
namespace block {
namespace cutlass {

namespace {

using CanImplementFn = std::function<
    bool(const BlockMatrix, bool,
         const Matrix, bool,
         Matrix)>;

using LaunchFn = std::function<
    cudaError_t(const BlockMatrix, bool,
                const Matrix, bool,
                Matrix, cudaStream_t)>;

using Kernel = std::pair<CanImplementFn, LaunchFn>;

using KernelRegistry = std::vector<Kernel>;

KernelRegistry& GetRegistry() {
  static KernelRegistry registry;
  return registry;
}

bool RegisterKernel(Kernel kernel) {
  GetRegistry().push_back(kernel);
  return true;
}

bool RegisterKernel(CanImplementFn can_implement, LaunchFn launch) {
  Kernel kernel(can_implement, launch);
  return RegisterKernel(kernel);
}

// TODO(tgale): Automate this with kernel generators.
static const bool k1 = RegisterKernel(can_launch_dsd_mixed_b128_128x128x32x5_nt_align8,
                                      launch_dsd_mixed_b128_128x128x32x5_nt_align8);
static const bool k2 = RegisterKernel(can_launch_dsd_mixed_b128_128x128x32x5_nn_align8,
                                      launch_dsd_mixed_b128_128x128x32x5_nn_align8);
static const bool k3 = RegisterKernel(can_launch_dsd_mixed_b128_128x128x32x5_tn_align8,
                                      launch_dsd_mixed_b128_128x128x32x5_tn_align8);
static const bool k4 = RegisterKernel(can_launch_dsd_mixed_b128_128x128x32x5_tt_align8,
                                      launch_dsd_mixed_b128_128x128x32x5_tt_align8);

}  // namespace

cudaError_t Matmul(const BlockMatrix a, bool transpose_a,
                   const Matrix b, bool transpose_b,
                   Matrix c, cudaStream_t stream) {
  for (auto &kernel : GetRegistry()) {
    // TODO(tgale): Do something smarter than launching the first
    // compatible kernel.
    if (kernel.first(a, transpose_a, b, transpose_b, c)) {
      return kernel.second(a, transpose_a, b, transpose_b, c, stream);
    }
  }

  MatmulShape shape(a, transpose_a, b, transpose_b);
  SPUTNIK_LOG(FATAL) << "No compatible kernel for dsd problem.\n" << "m = " << shape.m <<
      "\nn = " << shape.n << "\nk = " << shape.k << "\nblock_size = " <<
      AsInt(a.block_size) << "\ntrans_a = " << transpose_a << "\ntrans_b = " <<
      transpose_b << std::endl;
  return cudaGetLastError();
}

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik
