#include <functional>
#include <utility>
#include <vector>

#include "sputnik/block/dsd/cutlass/dsd.h"
#include "sputnik/block/cutlass/block_pitch_linear.h"
#include "sputnik/block/dsd/cutlass/dsd.h"

#include "glog/logging.h"

namespace sputnik {
namespace block {
namespace cutlass {

namespace {

using CanImplementFn = std::function<
    bool(int, int, int, int, int, bool, bool)>;

using LaunchFn = std::function<
    cudaError_t(int, int, int, int, int,
                const half*, const int*,
                const short*, const half*,
                bool, half*, cudaStream_t)>;

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
static const bool k1 = RegisterKernel(can_launch_dsd_mixed_b128_128x256x32x3_nt_align8,
                                      launch_dsd_mixed_b128_128x256x32x3_nt_align8);
static const bool k2 = RegisterKernel(can_launch_dsd_mixed_b128_128x256x32x3_nn_align8,
                                      launch_dsd_mixed_b128_128x256x32x3_nn_align8);

}  // namespace

cudaError_t Dsd(int m, int k, int n,
		int nonzeros, int block_dim,
		const half* a,
		const int* offsets_a,
		const short* indices_a,
		const half* b, bool transpose_b,
		half* c, cudaStream_t stream) {
  for (auto &kernel : GetRegistry()) {
    // TODO(tgale): Do something smarter than launching the first
    // compatible kernel.
    if (kernel.first(m, k, n, nonzeros, block_dim, false, transpose_b)) {
      return kernel.second(m, k, n, nonzeros, block_dim, a, offsets_a,
                           indices_a, b, transpose_b, c, stream);
    }
  }

  LOG(FATAL) << "No compatible kernel for problem. m/n/k/b = " <<
      m << "/" << n << "/" << k << "/" << block_dim << ".";
  return cudaGetLastError();
}

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik
