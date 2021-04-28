#include "sputnik/block/dsd/cutlass/dsd.h"
#include "sputnik/block/cutlass/block_pitch_linear.h"
#include "sputnik/block/dsd/cutlass/dsd.h"

#include "glog/logging.h"

namespace sputnik {
namespace block {
namespace cutlass {

cudaError_t Dsd(int m, int k, int n,
		int nonzeros, int block_dim,
		const half* a,
		const int* offsets_a,
		const short* indices_a,
		const half* b, bool transpose_b,
		half* c, cudaStream_t stream) {
  CHECK_EQ(transpose_b, true) <<
    "transpose_b must be set for block_size == 128.";
  if (can_launch_dsd_mixed_b128_128x256x32x3_nt_align8(
       m, k, n, nonzeros, block_dim)) {
    return launch_dsd_mixed_b128_128x256x32x3_nt_align8(
      m, k, n, nonzeros, block_dim, a, offsets_a,
      indices_a, b, transpose_b, c, stream);
  } else {
    LOG(FATAL) << "No compatible kernel for problem. m/n/k/b = " <<
      m << "/" << n << "/" << k << "/" << block_dim << ".";
  }
  return cudaGetLastError();
}

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik


