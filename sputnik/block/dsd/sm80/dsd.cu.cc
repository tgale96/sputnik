#include "sputnik/block/dsd/sm80/dsd.h"

namespace sputnik {
namespace block {
namespace sm80 {

cudaError_t Dsd(int m, int k, int n,
		int nonzeros, int block_dim,
		const half* a,
		const int* offsets_a,
		const short* indices_a,
		const half* b, bool transpose_b,
		half* c, cudaStream_t stream) {
  SPUTNIK_CHECK_EQ(transpose_b, false) <<
    "transpose_b not supported with block_size == 32.";
    
  // TODO(tgale): Add a proper kernel selector.
  if (can_launch_dsd_b32_m32n128k32_h8_h8(m, k, n, nonzeros, block_dim)) {
    return launch_dsd_b32_m32n128k32_h8_h8(
        m, k, n, nonzeros, block_dim,
	a, offsets_a, indices_a, b, c, stream);
  } else if (can_launch_dsd_b32_m32n64k32_h8_h8(m, k, n, nonzeros, block_dim)) {
    return launch_dsd_b32_m32n64k32_h8_h8(
        m, k, n, nonzeros, block_dim,
	a, offsets_a, indices_a, b, c, stream);    
  } else {
    SPUTNIK_LOG(FATAL) << "No compatible kernel for problem. m/n/k/b = " <<
      m << "/" << n << "/" << k << "/" << block_dim << ".";
  }
  return cudaGetLastError();
}

}  // namespace sm80
}  // namespace block
}  // namespace sputnik
