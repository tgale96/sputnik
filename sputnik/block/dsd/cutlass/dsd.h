#ifndef THIRD_PARTY_SPUTNIK_BLOCK_DSD_CUTLASS_DSD_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_DSD_CUTLASS_DSD_H_

#include "sputnik/cuda_utils.h"

namespace sputnik {
namespace block {
namespace cutlass {

bool can_launch_dsd_mixed_b128_128x256x32x3_nt_align8(
  int m, int k, int n, int nonzeros, int block_dim);

cudaError_t launch_dsd_mixed_b128_128x256x32x3_nt_align8(
  int m, int k, int n,
  int nonzeros, int block_dim,
  const half* a,
  const int* offsets_a,
  const short* indices_a,
  const half* b, bool transpose_b,
  half* c, cudaStream_t stream);
  
cudaError_t Dsd(int m, int k, int n,
		int nonzeros, int block_dim,
		const half* a,
		const int* offsets_a,
		const short* indices_a,
		const half* b, bool transpose_b,
		half* c, cudaStream_t stream); 
 
}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_DSD_CUTLASS_DSD_H_
