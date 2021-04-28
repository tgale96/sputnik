#ifndef THIRD_PARTY_SPUTNIK_BLOCK_DSD_DSD_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_DSD_DSD_H_

#include "sputnik/cuda_utils.h"

namespace sputnik {
namespace block {
  
cudaError_t Dsd(int m, int k, int n,
		int nonzeros, int block_dim,
		const half* a,
		const int* offsets_a,
		const short* indices_a,
		const half* b, bool transpose_b,
		half* c, cudaStream_t stream);

}  // namespace block
}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_DSD_DSD_H_
