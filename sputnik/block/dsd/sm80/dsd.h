#ifndef THIRD_PARTY_SPUTNIK_BLOCK_DSD_SM80_DSD_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_DSD_SM80_DSD_H_

#include "sputnik/cuda_utils.h"

#include "glog/logging.h"

namespace sputnik {
namespace block {
namespace sm80 {

bool can_launch_dsd_b32_m32n64k32_h8_h8(
    int m, int k, int n, int nonzeros, int block_dim);
  
cudaError_t launch_dsd_b32_m32n64k32_h8_h8(
    int m, int k, int n,
    int nonzeros, int block_dim,
    const half* __restrict__ values,
    const int* __restrict__ offsets,
    const short* __restrict__ indices,
    const half* __restrict__ dense_matrix,
    half* __restrict__ output_matrix,
    cudaStream_t stream);
  
bool can_launch_dsd_b32_m32n128k32_h8_h8(
    int m, int k, int n, int nonzeros, int block_dim);
  
cudaError_t launch_dsd_b32_m32n128k32_h8_h8(
    int m, int k, int n,
    int nonzeros, int block_dim,
    const half* __restrict__ values,
    const int* __restrict__ offsets,
    const short* __restrict__ indices,
    const half* __restrict__ dense_matrix,
    half* __restrict__ output_matrix,
    cudaStream_t stream);

cudaError_t Dsd(int m, int k, int n,
		int nonzeros, int block_dim,
		const half* a,
		const int* offsets_a,
		const short* indices_a,
		const half* b, bool transpose_b,
		half* c, cudaStream_t stream);		

}  // namespace sm80 
}  // namespace block
}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_DSD_SM80_DSD_H_
