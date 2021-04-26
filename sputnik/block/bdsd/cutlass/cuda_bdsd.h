#ifndef THIRD_PARTY_SPUTNIK_BLOCK_CUDA_BDSD_CUTLASS_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_CUDA_BDSD_CUTLASS_H_

#include "sputnik/cuda_utils.h"

namespace sputnik {
namespace block {
namespace cutlass {

cudaError_t hgemm_tn(
  int M,
  int N,
  int K,
  half const *A,
  half const *B,
  half *C);

cudaError_t hgemm_nt(
  int M,
  int N,
  int K,
  half const *A,
  half const *B,
  half *C);

cudaError_t dsd_nt(
  int M,
  int N,
  int K,
  half const *A,
  half const *B,
  half *C);
 
}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_CUDA_BDSD_CUTLASS_H_
