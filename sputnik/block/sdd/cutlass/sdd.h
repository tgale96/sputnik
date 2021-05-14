#ifndef SPUTNIK_BLOCK_SDD_CUTLASS_SDD_H_
#define SPUTNIK_BLOCK_SDD_CUTLASS_SDD_H_

#include "sputnik/block/arguments.h"
#include "sputnik/cuda_utils.h"

namespace sputnik {
namespace block {
namespace cutlass {

bool can_launch_sdd_mixed_b128_128x256x32x3_tt_align8(
    const Matrix a, bool transpose_a,
    const Matrix b, bool transpose_b, BlockMatrix c);

cudaError_t launch_sdd_mixed_b128_128x256x32x3_tt_align8(
    const Matrix a, bool transpose_a,
    const Matrix b, bool transpose_b,
    BlockMatrix c, cudaStream_t stream);

bool can_launch_sdd_mixed_b128_128x256x32x3_tn_align8(
    const Matrix a, bool transpose_a,
    const Matrix b, bool transpose_b, BlockMatrix c);

cudaError_t launch_sdd_mixed_b128_128x256x32x3_tn_align8(
    const Matrix a, bool transpose_a,
    const Matrix b, bool transpose_b,
    BlockMatrix c, cudaStream_t stream);

bool can_launch_sdd_mixed_b128_128x256x32x3_nt_align8(
    const Matrix a, bool transpose_a,
    const Matrix b, bool transpose_b, BlockMatrix c);

cudaError_t launch_sdd_mixed_b128_128x256x32x3_nt_align8(
    const Matrix a, bool transpose_a,
    const Matrix b, bool transpose_b,
    BlockMatrix c, cudaStream_t stream);

bool can_launch_sdd_mixed_b128_128x256x32x3_nn_align8(
    const Matrix a, bool transpose_a,
    const Matrix b, bool transpose_b, BlockMatrix c);

cudaError_t launch_sdd_mixed_b128_128x256x32x3_nn_align8(
    const Matrix a, bool transpose_a,
    const Matrix b, bool transpose_b,
    BlockMatrix c, cudaStream_t stream);

cudaError_t Matmul(const Matrix a, bool transpose_a,
                   const Matrix b, bool transpose_b,
                   BlockMatrix c, cudaStream_t stream);

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // SPUTNIK_BLOCK_SDD_CUTLASS_SDD_H_

