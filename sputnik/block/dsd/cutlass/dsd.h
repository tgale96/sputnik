#ifndef THIRD_PARTY_SPUTNIK_BLOCK_DSD_CUTLASS_DSD_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_DSD_CUTLASS_DSD_H_

#include "sputnik/block/arguments.h"
#include "sputnik/cuda_utils.h"

namespace sputnik {
namespace block {
namespace cutlass {

bool can_launch_dsd_mixed_b128_128x256x32x3_tt_align8(
    const BlockMatrix a, bool transpose_a,
    const Matrix b, bool transpose_b, Matrix c);

cudaError_t launch_dsd_mixed_b128_128x256x32x3_tt_align8(
    const BlockMatrix a, bool transpose_a,
    const Matrix b, bool transpose_b,
    Matrix c, cudaStream_t stream);

bool can_launch_dsd_mixed_b128_128x256x32x3_tn_align8(
    const BlockMatrix a, bool transpose_a,
    const Matrix b, bool transpose_b, Matrix c);

cudaError_t launch_dsd_mixed_b128_128x256x32x3_tn_align8(
    const BlockMatrix a, bool transpose_a,
    const Matrix b, bool transpose_b,
    Matrix c, cudaStream_t stream);

bool can_launch_dsd_mixed_b128_128x256x32x3_nt_align8(
    const BlockMatrix a, bool transpose_a,
    const Matrix b, bool transpose_b, Matrix c);

cudaError_t launch_dsd_mixed_b128_128x256x32x3_nt_align8(
    const BlockMatrix a, bool transpose_a,
    const Matrix b, bool transpose_b,
    Matrix c, cudaStream_t stream);

bool can_launch_dsd_mixed_b128_128x256x32x3_nn_align8(
    const BlockMatrix a, bool transpose_a,
    const Matrix b, bool transpose_b, Matrix c);

cudaError_t launch_dsd_mixed_b128_128x256x32x3_nn_align8(
    const BlockMatrix a, bool transpose_a,
    const Matrix b, bool transpose_b,
    Matrix c, cudaStream_t stream);

cudaError_t Matmul(const BlockMatrix a, bool transpose_a,
                   const Matrix b, bool transpose_b,
                   Matrix c, cudaStream_t stream);

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_DSD_CUTLASS_DSD_H_
