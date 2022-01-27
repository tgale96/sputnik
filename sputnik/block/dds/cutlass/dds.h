#ifndef SPUTNIK_BLOCK_DDS_CUTLASS_DDS_H_
#define SPUTNIK_BLOCK_DDS_CUTLASS_DDS_H_

#include "sputnik/block/arguments.h"
#include "sputnik/cuda_utils.h"

namespace sputnik {
namespace block {
namespace cutlass {

bool can_launch_dds_mixed_b128_128x128x32x5_tt_align8(
    const Matrix a, bool transpose_a,
    const BlockMatrix b, bool transpose_b, Matrix c);

cudaError_t launch_dds_mixed_b128_128x128x32x5_tt_align8(
    const Matrix a, bool transpose_a,
    const BlockMatrix b, bool transpose_b,
    Matrix c, cudaStream_t stream);

bool can_launch_dds_mixed_b128_128x128x32x5_tn_align8(
    const Matrix a, bool transpose_a,
    const BlockMatrix b, bool transpose_b, Matrix c);

cudaError_t launch_dds_mixed_b128_128x128x32x5_tn_align8(
    const Matrix a, bool transpose_a,
    const BlockMatrix b, bool transpose_b,
    Matrix c, cudaStream_t stream);

bool can_launch_dds_mixed_b128_128x128x32x5_nt_align8(
    const Matrix a, bool transpose_a,
    const BlockMatrix b, bool transpose_b, Matrix c);

cudaError_t launch_dds_mixed_b128_128x128x32x5_nt_align8(
    const Matrix a, bool transpose_a,
    const BlockMatrix b, bool transpose_b,
    Matrix c, cudaStream_t stream);

bool can_launch_dds_mixed_b128_128x128x32x5_nn_align8(
    const Matrix a, bool transpose_a,
    const BlockMatrix b, bool transpose_b, Matrix c);

cudaError_t launch_dds_mixed_b128_128x128x32x5_nn_align8(
    const Matrix a, bool transpose_a,
    const BlockMatrix b, bool transpose_b,
    Matrix c, cudaStream_t stream);

cudaError_t Matmul(const Matrix a, bool transpose_a,
                   const BlockMatrix b, bool transpose_b,
                   Matrix c, cudaStream_t stream);

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik

#endif  // SPUTNIK_BLOCK_DDS_CUTLASS_DDS_H_
