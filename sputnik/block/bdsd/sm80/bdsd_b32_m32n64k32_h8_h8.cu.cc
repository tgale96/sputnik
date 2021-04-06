#include "sputnik/block/bdsd/sm80/cuda_bdsd.h"
#include "sputnik/cuda_utils.h"
#include "sputnik/load_store.h"

#include "glog/logging.h"

namespace sputnik {
namespace block {
namespace sm80 {
    
namespace {

void __device__ __forceinline__ Transpose(half2 &x, half2 &y) {
  half tmp = x.y;
  x.y = y.x;
  y.x = tmp;
}

void __device__ __forceinline__ mma_m8n16k8(
  const half2 &lhs0, const half2 &rhs0, const half2 &rhs1,
  float &out0, float &out1, float &out2, float &out3) {
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=f"(out0), "=f"(out1), "=f"(out2), "=f"(out3)
      : "r"(BitCast<uint32_t>(rhs0)),
	"r"(BitCast<uint32_t>(rhs1)),
	"r"(BitCast<uint32_t>(lhs0)), 
	"f"(out0), "f"(out1), "f"(out2), "f"(out3));
}
    
struct bdsd_b32_m32n64k32_h8_h8 {

  // Block dimension parameters.
  static constexpr int kBlockDim = 32;
  static constexpr int kBlockSize = kBlockDim * kBlockDim;
  
  // Other tile dimension parameters.
  static constexpr int kTileX = 64;
  static constexpr int kTileK = 32;

  static void __device__ __forceinline__ KernelFn(
      int m, int k, int n,
      const half2* __restrict__ values,
      const int* __restrict__ offsets,
      const short2* __restrict__ indices,
      const half2* __restrict__ dense_matrix,
      half2* __restrict__ output_matrix) {
    // Calculate this thread block's indices into the
    // output matrices.
    const int m_index_block = blockIdx.x;
    const int n_index = blockIdx.y * kTileX;
    const int m_index = m_index_block * kBlockDim;

    // TODO(tgale): This should never execute. Delete this.
    //
    // Exit early if there's no work to do.
    if (m_index > m) return;
    
    // Row offset and nonzeros in scalar elements.
    const int row_offset = Load(
        offsets + m_index_block);
    int nonzeros = Load(
        offsets + m_index_block + 1) - row_offset;

    // Offset our sparse matrix pointers to the start of
    // this thread block's data and indices.
    //
    // TODO(tgale): Figure out if we can save registers by
    // recomputing these pointer rather than offsetting 
    // them directly.
    const int row_offset_block = row_offset / kBlockSize;
    short *indices_s1 = (short*)indices + row_offset_block;
    values = (half2*)((half*)values + row_offset);

    // Offset our dense matrix pointers to the start of
    // this thread block's data.
    //
    // TODO(tgale): Might be able to save a register by
    // recomputing 'n_index' here and when we use it for
    // the output pointer after the main loop.
    //
    // Thread's are grouped into sets of 8 to match the
    // mma thread layout requirements.
    //
    // TODO(tgale): Turn this modulo into a bitwise and
    // with '3'. Turn this divide into a shift-right by 2.
    dense_matrix = (half2*)((half*)dense_matrix + n_index);
    int tidy = threadIdx.x % 4;
    int tidx = threadIdx.x / 4;
    dense_matrix = (half2*)((half*)dense_matrix + tidy * n * 8);
    half8 *dense_matrix_h8 = (half8*)dense_matrix + tidx;

    // Register fragment for sparse matrix data. Each
    // thread owns 1/32nd of a 32x32 tile for a total
    // of 32 half-precision values - i.e., 16x half2s.
    half2 lhs_fragment[16];
    half8 *lhs_fragment_h8 = (half8*)lhs_fragment;

    // Offset the values pointer.
    //
    // Each thread loads 8x halfs in a single instruction.
    // We execute four loads per-thread per-tile. Threads
    // are strided in the standard way.
    half8 *values_h8 = (half8*)values + threadIdx.x;

    // Register fragment for dense matrix data. Each
    // thread owns 1/32nd of a 32x64 tile for a total
    // of 64 half-precision values - i.e., 32x half2s.
    half2 rhs_fragment[32];
    half8 *rhs_fragment_h8 = (half8*)rhs_fragment;

    // Register fragment for the results accumulators.
    //
    // Each thread ends up with an 8x8 square of results.
    // We accumulate in 32-bit as is standard.
    //
    // TODO(tgale): There are alternative techniques for
    // zero-ing our accumulators that we should explore.
    float out_fragment[64] = {};
    
    //
    /// Main loop.
    //

    for (; nonzeros >= kTileK; nonzeros -= kTileK * kBlockDim) {
      // Load the sparse block column index.
      //
      // TODO(tgale): We could prefetch this to avoid data-dependent
      // load latency within a loop iteration. We could also load
      // multiple (e.g., 4 or 8) indices in one go to help avoid
      // memory system waste.
      int lhs_index = Load(indices_s1) * n;
      ++indices_s1;

      // Load the sparse block data.
      //
      // We load an entire 32x32 sparse block in 4x 8-wide
      // vector loads.
#pragma unroll
      for (int i = 0; i < 4; ++i) {
	lhs_fragment_h8[i] = Load(values_h8);
	values_h8 += 32;
      }      

      // Load the dense matrix tile data.
      half8 *rhs_h8 = (half8*)((half*)dense_matrix_h8 + lhs_index);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
	rhs_fragment_h8[i] = Load(rhs_h8);
	rhs_h8 = (half8*)((half*)rhs_h8 + n);
      }

      // Transpose the dense matrix registers so that
      // 16-bit values are properly packed into 32-bit
      // registers for the mma instructions.
      //
      // TODO(tgale): This is likely the largest source
      // of instructions in the entire loop. Can we
      // reduce it by doing wider (e.g., 64-bit) swaps?
      //
      // Perhaps with the xor trick. This would be 3x
      // xors for every pair 4x halfs. We could also
      // try with the byte permute instructions, which
      // would be 1 instruction for every output.
#pragma unroll
      for (int i = 0; i < 4; ++i) {
#pragma unroll
	for (int j = 0; j < 4; ++j) {
	  Transpose(rhs_fragment[j + 8 * i], rhs_fragment[j + 8 * i + 4]);
	}
      }
      
      // Do the matrix multiplication. We re-use the lhs
      // values first. 
      //
      // TODO(tgale): Perhaps alternative orderings could
      // give us better latency hiding/liveness. The compiler
      // should schedule these instructions, but perhaps we
      // can tweak it.
      //
      // TODO(tgale): We should also explore whether it makes
      // sense to use the mma with a contraction dimension of
      // 16. This could increase performance, but it could
      // also increase register pressure and reduce our
      // ability to hide latency. It might make sense to use
      // the larger instruction when we use a larger k-dim
      // tile size.
#pragma unroll
      for (int k_item_idx = 0; k_item_idx < 4; ++k_item_idx) {
#pragma unroll
	for (int y_item_idx = 0; y_item_idx < 4; ++y_item_idx) {
#pragma unroll
	  for (int x_item_idx = 0; x_item_idx < 4; ++x_item_idx) {
	    const int lhs_idx = k_item_idx + y_item_idx * 4;
	    const int rhs_idx = x_item_idx + 8 * k_item_idx;
	    const int out_idx = 2 * x_item_idx + y_item_idx * 16;

	    mma_m8n16k8(lhs_fragment[lhs_idx],
			rhs_fragment[rhs_idx],
			rhs_fragment[rhs_idx + 4],
			out_fragment[out_idx],
			out_fragment[out_idx + 8],
			out_fragment[out_idx + 1],			
			out_fragment[out_idx + 9]);
	  }
	}
      }
    }
    
    // Convert the accumulators to half-precision.
    half2 out_fragment_h2[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      out_fragment_h2[i] = __float22half2_rn(make_float2(
          out_fragment[i * 2], out_fragment[i * 2 + 1]));
    }
    half8 *out_fragment_h8 = (half8*)out_fragment_h2;
    
    // Write the result to the output.
    //
    // Note that there are no residue tiles because
    // the k-dim tile size matches the block size.
    //
    // TODO(tgale): Make sure the thread indices are
    // re-computed here. Could also re-compute m/n
    // indices.
    output_matrix = (half2*)((half*)output_matrix + m_index * n + n_index);
    tidy = threadIdx.x % 4;
    tidx = threadIdx.x / 4;
    output_matrix = (half2*)((half*)output_matrix + tidy * n * 2);
    half8 *output_matrix_h8 = (half8*)output_matrix + tidx;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      half8* out_h8 = output_matrix_h8;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
	Store(out_fragment_h8[i * 2 + j], out_h8);
	out_h8 = (half8*)((half*)out_h8 + n);
      }
      output_matrix_h8 = (half8*)((half*)output_matrix_h8 + 8 * n);
    }
  }
  
};

__global__ void __launch_bounds__(32) bdsd_b32_m32n64k32_h8_h8_kernel(
    int m, int k, int n,
    const half2* __restrict__ values,
    const int* __restrict__ offsets,
    const short2* __restrict__ indices,
    const half2* __restrict__ dense_matrix,
    half2* __restrict__ output_matrix) {
  // TODO(tgale): Figure out a cleaner way to fail on unsupported hardware.
#if __CUDA_ARCH__ >= 800
  bdsd_b32_m32n64k32_h8_h8::KernelFn(
      m, k, n, values, offsets, indices,
      dense_matrix, output_matrix);
#endif
}

// TODO(tgale): Move this, along with common functions like
// transpose and mma into a separate file to be re-used.
int cdiv(int x, int y) {
  return (x + y - 1) / y;
}
  
} // namespace

bool can_launch_bdsd_b32_m32n64k32_h8_h8(
  int m, int k, int n, int nonzeros, int block_size) {
  bool can_launch = true;
  can_launch &= block_size == 32;
  can_launch &= (m % block_size) == 0;
  can_launch &= (k % block_size) == 0;
  can_launch &= (n % 64) == 0;
  return can_launch;
}
  
cudaError_t launch_bdsd_b32_m32n64k32_h8_h8(
    int m, int k, int n,
    int nonzeros, int block_size,
    const half* __restrict__ values,
    const int* __restrict__ offsets,
    const short* __restrict__ indices,
    const half* __restrict__ dense_matrix,
    half* __restrict__ output_matrix,
    cudaStream_t stream) {
  CHECK_EQ(block_size, 32);
  CHECK_EQ(m % block_size, 0);
  CHECK_EQ(k % block_size, 0);
  CHECK_EQ(n % 64, 0);
  dim3 grid_dim(cdiv(m, block_size), cdiv(n, 64), 1);
  dim3 block_dim(32, 1, 1);

  bdsd_b32_m32n64k32_h8_h8_kernel<<<grid_dim, block_dim, 0, stream>>>(
    m, k, n, (const half2*)values, offsets, (const short2*)indices,
    (const half2*)dense_matrix, (half2*)output_matrix);
  return cudaGetLastError();
}

}  // namespace sm80
}  // namespace block
}  // namespace sputnik
