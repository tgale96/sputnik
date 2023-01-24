#include "sputnik/block/dsd/sm80/dsd.h"
#include "sputnik/cuda_utils.h"
#include "sputnik/load_store.h"
#include "sputnik/logging.h"

namespace sputnik {
namespace block {
namespace sm80 {
    
namespace {

uint32_t __device__ __forceinline__ asuint(half2 x) {
  return *reinterpret_cast<uint32_t*>(&x);
}
  
void __device__ __forceinline__ store_h8(const half8 &x, half8 *p) {
  asm volatile("st.global.v4.b32 [%0], {%1,%2,%3,%4};"
	       :: "l"(p), "r"(asuint(x.x)), "r"(asuint(x.y)),
		"r"(asuint(x.z)), "r"(asuint(x.w)) : "memory");
}
  
void __device__ __forceinline__ Transpose(half2 &x, half2 &y) {
  half tmp = x.y;
  x.y = y.x;
  y.x = tmp;
}

half8 __device__ __forceinline__ load_nc_h8(const half8 *p) {
  half8 out;
  asm volatile("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];"
	       : "=r"(*reinterpret_cast<uint32_t*>(&out.x)),
		 "=r"(*reinterpret_cast<uint32_t*>(&out.y)),
		 "=r"(*reinterpret_cast<uint32_t*>(&out.z)),
		 "=r"(*reinterpret_cast<uint32_t*>(&out.w))
	       : "l"(p));
  return out;
}

void __device__ __forceinline__ mma_m8n16k16(
  const half2 &lhs0, const half2 &lhs1, const half2 &rhs0,
  const half2 &rhs1, const half2 &rhs2, const half2 &rhs3,
  float &out0, float &out1, float &out2, float &out3) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32  "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%10,%11,%12,%13};\n"
      : "=f"(out0), "=f"(out1), "=f"(out2), "=f"(out3)
      : "r"(asuint(rhs0)),
	"r"(asuint(rhs1)),
	"r"(asuint(rhs2)),
	"r"(asuint(rhs3)),
	"r"(asuint(lhs0)),
	"r"(asuint(lhs1)),
	"f"(out0), "f"(out1), "f"(out2), "f"(out3));
}

template <typename T>
__device__ __forceinline__ T* offset(T* p, int off) {
  // Calculate the byte offset as a 32-bit integer and offset
  // pointer directly in bytes to avoid 64-bit arithmetic.
  int off_bytes = off * sizeof(T);
  return reinterpret_cast<T*>(const_cast<char*>(
      reinterpret_cast<const char*>(p) + off_bytes));
}
  
struct dsd_b32_m32n128k32_h8_h8 {

  // Block dimension parameters.
  static constexpr int kBlockDim = 32;
  static constexpr int kBlockSize = kBlockDim * kBlockDim;
  
  // Other tile dimension parameters.
  static constexpr int kTileX = 128;
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
        offset(offsets, m_index_block));
    int nonzeros = Load(
        offset(offsets, m_index_block) + 1) - row_offset;

    // Offset our sparse matrix pointers to the start of
    // this thread block's data and indices.
    //
    // TODO(tgale): Figure out if we can save registers by
    // recomputing these pointer rather than offsetting 
    // them directly.
    const int row_offset_block = row_offset / kBlockSize;
    short *indices_s1 = offset((short*)indices, row_offset_block);
    values = (half2*)offset((half*)values, row_offset);

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
    dense_matrix = (half2*)offset((half*)dense_matrix, n_index);
    int tidy = threadIdx.x % 4;
    int tidx = threadIdx.x / 4;
    dense_matrix = (half2*)offset((half*)dense_matrix, tidy * n * 8);
    half8 *dense_matrix_h8 = offset((half8*)dense_matrix, tidx);

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
    half8 *values_h8 = offset((half8*)values, threadIdx.x);

    // Register fragment for dense matrix data. Each
    // thread owns 1/32nd of a 32x128 tile for a total
    // of 128 half-precision values - i.e., 64x half2s.
    half2 rhs_fragment[64];
    half8 *rhs_fragment_h8 = (half8*)rhs_fragment;

    // Register fragment for the results accumulators.
    //
    // Each thread ends up with an 8x16 square of results.
    // We accumulate in 32-bit as is standard.
    //
    // TODO(tgale): There are alternative techniques for
    // zero-ing our accumulators that we should explore.
    float out_fragment[128];
#pragma unroll
    for (int i = 0; i < 128; ++i) {
      // NOTE: The compiler always duplicates the zeroing instructions.
      // Doing this with asm avoids this issue.
      asm volatile("mov.f32 %0, 0F00000000;" : "=f"(out_fragment[i]));
    }

    // TODO(tgale): Figure out a way to stop the compiler from
    // reusing these values across the main loop.
    // asm("mov.b32 %0, %0;" : "+r"(tidx) :);
    // asm("mov.b32 %0, %0;" : "+r"(tidy) :);    

    // Prefetch the index for the next iteration.
    int lhs_index;
    if (nonzeros >= kTileK * kBlockDim) {
      lhs_index = Load(indices_s1) * n;
      ++indices_s1;
    }
    
    //
    /// Main loop.
    //
#pragma unroll 1
    for (; nonzeros >= kTileK * kBlockDim; nonzeros -= kTileK * kBlockDim) {
      // Load the sparse block data.
      //
      // We load an entire 32x32 sparse block in 4x 8-wide
      // vector loads.
#pragma unroll
      for (int i = 0; i < 4; ++i) {
	lhs_fragment_h8[i] = load_nc_h8(values_h8);
	values_h8 = offset(values_h8, 32);
      }      

      // Load the dense matrix tile data.
      half8 *rhs_h8 = (half8*)offset((half*)dense_matrix_h8, lhs_index);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
	half8 *inner_rhs_h8 = rhs_h8;
#pragma unroll
	for (int j = 0; j < 2; ++j) {
	  rhs_fragment_h8[i * 2 + j] = load_nc_h8(inner_rhs_h8);
	  inner_rhs_h8 = offset(inner_rhs_h8, 8);
	}
	rhs_h8 = (half8*)offset((half*)rhs_h8, n);
      }

      // Prefect the index for the next iteration.
      if (nonzeros >= 2 * kTileK * kBlockDim) {
	lhs_index = Load(indices_s1) * n;
	++indices_s1;
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
	for (int j = 0; j < 8; ++j) {
	  Transpose(rhs_fragment[j + 16 * i], rhs_fragment[j + 16 * i + 8]);
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
      for (int k_item_idx = 0; k_item_idx < 2; ++k_item_idx) {
#pragma unroll
	for (int y_item_idx = 0; y_item_idx < 4; ++y_item_idx) {
#pragma unroll
	  for (int x_item_idx = 0; x_item_idx < 8; ++x_item_idx) {
	    const int lhs_idx = k_item_idx * 2 + y_item_idx * 4;
	    const int rhs_idx = x_item_idx + 32 * k_item_idx;
	    const int out_idx = 2 * x_item_idx + y_item_idx * 32;

	    mma_m8n16k16(lhs_fragment[lhs_idx],
			 lhs_fragment[lhs_idx + 1],
			 rhs_fragment[rhs_idx],
			 rhs_fragment[rhs_idx + 8],
			 rhs_fragment[rhs_idx + 16],
			 rhs_fragment[rhs_idx + 24],
			 out_fragment[out_idx],
			 out_fragment[out_idx + 16],
			 out_fragment[out_idx + 1],
			 out_fragment[out_idx + 17]);
	  }
	}
      }
    }
    
    // Convert the accumulators to half-precision.
    half2 out_fragment_h2[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) {
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
    output_matrix = (half2*)offset((half*)output_matrix, m_index * n + n_index);
    tidy = threadIdx.x % 4;
    tidx = threadIdx.x / 4;
    output_matrix = (half2*)offset((half*)output_matrix, tidy * n * 2);
    half8 *output_matrix_h8 = offset((half8*)output_matrix, tidx);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      half8* out_h8 = output_matrix_h8;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
#pragma unroll
	for (int l = 0; l < 2; ++l) {
	  store_h8(out_fragment_h8[i * 4 + j * 2 + l], offset(out_h8, 8 * l));
	}
	out_h8 = (half8*)offset((half*)out_h8, n);	
      }
      output_matrix_h8 = (half8*)offset((half*)output_matrix_h8, 8 * n);
    }
  }
  
};

__global__ void __launch_bounds__(32, 8) dsd_b32_m32n128k32_h8_h8_kernel(
    int m, int k, int n,
    const half2* __restrict__ values,
    const int* __restrict__ offsets,
    const short2* __restrict__ indices,
    const half2* __restrict__ dense_matrix,
    half2* __restrict__ output_matrix) {
  // TODO(tgale): Figure out a cleaner way to fail on unsupported hardware.
#if __CUDA_ARCH__ >= 800
  dsd_b32_m32n128k32_h8_h8::KernelFn(
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
  
bool can_launch_dsd_b32_m32n128k32_h8_h8(
  int m, int k, int n, int nonzeros, int block_size) {
  bool can_launch = true;
  can_launch &= block_size == 32;
  can_launch &= (m % block_size) == 0;
  can_launch &= (k % block_size) == 0;
  can_launch &= (n % 128) == 0;
  return can_launch;
}
  
cudaError_t launch_dsd_b32_m32n128k32_h8_h8(
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
  CHECK_EQ(n % 128, 0);
  dim3 grid_dim(cdiv(m, block_size), cdiv(n, 128), 1);
  dim3 block_dim(32, 1, 1);

  dsd_b32_m32n128k32_h8_h8_kernel<<<grid_dim, block_dim, 0, stream>>>(
    m, k, n, (const half2*)values, offsets, (const short2*)indices,
    (const half2*)dense_matrix, (half2*)output_matrix);
  return cudaGetLastError();
}
  
}  // namespace sm80
}  // namespace block
}  // namespace sputnik
