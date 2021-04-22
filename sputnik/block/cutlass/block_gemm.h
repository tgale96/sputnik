#ifndef THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_GEMM_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_GEMM_H_

#include "cutlass/gemm/kernel/gemm_universal.h"

namespace sputnik {
namespace block {
namespace cutlass {
  
template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_    ///! Threadblock swizzling function
>
struct BlockGemm {
public:

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using ElementA = typename Mma::IteratorA::Element;
  using ElementB = typename Mma::IteratorB::Element;
  using ElementC = typename Epilogue::OutputTileIterator::Element;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    ::cutlass::gemm::GemmCoord problem_size;
    typename EpilogueOutputOp::Params epilogue;

    void const * ptr_A;
    void const * ptr_B;
    void const * ptr_C;
    void * ptr_D;

    int lda;
    int ldb;
    int ldc;
    int ldd;

    //
    // Methods
    //

    Arguments(): 
      ptr_A(nullptr), ptr_B(nullptr), ptr_C(nullptr), ptr_D(nullptr) { }

    /// constructs an arguments structure
    Arguments(
      ::cutlass::gemm::GemmCoord problem_size,
      typename EpilogueOutputOp::Params epilogue,
      void const * ptr_A,
      void const * ptr_B,
      void const * ptr_C,
      void * ptr_D,
      int lda,
      int ldb,
      int ldc,
      int ldd
    ):
      problem_size(problem_size), 
      epilogue(epilogue), 
      ptr_A(ptr_A), ptr_B(ptr_B), ptr_C(ptr_C), ptr_D(ptr_D), 
      lda(lda), ldb(ldb), ldc(ldc), ldd(ldd) {}

    /// Returns arguments for the transposed problem
    Arguments transposed_problem() const {
      Arguments args(*this);
      std::swap(args.problem_size.m(), args.problem_size.n());
      std::swap(args.ptr_A, args.ptr_B);
      std::swap(args.lda, args.ldb);
      return args;
    }
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {

    ::cutlass::gemm::GemmCoord problem_size;
    ::cutlass::gemm::GemmCoord grid_tiled_shape;
    
    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorB::Params params_B;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    
    typename EpilogueOutputOp::Params output_op;

    void * ptr_A;
    void * ptr_B;
    void * ptr_C;
    void * ptr_D;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params():
      params_A(0),
      params_B(0),
      params_C(0),
      params_D(0),
      ptr_A(nullptr),
      ptr_B(nullptr),
      ptr_C(nullptr),
      ptr_D(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(
      Arguments const &args,
      ::cutlass::gemm::GemmCoord const & grid_tiled_shape):
      problem_size(args.problem_size),
      grid_tiled_shape(grid_tiled_shape),
      params_A(args.lda),
      params_B(args.ldb),
      params_C(args.ldc),
      params_D(args.ldd),
      output_op(args.epilogue),
      ptr_A(const_cast<void *>(args.ptr_A)),
      ptr_B(const_cast<void *>(args.ptr_B)),
      ptr_C(const_cast<void *>(args.ptr_C)),
      ptr_D(args.ptr_D) {}

    CUTLASS_HOST_DEVICE
    void update(Arguments const &args) {
      ptr_A = const_cast<void *>(args.ptr_A);
      ptr_B = const_cast<void *>(args.ptr_B);
      ptr_C = const_cast<void *>(args.ptr_C);
      ptr_D = args.ptr_D;
      output_op = args.epilogue;
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

public:

  //
  // Methods
  //

  CUTLASS_DEVICE
  BlockGemm() {} 

  /// Determines whether kernel satisfies alignment
  static ::cutlass::Status can_implement(
    ::cutlass::gemm::GemmCoord const & problem_size) {
    // NOTE: We don't support interleaved layouts.
    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    if ((problem_size.m() % kAlignmentA) || (problem_size.k() % kAlignmentA) ||
      (problem_size.n() % kAlignmentB) || (problem_size.k() % kAlignmentB) ||
      (problem_size.m() % kAlignmentC) || (problem_size.n() % kAlignmentC)) {
      return ::cutlass::Status::kErrorMisalignedOperand;
    }
    return ::cutlass::Status::kSuccess;
  }

  static ::cutlass::Status can_implement(Arguments const &args) {
    return can_implement(args.problem_size);
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    ::cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.grid_tiled_shape);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }

    int problem_size_k = params.problem_size.k();

    ElementA *ptr_A = static_cast<ElementA *>(params.ptr_A); 
    ElementB *ptr_B = static_cast<ElementB *>(params.ptr_B);

    // TODO(tgale): Do we need to synchronize here?
    __syncthreads();

    // Compute initial location in logical coordinates
    ::cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM, 0
    };

    ::cutlass::MatrixCoord tb_offset_B{
      0, threadblock_tile_offset.n() * Mma::Shape::kN
    };


    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      params.params_A,
      ptr_A,
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A);

    typename Mma::IteratorB iterator_B(
      params.params_B,
      ptr_B,
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B);

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k + Mma::Shape::kK - 1) / Mma::Shape::kK;

    // Compute threadblock-scoped matrix multiply-add
    mma(
      gemm_k_iterations, 
      accumulators, 
      iterator_A, 
      iterator_B, 
      accumulators);

    //
    // Epilogue
    //

    EpilogueOutputOp output_op(params.output_op);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.grid_tiled_shape);

    //assume identity swizzle
    ::cutlass::MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.n() * Mma::Shape::kN
    );

    ElementC *ptr_C = static_cast<ElementC *>(params.ptr_C); 
    ElementC *ptr_D = static_cast<ElementC *>(params.ptr_D);

    //
    // Fetch pointers based on mode.
    //

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
      params.params_C,
      ptr_C,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,
      ptr_D,
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    Epilogue epilogue(
      shared_storage.epilogue, 
      thread_idx, 
      warp_idx, 
      lane_idx);
    
    // Execute the epilogue operator to update the destination tensor.
    epilogue(
      output_op, 
      iterator_D, 
      accumulators, 
      iterator_C); 
  }
};

}  // namespace cutlass
}  // namespace block
}  // namespace sputnik 

#endif  // THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_GEMM_H_ 
