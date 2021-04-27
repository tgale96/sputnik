#ifndef THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_GEMM_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_GEMM_H_

#include "sputnik/block/cutlass/block_pitch_linear.h"
#include "cutlass/gemm/kernel/gemm_universal.h"

namespace sputnik {
namespace block {
namespace cutlass {

// Forward declare for Filter.
template <
  typename Mma_,
  typename Epilogue_,
  typename ThreadblockSwizzle_>
struct BlockGemm;
  
template <
  typename Arguments,
  typename LayoutA,
  typename LayoutB>
struct FilterHelper {

  using RetA = int;
  using RetB = int;

  CUTLASS_HOST_DEVICE
  static RetA ParamsA(Arguments args) {
    return args.lda;
  }

  CUTLASS_HOST_DEVICE
  static RetB ParamsB(Arguments args) {
    return args.ldb;
  }
};

// TODO(tgale): Augment this to pass all information about a
// matrix into the params.
template <
  typename Arguments,
  typename LayoutB>
struct FilterHelper<Arguments, BlockPitchLinear, LayoutB> {

  using RetA = void *;
  using RetB = int;

  CUTLASS_HOST_DEVICE
  static RetA ParamsA(Arguments args) {
    return args.op_A.offsets;
  }

  CUTLASS_HOST_DEVICE
  static RetB ParamsB(Arguments args) {
    return args.ldb;
  }
};  
  
// Helper to handle mixes of sparse and dense arguments.
template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_>
struct Filter {

  using Gemm = BlockGemm<Mma_, Epilogue_, ThreadblockSwizzle_>;

  using Arguments = typename Gemm::Arguments;
  using LayoutA = typename Gemm::Mma::IteratorA::Layout;
  using ElementA = typename Gemm::Mma::IteratorA::Element;
  using LayoutB = typename Gemm::Mma::IteratorB::Layout;
  using ElementB = typename Gemm::Mma::IteratorB::Element;

  using Helper = FilterHelper<Arguments, LayoutA, LayoutB>;
  
  // Default config - no blocksparse arguments.

  CUTLASS_HOST_DEVICE
  static typename Helper::RetA ParamsA(Arguments args) {
    return Helper::ParamsA(args);
  }

  CUTLASS_HOST_DEVICE
  static typename Helper::RetB ParamsB(Arguments args) {
    return Helper::ParamsB(args);
  }  
};

// Gemm class.
  
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

  // Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using ArgFilter = Filter<Mma, Epilogue, ThreadblockSwizzle>;
  
  //
  // Structures
  //

  // Operand structure.
  struct Op {
    void * data;
    void * offsets;
    void * indices;

    CUTLASS_HOST_DEVICE
    Op(void const *data_, void const *offsets_, void const *indices_) :
      data(const_cast<void*>(data_)),
      offsets(const_cast<void*>(offsets_)),
      indices(const_cast<void*>(indices_)) {}

    CUTLASS_HOST_DEVICE
    Op(void const *data_) :
      data(const_cast<void*>(data_)), offsets(nullptr), indices(nullptr) {}
  };
  
  /// Argument structure
  struct Arguments {
      
    //
    // Data members
    //

    ::cutlass::gemm::GemmCoord problem_size;
    typename EpilogueOutputOp::Params epilogue;

    Op op_A;
    Op op_B;
    Op op_C;
    Op op_D;

    int lda;
    int ldb;
    int ldc;
    int ldd;

    //
    // Methods
    //

    Arguments(): 
      op_A(nullptr), op_B(nullptr), op_C(nullptr), op_D(nullptr) { }

    /// constructs an arguments structure
    Arguments(
      ::cutlass::gemm::GemmCoord problem_size,
      typename EpilogueOutputOp::Params epilogue,
      Op op_A,
      Op op_B,
      Op op_C,
      Op op_D,
      int lda,
      int ldb,
      int ldc,
      int ldd
    ):
      problem_size(problem_size), 
      epilogue(epilogue), 
      op_A(op_A), op_B(op_B), op_C(op_C), op_D(op_D), 
      lda(lda), ldb(ldb), ldc(ldc), ldd(ldd) {}

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
      op_A(ptr_A), op_B(ptr_B), op_C(ptr_C), op_D(ptr_D), 
      lda(lda), ldb(ldb), ldc(ldc), ldd(ldd) {}    

    /// Returns arguments for the transposed problem
    Arguments transposed_problem() const {
      Arguments args(*this);
      std::swap(args.problem_size.m(), args.problem_size.n());
      std::swap(args.op_A, args.op_B);
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

    Op op_A;
    Op op_B;
    Op op_C;
    Op op_D;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params():
      params_C(0),
      params_D(0),
      op_A(nullptr),
      op_B(nullptr),
      op_C(nullptr),
      op_D(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(
      Arguments const &args,
      ::cutlass::gemm::GemmCoord const & grid_tiled_shape):
      problem_size(args.problem_size),
      grid_tiled_shape(grid_tiled_shape),
      params_A(ArgFilter::ParamsA(args)),
      params_B(ArgFilter::ParamsB(args)),
      params_C(args.ldc),
      params_D(args.ldd),
      output_op(args.epilogue),
      op_A(args.op_A),
      op_B(args.op_B),
      op_C(args.op_C),
      op_D(args.op_D) {}

    CUTLASS_HOST_DEVICE
    void update(Arguments const &args) {
      op_A = args.op_A;
      op_B = args.op_B;
      op_C = args.op_C;
      op_D = args.op_D;
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

    ElementA *ptr_A = static_cast<ElementA *>(params.op_A.data); 
    ElementB *ptr_B = static_cast<ElementB *>(params.op_B.data);

    // TODO(tgale): Do we need to synchronize here?
    __syncthreads();

    // Config::OffsetA(threadblock_tile_offset.m(), Mma::Shape::kM, params.op_A);
    // Config::OffsetB(threadblock_tile_offset.n(), Mma::Shape::kN, params.op_B);
    // Config::ParamsA(params.op_A);
    // Config::ParamsB(params.op_B);
    
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

    ElementC *ptr_C = static_cast<ElementC *>(params.op_C.data); 
    ElementC *ptr_D = static_cast<ElementC *>(params.op_D.data);

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
