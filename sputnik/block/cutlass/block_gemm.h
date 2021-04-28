#ifndef THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_GEMM_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_GEMM_H_

#include "sputnik/block/cutlass/block_pitch_linear.h"
#include "sputnik/block/cutlass/op.h"

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
  typename Gemm,
  typename LayoutA,
  typename LayoutB>
struct ConfigHelper {

  using Arguments = typename Gemm::Arguments;
  using Params = typename Gemm::Params;
  
  using RetParamsA = int;
  using RetParamsB = int;
  using RetOffsetA = ::cutlass::MatrixCoord;
  using RetOffsetB = ::cutlass::MatrixCoord;
  
  Params const &params;
  const ::cutlass::gemm::GemmCoord &threadblock_tile_offset;
  
  CUTLASS_DEVICE
  ConfigHelper(Params const &params_,
	       const ::cutlass::gemm::GemmCoord &threadblock_tile_offset_) :
    params(params_), threadblock_tile_offset(threadblock_tile_offset_) {}
    
  CUTLASS_HOST_DEVICE
  static RetParamsA ParamsA(Arguments args) {
    return args.op_A.ld;
  }

  CUTLASS_HOST_DEVICE
  static RetParamsB ParamsB(Arguments args) {
    return args.op_B.ld;
  }

  CUTLASS_DEVICE
  RetOffsetA OffsetA() const {
    RetOffsetA tb_offset_A{
      threadblock_tile_offset.m() * Gemm::Mma::Shape::kM, 0
    };
    return tb_offset_A;
  }

  CUTLASS_DEVICE
  RetOffsetB OffsetB() const {
    RetOffsetB tb_offset_B{
      0, threadblock_tile_offset.n() * Gemm::Mma::Shape::kN
    };
    return tb_offset_B;
  }
			    
  CUTLASS_DEVICE
  int StepsK() const {
    return (params.problem_size.k() + Gemm::Mma::Shape::kK - 1) /
      Gemm::Mma::Shape::kK;
  }
};

template <
  typename Gemm,
  typename LayoutB>
struct ConfigHelper<Gemm, BlockPitchLinear, LayoutB> {
  using Arguments = typename Gemm::Arguments;
  using Params = typename Gemm::Params;
  
  using RetParamsA = Op;
  using RetParamsB = Op;
  using RetOffsetA = int;
  using RetOffsetB = ::cutlass::MatrixCoord;

  Params const &params;
  const ::cutlass::gemm::GemmCoord &threadblock_tile_offset;
  int offset_a, nnz_a;
  
  CUTLASS_DEVICE
  ConfigHelper(Params const &params_,
	       const ::cutlass::gemm::GemmCoord &threadblock_tile_offset_) :
    params(params_), threadblock_tile_offset(threadblock_tile_offset_) {
    // Load the offset and number of nonzeros.
    int *offset_ptr_a = (int*)params_.op_A.offsets;
    int block_row_idx = threadblock_tile_offset.m();    
    offset_a = __ldg(offset_ptr_a + block_row_idx);

    // In scalar elements. Divide by the block size to get
    // the number of columns to process.
    nnz_a = __ldg(offset_ptr_a + block_row_idx + 1) - offset_a;
  }
  
  CUTLASS_HOST_DEVICE
  static RetParamsA ParamsA(Arguments args) {
    return args.op_A;
  }

  CUTLASS_HOST_DEVICE
  static RetParamsB ParamsB(Arguments args) {
    return args.op_B;
  }

  CUTLASS_DEVICE
  RetOffsetA OffsetA() const {
    return offset_a;
  }

  CUTLASS_DEVICE
  RetOffsetB OffsetB() const {
    RetOffsetB tb_offset_B{
      0, threadblock_tile_offset.n() * Gemm::Mma::Shape::kN
    };
    return tb_offset_B;
  }
			    
  CUTLASS_DEVICE
  int StepsK() const {
    int nnz_cols_a = nnz_a / Gemm::Mma::IteratorA::Shape::kBlock;    
    return (nnz_cols_a + Gemm::Mma::Shape::kK - 1) / Gemm::Mma::Shape::kK;
  }  
};  
  
// Helper to handle mixes of sparse and dense arguments.
template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_>
struct Config {

  using Gemm = BlockGemm<Mma_, Epilogue_, ThreadblockSwizzle_>;

  using Arguments = typename Gemm::Arguments;
  using Params = typename Gemm::Params;
  
  using LayoutA = typename Gemm::Mma::IteratorA::Layout;
  using ElementA = typename Gemm::Mma::IteratorA::Element;
  using LayoutB = typename Gemm::Mma::IteratorB::Layout;
  using ElementB = typename Gemm::Mma::IteratorB::Element;

  using Helper = ConfigHelper<Gemm, LayoutA, LayoutB>;

  // Underlying helper.
  Helper helper;
  
  CUTLASS_DEVICE
  Config(Params const &params_,
	 const ::cutlass::gemm::GemmCoord &threadblock_tile_offset_) :
    helper(params_, threadblock_tile_offset_) {}

  CUTLASS_HOST_DEVICE
  static typename Helper::RetParamsA ParamsA(Arguments args) {
    return Helper::ParamsA(args);
  }

  CUTLASS_HOST_DEVICE
  static typename Helper::RetParamsB ParamsB(Arguments args) {
    return Helper::ParamsB(args);
  }

  CUTLASS_DEVICE
  typename Helper::RetOffsetA OffsetA() const { return helper.OffsetA(); }

  CUTLASS_DEVICE
  typename Helper::RetOffsetB OffsetB() const { return helper.OffsetB(); }
			    
  CUTLASS_DEVICE
  int StepsK() const { return helper.StepsK(); }
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

  using Config = Config<Mma, Epilogue, ThreadblockSwizzle>;
  
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

    Op op_A;
    Op op_B;
    Op op_C;
    Op op_D;

    //
    // Methods
    //

    Arguments(): 
      op_A(nullptr, 0), op_B(nullptr, 0), op_C(nullptr, 0), op_D(nullptr, 0) { }

    /// constructs an arguments structure
    Arguments(
      ::cutlass::gemm::GemmCoord problem_size,
      typename EpilogueOutputOp::Params epilogue,
      Op op_A,
      Op op_B,
      Op op_C,
      Op op_D
    ):
      problem_size(problem_size), 
      epilogue(epilogue), 
      op_A(op_A), op_B(op_B), op_C(op_C), op_D(op_D) {}

    /// Returns arguments for the transposed problem
    Arguments transposed_problem() const {
      Arguments args(*this);
      std::swap(args.problem_size.m(), args.problem_size.n());
      std::swap(args.op_A, args.op_B);
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
      op_A(nullptr, 0),
      op_B(nullptr, 0),
      op_C(nullptr, 0),
      op_D(nullptr, 0) {}

    CUTLASS_HOST_DEVICE
    Params(
      Arguments const &args,
      ::cutlass::gemm::GemmCoord const & grid_tiled_shape):
      problem_size(args.problem_size),
      grid_tiled_shape(grid_tiled_shape),
      params_A(Config::ParamsA(args)),
      params_B(Config::ParamsB(args)),
      params_C(args.op_C.ld),
      params_D(args.op_D.ld),
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

    Config config(params, threadblock_tile_offset);

    ElementA *ptr_A = static_cast<ElementA *>(params.op_A.data); 
    ElementB *ptr_B = static_cast<ElementB *>(params.op_B.data);

    // TODO(tgale): Do we need to synchronize here?
    __syncthreads();

    // Config::OffsetA(threadblock_tile_offset.m(), Mma::Shape::kM, params.op_A);
    // Config::OffsetB(threadblock_tile_offset.n(), Mma::Shape::kN, params.op_B);
    // Config::ParamsA(params.op_A);
    // Config::ParamsB(params.op_B);
    
    // Compute initial location in logical coordinates
    auto tb_offset_A = config.OffsetA();
    auto tb_offset_B = config.OffsetB();

    // Compute position within threadblock
    int thread_idx = threadIdx.x;
    int problem_size_k = params.problem_size.k();
    
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
    int gemm_k_iterations = config.StepsK();

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
