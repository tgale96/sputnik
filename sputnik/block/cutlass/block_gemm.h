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
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename Epilogue::OutputTileIterator::Layout;

  static ::cutlass::ComplexTransform const kTransformA = Mma::kTransformA;
  static ::cutlass::ComplexTransform const kTransformB = Mma::kTransformB;
  using Operator = typename Mma::Operator;

  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
  static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
  static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

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

    ::cutlass::gemm::GemmUniversalMode mode;
    ::cutlass::gemm::GemmCoord problem_size;
    int batch_count;

    typename EpilogueOutputOp::Params epilogue;

    void const * ptr_A;
    void const * ptr_B;
    void const * ptr_C;
    void * ptr_D;

    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C;
    int64_t batch_stride_D;

    int lda;
    int ldb;
    int ldc;
    int ldd;

    //
    // Methods
    //

    Arguments(): 
      mode(::cutlass::gemm::GemmUniversalMode::kGemm), 
      batch_count(1), 
      ptr_A(nullptr), ptr_B(nullptr), ptr_C(nullptr), ptr_D(nullptr) { }

    /// constructs an arguments structure
    Arguments(
      ::cutlass::gemm::GemmUniversalMode mode,
      ::cutlass::gemm::GemmCoord problem_size,
      int batch_count,
      typename EpilogueOutputOp::Params epilogue,
      void const * ptr_A,
      void const * ptr_B,
      void const * ptr_C,
      void * ptr_D,
      int64_t batch_stride_A,
      int64_t batch_stride_B,
      int64_t batch_stride_C,
      int64_t batch_stride_D,
      int lda,
      int ldb,
      int ldc,
      int ldd
    ):
      mode(mode), 
      problem_size(problem_size), 
      batch_count(batch_count),
      epilogue(epilogue), 
      ptr_A(ptr_A), ptr_B(ptr_B), ptr_C(ptr_C), ptr_D(ptr_D), 
      batch_stride_A(batch_stride_A), batch_stride_B(batch_stride_B), batch_stride_C(batch_stride_C), batch_stride_D(batch_stride_D), 
      lda(lda), ldb(ldb), ldc(ldc), ldd(ldd) {

      CUTLASS_TRACE_HOST("GemmUniversal::Arguments::Arguments() - problem_size: " << problem_size);
      }

    /// Returns arguments for the transposed problem
    Arguments transposed_problem() const {
      Arguments args(*this);
      
      std::swap(args.problem_size.m(), args.problem_size.n());
      std::swap(args.ptr_A, args.ptr_B);
      std::swap(args.lda, args.ldb);
      std::swap(args.batch_stride_A, args.batch_stride_B);

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

    ::cutlass::gemm::GemmUniversalMode mode;
    int batch_count;
    int gemm_k_size;

    void * ptr_A;
    void * ptr_B;
    void * ptr_C;
    void * ptr_D;

    int64_t batch_stride_A;
    int64_t batch_stride_B;
    int64_t batch_stride_C;
    int64_t batch_stride_D;

    int *semaphore;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params():
      params_A(0),
      params_B(0),
      params_C(0),
      params_D(0),
      batch_count(0),
      gemm_k_size(0),
      mode(::cutlass::gemm::GemmUniversalMode::kGemm),
      ptr_A(nullptr),
      ptr_B(nullptr),
      ptr_C(nullptr),
      ptr_D(nullptr),
      batch_stride_A(0),
      batch_stride_B(0),
      batch_stride_C(0),
      batch_stride_D(0),
      semaphore(nullptr) { }

    CUTLASS_HOST_DEVICE
    Params(
      Arguments const &args,
      ::cutlass::gemm::GemmCoord const & grid_tiled_shape,
      int gemm_k_size,
      void *workspace = nullptr
    ):
      problem_size(args.problem_size),
      grid_tiled_shape(grid_tiled_shape),
      params_A(args.lda),
      params_B(args.ldb),
      params_C(args.ldc),
      params_D(args.ldd),
      output_op(args.epilogue),
      mode(args.mode),
      batch_count(args.batch_count),
      gemm_k_size(gemm_k_size),
      ptr_A(const_cast<void *>(args.ptr_A)),
      ptr_B(const_cast<void *>(args.ptr_B)),
      ptr_C(const_cast<void *>(args.ptr_C)),
      ptr_D(args.ptr_D),
      batch_stride_A(args.batch_stride_A),
      batch_stride_B(args.batch_stride_B),
      batch_stride_C(args.batch_stride_C),
      batch_stride_D(args.batch_stride_D),
      semaphore(static_cast<int *>(workspace)) {

      CUTLASS_TRACE_HOST("GemmUniversal::Params::Params() - problem_size: " << problem_size);
    }

    CUTLASS_HOST_DEVICE
    void update(
      Arguments const &args,
      void *workspace = nullptr) {

      ptr_A = const_cast<void *>(args.ptr_A);
      ptr_B = const_cast<void *>(args.ptr_B);
      ptr_C = const_cast<void *>(args.ptr_C);
      ptr_D = args.ptr_D;

      batch_stride_A = args.batch_stride_A;
      batch_stride_B = args.batch_stride_B;
      batch_stride_C = args.batch_stride_C;
      batch_stride_D = args.batch_stride_D;

      output_op = args.epilogue;

      semaphore = static_cast<int *>(workspace);

      CUTLASS_TRACE_HOST("GemmUniversal::Params::update()");
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
  BlockGemm() { } 

  /// Determines whether kernel satisfies alignment
  static ::cutlass::Status can_implement(
    ::cutlass::gemm::GemmCoord const & problem_size) {

    CUTLASS_TRACE_HOST("GemmUniversal::can_implement()");

    static int const kAlignmentA = (::cutlass::platform::is_same<typename Mma::IteratorA::Layout,
                                                      ::cutlass::layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (::cutlass::platform::is_same<typename Mma::IteratorA::Layout,
                                                        ::cutlass::layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = (::cutlass::platform::is_same<typename Mma::IteratorB::Layout,
                                                       ::cutlass::layout::RowMajorInterleaved<32>>::value)
                                   ? 32
                                   : (::cutlass::platform::is_same<typename Mma::IteratorB::Layout,
                                                        ::cutlass::layout::RowMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

    if ((problem_size.m() % kAlignmentA) || (problem_size.k() % kAlignmentA) ||
      (problem_size.n() % kAlignmentB) || (problem_size.k() % kAlignmentB) ||
      (problem_size.m() % kAlignmentC) || (problem_size.n() % kAlignmentC)) {

      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand");
      return ::cutlass::Status::kErrorMisalignedOperand;
    }

    CUTLASS_TRACE_HOST("  returning kSuccess");

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

    //
    // Fetch pointers based on mode.
    //
    
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
