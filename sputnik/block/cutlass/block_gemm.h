#ifndef THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_GEMM_H_
#define THIRD_PARTY_SPUTNIK_BLOCK_CUTLASS_BLOCK_GEMM_H_

#include "sputnik/block/cutlass/block_pitch_linear.h"
#include "sputnik/block/cutlass/index_merge.h"
#include "sputnik/block/cutlass/op.h"
#include "sputnik/block/cutlass/type_utils.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm.h"

namespace sputnik {
namespace block {
namespace cutlass {

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

  using GemmCoord = ::cutlass::gemm::GemmCoord;
  using Arguments = typename Gemm::Arguments;
  using Params = typename Gemm::Params;

  using RetParamsA = int;
  using RetParamsB = int;
  using RetOffsetA = ::cutlass::MatrixCoord;
  using RetOffsetB = ::cutlass::MatrixCoord;

  using ParamsA = typename Gemm::Mma::IteratorA::Params;
  using ParamsB = typename Gemm::Mma::IteratorB::Params;

  static constexpr int kSmemBytes = 1;

  Params const &params;
  const GemmCoord &offset;

  CUTLASS_DEVICE
  ConfigHelper(Params const &params_,
	       const GemmCoord &offset_,
               uint8_t *smem) :
    params(params_), offset(offset_) {}

  CUTLASS_HOST_DEVICE
  static RetParamsA ItArgsA(Arguments args) {
    return args.op_A.ld;
  }

  CUTLASS_HOST_DEVICE
  static RetParamsB ItArgsB(Arguments args) {
    return args.op_B.ld;
  }

  CUTLASS_DEVICE
  ParamsA UpdateParamsA(ParamsA const &params) const {
    return params;
  }

  CUTLASS_DEVICE
  ParamsB UpdateParamsB(ParamsB const &params) const {
    return params;
  }

  CUTLASS_DEVICE
  RetOffsetA OffsetA() const {
    RetOffsetA tb_offset_A{
      offset.m() * Gemm::Mma::Shape::kM, 0
    };
    return tb_offset_A;
  }

  CUTLASS_DEVICE
  RetOffsetB OffsetB() const {
    RetOffsetB tb_offset_B{
      0, offset.n() * Gemm::Mma::Shape::kN
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

  using GemmCoord = ::cutlass::gemm::GemmCoord;

  using Arguments = typename Gemm::Arguments;
  using Params = typename Gemm::Params;

  using RetParamsA = Op;
  using RetParamsB = Op;
  using RetOffsetA = int;
  using RetOffsetB = ::cutlass::MatrixCoord;

  using ParamsA = typename Gemm::Mma::IteratorA::Params;
  using ParamsB = typename Gemm::Mma::IteratorB::Params;

  using ElementA = typename Gemm::Mma::IteratorA::Element;
  using MetaA = typename Type<ElementA>::Meta;

  static const int kBlockSize = Gemm::Mma::IteratorA::Shape::kBlock *
    Gemm::Mma::IteratorA::Shape::kBlock;

  static constexpr int kSmemBytes = 1;

  Params const &params;
  const GemmCoord &offset;
  int offset_a, nnz_a;

  CUTLASS_DEVICE
  ConfigHelper(Params const &params_,
	       const GemmCoord &offset_,
               uint8_t *smem) :
    params(params_), offset(offset_) {
    // Load the offset and number of nonzeros.
    int *offset_ptr_a = (int*)params_.op_A.offsets;
    int block_row_idx = offset.m();

    offset_a = __ldg(offset_ptr_a + block_row_idx);

    // In blocks. Multiply by the block size to get
    // the number of columns to process.
    nnz_a = __ldg(offset_ptr_a + block_row_idx + 1) - offset_a;
  }

  CUTLASS_HOST_DEVICE
  static RetParamsA ItArgsA(Arguments args) {
    return args.op_A;
  }

  CUTLASS_HOST_DEVICE
  static RetParamsB ItArgsB(Arguments args) {
    // Set the meta-data pointer for operand B.
    args.op_B.indices = args.op_A.indices;
    return args.op_B;
  }

  CUTLASS_DEVICE
  ParamsA UpdateParamsA(ParamsA const &params) const {
    // NOTE: This should be elided by the compiler for
    // the non-transposed kernels where we do not use
    // explicit block offsets.
    ParamsA out = params;
    out.block_offsets += offset_a;

    // Set the number of steps for predicated index loads.
    out.steps_k = StepsK();
    return out;
  }

  CUTLASS_DEVICE
  ParamsB UpdateParamsB(ParamsB const &params) const {
    ParamsB out = params;
    out.indices += offset_a;

    // Set the number of steps for predicated index loads.
    out.steps_k = StepsK();
    return out;
  }

  CUTLASS_DEVICE
  RetOffsetA OffsetA() const {
    return offset_a * kBlockSize;
  }

  CUTLASS_DEVICE
  RetOffsetB OffsetB() const {
    RetOffsetB tb_offset_B{
      0, offset.n() * Gemm::Mma::Shape::kN
    };
    return tb_offset_B;
  }

  CUTLASS_DEVICE
  int StepsK() const {
    int nnz_cols_a = nnz_a * Gemm::Mma::IteratorA::Shape::kBlock;
    return (nnz_cols_a + Gemm::Mma::Shape::kK - 1) / Gemm::Mma::Shape::kK;
  }
};

template <
  typename Gemm,
  typename LayoutA>
struct ConfigHelper<Gemm, LayoutA, BlockPitchLinear> {
  using GemmCoord = ::cutlass::gemm::GemmCoord;

  using Arguments = typename Gemm::Arguments;
  using Params = typename Gemm::Params;

  using RetParamsA = Op;
  using RetParamsB = Op;
  using RetOffsetA = ::cutlass::MatrixCoord;
  using RetOffsetB = int;

  using ParamsA = typename Gemm::Mma::IteratorA::Params;
  using ParamsB = typename Gemm::Mma::IteratorB::Params;

  using ElementB = typename Gemm::Mma::IteratorB::Element;
  using MetaB = typename Type<ElementB>::Meta;

  static const int kBlockSize = Gemm::Mma::IteratorB::Shape::kBlock *
    Gemm::Mma::IteratorB::Shape::kBlock;

  static constexpr int kSmemBytes = 1;

  Params const &params;
  const GemmCoord &offset;
  int offset_b, nnz_b;

  CUTLASS_DEVICE
  ConfigHelper(Params const &params_,
	       const GemmCoord &offset_,
               uint8_t *smem) :
    params(params_), offset(offset_) {
    // Load the offset and number of nonzeros.
    int *offset_ptr_b = (int*)params_.op_B.offsets;
    int block_cow_idx = offset.n();

    offset_b = __ldg(offset_ptr_b + block_cow_idx);

    // In blocks. Multiply by the block size to get
    // the number of columns to process.
    nnz_b = __ldg(offset_ptr_b + block_cow_idx + 1) - offset_b;
  }

  CUTLASS_HOST_DEVICE
  static RetParamsA ItArgsA(Arguments args) {
    // Set the meta-data pointer for operand A.
    args.op_A.indices = args.op_B.indices;
    return args.op_A;
  }

  CUTLASS_HOST_DEVICE
  static RetParamsB ItArgsB(Arguments args) {
    return args.op_B;
  }

  CUTLASS_DEVICE
  ParamsA UpdateParamsA(ParamsA const &params) const {
    ParamsA out = params;
    out.indices += offset_b;

    // Set the number of steps for predicated index loads.
    out.steps_k = StepsK();
    return out;
  }

  CUTLASS_DEVICE
  ParamsB UpdateParamsB(ParamsB const &params) const {
    // NOTE: This should be elided by the compiler for
    // the non-transposed kernels where we do not use
    // explicit block offsets.
    ParamsB out = params;
    out.block_offsets += offset_b;

    // Set the number of steps for predicated index loads.
    out.steps_k = StepsK();
    return out;
  }

  CUTLASS_DEVICE
  RetOffsetA OffsetA() const {
    RetOffsetA tb_offset_A{
      offset.m() * Gemm::Mma::Shape::kM, 0
    };
    return tb_offset_A;
  }

  CUTLASS_DEVICE
  RetOffsetB OffsetB() const {
    return offset_b * kBlockSize;
  }

  CUTLASS_DEVICE
  int StepsK() const {
    // TODO(tgale): We now call this function in multiple places.
    // If the compiler doesn't already, we could calculate this value
    // in the constructor.
    int nnz_rows_b = nnz_b * Gemm::Mma::IteratorB::Shape::kBlock;
    return (nnz_rows_b + Gemm::Mma::Shape::kK - 1) / Gemm::Mma::Shape::kK;
  }
};

template <typename Gemm>
struct ConfigHelper<Gemm, BlockPitchLinear, BlockPitchLinear> {
  using Arguments = typename Gemm::Arguments;
  using Params = typename Gemm::Params;

  using RetParamsA = Op;
  using RetParamsB = Op;
  using RetOffsetA = int;
  using RetOffsetB = int;

  using ParamsA = typename Gemm::Mma::IteratorA::Params;
  using ParamsB = typename Gemm::Mma::IteratorB::Params;

  using ElementA = typename Gemm::Mma::IteratorA::Element;
  using MetaA = typename Type<ElementA>::Meta;

  using GemmCoord = ::cutlass::gemm::GemmCoord;

  // Dimension of the sparse blocks.
  static constexpr int kBlockSize = Gemm::Mma::IteratorA::Shape::kBlock;

  // The number of elements in a sparse block
  static constexpr int kBlockElements = kBlockSize * kBlockSize;

  // Helper to perform the index merge.
  using IndexMerge = IndexMerge<Gemm, kBlockSize>;

  static constexpr int kSmemBytes = IndexMerge::kSmemBytes;

  Params const &params;
  int offset_a, nnz_a, offset_b, nnz_b;

  IndexMerge merger;

  CUTLASS_DEVICE
  ConfigHelper(Params const &params_,
	       const GemmCoord &offset,
               uint8_t *smem) : params(params_) {
    // Load the offset and number of nonzeros.
    int *offset_ptr_a = (int*)params_.op_A.offsets;
    int block_row_idx = offset.m();

    // In blocks.
    offset_a = __ldg(offset_ptr_a + block_row_idx);
    nnz_a = __ldg(offset_ptr_a + block_row_idx + 1) - offset_a;

    int *offset_ptr_b = (int*)params_.op_B.offsets;
    int block_column_idx = offset.n();

    // In blocks.
    offset_b = __ldg(offset_ptr_b + block_column_idx);
    nnz_b = __ldg(offset_ptr_b + block_column_idx + 1) - offset_b;

    // Initialize the index merger.
    merger = IndexMerge(params_.op_A,
			params_.op_B,
			params_.problem_size.k(),
			offset_a, nnz_a,
			offset_b, nnz_b,
			offset, smem);
  }

  CUTLASS_HOST_DEVICE
  static RetParamsA ItArgsA(Arguments args) {
    return args.op_A;
  }

  CUTLASS_HOST_DEVICE
  static RetParamsB ItArgsB(Arguments args) {
    return args.op_B;
  }

  CUTLASS_DEVICE
  ParamsA UpdateParamsA(ParamsA const &params) const {
    // NOTE: This should be elided by the compiler for
    // the non-transposed kernels where we do not use
    // explicit block offsets.
    ParamsA out = params;
    out.base_params.block_offsets += offset_a;

    out.offsets = merger.OffsetPtrA();

    // Set the number of steps for predicated index loads.
    out.base_params.steps_k = StepsK();
    return out;
  }

  CUTLASS_DEVICE
  ParamsB UpdateParamsB(ParamsB const &params) const {
    // NOTE: This should be elided by the compiler for
    // the non-transposed kernels where we do not use
    // explicit block offsets.
    ParamsB out = params;
    out.base_params.block_offsets += offset_b;

    out.offsets = merger.OffsetPtrB();

    // Set the number of steps for predicated index loads.
    out.base_params.steps_k = StepsK();
    return out;
  }

  CUTLASS_DEVICE
  RetOffsetA OffsetA() const {
    return offset_a * kBlockElements;
  }

  CUTLASS_DEVICE
  RetOffsetB OffsetB() const {
    return offset_b * kBlockElements;
  }

  CUTLASS_DEVICE
  int StepsK() const {
    return merger.StepsK();
  }
};

template <typename Gemm, typename LayoutC>
struct OutputConfig {

  using Params = typename Gemm::Params;
  using GemmCoord = ::cutlass::gemm::GemmCoord;

  using RetOffsetC = ::cutlass::MatrixCoord;
  using RetExtentC = typename LayoutC::TensorCoord;

  Params const &params;

  CUTLASS_DEVICE
  OutputConfig(Params const &params_, const GemmCoord &offset) : params(params_) {}

  CUTLASS_DEVICE
  bool EarlyExit(const GemmCoord &offset) const {
    return params.grid_tiled_shape.m() <= offset.m() ||
        params.grid_tiled_shape.n() <= offset.n();
  }

  CUTLASS_DEVICE
  GemmCoord UpdateTileOffset(const GemmCoord &offset) const {
    return offset;
  }

  CUTLASS_DEVICE
  RetOffsetC OffsetC(const GemmCoord &offset) const {
    RetOffsetC threadblock_offset(
         offset.m() * Gemm::Mma::Shape::kM,
         offset.n() * Gemm::Mma::Shape::kN
    );
    return threadblock_offset;
  }

  CUTLASS_DEVICE
  RetExtentC ExtentC() const {
    return params.problem_size.mn();
  }
};

// Specialization for blocksparse output.
template <typename Gemm>
struct OutputConfig<Gemm, BlockRowMajor> {
  using Params = typename Gemm::Params;
  using GemmCoord = ::cutlass::gemm::GemmCoord;

  using RetOffsetC = int;
  using RetExtentC = BlockRowMajor::TensorCoord;

  // The block dimension as an int.
  static constexpr int kBlockSize =
      Gemm::Epilogue::OutputTileIterator::kBlockSize;
  static constexpr int kValuesPerBlock =
      kBlockSize * kBlockSize;

  // Element type for output matrix.
  using Element = typename Gemm::Epilogue::OutputTileIterator::Element;

  // Meta-data type for the output matrix.
  using Meta = typename Type<Element>::Meta;

  Params const &params;

  CUTLASS_DEVICE
  OutputConfig(Params const & params_, const GemmCoord &offset) :
    params(params_) {}

  CUTLASS_DEVICE
  bool EarlyExit(const GemmCoord &offset) const {
    // NOTE: We always launch the exact number of
    // threadblocks needed for the output.
    return false;
  }

  CUTLASS_DEVICE
  GemmCoord UpdateTileOffset(const GemmCoord &offset) const {
    // NOTE: It's required that C & D have the same topology.
    int block_index = offset.m();

    Meta column_index = __ldg((Meta*)params.op_C.indices + block_index);
    Meta row_index = __ldg((Meta*)params.op_C.row_indices + block_index);
    return {row_index, column_index, offset.k()};
  }

  CUTLASS_DEVICE
  RetOffsetC OffsetC(const GemmCoord &offset) const {
    return offset.m() * kValuesPerBlock;
  }

  CUTLASS_DEVICE
  RetExtentC ExtentC() const {
    // NOTE: This is unused.
    return params.problem_size.mn();
  }
};

// Helper to handle mixes of sparse and dense arguments.
template <typename Gemm>
struct Config {

  using GemmCoord = ::cutlass::gemm::GemmCoord;

  using Arguments = typename Gemm::Arguments;
  using Params = typename Gemm::Params;

  using LayoutA = typename Gemm::Mma::IteratorA::Layout;
  using ElementA = typename Gemm::Mma::IteratorA::Element;
  using LayoutB = typename Gemm::Mma::IteratorB::Layout;
  using ElementB = typename Gemm::Mma::IteratorB::Element;

  using ParamsA = typename Gemm::Mma::IteratorA::Params;
  using ParamsB = typename Gemm::Mma::IteratorB::Params;

  using Helper = ConfigHelper<Gemm, LayoutA, LayoutB>;

  // The number of bytes of shared memory needed.
  static constexpr int kSmemBytes = Helper::kSmemBytes;

  // Underlying helper.
  Helper helper;

  CUTLASS_DEVICE
  Config(Params const &params_,
	 const GemmCoord &offset_,
         uint8_t* smem) :
      helper(params_, offset_, smem) {}

  CUTLASS_HOST_DEVICE
  static typename Helper::RetParamsA ItArgsA(Arguments args) {
    return Helper::ItArgsA(args);
  }

  CUTLASS_HOST_DEVICE
  static typename Helper::RetParamsB ItArgsB(Arguments args) {
    return Helper::ItArgsB(args);
  }

  CUTLASS_DEVICE
  ParamsA UpdateParamsA(ParamsA const &params) const {
    return helper.UpdateParamsA(params);
  }

  CUTLASS_DEVICE
  ParamsB UpdateParamsB(ParamsB const &params) const {
    return helper.UpdateParamsB(params);
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

  // Configuration for input/output iterators.
  using Config = Config<
      BlockGemm<Mma, Epilogue, ThreadblockSwizzle>>;
  using OutputConfig = OutputConfig<
      BlockGemm<Mma, Epilogue, ThreadblockSwizzle>,
      typename Epilogue::OutputTileIterator::Layout>;

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
      params_A(Config::ItArgsA(args)),
      params_B(Config::ItArgsB(args)),
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
    __shared__ uint8_t config_shared[Config::kSmemBytes];
    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    ::cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.grid_tiled_shape);

    // Early exit if CTA is out of range
    OutputConfig bounds_checker(params, threadblock_tile_offset);
    if (bounds_checker.EarlyExit(threadblock_tile_offset)) return;

    // Update the tile offset if necessary.
    ::cutlass::gemm::GemmCoord tile_offset =
	bounds_checker.UpdateTileOffset(threadblock_tile_offset);

    Config config(params, tile_offset, config_shared);

    ElementA *ptr_A = static_cast<ElementA *>(params.op_A.data);
    ElementB *ptr_B = static_cast<ElementB *>(params.op_B.data);

    // TODO(tgale): Do we need to synchronize here?
    __syncthreads();

    // Compute initial location in logical coordinates
    auto tb_offset_A = config.OffsetA();
    auto tb_offset_B = config.OffsetB();

    // Compute position within threadblock
    int thread_idx = threadIdx.x;
    int problem_size_k = params.problem_size.k();

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
      config.UpdateParamsA(params.params_A),
      ptr_A,
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A);

    typename Mma::IteratorB iterator_B(
      config.UpdateParamsB(params.params_B),
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

    // Re-calculate the output meta-data to avoid wasting registers.
    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.grid_tiled_shape);

    OutputConfig output_config(params, threadblock_tile_offset);
    auto threadblock_offset = output_config.OffsetC(threadblock_tile_offset);
    auto extent_c = output_config.ExtentC();

    ElementC *ptr_C = static_cast<ElementC *>(params.op_C.data);
    ElementC *ptr_D = static_cast<ElementC *>(params.op_D.data);

    //
    // Fetch pointers based on mode.
    //

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C(
      params.params_C,
      ptr_C,
      extent_c,
      thread_idx,
      threadblock_offset
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,
      ptr_D,
      extent_c,
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
