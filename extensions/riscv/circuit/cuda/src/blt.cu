#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/constants.h"
#include "riscv/adapters/branch.cuh" // Rv64BranchAdapterCols, Rv64BranchAdapterRecord, Rv64BranchAdapter
#include "riscv/cores/blt.cuh"
#include "riscv/cores/less_than.cuh"
#include "system/memory/params.cuh" // BLOCK_FE_WIDTH

using namespace riscv;

using Rv64BranchLessThanCoreRecord =
    BranchLessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>;
using Rv64BranchLessThanCore = BranchLessThanCore<BLOCK_FE_WIDTH, U16_BITS>;
template <typename T>
using Rv64BranchLessThanCoreCols =
    BranchLessThanCoreCols<T, BLOCK_FE_WIDTH, U16_BITS>;

template <typename T> struct BranchLessThanCols {
    Rv64BranchAdapterCols<T> adapter;
    Rv64BranchLessThanCoreCols<T> core;
};

struct BranchLessThanRecord {
    Rv64BranchAdapterRecord adapter;
    Rv64BranchLessThanCoreRecord core;
};

#include "../rvr/src/blt.inc.cuh"
