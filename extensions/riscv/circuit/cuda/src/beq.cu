#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/branch.cuh" // Rv64BranchAdapterCols, Rv64BranchAdapterRecord, Rv64BranchAdapter
#include "riscv/cores/beq.cuh"
#include "system/memory/params.cuh" // BLOCK_FE_WIDTH

using namespace riscv;

using Rv64BranchEqualCore = BranchEqualCore<BLOCK_FE_WIDTH>;
template <typename T>
using Rv64BranchEqualCoreCols = BranchEqualCoreCols<T, BLOCK_FE_WIDTH>;
using Rv64BranchEqualCoreRecord = BranchEqualCoreRecord<BLOCK_FE_WIDTH>;

template <typename T> struct BranchEqualCols {
    Rv64BranchAdapterCols<T> adapter;
    Rv64BranchEqualCoreCols<T> core;
};

struct BranchEqualRecord {
    Rv64BranchAdapterRecord adapter;
    Rv64BranchEqualCoreRecord core;
};

#include "../rvr/src/beq.inc.cuh"
