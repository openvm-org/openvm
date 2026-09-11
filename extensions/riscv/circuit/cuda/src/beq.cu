#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/branch.cuh" // BranchAdapterCols, BranchAdapterRecord, BranchAdapter
#include "riscv/cores/beq.cuh"
#include "system/memory/params.cuh" // BLOCK_FE_WIDTH

using namespace riscv;


template <typename T> struct BranchEqualCols {
    BranchAdapterCols<T> adapter;
    BranchEqualCoreCols<T, BLOCK_FE_WIDTH> core;
};

#include "../rvr/src/beq.inc.cuh"
