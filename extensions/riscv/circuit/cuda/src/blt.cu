#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/constants.h"
#include "riscv/adapters/branch.cuh" // BranchAdapterCols, BranchAdapterRecord, BranchAdapter
#include "riscv/cores/blt.cuh"
#include "riscv/cores/less_than.cuh"
#include "system/memory/params.cuh" // BLOCK_FE_WIDTH

using namespace riscv;


template <typename T> struct BranchLessThanCols {
    BranchAdapterCols<T> adapter;
    BranchLessThanCoreCols<T, BLOCK_FE_WIDTH, U16_BITS> core;
};

#include "../rvr/src/blt.inc.cuh"
