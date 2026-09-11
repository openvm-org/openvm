#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_reg_u16.cuh"
#include "riscv/cores/less_than.cuh"
#include "system/memory/params.cuh"

using namespace riscv;


template <typename T> struct LessThanCols {
    BaseAluRegU16AdapterCols<T> adapter;
    LessThanCoreCols<T, BLOCK_FE_WIDTH, U16_BITS> core;
};

#include "../rvr/src/less_than.inc.cuh"
