#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_imm_u16.cuh"
#include "riscv/cores/less_than_imm.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

// SLTI/SLTIU use u16 limbs and the single-read immediate adapter.

template <typename T> struct LessThanImmCols {
    BaseAluImmU16AdapterCols<T> adapter;
    LessThanImmCoreCols<T, BLOCK_FE_WIDTH, U16_BITS> core;
};

#include "../rvr/src/less_than_imm.inc.cuh"
