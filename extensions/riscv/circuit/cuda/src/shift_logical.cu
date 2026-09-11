#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_reg_u16.cuh"
#include "riscv/cores/shift_logical.cuh"
#include "system/memory/params.cuh"

using namespace riscv;
using namespace program;

// SLL/SRL use u16 limbs (4 limbs of 16 bits) and the u16 ALU adapter.

template <typename T> struct ShiftLogicalCols {
    BaseAluRegU16AdapterCols<T> adapter;
    ShiftLogicalCoreCols<T, BLOCK_FE_WIDTH, U16_BITS> core;
};

#include "../rvr/src/shift_logical.inc.cuh"
