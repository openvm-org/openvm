#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_reg_u16.cuh"
#include "riscv/cores/shift_right_arithmetic.cuh"
#include "system/memory/params.cuh"

using namespace riscv;
using namespace program;

// SRA uses u16 limbs (4 limbs of 16 bits) and the u16 ALU adapter.

template <typename T> struct ShiftRightArithmeticCols {
    BaseAluRegU16AdapterCols<T> adapter;
    ShiftRightArithmeticCoreCols<T, BLOCK_FE_WIDTH, U16_BITS> core;
};

#include "../rvr/src/shift_right_arithmetic.inc.cuh"
