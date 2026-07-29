#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_w_reg_u16.cuh"
#include "riscv/cores/shift_right_arithmetic.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

// SRAW uses the u16 shift-right-arithmetic core (RV64_WORD_U16_LIMBS limbs of 16 bits) over the low
// 32-bit word and the u16 W adapter.
using Rv64ShiftWRightArithmeticCore = ShiftRightArithmeticCore<RV64_WORD_U16_LIMBS, U16_BITS>;
template <typename T>
using Rv64ShiftWRightArithmeticCoreCols =
    ShiftRightArithmeticCoreCols<T, RV64_WORD_U16_LIMBS, U16_BITS>;

template <typename T> struct ShiftWRightArithmeticCols {
    Rv64BaseAluWRegU16AdapterCols<T> adapter;
    Rv64ShiftWRightArithmeticCoreCols<T> core;
};

#include "../rvr/src/shift_w_right_arithmetic.inc.cuh"
