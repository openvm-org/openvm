#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_w_imm_u16.cuh"
#include "riscv/cores/shift_right_arithmetic_imm.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

using Rv64ShiftWRightArithmeticImmCore = ShiftRightArithmeticImmCore<RV64_WORD_U16_LIMBS, U16_BITS>;
template <typename T>
using Rv64ShiftWRightArithmeticImmCoreCols =
    ShiftRightArithmeticImmCoreCols<T, RV64_WORD_U16_LIMBS, U16_BITS>;

template <typename T> struct Rv64ShiftWRightArithmeticImmCols {
    Rv64BaseAluWImmU16AdapterCols<T> adapter;
    Rv64ShiftWRightArithmeticImmCoreCols<T> core;
};

#include "../rvr/src/shift_w_right_arithmetic_imm.inc.cuh"
