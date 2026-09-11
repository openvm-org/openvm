#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_w_imm_u16.cuh"
#include "riscv/cores/shift_logical_imm.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

using ShiftWLogicalImmCore = ShiftLogicalImmCore<WORD_U16_LIMBS, U16_BITS>;
template <typename T>
using ShiftWLogicalImmCoreCols = ShiftLogicalImmCoreCols<T, WORD_U16_LIMBS, U16_BITS>;

template <typename T> struct ShiftWLogicalImmCols {
    BaseAluWImmU16AdapterCols<T> adapter;
    ShiftWLogicalImmCoreCols<T> core;
};

#include "../rvr/src/shift_w_logical_imm.inc.cuh"
