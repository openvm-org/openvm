#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_w_reg_u16.cuh"
#include "riscv/cores/shift_logical.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

// SLLW/SRLW use the u16 shift-logical core (WORD_U16_LIMBS limbs of 16 bits) over the low
// 32-bit word and the u16 W adapter.
using ShiftWLogicalCore = ShiftLogicalCore<WORD_U16_LIMBS, U16_BITS>;
template <typename T>
using ShiftWLogicalCoreCols = ShiftLogicalCoreCols<T, WORD_U16_LIMBS, U16_BITS>;

template <typename T> struct ShiftWLogicalCols {
    BaseAluWRegU16AdapterCols<T> adapter;
    ShiftWLogicalCoreCols<T> core;
};

#include "../rvr/src/shift_w_logical.inc.cuh"
