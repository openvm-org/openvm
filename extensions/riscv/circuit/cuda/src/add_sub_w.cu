#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_w_reg_u16.cuh"
#include "riscv/cores/add_sub.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

// Concrete type aliases for the 32-bit word variant on RV64. The low word is two u16 limbs and
// reuses the add_sub core; the adapter rebuilds the sign-extended 64-bit register write.
using AddSubWCore = AddSubCore<WORD_U16_LIMBS, U16_BITS, false>;
template <typename T> using AddSubWCoreCols = AddSubCoreCols<T, WORD_U16_LIMBS>;

template <typename T> struct AddSubWCols {
    BaseAluWRegU16AdapterCols<T> adapter;
    AddSubWCoreCols<T> core;
};

#include "../rvr/src/add_sub_w.inc.cuh"
