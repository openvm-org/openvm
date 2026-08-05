#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_w_imm_u16.cuh"
#include "riscv/cores/addi.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

using AddIWCore = AddICore<WORD_U16_LIMBS, U16_BITS, false>;
template <typename T> using AddIWCoreCols = AddICoreCols<T, WORD_U16_LIMBS>;

template <typename T> struct AddIWCols {
    BaseAluWImmU16AdapterCols<T> adapter;
    AddIWCoreCols<T> core;
};

#include "../rvr/src/addi_w.inc.cuh"
