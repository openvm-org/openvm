#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_imm_u16.cuh"
#include "riscv/cores/addi.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

// Concrete type aliases for RV64

template <typename T> struct AddICols {
    BaseAluImmU16AdapterCols<T> adapter;
    AddICoreCols<T, BLOCK_FE_WIDTH> core;
};

#include "../rvr/src/addi.inc.cuh"
