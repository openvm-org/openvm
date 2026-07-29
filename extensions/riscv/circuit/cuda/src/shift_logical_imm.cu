#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_imm_u16.cuh"
#include "riscv/cores/shift_logical_imm.cuh"
#include "system/memory/params.cuh"

using namespace riscv;
using namespace program;

// SLLI/SRLI use u16 limbs (4 limbs of 16 bits) and the immediate u16 ALU adapter. The
// immediate operand is reconstructed from the core's marker columns.
using Rv64ShiftLogicalImmCore = ShiftLogicalImmCore<BLOCK_FE_WIDTH, U16_BITS>;
template <typename T>
using Rv64ShiftLogicalImmCoreCols = ShiftLogicalImmCoreCols<T, BLOCK_FE_WIDTH, U16_BITS>;

template <typename T> struct ShiftLogicalImmCols {
    Rv64BaseAluImmU16AdapterCols<T> adapter;
    Rv64ShiftLogicalImmCoreCols<T> core;
};

#include "../rvr/src/shift_logical_imm.inc.cuh"
