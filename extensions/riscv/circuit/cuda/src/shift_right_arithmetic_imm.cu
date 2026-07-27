#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_imm_u16.cuh"
#include "riscv/cores/shift_right_arithmetic_imm.cuh"
#include "system/memory/params.cuh"

using namespace riscv;
using namespace program;

// SRAI uses u16 limbs (4 limbs of 16 bits) and the immediate-only u16 ALU adapter.
using Rv64ShiftRightArithmeticImmCoreRecord =
    ShiftRightArithmeticImmCoreRecord<BLOCK_FE_WIDTH, U16_BITS>;
using Rv64ShiftRightArithmeticImmCore =
    ShiftRightArithmeticImmCore<BLOCK_FE_WIDTH, U16_BITS>;
template <typename T>
using Rv64ShiftRightArithmeticImmCoreCols =
    ShiftRightArithmeticImmCoreCols<T, BLOCK_FE_WIDTH, U16_BITS>;

template <typename T> struct ShiftRightArithmeticImmCols {
    Rv64BaseAluImmU16AdapterCols<T> adapter;
    Rv64ShiftRightArithmeticImmCoreCols<T> core;
};

struct ShiftRightArithmeticImmRecord {
    Rv64BaseAluImmU16AdapterRecord adapter;
    Rv64ShiftRightArithmeticImmCoreRecord core;
};

static_assert(sizeof(ShiftRightArithmeticImmRecord) == 44);
static_assert(offsetof(ShiftRightArithmeticImmRecord, core) == 32);

#include "../rvr/src/shift_right_arithmetic_imm.inc.cuh"
