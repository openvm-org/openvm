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
using Rv64ShiftRightArithmeticCoreRecord = ShiftRightArithmeticCoreRecord<BLOCK_FE_WIDTH, U16_BITS>;
using Rv64ShiftRightArithmeticCore = ShiftRightArithmeticCore<BLOCK_FE_WIDTH, U16_BITS>;
template <typename T>
using Rv64ShiftRightArithmeticCoreCols = ShiftRightArithmeticCoreCols<T, BLOCK_FE_WIDTH, U16_BITS>;

template <typename T> struct ShiftRightArithmeticCols {
    Rv64BaseAluRegU16AdapterCols<T> adapter;
    Rv64ShiftRightArithmeticCoreCols<T> core;
};

struct ShiftRightArithmeticRecord {
    Rv64BaseAluRegU16AdapterRecord adapter;
    Rv64ShiftRightArithmeticCoreRecord core;
};

static_assert(sizeof(Rv64ShiftRightArithmeticCoreRecord) == 16);
static_assert(sizeof(ShiftRightArithmeticRecord) == 56);
static_assert(offsetof(ShiftRightArithmeticRecord, core) == 40);

#include "../rvr/src/shift_right_arithmetic.inc.cuh"
