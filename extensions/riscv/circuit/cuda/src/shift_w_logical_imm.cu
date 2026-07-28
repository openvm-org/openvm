#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_w_imm_u16.cuh"
#include "riscv/cores/shift_logical_imm.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

using Rv64ShiftWLogicalImmCore = ShiftLogicalImmCore<RV64_WORD_U16_LIMBS, U16_BITS>;
using Rv64ShiftWLogicalImmCoreRecord = ShiftLogicalImmCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>;
template <typename T>
using Rv64ShiftWLogicalImmCoreCols = ShiftLogicalImmCoreCols<T, RV64_WORD_U16_LIMBS, U16_BITS>;

template <typename T> struct Rv64ShiftWLogicalImmCols {
    Rv64BaseAluWImmU16AdapterCols<T> adapter;
    Rv64ShiftWLogicalImmCoreCols<T> core;
};

struct Rv64ShiftWLogicalImmRecord {
    Rv64BaseAluWImmU16AdapterRecord adapter;
    Rv64ShiftWLogicalImmCoreRecord core;
};

static_assert(sizeof(Rv64ShiftWLogicalImmRecord) == 48);
static_assert(offsetof(Rv64ShiftWLogicalImmRecord, core) == 40);

#include "../rvr/src/shift_w_logical_imm.inc.cuh"
