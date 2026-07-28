#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_w_reg_u16.cuh"
#include "riscv/cores/shift_logical.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

// SLLW/SRLW use the u16 shift-logical core (RV64_WORD_U16_LIMBS limbs of 16 bits) over the low
// 32-bit word and the u16 W adapter.
using Rv64ShiftWLogicalCore = ShiftLogicalCore<RV64_WORD_U16_LIMBS, U16_BITS>;
using Rv64ShiftWLogicalCoreRecord = ShiftLogicalCoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>;
template <typename T>
using Rv64ShiftWLogicalCoreCols = ShiftLogicalCoreCols<T, RV64_WORD_U16_LIMBS, U16_BITS>;

template <typename T> struct ShiftWLogicalCols {
    Rv64BaseAluWRegU16AdapterCols<T> adapter;
    Rv64ShiftWLogicalCoreCols<T> core;
};

struct ShiftWLogicalRecord {
    Rv64BaseAluWRegU16AdapterRecord adapter;
    Rv64ShiftWLogicalCoreRecord core;
};

static_assert(sizeof(Rv64ShiftWLogicalCoreRecord) == 10);
static_assert(sizeof(ShiftWLogicalRecord) == 64);
static_assert(offsetof(ShiftWLogicalRecord, core) == 52);

#include "../rvr/src/shift_w_logical.inc.cuh"
