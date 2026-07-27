#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_w_imm_u16.cuh"
#include "riscv/cores/addi.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

using Rv64AddIWCoreRecord = AddICoreRecord<RV64_WORD_U16_LIMBS>;
using Rv64AddIWCore = AddICore<RV64_WORD_U16_LIMBS, U16_BITS, false>;
template <typename T> using Rv64AddIWCoreCols = AddICoreCols<T, RV64_WORD_U16_LIMBS>;

template <typename T> struct Rv64AddIWCols {
    Rv64BaseAluWImmU16AdapterCols<T> adapter;
    Rv64AddIWCoreCols<T> core;
};

struct Rv64AddIWRecord {
    Rv64BaseAluWImmU16AdapterRecord adapter;
    Rv64AddIWCoreRecord core;
};

static_assert(sizeof(Rv64AddIWRecord) == 48);
static_assert(offsetof(Rv64AddIWRecord, core) == 40);

#include "../rvr/src/addi_w.inc.cuh"
