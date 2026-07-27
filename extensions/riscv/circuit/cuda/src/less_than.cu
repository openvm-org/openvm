#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_reg_u16.cuh"
#include "riscv/cores/less_than.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

using Rv64LessThanCoreRecord = LessThanCoreRecord<BLOCK_FE_WIDTH, U16_BITS>;
using Rv64LessThanCore = LessThanCore<BLOCK_FE_WIDTH, U16_BITS>;
template <typename T>
using Rv64LessThanCoreCols =
    LessThanCoreCols<T, BLOCK_FE_WIDTH, U16_BITS>;

template <typename T> struct LessThanCols {
    Rv64BaseAluRegU16AdapterCols<T> adapter;
    Rv64LessThanCoreCols<T> core;
};

struct LessThanRecord {
    Rv64BaseAluRegU16AdapterRecord adapter;
    Rv64LessThanCoreRecord core;
};

#include "../rvr/src/less_than.inc.cuh"
