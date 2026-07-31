#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_reg_u16.cuh"
#include "riscv/cores/add_sub.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

// Concrete type aliases for RV64
using Rv64AddSubCore = AddSubCore<BLOCK_FE_WIDTH, U16_BITS, true>;
template <typename T> using Rv64AddSubCoreCols = AddSubCoreCols<T, BLOCK_FE_WIDTH>;

template <typename T> struct Rv64AddSubCols {
    Rv64BaseAluRegU16AdapterCols<T> adapter;
    Rv64AddSubCoreCols<T> core;
};

#include "../rvr/src/add_sub.inc.cuh"
