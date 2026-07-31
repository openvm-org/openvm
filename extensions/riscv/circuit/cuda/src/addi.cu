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
using Rv64AddICore = AddICore<BLOCK_FE_WIDTH, U16_BITS, true>;
template <typename T> using Rv64AddICoreCols = AddICoreCols<T, BLOCK_FE_WIDTH>;

template <typename T> struct Rv64AddICols {
    Rv64BaseAluImmU16AdapterCols<T> adapter;
    Rv64AddICoreCols<T> core;
};

#include "../rvr/src/addi.inc.cuh"
