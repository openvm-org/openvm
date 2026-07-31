#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_imm.cuh"
#include "riscv/cores/bitwise_logic_imm.cuh"

using namespace riscv;

// XORI/ORI/ANDI use byte limbs and the immediate-only byte ALU adapter.
using Rv64BitwiseLogicImmCore = BitwiseLogicImmCore<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>;
template <typename T>
using Rv64BitwiseLogicImmCoreCols = BitwiseLogicImmCoreCols<T, RV64_REGISTER_NUM_LIMBS>;

template <typename T> struct Rv64BitwiseLogicImmCols {
    Rv64BaseAluImmAdapterCols<T> adapter;
    Rv64BitwiseLogicImmCoreCols<T> core;
};

#include "../rvr/src/bitwise_logic_imm.inc.cuh"
