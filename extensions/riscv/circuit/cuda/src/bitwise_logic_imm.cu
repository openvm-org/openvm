#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_imm.cuh"
#include "riscv/cores/bitwise_logic_imm.cuh"

using namespace riscv;

// XORI/ORI/ANDI use byte limbs and the immediate-only byte ALU adapter.

template <typename T> struct BitwiseLogicImmCols {
    BaseAluImmAdapterCols<T> adapter;
    BitwiseLogicImmCoreCols<T, REGISTER_NUM_LIMBS> core;
};

#include "../rvr/src/bitwise_logic_imm.inc.cuh"
