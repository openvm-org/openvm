#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_reg.cuh"
#include "riscv/cores/bitwise_logic.cuh"

using namespace riscv;

// Concrete type aliases for RV64

template <typename T> struct BitwiseLogicCols {
    BaseAluRegAdapterCols<T> adapter;
    BitwiseLogicCoreCols<T, REGISTER_NUM_LIMBS> core;
};

#include "../rvr/src/bitwise_logic.inc.cuh"
