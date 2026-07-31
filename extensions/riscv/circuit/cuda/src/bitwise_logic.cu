#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "riscv/adapters/alu_reg.cuh"
#include "riscv/cores/bitwise_logic.cuh"

using namespace riscv;

// Concrete type aliases for RV64
using Rv64BitwiseLogicCore = BitwiseLogicCore<RV64_REGISTER_NUM_LIMBS>;
template <typename T> using Rv64BitwiseLogicCoreCols = BitwiseLogicCoreCols<T, RV64_REGISTER_NUM_LIMBS>;

template <typename T> struct Rv64BitwiseLogicCols {
    Rv64BaseAluRegAdapterCols<T> adapter;
    Rv64BitwiseLogicCoreCols<T> core;
};

#include "../rvr/src/bitwise_logic.inc.cuh"
