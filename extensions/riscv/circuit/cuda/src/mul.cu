#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/mul.cuh"
#include "riscv/cores/mul.cuh"

using namespace riscv;

// Concrete type aliases for 64-bit
using Rv64MultiplicationCoreRecord = MultiplicationCoreRecord<RV64_REGISTER_NUM_LIMBS>;
using Rv64MultiplicationCore = MultiplicationCore<RV64_REGISTER_NUM_LIMBS>;
template <typename T>
using Rv64MultiplicationCoreCols = MultiplicationCoreCols<T, RV64_REGISTER_NUM_LIMBS>;

template <typename T> struct Rv64MultiplicationCols {
    Rv64MultAdapterCols<T> adapter;
    Rv64MultiplicationCoreCols<T> core;
};

struct Rv64MultiplicationRecord {
    Rv64MultAdapterRecord adapter;
    Rv64MultiplicationCoreRecord core;
};

#include "../rvr/src/mul.inc.cuh"
