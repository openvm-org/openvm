#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/mul.cuh"
#include "riscv/cores/mul.cuh"

using namespace riscv;

// Concrete type aliases for 64-bit

template <typename T> struct MultiplicationCols {
    MultAdapterCols<T> adapter;
    MultiplicationCoreCols<T, REGISTER_NUM_LIMBS> core;
};

#include "../rvr/src/mul.inc.cuh"
