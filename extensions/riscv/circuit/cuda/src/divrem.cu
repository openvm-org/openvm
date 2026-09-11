#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/mul.cuh"
#include "riscv/cores/divrem.cuh"

using namespace riscv;

template <typename T> struct DivRemCols {
    MultAdapterCols<T> adapter;
    DivRemCoreCols<T, REGISTER_NUM_LIMBS> core;
};

#include "../rvr/src/divrem.inc.cuh"
