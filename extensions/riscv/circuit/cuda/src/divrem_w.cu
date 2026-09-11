#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/mul_w.cuh"
#include "riscv/cores/divrem.cuh"

using namespace riscv;

template <typename T> struct DivRemWCols {
    MultWAdapterCols<T> adapter;
    DivRemCoreCols<T, WORD_NUM_LIMBS> core;
};

#include "../rvr/src/divrem_w.inc.cuh"
