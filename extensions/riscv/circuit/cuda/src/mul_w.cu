#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/mul_w.cuh"
#include "riscv/cores/mul.cuh"

using namespace riscv;

// Concrete type aliases for the 32-bit word variant on RV64.
using MulWCoreRecord = MultiplicationCoreRecord<WORD_NUM_LIMBS>;
using MulWCore = MultiplicationCore<WORD_NUM_LIMBS>;
template <typename T> using MulWCoreCols = MultiplicationCoreCols<T, WORD_NUM_LIMBS>;

template <typename T> struct MulWCols {
    MultWAdapterCols<T> adapter;
    MulWCoreCols<T> core;
};

#include "../rvr/src/mul_w.inc.cuh"
