#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/mul_w.cuh"
#include "riscv/cores/mul.cuh"

using namespace riscv;

// Concrete type aliases for the 32-bit word variant on RV64.
using Rv64MulWCoreRecord = MultiplicationCoreRecord<RV64_WORD_NUM_LIMBS>;
using Rv64MulWCore = MultiplicationCore<RV64_WORD_NUM_LIMBS>;
template <typename T> using Rv64MulWCoreCols = MultiplicationCoreCols<T, RV64_WORD_NUM_LIMBS>;

template <typename T> struct Rv64MulWCols {
    Rv64MultWAdapterCols<T> adapter;
    Rv64MulWCoreCols<T> core;
};

#include "../rvr/src/mul_w.inc.cuh"
