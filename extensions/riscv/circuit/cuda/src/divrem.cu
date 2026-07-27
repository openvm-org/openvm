#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "riscv/adapters/mul.cuh"
#include "riscv/cores/divrem.cuh"

using namespace riscv;

template <typename T> struct Rv64DivRemCols {
    Rv64MultAdapterCols<T> adapter;
    DivRemCoreCols<T, RV64_REGISTER_NUM_LIMBS> core;
};

struct Rv64DivRemRecord {
    Rv64MultAdapterRecord adapter;
    DivRemCoreRecords<RV64_REGISTER_NUM_LIMBS> core;
};

#include "../rvr/src/divrem.inc.cuh"
