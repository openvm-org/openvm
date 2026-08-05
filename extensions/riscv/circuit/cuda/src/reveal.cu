#include "launcher.cuh"
#include "primitives/constants.h"
#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;
using namespace program;

template <typename T> struct RevealCols {
    T is_valid;
    ExecutionState<T> from_state;
    T src_ptr;
    T src_data[BLOCK_FE_WIDTH];
    MemoryReadAuxCols<T> src_aux;
    T ordinal;
    T has_next;
    T timestamp_delta_low;
};

#include "../rvr/src/reveal.inc.cuh"
