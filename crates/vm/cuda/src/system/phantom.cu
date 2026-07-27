#include "launcher.cuh"
#include "primitives/trace_access.h"

static constexpr uint32_t NUM_PHANTOM_OPERANDS = 3;

template <typename T> struct PhantomCols {
    T pc;
    T operands[NUM_PHANTOM_OPERANDS];
    T timestamp;
    T is_valid;
};

#include "../../rvr/src/system/phantom.inc.cuh"
