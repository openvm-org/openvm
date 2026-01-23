#include <cstddef>

template <typename T, size_t DEPTH_MINUS_TWO> struct NestedForLoopAuxCols {
    T is_transition[DEPTH_MINUS_TWO];
};
