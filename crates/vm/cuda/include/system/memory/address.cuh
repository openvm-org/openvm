#pragma once

#include "system/memory/params.cuh"

template <typename T> struct MemoryAddress {
    T address_space;
    T pointer_limbs[POINTER_LIMBS];
};
