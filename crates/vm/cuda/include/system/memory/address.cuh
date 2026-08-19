#pragma once

#include "system/memory/params.cuh"

template <typename T> struct MemoryAddress {
    T address_space;
    T block_index;
};
