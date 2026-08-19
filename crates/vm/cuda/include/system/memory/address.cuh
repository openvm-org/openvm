#pragma once

#include "system/memory/params.cuh"

// Mirrors `openvm_circuit::system::memory::MemoryAddress`: the pointer is expressed at
// memory-bus block granularity (AS-native cell pointer / BLOCK_FE_WIDTH).
template <typename T> struct MemoryAddress {
    T address_space;
    T pointer;
};
