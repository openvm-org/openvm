#pragma once

#include <cstdint>
#include <cstddef>

template <size_t CHUNK, size_t BLOCKS> struct MemoryInventoryRecord {
    uint32_t address_space;
    uint32_t ptr;
    uint32_t timestamps[BLOCKS];
    uint32_t values[CHUNK];
};
