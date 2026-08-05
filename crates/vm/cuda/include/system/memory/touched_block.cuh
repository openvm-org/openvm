#pragma once

#include "system/memory/params.cuh"
#include <cstddef>
#include <cstdint>

// CUDA layout of `TouchedBlock`: one fixed memory-bus block, its last
// timestamp, and whether any access to the block in this segment was a write.
// Values are canonical field representatives; the memory-inventory kernel
// converts them to Montgomery form at the proof boundary.
struct MemoryTouchedBlock {
    uint32_t address_space;
    uint32_t ptr;
    uint32_t is_dirty;
    uint32_t timestamp;
    uint32_t values[BLOCK_FE_WIDTH];
};

static_assert(BLOCK_FE_WIDTH == 4);
static_assert(offsetof(MemoryTouchedBlock, address_space) == 0);
static_assert(offsetof(MemoryTouchedBlock, ptr) == sizeof(uint32_t));
static_assert(offsetof(MemoryTouchedBlock, is_dirty) == 2 * sizeof(uint32_t));
static_assert(offsetof(MemoryTouchedBlock, timestamp) == 3 * sizeof(uint32_t));
static_assert(offsetof(MemoryTouchedBlock, values) == 4 * sizeof(uint32_t));
static_assert(sizeof(MemoryTouchedBlock) == 8 * sizeof(uint32_t));
