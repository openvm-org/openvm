#pragma once

#include "primitives/constants.h"
#include "primitives/less_than.cuh"
#include "system/memory/params.cuh"

using namespace riscv;

template <typename T> struct MemoryBaseAuxCols {
    /// The previous timestamps in which the cells were accessed.
    T prev_timestamp;
    /// The auxiliary columns to perform the less than check.
    LessThanAuxCols<T, AUX_LEN> timestamp_lt_aux; // lower_decomp [T; AUX_LEN]
};

template <typename T> struct MemoryReadAuxCols {
    MemoryBaseAuxCols<T> base;
};

template <typename T, size_t NUM_LIMBS = BLOCK_FE_WIDTH> struct MemoryWriteAuxCols {
    MemoryBaseAuxCols<T> base;
    T prev_data[NUM_LIMBS];
};

struct MemoryReadAuxRecord {
    uint32_t prev_timestamp;
};

template <typename T, size_t NUM_LIMBS> struct MemoryWriteAuxRecord {
    uint32_t prev_timestamp;
    T prev_data[NUM_LIMBS];
};

template <size_t NUM_LIMBS>
using MemoryWriteBytesAuxRecord = MemoryWriteAuxRecord<uint8_t, NUM_LIMBS>;

template <size_t NUM_LIMBS>
using MemoryWriteU16AuxRecord = MemoryWriteAuxRecord<uint16_t, NUM_LIMBS>;

template <typename T>
__device__ inline void pack_u8_block_bytes(T (&out)[BLOCK_FE_WIDTH], uint8_t const (&data)[MEMORY_BLOCK_BYTES]) {
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        out[i] = T(uint32_t(data[2 * i]) + 256u * uint32_t(data[2 * i + 1]));
    }
}

template <typename T, size_t NUM_U16_CELLS>
__device__ inline void copy_u16_cells(
    T (&out)[NUM_U16_CELLS],
    uint16_t const (&data)[NUM_U16_CELLS]
) {
#pragma unroll
    for (size_t i = 0; i < NUM_U16_CELLS; i++) {
        out[i] = T(uint32_t(data[i]));
    }
}
