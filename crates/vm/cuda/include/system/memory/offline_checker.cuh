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

template <typename T, size_t NUM_LIMBS = RV64_REGISTER_NUM_LIMBS> struct MemoryWriteAuxCols {
    MemoryBaseAuxCols<T> base;
    T prev_data[NUM_LIMBS];
};

template <typename T> struct MemoryReadOrImmediateAuxCols {
    MemoryBaseAuxCols<T> base;
    T is_immediate;
    T is_zero_aux;
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

template <typename T, size_t NUM_LIMBS>
__device__ inline void pack_u8_block_bytes(T (&out)[BLOCK_FE_WIDTH], uint8_t const (&data)[NUM_LIMBS]) {
    static_assert(NUM_LIMBS == MEMORY_BLOCK_BYTES, "pack_u8_block_bytes expects one memory block");
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        out[i] = T(uint32_t(data[2 * i]) + 256u * uint32_t(data[2 * i + 1]));
    }
}

template <typename T, size_t NUM_U16_CELLS>
__device__ inline void pack_u32_to_u16_cells_le(T (&out)[NUM_U16_CELLS], uint32_t value) {
    static_assert(NUM_U16_CELLS == sizeof(uint32_t) / sizeof(uint16_t));
#pragma unroll
    for (size_t i = 0; i < NUM_U16_CELLS; i++) {
        out[i] = T((value >> ((8 * sizeof(uint16_t)) * i)) & 0xffffu);
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
