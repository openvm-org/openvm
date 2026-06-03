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

template <typename T>
__device__ inline void pack_u8_block_bytes(T (&out)[BLOCK_FE_WIDTH], uint8_t const (&data)[MEMORY_BLOCK_BYTES]) {
#pragma unroll
    for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
        out[i] = T(uint32_t(data[2 * i]) + 256u * uint32_t(data[2 * i + 1]));
    }
}
