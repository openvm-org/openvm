#pragma once

#include "primitives/constants.h"
#include "primitives/less_than.cuh"

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

template <typename T, size_t NUM_LIMBS = RV32_REGISTER_NUM_LIMBS> struct MemoryWriteAuxCols {
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