#pragma once

#include "constants.h"
#include "controller.cuh"
#include "less_than.cuh"
#include "trace_access.h"

using namespace riscv;

template <typename T> struct MemoryBaseAuxCols {
    /// The previous timestamps in which the cells were accessed.
    T prev_timestamp;
    /// The auxiliary columns to perform the less than check.
    LessThanAuxCols<T, AUX_LEN> timestamp_lt_aux; // lower_decomp [T; AUX_LEN]
};

template <typename T, size_t NUM_LIMBS = RV32_REGISTER_NUM_LIMBS> struct MemoryWriteAuxCols {
    MemoryBaseAuxCols<T> base;
    T prev_data[NUM_LIMBS];
};

template <size_t NUM_LIMBS> struct MemoryWriteAuxRecord {
    uint32_t prev_timestamp;
    uint8_t prev_data[NUM_LIMBS];
};

template <size_t NUM_LIMBS> struct MemoryWriteAuxAdapter {
    MemoryAuxColsFactory mem_helper;

    __device__ MemoryWriteAuxAdapter(VariableRangeChecker range_checker)
        : mem_helper(range_checker) {}

    __device__ void fill_trace_row(
        RowSlice row,
        MemoryWriteAuxRecord<NUM_LIMBS> record,
        uint32_t timestamp
    ) {
        COL_WRITE_ARRAY(row, MemoryWriteAuxCols, prev_data, record.prev_data);
        COL_WRITE_VALUE(row, MemoryWriteAuxCols, base.prev_timestamp, record.prev_timestamp);
        mem_helper.fill(
            row.slice_from(COL_INDEX(MemoryWriteAuxCols, base.timestamp_lt_aux)),
            AUX_LEN,
            record.prev_timestamp,
            timestamp
        );
    }
};
