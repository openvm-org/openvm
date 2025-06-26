#pragma once

#include "histogram.cuh"
#include "less_than.cuh"
#include "offline_checker.cuh"
#include "trace_access.h"

struct MemoryAuxColsFactory {
    VariableRangeChecker range_checker;

    __device__ MemoryAuxColsFactory(VariableRangeChecker range_checker)
        : range_checker(range_checker) {}

    __device__ void fill(RowSlice row, uint32_t prev_timestamp, uint32_t timestamp) {
        AssertLessThan::generate_subrow(
            range_checker,
            range_checker.max_bits(),
            prev_timestamp,
            timestamp,
            AUX_LEN,
            row.slice_from(COL_INDEX(MemoryBaseAuxCols, timestamp_lt_aux))
        );
        COL_WRITE_VALUE(row, MemoryBaseAuxCols, prev_timestamp, prev_timestamp);
    }

    __device__ void fill_zero(RowSlice row) {
        row.fill_zero(0, sizeof(MemoryBaseAuxCols<uint8_t>));
    }
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
        mem_helper.fill(
            row.slice_from(COL_INDEX(MemoryWriteAuxCols, base)), record.prev_timestamp, timestamp
        );
    }
};
