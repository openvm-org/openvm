#pragma once

#include "offline_checker.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"

struct MemoryAuxColsFactory {
    VariableRangeChecker range_checker;
    uint32_t timestamp_max_bits;

    __device__ MemoryAuxColsFactory(VariableRangeChecker range_checker, uint32_t timestamp_max_bits)
        : range_checker(range_checker), timestamp_max_bits(timestamp_max_bits) {}

    __device__ void fill(RowSlice row, uint32_t prev_timestamp, uint32_t timestamp) {
        AssertLessThan::generate_subrow(
            range_checker,
            timestamp_max_bits,
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
