#pragma once

#include "histogram.cuh"
#include "less_than.cuh"
#include "trace_access.h"

struct MemoryAuxColsFactory {
    VariableRangeChecker range_checker;

    __device__ MemoryAuxColsFactory(VariableRangeChecker range_checker)
        : range_checker(range_checker) {}

    __device__ void fill(RowSlice row, size_t length, uint32_t prev_timestamp, uint32_t timestamp) {
        AssertLessThan::generate_subrow(
            range_checker, range_checker.max_bits(), prev_timestamp, timestamp, length, row
        );
    }
};