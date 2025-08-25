#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace native;

template <typename T, size_t WRITE_SIZE> struct ConvertAdapterCols {
    ExecutionState<T> from_state;
    T a_pointer;
    T b_pointer;
    MemoryWriteAuxCols<T, WRITE_SIZE> writes_aux;
    MemoryReadAuxCols<T> reads_aux;
};

template <typename F, size_t WRITE_SIZE> struct ConvertAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    F a_ptr;
    F b_ptr;

    MemoryReadAuxRecord read_aux;
    MemoryWriteBytesAuxRecord<WRITE_SIZE> write_aux;
};

template <typename F, size_t WRITE_SIZE> struct ConvertAdapter {
    MemoryAuxColsFactory mem_helper;

    __device__ ConvertAdapter(VariableRangeChecker range_checker, uint32_t timestamp_max_bits)
        : mem_helper(range_checker, timestamp_max_bits) {}

    template <typename T> using Cols = ConvertAdapterCols<T, WRITE_SIZE>;

    __device__ void fill_trace_row(
        RowSlice row,
        ConvertAdapterRecord<F, WRITE_SIZE> const &record
    ) {
        COL_WRITE_VALUE(row, Cols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, Cols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Cols, a_pointer, record.a_ptr);
        COL_WRITE_VALUE(row, Cols, b_pointer, record.b_ptr);

        mem_helper.fill(
            row.slice_from(COL_INDEX(Cols, reads_aux)),
            record.read_aux.prev_timestamp,
            record.from_timestamp
        );

        COL_WRITE_ARRAY(row, Cols, writes_aux.prev_data, record.write_aux.prev_data);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Cols, writes_aux)),
            record.write_aux.prev_timestamp,
            record.from_timestamp + 1
        );
    }
};