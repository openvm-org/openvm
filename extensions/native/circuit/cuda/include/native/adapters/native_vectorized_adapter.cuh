#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace native;

template <typename T, size_t N> struct NativeVectorizedAdapterCols {
    ExecutionState<T> from_state;
    T a_pointer;
    T b_pointer;
    T c_pointer;
    MemoryReadAuxCols<T> reads_aux[2];
    MemoryWriteAuxCols<T, N> writes_aux;
};

template <typename F, size_t N> struct NativeVectorizedAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    F a_ptr;
    F b_ptr;
    F c_ptr;

    MemoryReadAuxRecord reads_aux[2];
    MemoryWriteAuxRecord<F, N> write_aux;
};

template <typename F, size_t N> struct NativeVectorizedAdapter {
    MemoryAuxColsFactory mem_helper;

    __device__ NativeVectorizedAdapter(
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : mem_helper(range_checker, timestamp_max_bits) {}

    template <typename T> using Cols = NativeVectorizedAdapterCols<T, N>;

    __device__ void fill_trace_row(
        RowSlice row,
        NativeVectorizedAdapterRecord<F, N> const &record
    ) {
        COL_WRITE_VALUE(row, Cols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, Cols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Cols, a_pointer, record.a_ptr);
        COL_WRITE_VALUE(row, Cols, b_pointer, record.b_ptr);
        COL_WRITE_VALUE(row, Cols, c_pointer, record.c_ptr);

        for (int i = 0; i < 2; i++) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Cols, reads_aux[i])),
                record.reads_aux[i].prev_timestamp,
                record.from_timestamp + i
            );
        }

        COL_WRITE_ARRAY(row, Cols, writes_aux.prev_data, record.write_aux.prev_data);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Cols, writes_aux)),
            record.write_aux.prev_timestamp,
            record.from_timestamp + 2
        );
    }
};