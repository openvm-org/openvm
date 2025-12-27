#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

template <typename T, uint32_t NUM_CELLS> struct NativeLoadStoreAdapterCols {
    ExecutionState<T> from_state;
    T a;
    T b;
    T c;

    T data_write_pointer;

    MemoryReadAuxCols<T> pointer_read_aux_cols;
    MemoryReadAuxCols<T> data_read_aux_cols;
    MemoryWriteAuxCols<T, NUM_CELLS> data_write_aux_cols;
};

template <typename F, uint32_t NUM_CELLS> struct NativeLoadStoreAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    F a;
    F b;
    F c;
    F write_ptr;

    MemoryReadAuxRecord ptr_read;
    MemoryReadAuxRecord data_read;
    MemoryWriteAuxRecord<F, NUM_CELLS> data_write;
};

template <typename F, uint32_t NUM_CELLS> struct NativeLoadStoreAdapter {
    template <typename T> using Cols = NativeLoadStoreAdapterCols<T, NUM_CELLS>;

    MemoryAuxColsFactory mem_helper;

    __device__ NativeLoadStoreAdapter(
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_trace_row(
        RowSlice &row,
        NativeLoadStoreAdapterRecord<F, NUM_CELLS> const &record
    ) {
        bool is_hint_storew = (record.data_read.prev_timestamp == UINT32_MAX);

        COL_WRITE_VALUE(row, Cols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, Cols, from_state.timestamp, record.from_timestamp);

        COL_WRITE_VALUE(row, Cols, a, record.a);
        COL_WRITE_VALUE(row, Cols, b, record.b);
        COL_WRITE_VALUE(row, Cols, c, record.c);
        COL_WRITE_VALUE(row, Cols, data_write_pointer, record.write_ptr);

        mem_helper.fill(
            row.slice_from(COL_INDEX(Cols, pointer_read_aux_cols.base)),
            record.ptr_read.prev_timestamp,
            record.from_timestamp
        );

        if (!is_hint_storew) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Cols, data_read_aux_cols.base)),
                record.data_read.prev_timestamp,
                record.from_timestamp + 1
            );
        } else {
            mem_helper.fill_zero(row.slice_from(COL_INDEX(Cols, data_read_aux_cols)));
        }

        COL_WRITE_ARRAY(row, Cols, data_write_aux_cols.prev_data, record.data_write.prev_data);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Cols, data_write_aux_cols.base)),
            record.data_write.prev_timestamp,
            record.from_timestamp + 2 - (is_hint_storew ? 1 : 0)
        );
    }
};