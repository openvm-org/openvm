#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <typename T> struct RdWriteAdapterCols {
    ExecutionState<T> from_state; // { pub pc: T, pub timestamp: T}
    T rd_ptr;
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> rd_aux_cols;
};

struct RdWriteAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rd_ptr;
    MemoryWriteU16AuxRecord<BLOCK_FE_WIDTH> rd_aux_record;
};

struct RdWriteAdapter {
    MemoryAuxColsFactory mem_helper;

    __device__ RdWriteAdapter(VariableRangeChecker range_checker, uint32_t timestamp_max_bits)
        : mem_helper(range_checker, timestamp_max_bits) {}

    __device__ inline void fill_trace_row(
        RowSlice row,
        uint32_t from_pc,
        uint32_t from_timestamp,
        uint32_t rd_ptr,
        uint32_t prev_timestamp,
        uint16_t const (&prev_data)[BLOCK_FE_WIDTH]
    ) {
        COL_WRITE_VALUE(row, RdWriteAdapterCols, from_state.pc, ::program::pc_to_idx(from_pc));
        COL_WRITE_VALUE(row, RdWriteAdapterCols, from_state.timestamp, from_timestamp);
        COL_WRITE_VALUE(row, RdWriteAdapterCols, rd_ptr, rd_ptr);

        Fp prev[BLOCK_FE_WIDTH];
        copy_u16_cells(prev, prev_data);
        COL_WRITE_ARRAY(row, RdWriteAdapterCols, rd_aux_cols.prev_data, prev);
        mem_helper.fill(
            row.slice_from(COL_INDEX(RdWriteAdapterCols, rd_aux_cols.base)),
            prev_timestamp,
            from_timestamp
        );
    }

    __device__ inline void fill_trace_row(RowSlice row, RdWriteAdapterRecord record) {
        fill_trace_row(
            row,
            record.from_pc,
            record.from_timestamp,
            record.rd_ptr,
            record.rd_aux_record.prev_timestamp,
            record.rd_aux_record.prev_data
        );
    }
};

template <typename T> struct CondRdWriteAdapterCols {
    RdWriteAdapterCols<T> inner;
    T needs_write;
};

struct CondRdWriteAdapter {
    MemoryAuxColsFactory mem_helper;
    uint32_t timestamp_max_bits;

    __device__ CondRdWriteAdapter(
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : mem_helper(range_checker, timestamp_max_bits), timestamp_max_bits(timestamp_max_bits) {}

    __device__ inline void fill_trace_row(
        RowSlice row,
        uint32_t from_pc,
        uint32_t from_timestamp,
        uint32_t rd_ptr,
        bool do_write,
        uint32_t prev_timestamp,
        uint16_t const (&prev_data)[BLOCK_FE_WIDTH]
    ) {
        COL_WRITE_VALUE(row, CondRdWriteAdapterCols, needs_write, do_write);

        RowSlice inner = row.slice_from(COL_INDEX(CondRdWriteAdapterCols, inner));

        if (do_write) {
            RdWriteAdapter adapter(mem_helper.range_checker, timestamp_max_bits);
            adapter.fill_trace_row(
                inner, from_pc, from_timestamp, rd_ptr, prev_timestamp, prev_data
            );
        } else {
            inner.fill_zero(0, sizeof(RdWriteAdapterCols<uint8_t>));
            COL_WRITE_VALUE(inner, RdWriteAdapterCols, from_state.timestamp, from_timestamp);
            COL_WRITE_VALUE(inner, RdWriteAdapterCols, from_state.pc, ::program::pc_to_idx(from_pc));
        }
    }

    __device__ inline void fill_trace_row(RowSlice row, RdWriteAdapterRecord record) {
        bool do_write = (record.rd_ptr != UINT32_MAX);
        if (do_write) {
            fill_trace_row(
                row,
                record.from_pc,
                record.from_timestamp,
                record.rd_ptr,
                true,
                record.rd_aux_record.prev_timestamp,
                record.rd_aux_record.prev_data
            );
        } else {
            uint16_t unused_prev_data[BLOCK_FE_WIDTH] = {};
            fill_trace_row(
                row,
                record.from_pc,
                record.from_timestamp,
                0,
                false,
                0,
                unused_prev_data
            );
        }
    }
};
