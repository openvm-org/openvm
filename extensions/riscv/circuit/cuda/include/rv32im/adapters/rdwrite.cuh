#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"

using namespace riscv;

template <typename T> struct Rv64RdWriteAdapterCols {
    ExecutionState<T> from_state; // { pub pc: T, pub timestamp: T}
    T rd_ptr;
    MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS> rd_aux_cols;
};

struct Rv64RdWriteAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rd_ptr;
    MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS> rd_aux_record;
};

struct Rv64RdWriteAdapter {
    MemoryAuxColsFactory mem_helper;

    __device__ Rv64RdWriteAdapter(VariableRangeChecker range_checker, uint32_t timestamp_max_bits)
        : mem_helper(range_checker, timestamp_max_bits) {}

    __device__ inline void fill_trace_row(RowSlice row, Rv64RdWriteAdapterRecord record) {
        COL_WRITE_VALUE(row, Rv64RdWriteAdapterCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, Rv64RdWriteAdapterCols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Rv64RdWriteAdapterCols, rd_ptr, record.rd_ptr);

        RowSlice aux_row = row.slice_from(COL_INDEX(Rv64RdWriteAdapterCols, rd_aux_cols));
        COL_WRITE_ARRAY(aux_row, MemoryWriteAuxCols, prev_data, record.rd_aux_record.prev_data);
        mem_helper.fill(
            aux_row.slice_from(COL_INDEX(MemoryWriteAuxCols, base)),
            record.rd_aux_record.prev_timestamp,
            record.from_timestamp
        );
    }
};

template <typename T> struct Rv64CondRdWriteAdapterCols {
    Rv64RdWriteAdapterCols<T> inner;
    T needs_write;
};

struct Rv64CondRdWriteAdapter {
    MemoryAuxColsFactory mem_helper;
    uint32_t timestamp_max_bits;

    __device__ Rv64CondRdWriteAdapter(
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : mem_helper(range_checker, timestamp_max_bits), timestamp_max_bits(timestamp_max_bits) {}

    __device__ inline void fill_trace_row(RowSlice row, Rv64RdWriteAdapterRecord record) {
        bool do_write = (record.rd_ptr != UINT32_MAX);
        COL_WRITE_VALUE(row, Rv64CondRdWriteAdapterCols, needs_write, do_write);

        RowSlice inner = row.slice_from(COL_INDEX(Rv64CondRdWriteAdapterCols, inner));

        if (do_write) {
            Rv64RdWriteAdapter adapter(mem_helper.range_checker, timestamp_max_bits);
            adapter.fill_trace_row(inner, record);
        } else {
            inner.fill_zero(0, sizeof(Rv64RdWriteAdapterCols<uint8_t>));
            COL_WRITE_VALUE(
                inner, Rv64RdWriteAdapterCols, from_state.timestamp, record.from_timestamp
            );
            COL_WRITE_VALUE(inner, Rv64RdWriteAdapterCols, from_state.pc, record.from_pc);
        }
    }
};
