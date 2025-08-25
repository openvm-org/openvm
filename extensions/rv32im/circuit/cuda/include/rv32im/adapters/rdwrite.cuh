#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"

using namespace riscv;

template <typename T> struct Rv32RdWriteAdapterCols {
    ExecutionState<T> from_state; // { pub pc: T, pub timestamp: T}
    T rd_ptr;
    MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS> rd_aux_cols;
};

struct Rv32RdWriteAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rd_ptr;
    MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS> rd_aux_record;
};

struct Rv32RdWriteAdapter {
    MemoryAuxColsFactory mem_helper;

    __device__ Rv32RdWriteAdapter(VariableRangeChecker range_checker, uint32_t timestamp_max_bits)
        : mem_helper(range_checker, timestamp_max_bits) {}

    __device__ inline void fill_trace_row(RowSlice row, Rv32RdWriteAdapterRecord record) {
        COL_WRITE_VALUE(row, Rv32RdWriteAdapterCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, Rv32RdWriteAdapterCols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Rv32RdWriteAdapterCols, rd_ptr, record.rd_ptr);

        RowSlice aux_row = row.slice_from(COL_INDEX(Rv32RdWriteAdapterCols, rd_aux_cols));
        COL_WRITE_ARRAY(aux_row, MemoryWriteAuxCols, prev_data, record.rd_aux_record.prev_data);
        mem_helper.fill(
            aux_row.slice_from(COL_INDEX(MemoryWriteAuxCols, base)),
            record.rd_aux_record.prev_timestamp,
            record.from_timestamp
        );
    }
};

template <typename T> struct Rv32CondRdWriteAdapterCols {
    Rv32RdWriteAdapterCols<T> inner;
    T needs_write;
};

struct Rv32CondRdWriteAdapter {
    MemoryAuxColsFactory mem_helper;
    uint32_t timestamp_max_bits;

    __device__ Rv32CondRdWriteAdapter(
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : mem_helper(range_checker, timestamp_max_bits), timestamp_max_bits(timestamp_max_bits) {}

    __device__ inline void fill_trace_row(RowSlice row, Rv32RdWriteAdapterRecord record) {
        bool do_write = (record.rd_ptr != UINT32_MAX);
        COL_WRITE_VALUE(row, Rv32CondRdWriteAdapterCols, needs_write, do_write);

        RowSlice inner = row.slice_from(COL_INDEX(Rv32CondRdWriteAdapterCols, inner));

        if (do_write) {
            Rv32RdWriteAdapter adapter(mem_helper.range_checker, timestamp_max_bits);
            adapter.fill_trace_row(inner, record);
        } else {
            inner.fill_zero(0, sizeof(Rv32RdWriteAdapterCols<uint8_t>));
            COL_WRITE_VALUE(
                inner, Rv32RdWriteAdapterCols, from_state.timestamp, record.from_timestamp
            );
            COL_WRITE_VALUE(inner, Rv32RdWriteAdapterCols, from_state.pc, record.from_pc);
        }
    }
};
