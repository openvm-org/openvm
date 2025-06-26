#pragma once

#include "execution.h"
#include "system/memory/controller.cuh"
#include "trace_access.h"

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
    MemoryWriteAuxRecord<RV32_REGISTER_NUM_LIMBS> rd_aux_record;
};

struct Rv32RdWriteAdapter {
    MemoryWriteAuxAdapter<RV32_REGISTER_NUM_LIMBS> mem_write_aux_adapter;

    __device__ Rv32RdWriteAdapter(VariableRangeChecker range_checker)
        : mem_write_aux_adapter(range_checker) {}

    __device__ void fill_trace_row(RowSlice row, Rv32RdWriteAdapterRecord record) {
        COL_WRITE_VALUE(row, Rv32RdWriteAdapterCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, Rv32RdWriteAdapterCols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Rv32RdWriteAdapterCols, rd_ptr, record.rd_ptr);

        mem_write_aux_adapter.fill_trace_row(
            row.slice_from(COL_INDEX(Rv32RdWriteAdapterCols, rd_aux_cols)),
            record.rd_aux_record,
            record.from_timestamp
        );
    }
};