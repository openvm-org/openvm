#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <typename T> struct Rv64MultAdapterCols {
    ExecutionState<T> from_state;
    T rd_ptr;
    T rs1_ptr;
    T rs2_ptr;
    MemoryReadAuxCols<T> reads_aux[2];
    MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS> writes_aux;
};

struct Rv64MultAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rd_ptr;
    uint32_t rs1_ptr;
    uint32_t rs2_ptr;
    MemoryReadAuxRecord reads_aux[2];
    MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS> writes_aux;
};

struct Rv64MultAdapter {
    MemoryAuxColsFactory mem_helper;

    __device__ Rv64MultAdapter(VariableRangeChecker range_checker, uint32_t timestamp_max_bits)
        : mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_trace_row(RowSlice row, Rv64MultAdapterRecord record);
};

__device__ inline void Rv64MultAdapter::fill_trace_row(RowSlice row, Rv64MultAdapterRecord record) {
    uint32_t ts = record.from_timestamp;

    COL_WRITE_ARRAY(row, Rv64MultAdapterCols, writes_aux.prev_data, record.writes_aux.prev_data);
    mem_helper.fill(
        row.slice_from(COL_INDEX(Rv64MultAdapterCols, writes_aux)),
        record.writes_aux.prev_timestamp,
        ts + 2
    );

    for (int i = 0; i < 2; i++) {
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64MultAdapterCols, reads_aux[i])),
            record.reads_aux[i].prev_timestamp,
            ts + i
        );
    }

    COL_WRITE_VALUE(row, Rv64MultAdapterCols, rs2_ptr, record.rs2_ptr);
    COL_WRITE_VALUE(row, Rv64MultAdapterCols, rs1_ptr, record.rs1_ptr);
    COL_WRITE_VALUE(row, Rv64MultAdapterCols, rd_ptr, record.rd_ptr);
    COL_WRITE_VALUE(row, Rv64MultAdapterCols, from_state.pc, record.from_pc);
    COL_WRITE_VALUE(row, Rv64MultAdapterCols, from_state.timestamp, record.from_timestamp);
}