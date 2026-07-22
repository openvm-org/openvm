#pragma once

#include "primitives/execution.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/constants.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

// U16-limb adapter with one register read and one register write.
template <typename T> struct Rv64BaseAluImmU16AdapterCols {
    ExecutionState<T> from_state;
    T rd_ptr;
    T rs1_ptr;
    MemoryReadAuxCols<T> reads_aux;
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> writes_aux;
};

struct Rv64BaseAluImmU16AdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rd_ptr;
    uint32_t rs1_ptr;
    MemoryReadAuxRecord reads_aux;
    MemoryWriteU16AuxRecord<BLOCK_FE_WIDTH> writes_aux;
};

static_assert(sizeof(Rv64BaseAluImmU16AdapterRecord) == 32);
static_assert(offsetof(Rv64BaseAluImmU16AdapterRecord, reads_aux) == 16);
static_assert(offsetof(Rv64BaseAluImmU16AdapterRecord, writes_aux) == 20);

struct Rv64BaseAluImmU16Adapter {
    MemoryAuxColsFactory mem_helper;

    __device__ Rv64BaseAluImmU16Adapter(VariableRangeChecker rc, uint32_t timestamp_max_bits)
        : mem_helper(rc, timestamp_max_bits) {}

    __device__ void fill_trace_row(
        RowSlice row,
        uint32_t from_pc,
        uint32_t from_timestamp,
        uint32_t rd_ptr,
        uint32_t rs1_ptr,
        uint32_t read_prev_timestamp,
        uint32_t write_prev_timestamp,
        uint16_t const (&write_prev_data)[BLOCK_FE_WIDTH]
    ) {
        COL_WRITE_VALUE(row, Rv64BaseAluImmU16AdapterCols, from_state.pc, from_pc);
        COL_WRITE_VALUE(row, Rv64BaseAluImmU16AdapterCols, from_state.timestamp, from_timestamp);
        COL_WRITE_VALUE(row, Rv64BaseAluImmU16AdapterCols, rd_ptr, rd_ptr);
        COL_WRITE_VALUE(row, Rv64BaseAluImmU16AdapterCols, rs1_ptr, rs1_ptr);

        // rs1 register read at timestamp slot 0.
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluImmU16AdapterCols, reads_aux)),
            read_prev_timestamp,
            from_timestamp
        );

        // rd write at timestamp slot 1.
        Fp prev[BLOCK_FE_WIDTH];
        copy_u16_cells(prev, write_prev_data);
        COL_WRITE_ARRAY(row, Rv64BaseAluImmU16AdapterCols, writes_aux.prev_data, prev);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluImmU16AdapterCols, writes_aux)),
            write_prev_timestamp,
            from_timestamp + 1
        );
    }

    __device__ void fill_trace_row(RowSlice row, Rv64BaseAluImmU16AdapterRecord record) {
        fill_trace_row(
            row,
            record.from_pc,
            record.from_timestamp,
            record.rd_ptr,
            record.rs1_ptr,
            record.reads_aux.prev_timestamp,
            record.writes_aux.prev_timestamp,
            record.writes_aux.prev_data
        );
    }
};
