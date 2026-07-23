#pragma once

#include "primitives/execution.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/constants.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

// Adapter for base ALU instructions with two register operands.
template <typename T> struct Rv64BaseAluRegU16AdapterCols {
    ExecutionState<T> from_state;
    T rd_ptr;
    T rs1_ptr;
    T rs2_ptr;
    MemoryReadAuxCols<T> reads_aux[2];
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> writes_aux;
};

struct Rv64BaseAluRegU16AdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rd_ptr;
    uint32_t rs1_ptr;
    uint32_t rs2_ptr;
    MemoryReadAuxRecord reads_aux[2];
    MemoryWriteU16AuxRecord<BLOCK_FE_WIDTH> writes_aux;
};

static_assert(sizeof(Rv64BaseAluRegU16AdapterRecord) == 40);
static_assert(offsetof(Rv64BaseAluRegU16AdapterRecord, reads_aux) == 20);
static_assert(offsetof(Rv64BaseAluRegU16AdapterRecord, writes_aux) == 28);

struct Rv64BaseAluRegU16Adapter {
    MemoryAuxColsFactory mem_helper;

    __device__ Rv64BaseAluRegU16Adapter(VariableRangeChecker rc, uint32_t timestamp_max_bits)
        : mem_helper(rc, timestamp_max_bits) {}

    __device__ void fill_trace_row(
        RowSlice row,
        uint32_t from_pc,
        uint32_t from_timestamp,
        uint32_t rd_ptr,
        uint32_t rs1_ptr,
        uint32_t rs2_ptr,
        uint32_t rs1_prev_timestamp,
        uint32_t rs2_prev_timestamp,
        uint32_t write_prev_timestamp,
        uint16_t const (&write_prev_data)[BLOCK_FE_WIDTH]
    ) {
        COL_WRITE_VALUE(row, Rv64BaseAluRegU16AdapterCols, from_state.pc, from_pc);
        COL_WRITE_VALUE(row, Rv64BaseAluRegU16AdapterCols, from_state.timestamp, from_timestamp);
        COL_WRITE_VALUE(row, Rv64BaseAluRegU16AdapterCols, rd_ptr, rd_ptr);
        COL_WRITE_VALUE(row, Rv64BaseAluRegU16AdapterCols, rs1_ptr, rs1_ptr);
        COL_WRITE_VALUE(row, Rv64BaseAluRegU16AdapterCols, rs2_ptr, rs2_ptr);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluRegU16AdapterCols, reads_aux[0])),
            rs1_prev_timestamp,
            from_timestamp
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluRegU16AdapterCols, reads_aux[1])),
            rs2_prev_timestamp,
            from_timestamp + 1
        );
        Fp prev[BLOCK_FE_WIDTH];
        copy_u16_cells(prev, write_prev_data);
        COL_WRITE_ARRAY(row, Rv64BaseAluRegU16AdapterCols, writes_aux.prev_data, prev);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluRegU16AdapterCols, writes_aux)),
            write_prev_timestamp,
            from_timestamp + 2
        );
    }

    __device__ void fill_trace_row(RowSlice row, Rv64BaseAluRegU16AdapterRecord record) {
        COL_WRITE_VALUE(row, Rv64BaseAluRegU16AdapterCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(
            row, Rv64BaseAluRegU16AdapterCols, from_state.timestamp, record.from_timestamp
        );

        COL_WRITE_VALUE(row, Rv64BaseAluRegU16AdapterCols, rd_ptr, record.rd_ptr);
        COL_WRITE_VALUE(row, Rv64BaseAluRegU16AdapterCols, rs1_ptr, record.rs1_ptr);
        COL_WRITE_VALUE(row, Rv64BaseAluRegU16AdapterCols, rs2_ptr, record.rs2_ptr);

        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluRegU16AdapterCols, reads_aux[0])),
            record.reads_aux[0].prev_timestamp,
            record.from_timestamp
        );

        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluRegU16AdapterCols, reads_aux[1])),
            record.reads_aux[1].prev_timestamp,
            record.from_timestamp + 1
        );

        Fp prev[BLOCK_FE_WIDTH];
        copy_u16_cells(prev, record.writes_aux.prev_data);
        COL_WRITE_ARRAY(row, Rv64BaseAluRegU16AdapterCols, writes_aux.prev_data, prev);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluRegU16AdapterCols, writes_aux)),
            record.writes_aux.prev_timestamp,
            record.from_timestamp + 2
        );
    }
};
