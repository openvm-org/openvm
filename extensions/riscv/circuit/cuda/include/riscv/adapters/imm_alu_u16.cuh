#pragma once

#include "primitives/execution.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/constants.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

// Adapter columns for immediate-operand base-ALU instructions (I-type: read rs1,
// write rd, immediate operand). Immediate-only variant of Rv64BaseAluU16AdapterCols:
// rs2 is always an immediate, so there is no rs2_as / rs2_imm_sign and only one reads_aux.
template <typename T> struct Rv64ImmBaseAluU16AdapterCols {
    ExecutionState<T> from_state;
    T rd_ptr;
    T rs1_ptr;
    MemoryReadAuxCols<T> reads_aux;
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> writes_aux;
};

struct Rv64ImmBaseAluU16AdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rd_ptr;
    uint32_t rs1_ptr;
    MemoryReadAuxRecord reads_aux;
    MemoryWriteU16AuxRecord<BLOCK_FE_WIDTH> writes_aux;
};

struct Rv64ImmBaseAluU16Adapter {
    MemoryAuxColsFactory mem_helper;

    __device__ Rv64ImmBaseAluU16Adapter(VariableRangeChecker rc, uint32_t timestamp_max_bits)
        : mem_helper(rc, timestamp_max_bits) {}

    __device__ void fill_trace_row(RowSlice row, Rv64ImmBaseAluU16AdapterRecord record) {
        COL_WRITE_VALUE(row, Rv64ImmBaseAluU16AdapterCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, Rv64ImmBaseAluU16AdapterCols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Rv64ImmBaseAluU16AdapterCols, rd_ptr, record.rd_ptr);
        COL_WRITE_VALUE(row, Rv64ImmBaseAluU16AdapterCols, rs1_ptr, record.rs1_ptr);

        // rs1 register read at timestamp slot 0.
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64ImmBaseAluU16AdapterCols, reads_aux)),
            record.reads_aux.prev_timestamp,
            record.from_timestamp
        );

        // rd write at timestamp slot 1.
        Fp prev[BLOCK_FE_WIDTH];
        copy_u16_cells(prev, record.writes_aux.prev_data);
        COL_WRITE_ARRAY(row, Rv64ImmBaseAluU16AdapterCols, writes_aux.prev_data, prev);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64ImmBaseAluU16AdapterCols, writes_aux)),
            record.writes_aux.prev_timestamp,
            record.from_timestamp + 1
        );
    }
};
