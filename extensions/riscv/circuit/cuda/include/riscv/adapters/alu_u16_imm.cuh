#pragma once

#include "primitives/execution.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/constants.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

// Immediate-only variant of Rv64BaseAluU16AdapterCols: the second operand always comes from the
// instruction, so there is no rs2_as selector and only one reads_aux.
template <typename T> struct Rv64BaseAluU16ImmAdapterCols {
    ExecutionState<T> from_state;
    T rd_ptr;
    T rs1_ptr;
    T rs2_imm;      // The 24-bit immediate operand.
    T rs2_imm_sign; // Sign bit of the immediate (0 or 1).
    MemoryReadAuxCols<T> reads_aux;
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> writes_aux;
};

struct Rv64BaseAluU16ImmAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rd_ptr;
    uint32_t rs1_ptr;
    uint32_t rs2_imm;
    uint8_t rs2_imm_sign;
    MemoryReadAuxRecord reads_aux;
    MemoryWriteU16AuxRecord<BLOCK_FE_WIDTH> writes_aux;
};

struct Rv64BaseAluU16ImmAdapter {
    MemoryAuxColsFactory mem_helper;
    VariableRangeChecker range_checker;

    __device__ Rv64BaseAluU16ImmAdapter(VariableRangeChecker rc, uint32_t timestamp_max_bits)
        : mem_helper(rc, timestamp_max_bits), range_checker(rc) {}

    __device__ void fill_trace_row(RowSlice row, Rv64BaseAluU16ImmAdapterRecord record) {
        COL_WRITE_VALUE(row, Rv64BaseAluU16ImmAdapterCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(
            row, Rv64BaseAluU16ImmAdapterCols, from_state.timestamp, record.from_timestamp
        );
        COL_WRITE_VALUE(row, Rv64BaseAluU16ImmAdapterCols, rd_ptr, record.rd_ptr);
        COL_WRITE_VALUE(row, Rv64BaseAluU16ImmAdapterCols, rs1_ptr, record.rs1_ptr);
        COL_WRITE_VALUE(row, Rv64BaseAluU16ImmAdapterCols, rs2_imm, record.rs2_imm);
        COL_WRITE_VALUE(row, Rv64BaseAluU16ImmAdapterCols, rs2_imm_sign, record.rs2_imm_sign);

        // rs1 register read at timestamp slot 0.
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluU16ImmAdapterCols, reads_aux)),
            record.reads_aux.prev_timestamp,
            record.from_timestamp
        );

        // Range check the low u16 immediate limb.
        range_checker.add_count(record.rs2_imm & uint32_t(UINT16_MAX), U16_BITS);

        // rd write at timestamp slot 1.
        Fp prev[BLOCK_FE_WIDTH];
        copy_u16_cells(prev, record.writes_aux.prev_data);
        COL_WRITE_ARRAY(row, Rv64BaseAluU16ImmAdapterCols, writes_aux.prev_data, prev);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluU16ImmAdapterCols, writes_aux)),
            record.writes_aux.prev_timestamp,
            record.from_timestamp + 1
        );
    }
};
