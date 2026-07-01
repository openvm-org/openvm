#pragma once

#include "primitives/execution.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/constants.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

// Adapter columns for ADDI — immediate-only variant of Rv64BaseAluU16AdapterCols.
// Differences vs alu_u16: no rs2_as field, single reads_aux (rs2 is never a register read).
template <typename T> struct Rv64AddIAdapterCols {
    ExecutionState<T> from_state;
    T rd_ptr;
    T rs1_ptr;
    T rs2;
    T rs2_imm_sign;
    MemoryReadAuxCols<T> reads_aux;
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> writes_aux;
};

struct Rv64AddIAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rd_ptr;
    uint32_t rs1_ptr;
    uint32_t rs2;
    uint8_t rs2_imm_sign;
    MemoryReadAuxRecord reads_aux;
    MemoryWriteU16AuxRecord<BLOCK_FE_WIDTH> writes_aux;
};

struct Rv64AddIAdapter {
    MemoryAuxColsFactory mem_helper;
    VariableRangeChecker range_checker;

    __device__ Rv64AddIAdapter(
        VariableRangeChecker rc,
        uint32_t timestamp_max_bits
    )
        : mem_helper(rc, timestamp_max_bits), range_checker(rc) {}

    __device__ void fill_trace_row(RowSlice row, Rv64AddIAdapterRecord record) {
        COL_WRITE_VALUE(row, Rv64AddIAdapterCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(
            row, Rv64AddIAdapterCols, from_state.timestamp, record.from_timestamp
        );

        COL_WRITE_VALUE(row, Rv64AddIAdapterCols, rd_ptr, record.rd_ptr);
        COL_WRITE_VALUE(row, Rv64AddIAdapterCols, rs1_ptr, record.rs1_ptr);
        COL_WRITE_VALUE(row, Rv64AddIAdapterCols, rs2, record.rs2);
        COL_WRITE_VALUE(row, Rv64AddIAdapterCols, rs2_imm_sign, record.rs2_imm_sign);

        // rs1 register read at timestamp slot 0.
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64AddIAdapterCols, reads_aux)),
            record.reads_aux.prev_timestamp,
            record.from_timestamp
        );

        // rs2 is always an immediate — range-check the low u16 limb unconditionally.
        range_checker.add_count(record.rs2 & uint32_t(UINT16_MAX), U16_BITS);

        // rd write at timestamp slot 1 (no rs2 register read slot).
        Fp prev[BLOCK_FE_WIDTH];
        copy_u16_cells(prev, record.writes_aux.prev_data);
        COL_WRITE_ARRAY(row, Rv64AddIAdapterCols, writes_aux.prev_data, prev);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64AddIAdapterCols, writes_aux)),
            record.writes_aux.prev_timestamp,
            record.from_timestamp + 1
        );
    }
};
