#pragma once

#include "primitives/execution.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/constants.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

// Immediate-only byte-limb adapter (single register read + register write). The immediate itself
// lives in the core, which passes it back as the ImmInstruction immediate expression, so there
// is no rs2_imm column and no bitwise lookup here.
template <typename T> struct Rv64BaseAluImmAdapterCols {
    ExecutionState<T> from_state;
    T rd_ptr;
    T rs1_ptr;
    MemoryReadAuxCols<T> reads_aux;
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> writes_aux;
};

struct Rv64BaseAluImmAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rd_ptr;
    uint32_t rs1_ptr;
    MemoryReadAuxRecord reads_aux;
    MemoryWriteBytesAuxRecord<RV64_REGISTER_NUM_LIMBS> writes_aux;
};

struct Rv64BaseAluImmAdapter {
    MemoryAuxColsFactory mem_helper;

    __device__ Rv64BaseAluImmAdapter(VariableRangeChecker rc, uint32_t timestamp_max_bits)
        : mem_helper(rc, timestamp_max_bits) {}

    __device__ void fill_trace_row(RowSlice row, Rv64BaseAluImmAdapterRecord record) {
        COL_WRITE_VALUE(row, Rv64BaseAluImmAdapterCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, Rv64BaseAluImmAdapterCols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Rv64BaseAluImmAdapterCols, rd_ptr, record.rd_ptr);
        COL_WRITE_VALUE(row, Rv64BaseAluImmAdapterCols, rs1_ptr, record.rs1_ptr);

        // rs1 register read at timestamp slot 0.
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluImmAdapterCols, reads_aux)),
            record.reads_aux.prev_timestamp,
            record.from_timestamp
        );

        // rd write at timestamp slot 1.
        Fp packed_prev[BLOCK_FE_WIDTH];
        pack_u8_block_bytes(packed_prev, record.writes_aux.prev_data);
        COL_WRITE_ARRAY(row, Rv64BaseAluImmAdapterCols, writes_aux.prev_data, packed_prev);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluImmAdapterCols, writes_aux)),
            record.writes_aux.prev_timestamp,
            record.from_timestamp + 1
        );
    }
};
