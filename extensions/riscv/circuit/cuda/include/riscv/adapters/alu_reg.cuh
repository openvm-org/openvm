#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <typename T> struct Rv64BaseAluRegAdapterCols {
    ExecutionState<T> from_state; // { pub pc: T, pub timestamp: T}
    T rd_ptr;
    T rs1_ptr;
    T rs2_ptr;
    MemoryReadAuxCols<T> reads_aux[2];
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> writes_aux;
};

struct Rv64BaseAluRegAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rd_ptr;
    uint32_t rs1_ptr;
    uint32_t rs2_ptr;
    MemoryReadAuxRecord reads_aux[2];
    MemoryWriteBytesAuxRecord<RV64_REGISTER_NUM_LIMBS> writes_aux;
};

static_assert(sizeof(Rv64BaseAluRegAdapterRecord) == 40);
static_assert(offsetof(Rv64BaseAluRegAdapterRecord, reads_aux) == 20);
static_assert(offsetof(Rv64BaseAluRegAdapterRecord, writes_aux) == 28);

struct Rv64BaseAluRegAdapter {
    MemoryAuxColsFactory mem_helper;

    __device__ Rv64BaseAluRegAdapter(
        VariableRangeChecker range_checker, uint32_t timestamp_max_bits
    )
        : mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_trace_row(RowSlice row, Rv64BaseAluRegAdapterRecord record) {
        COL_WRITE_VALUE(row, Rv64BaseAluRegAdapterCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(
            row, Rv64BaseAluRegAdapterCols, from_state.timestamp, record.from_timestamp
        );

        COL_WRITE_VALUE(row, Rv64BaseAluRegAdapterCols, rd_ptr, record.rd_ptr);
        COL_WRITE_VALUE(row, Rv64BaseAluRegAdapterCols, rs1_ptr, record.rs1_ptr);
        COL_WRITE_VALUE(row, Rv64BaseAluRegAdapterCols, rs2_ptr, record.rs2_ptr);

        // Read auxiliary for rs1
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluRegAdapterCols, reads_aux[0])),
            record.reads_aux[0].prev_timestamp,
            record.from_timestamp
        );

        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluRegAdapterCols, reads_aux[1])),
            record.reads_aux[1].prev_timestamp,
            record.from_timestamp + 1
        );

        Fp packed_prev[BLOCK_FE_WIDTH];
        pack_u8_block_bytes(packed_prev, record.writes_aux.prev_data);
        COL_WRITE_ARRAY(row, Rv64BaseAluRegAdapterCols, writes_aux.prev_data, packed_prev);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluRegAdapterCols, writes_aux)),
            record.writes_aux.prev_timestamp,
            record.from_timestamp + 2
        );
    }
};
