#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <typename T> struct Rv32JalrAdapterCols {
    ExecutionState<T> from_state; // { pc, timestamp }
    T rs1_ptr;
    MemoryReadAuxCols<T> rs1_aux_cols;
    T rd_ptr;
    MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS> rd_aux_cols;
    T needs_write;
};

struct Rv32JalrAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    uint32_t rs1_ptr;
    // rd_ptr == UINT32_MAX means “no write”
    uint32_t rd_ptr;

    MemoryReadAuxRecord reads_aux;
    MemoryWriteBytesAuxRecord<RV32_REGISTER_NUM_LIMBS> writes_aux;
};

struct Rv32JalrAdapter {
    MemoryAuxColsFactory mem_helper;

    __device__ Rv32JalrAdapter(VariableRangeChecker range_checker, uint32_t timestamp_max_bits)
        : mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_trace_row(RowSlice row, Rv32JalrAdapterRecord record) {
        bool do_write = record.rd_ptr != UINT32_MAX;
        COL_WRITE_VALUE(row, Rv32JalrAdapterCols, needs_write, do_write);

        if (do_write) {
            RowSlice aux_row = row.slice_from(COL_INDEX(Rv32JalrAdapterCols, rd_aux_cols));
            // NOTE: COL_WRITE_ARRAY uses the default NUM_LIMBS = RV32_REGISTER_NUM_LIMBS in MemoryWriteAuxCols template definition for size calculations, which is correct in this case for Rv32JalrAdapterCols
            COL_WRITE_ARRAY(aux_row, MemoryWriteAuxCols, prev_data, record.writes_aux.prev_data);
            mem_helper.fill(
                aux_row.slice_from(COL_INDEX(MemoryWriteAuxCols, base)),
                record.writes_aux.prev_timestamp,
                record.from_timestamp + 1
            );
            COL_WRITE_VALUE(row, Rv32JalrAdapterCols, rd_ptr, record.rd_ptr);
        } else {
            // NOTE: see note above on size calculation for MemoryWriteAuxCols
            COL_FILL_ZERO(row, Rv32JalrAdapterCols, rd_aux_cols);
            COL_WRITE_VALUE(row, Rv32JalrAdapterCols, rd_ptr, 0u);
        }

        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv32JalrAdapterCols, rs1_aux_cols)),
            record.reads_aux.prev_timestamp,
            record.from_timestamp
        );

        COL_WRITE_VALUE(row, Rv32JalrAdapterCols, rs1_ptr, record.rs1_ptr);
        COL_WRITE_VALUE(row, Rv32JalrAdapterCols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Rv32JalrAdapterCols, from_state.pc, record.from_pc);
    }
};
