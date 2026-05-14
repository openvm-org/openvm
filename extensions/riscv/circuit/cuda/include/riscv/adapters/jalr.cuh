#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <typename T> struct Rv64JalrAdapterCols {
    ExecutionState<T> from_state; // { pc, timestamp }
    T rs1_ptr;
    MemoryReadAuxCols<T> rs1_aux_cols;
    T rd_ptr;
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> rd_aux_cols;
    T needs_write;
};

/// Pattern B u16: stored prev_data is the BLOCK_FE_WIDTH u16 cells written to the register, not
/// raw bytes. Matches the Rust `Rv64JalrAdapterWriteAuxRecord` shape.
struct Rv64JalrAdapterWriteAuxRecord {
    uint32_t prev_timestamp;
    uint16_t prev_data[BLOCK_FE_WIDTH];
};

struct Rv64JalrAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    uint32_t rs1_ptr;
    // rd_ptr == UINT32_MAX means “no write”
    uint32_t rd_ptr;

    MemoryReadAuxRecord reads_aux;
    Rv64JalrAdapterWriteAuxRecord writes_aux;
};

struct Rv64JalrAdapter {
    MemoryAuxColsFactory mem_helper;

    __device__ Rv64JalrAdapter(VariableRangeChecker range_checker, uint32_t timestamp_max_bits)
        : mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_trace_row(RowSlice row, Rv64JalrAdapterRecord record) {
        bool do_write = record.rd_ptr != UINT32_MAX;
        COL_WRITE_VALUE(row, Rv64JalrAdapterCols, needs_write, do_write);

        if (do_write) {
            Fp prev[BLOCK_FE_WIDTH];
#pragma unroll
            for (size_t i = 0; i < BLOCK_FE_WIDTH; i++) {
                prev[i] = Fp(uint32_t(record.writes_aux.prev_data[i]));
            }
            COL_WRITE_ARRAY(row, Rv64JalrAdapterCols, rd_aux_cols.prev_data, prev);
            mem_helper.fill(
                row.slice_from(COL_INDEX(Rv64JalrAdapterCols, rd_aux_cols.base)),
                record.writes_aux.prev_timestamp,
                record.from_timestamp + 1
            );
            COL_WRITE_VALUE(row, Rv64JalrAdapterCols, rd_ptr, record.rd_ptr);
        } else {
            COL_FILL_ZERO(row, Rv64JalrAdapterCols, rd_aux_cols);
            COL_WRITE_VALUE(row, Rv64JalrAdapterCols, rd_ptr, 0u);
        }

        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64JalrAdapterCols, rs1_aux_cols)),
            record.reads_aux.prev_timestamp,
            record.from_timestamp
        );

        COL_WRITE_VALUE(row, Rv64JalrAdapterCols, rs1_ptr, record.rs1_ptr);
        COL_WRITE_VALUE(row, Rv64JalrAdapterCols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Rv64JalrAdapterCols, from_state.pc, record.from_pc);
    }
};
