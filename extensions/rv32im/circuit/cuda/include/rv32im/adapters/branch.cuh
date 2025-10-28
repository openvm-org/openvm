#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <typename T> struct Rv32BranchAdapterCols {
    ExecutionState<T> from_state; // { pc, timestamp }
    T rs1_ptr;
    T rs2_ptr;
    MemoryReadAuxCols<T> reads_aux_0;
    MemoryReadAuxCols<T> reads_aux_1;
};

struct Rv32BranchAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rs1_ptr;
    uint32_t rs2_ptr;
    MemoryReadAuxRecord reads_aux[2];
};

struct Rv32BranchAdapter {
    MemoryAuxColsFactory mem_helper;

    __device__ Rv32BranchAdapter(VariableRangeChecker rc, uint32_t timestamp_max_bits)
        : mem_helper(rc, timestamp_max_bits) {}

    __device__ void fill_trace_row(RowSlice row, Rv32BranchAdapterRecord rec) {

        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv32BranchAdapterCols, reads_aux_1)),
            rec.reads_aux[1].prev_timestamp,
            rec.from_timestamp + 1
        );

        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv32BranchAdapterCols, reads_aux_0)),
            rec.reads_aux[0].prev_timestamp,
            rec.from_timestamp
        );

        COL_WRITE_VALUE(row, Rv32BranchAdapterCols, from_state.pc, rec.from_pc);
        COL_WRITE_VALUE(row, Rv32BranchAdapterCols, from_state.timestamp, rec.from_timestamp);
        COL_WRITE_VALUE(row, Rv32BranchAdapterCols, rs1_ptr, rec.rs1_ptr);
        COL_WRITE_VALUE(row, Rv32BranchAdapterCols, rs2_ptr, rec.rs2_ptr);
    }
};
