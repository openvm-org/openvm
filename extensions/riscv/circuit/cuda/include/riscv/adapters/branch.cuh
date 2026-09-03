#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <typename T> struct BranchAdapterCols {
    ExecutionState<T> from_state; // { pc, timestamp }
    T rs1_ptr;
    T rs2_ptr;
    MemoryReadAuxCols<T> reads_aux_0;
    MemoryReadAuxCols<T> reads_aux_1;
};

struct BranchAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rs1_ptr;
    uint32_t rs2_ptr;
    MemoryReadAuxRecord reads_aux[2];
};

struct BranchAdapter {
    MemoryAuxColsFactory mem_helper;

    __device__ BranchAdapter(VariableRangeChecker rc, uint32_t timestamp_max_bits)
        : mem_helper(rc, timestamp_max_bits) {}

    __device__ void fill_trace_row(
        RowSlice row,
        uint32_t from_pc,
        uint32_t from_timestamp,
        uint32_t rs1_ptr,
        uint32_t rs2_ptr,
        uint32_t rs1_prev_timestamp,
        uint32_t rs2_prev_timestamp
    ) {
        mem_helper.fill(
            row.slice_from(COL_INDEX(BranchAdapterCols, reads_aux_1)),
            rs2_prev_timestamp,
            from_timestamp + 1
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(BranchAdapterCols, reads_aux_0)),
            rs1_prev_timestamp,
            from_timestamp
        );
        COL_WRITE_VALUE(row, BranchAdapterCols, from_state.pc, ::program::pc_to_idx(from_pc));
        COL_WRITE_VALUE(row, BranchAdapterCols, from_state.timestamp, from_timestamp);
        COL_WRITE_VALUE(row, BranchAdapterCols, rs1_ptr, rs1_ptr);
        COL_WRITE_VALUE(row, BranchAdapterCols, rs2_ptr, rs2_ptr);
    }

    __device__ void fill_trace_row(RowSlice row, BranchAdapterRecord rec) {
        fill_trace_row(
            row,
            rec.from_pc,
            rec.from_timestamp,
            rec.rs1_ptr,
            rec.rs2_ptr,
            rec.reads_aux[0].prev_timestamp,
            rec.reads_aux[1].prev_timestamp
        );
    }
};
