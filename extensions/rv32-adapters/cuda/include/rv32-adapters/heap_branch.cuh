#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

// Heap branch adapter with explicit READ_BLOCKS parameter.
// For 256-bit (32-byte) operations with 4-byte block size: READ_SIZE=32, READ_BLOCKS=8
template <typename T, size_t NUM_READS, size_t READ_SIZE, size_t READ_BLOCKS> struct Rv32HeapBranchAdapterCols {
    ExecutionState<T> from_state;

    T rs_ptr[NUM_READS];
    T rs_val[NUM_READS][RV32_REGISTER_NUM_LIMBS];
    MemoryReadAuxCols<T> rs_read_aux[NUM_READS];

    // READ_BLOCKS reads per pointer
    MemoryReadAuxCols<T> heap_read_aux[NUM_READS][READ_BLOCKS];
};

template <size_t NUM_READS, size_t READ_BLOCKS> struct Rv32HeapBranchAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    uint32_t rs_ptr[NUM_READS];
    uint32_t rs_vals[NUM_READS];

    MemoryReadAuxRecord rs_read_aux[NUM_READS];
    MemoryReadAuxRecord heap_read_aux[NUM_READS][READ_BLOCKS];
};

template <size_t NUM_READS, size_t READ_SIZE, size_t READ_BLOCKS> struct Rv32HeapBranchAdapter {
    size_t pointer_max_bits;
    BitwiseOperationLookup bitwise_lookup;
    MemoryAuxColsFactory mem_helper;

    static constexpr size_t RV32_REGISTER_TOTAL_BITS = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS;
    static constexpr size_t MSL_SHIFT = RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);

    __device__ Rv32HeapBranchAdapter(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        BitwiseOperationLookup bitwise_lookup,
        uint32_t timestamp_max_bits
    )
        : pointer_max_bits(pointer_max_bits), bitwise_lookup(bitwise_lookup),
          mem_helper(range_checker, timestamp_max_bits) {}

    template <typename T> using Cols = Rv32HeapBranchAdapterCols<T, NUM_READS, READ_SIZE, READ_BLOCKS>;

    __device__ void fill_trace_row(RowSlice row, Rv32HeapBranchAdapterRecord<NUM_READS, READ_BLOCKS> record) {
        const size_t limb_shift_bits = RV32_REGISTER_TOTAL_BITS - pointer_max_bits;

        bitwise_lookup.add_range(
            (record.rs_vals[0] >> MSL_SHIFT) << limb_shift_bits,
            NUM_READS > 1 ? (record.rs_vals[1] >> MSL_SHIFT) << limb_shift_bits : 0
        );

        // Fill heap read aux in reverse order
        for (int i = NUM_READS - 1; i >= 0; i--) {
            for (int j = READ_BLOCKS - 1; j >= 0; j--) {
                mem_helper.fill(
                    row.slice_from(COL_INDEX(Cols, heap_read_aux[i][j])),
                    record.heap_read_aux[i][j].prev_timestamp,
                    record.from_timestamp + NUM_READS + i * READ_BLOCKS + j
                );
            }
        }

        for (int i = NUM_READS - 1; i >= 0; i--) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Cols, rs_read_aux[i])),
                record.rs_read_aux[i].prev_timestamp,
                record.from_timestamp + i
            );
        }

        for (int i = NUM_READS - 1; i >= 0; i--) {
            COL_WRITE_ARRAY(row, Cols, rs_val[i], (uint8_t *)&record.rs_vals[i]);
        }

        for (int i = NUM_READS - 1; i >= 0; i--) {
            COL_WRITE_VALUE(row, Cols, rs_ptr[i], record.rs_ptr[i]);
        }

        COL_WRITE_VALUE(row, Cols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Cols, from_state.pc, record.from_pc);
    }
};
