#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "riscv-adapters/constants.cuh"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <typename T, size_t NUM_READS, size_t BLOCKS_PER_READ>
struct Rv64VecHeapBranchU16AdapterCols {
    ExecutionState<T> from_state;

    T rs_ptr[NUM_READS];
    // Low 32 bits of each source pointer register as u16 limbs.
    T rs_val[NUM_READS][RV64_PTR_U16_LIMBS];
    MemoryReadAuxCols<T> rs_read_aux[NUM_READS];

    MemoryReadAuxCols<T> reads_aux[NUM_READS][BLOCKS_PER_READ];
};

template <size_t NUM_READS, size_t BLOCKS_PER_READ>
struct Rv64VecHeapBranchU16AdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    uint32_t rs_ptr[NUM_READS];
    uint32_t rs_vals[NUM_READS];

    MemoryReadAuxRecord rs_read_aux[NUM_READS];
    MemoryReadAuxRecord reads_aux[NUM_READS][BLOCKS_PER_READ];
};

template <size_t NUM_READS, size_t BLOCKS_PER_READ> struct Rv64VecHeapBranchU16Adapter {
    size_t pointer_max_bits;
    VariableRangeChecker range_checker;
    MemoryAuxColsFactory mem_helper;

    __device__ Rv64VecHeapBranchU16Adapter(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : pointer_max_bits(pointer_max_bits), range_checker(range_checker),
          mem_helper(range_checker, timestamp_max_bits) {}

    template <typename T>
    using Cols = Rv64VecHeapBranchU16AdapterCols<T, NUM_READS, BLOCKS_PER_READ>;

    __device__ void fill_trace_row(
        RowSlice row,
        Rv64VecHeapBranchU16AdapterRecord<NUM_READS, BLOCKS_PER_READ> record
    ) {
        // Bound each source pointer to pointer_max_bits by narrowing the high u16 limb.
        const size_t limb_shift_bits = RV64_PTR_BITS - pointer_max_bits;
        for (size_t i = 0; i < NUM_READS; i++) {
            range_checker.add_count(
                (record.rs_vals[i] >> U16_BITS) << limb_shift_bits, U16_BITS
            );
        }

        uint32_t timestamp = record.from_timestamp + NUM_READS + NUM_READS * BLOCKS_PER_READ;

        for (int i = NUM_READS - 1; i >= 0; i--) {
            for (int j = BLOCKS_PER_READ - 1; j >= 0; j--) {
                timestamp--;
                mem_helper.fill(
                    row.slice_from(COL_INDEX(Cols, reads_aux[i][j])),
                    record.reads_aux[i][j].prev_timestamp,
                    timestamp
                );
            }
        }

        for (int i = NUM_READS - 1; i >= 0; i--) {
            timestamp--;
            mem_helper.fill(
                row.slice_from(COL_INDEX(Cols, rs_read_aux[i])),
                record.rs_read_aux[i].prev_timestamp,
                timestamp
            );
        }

        for (int i = NUM_READS - 1; i >= 0; i--) {
            Fp rs_val_packed[RV64_PTR_U16_LIMBS];
            pack_u32_to_u16_limbs(rs_val_packed, record.rs_vals[i]);
            COL_WRITE_ARRAY(row, Cols, rs_val[i], rs_val_packed);
        }

        for (int i = NUM_READS - 1; i >= 0; i--) {
            COL_WRITE_VALUE(row, Cols, rs_ptr[i], record.rs_ptr[i]);
        }

        COL_WRITE_VALUE(row, Cols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Cols, from_state.pc, record.from_pc);
    }
};
