#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <
    typename T,
    size_t NUM_READS,
    size_t BLOCKS_PER_READ,
    size_t BLOCKS_PER_WRITE,
    size_t READ_SIZE,
    size_t WRITE_SIZE>
struct Rv32VecHeapAdapterCols {
    ExecutionState<T> from_state;

    T rs_ptr[NUM_READS];
    T rd_ptr;

    T rs_val[NUM_READS][RV32_REGISTER_NUM_LIMBS];
    T rd_val[RV32_REGISTER_NUM_LIMBS];

    MemoryReadAuxCols<T> rs_read_aux[NUM_READS];
    MemoryReadAuxCols<T> rd_read_aux;

    MemoryReadAuxCols<T> reads_aux[NUM_READS][BLOCKS_PER_READ];
    MemoryWriteAuxCols<T, WRITE_SIZE> writes_aux[BLOCKS_PER_WRITE];
};

template <
    size_t NUM_READS,
    size_t BLOCKS_PER_READ,
    size_t BLOCKS_PER_WRITE,
    size_t READ_SIZE,
    size_t WRITE_SIZE>
struct Rv32VecHeapAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    uint32_t rs_ptrs[NUM_READS];
    uint32_t rd_ptr;

    uint32_t rs_vals[NUM_READS];
    uint32_t rd_val;

    MemoryReadAuxRecord rs_read_aux[NUM_READS];
    MemoryReadAuxRecord rd_read_aux;

    MemoryReadAuxRecord reads_aux[NUM_READS][BLOCKS_PER_READ];
    MemoryWriteAuxRecord<uint8_t, WRITE_SIZE>
        writes_aux[BLOCKS_PER_WRITE]; // MemoryWriteBytesAuxRecord
};

template <
    size_t NUM_READS,
    size_t BLOCKS_PER_READ,
    size_t BLOCKS_PER_WRITE,
    size_t READ_SIZE,
    size_t WRITE_SIZE>
struct Rv32VecHeapAdapter {
    size_t pointer_max_bits;
    BitwiseOperationLookup bitwise_lookup;
    MemoryAuxColsFactory mem_helper;

    static constexpr size_t RV32_REGISTER_TOTAL_BITS = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS;
    static constexpr size_t MSL_SHIFT = RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);

    __device__ Rv32VecHeapAdapter(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        BitwiseOperationLookup bitwise_lookup,
        uint32_t timestamp_max_bits
    )
        : pointer_max_bits(pointer_max_bits), bitwise_lookup(bitwise_lookup),
          mem_helper(range_checker, timestamp_max_bits) {}

    template <typename T>
    using Cols = Rv32VecHeapAdapterCols<
        T,
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE>;

    __device__ void fill_trace_row(
        RowSlice row,
        Rv32VecHeapAdapterRecord<
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE> record
    ) {
        const size_t limb_shift_bits = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - pointer_max_bits;

        if (NUM_READS == 2) {
            bitwise_lookup.add_range(
                (record.rs_vals[0] >> MSL_SHIFT) << limb_shift_bits,
                (record.rs_vals[1] >> MSL_SHIFT) << limb_shift_bits
            );
            bitwise_lookup.add_range(
                (record.rd_val >> MSL_SHIFT) << limb_shift_bits,
                (record.rd_val >> MSL_SHIFT) << limb_shift_bits
            );
        } else if (NUM_READS == 1) {
            bitwise_lookup.add_range(
                (record.rs_vals[0] >> MSL_SHIFT) << limb_shift_bits,
                (record.rd_val >> MSL_SHIFT) << limb_shift_bits
            );
        } else {
            assert(false);
        }

        uint32_t timestamp =
            record.from_timestamp + NUM_READS + 1 + NUM_READS * BLOCKS_PER_READ + BLOCKS_PER_WRITE;

        for (int i = BLOCKS_PER_WRITE - 1; i >= 0; i--) {
            timestamp--;
            COL_WRITE_ARRAY(row, Cols, writes_aux[i].prev_data, record.writes_aux[i].prev_data);
            mem_helper.fill(
                row.slice_from(COL_INDEX(Cols, writes_aux[i])),
                record.writes_aux[i].prev_timestamp,
                timestamp
            );
        }

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

        timestamp--;
        mem_helper.fill(
            row.slice_from(COL_INDEX(Cols, rd_read_aux)),
            record.rd_read_aux.prev_timestamp,
            timestamp
        );

        for (int i = NUM_READS - 1; i >= 0; i--) {
            timestamp--;
            mem_helper.fill(
                row.slice_from(COL_INDEX(Cols, rs_read_aux[i])),
                record.rs_read_aux[i].prev_timestamp,
                timestamp
            );
        }

        COL_WRITE_ARRAY(row, Cols, rd_val, (uint8_t *)&record.rd_val);

        for (int i = NUM_READS - 1; i >= 0; i--) {
            COL_WRITE_ARRAY(row, Cols, rs_val[i], (uint8_t *)&record.rs_vals[i]);
        }

        COL_WRITE_VALUE(row, Cols, rd_ptr, record.rd_ptr);

        for (int i = NUM_READS - 1; i >= 0; i--) {
            COL_WRITE_VALUE(row, Cols, rs_ptr[i], record.rs_ptrs[i]);
        }

        COL_WRITE_VALUE(row, Cols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Cols, from_state.pc, record.from_pc);
    }
};
