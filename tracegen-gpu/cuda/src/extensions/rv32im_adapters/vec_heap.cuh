#pragma once

#include "execution.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"
#include "trace_access.h"

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
struct Rv32VecHeapAdapterStep {
    size_t pointer_max_bits;
    BitwiseOperationLookup bitwise_lookup;
    MemoryAuxColsFactory mem_helper;

    __device__ Rv32VecHeapAdapterStep(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        BitwiseOperationLookup bitwise_lookup
    )
        : pointer_max_bits(pointer_max_bits), bitwise_lookup(bitwise_lookup),
          mem_helper(range_checker) {}

    __device__ void fill_trace_row(
        RowSlice row,
        Rv32VecHeapAdapterRecord<
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE> record
    ) {
        // Create a type alias that works with COL_INDEX macro
        using ConcreteAdapterCols = Rv32VecHeapAdapterCols<
            uint8_t,
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE>;

        const size_t limb_shift_bits = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - pointer_max_bits;
        const size_t MSL_SHIFT = RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);

        if (NUM_READS > 1) {
            bitwise_lookup.add_range(
                (record.rs_vals[0] >> MSL_SHIFT) << limb_shift_bits,
                (record.rs_vals[1] >> MSL_SHIFT) << limb_shift_bits
            );
            bitwise_lookup.add_range(
                (record.rd_val >> MSL_SHIFT) << limb_shift_bits,
                (record.rd_val >> MSL_SHIFT) << limb_shift_bits
            );
        } else {
            bitwise_lookup.add_range(
                (record.rs_vals[0] >> MSL_SHIFT) << limb_shift_bits,
                (record.rd_val >> MSL_SHIFT) << limb_shift_bits
            );
        }

        size_t timestamp_delta = NUM_READS + 1 + NUM_READS * BLOCKS_PER_READ + BLOCKS_PER_WRITE;
        uint32_t timestamp = record.from_timestamp + timestamp_delta;

        for (int i = BLOCKS_PER_WRITE - 1; i >= 0; i--) {
            timestamp--;
            size_t write_aux_offset = offsetof(ConcreteAdapterCols, writes_aux) +
                                      i * sizeof(MemoryWriteAuxCols<uint8_t, WRITE_SIZE>);
            RowSlice write_aux_row = row.slice_from(write_aux_offset);

            COL_WRITE_ARRAY(
                write_aux_row, MemoryWriteAuxCols, prev_data, record.writes_aux[i].prev_data
            );
            mem_helper.fill(
                write_aux_row.slice_from(COL_INDEX(MemoryWriteAuxCols, base)),
                record.writes_aux[i].prev_timestamp,
                timestamp
            );
        }

        for (int i = NUM_READS - 1; i >= 0; i--) {
            for (int j = BLOCKS_PER_READ - 1; j >= 0; j--) {
                timestamp--;
                size_t read_aux_offset =
                    offsetof(ConcreteAdapterCols, reads_aux) +
                    (i * BLOCKS_PER_READ + j) * sizeof(MemoryReadAuxCols<uint8_t>);
                mem_helper.fill(
                    row.slice_from(read_aux_offset),
                    record.reads_aux[i][j].prev_timestamp,
                    timestamp
                );
            }
        }

        timestamp--;
        mem_helper.fill(
            row.slice_from(offsetof(ConcreteAdapterCols, rd_read_aux)),
            record.rd_read_aux.prev_timestamp,
            timestamp
        );

        for (int i = NUM_READS - 1; i >= 0; i--) {
            timestamp--;
            size_t rs_aux_offset =
                offsetof(ConcreteAdapterCols, rs_read_aux) + i * sizeof(MemoryReadAuxCols<uint8_t>);
            mem_helper.fill(
                row.slice_from(rs_aux_offset), record.rs_read_aux[i].prev_timestamp, timestamp
            );
        }

        uint8_t rd_bytes[RV32_REGISTER_NUM_LIMBS];
        memcpy(rd_bytes, &record.rd_val, sizeof(rd_bytes));
        row.write_array(
            offsetof(ConcreteAdapterCols, rd_val), sizeof(ConcreteAdapterCols::rd_val), rd_bytes
        );

        for (int i = NUM_READS - 1; i >= 0; i--) {
            uint8_t rs_bytes[RV32_REGISTER_NUM_LIMBS];
            memcpy(rs_bytes, &record.rs_vals[i], sizeof(rs_bytes));
            size_t rs_val_offset =
                offsetof(ConcreteAdapterCols, rs_val) + i * RV32_REGISTER_NUM_LIMBS;
            row.write_array(rs_val_offset, RV32_REGISTER_NUM_LIMBS, rs_bytes);
        }

        row.write(offsetof(ConcreteAdapterCols, rd_ptr), record.rd_ptr);

        for (int i = NUM_READS - 1; i >= 0; i--) {
            size_t rs_ptr_offset = offsetof(ConcreteAdapterCols, rs_ptr) + i;
            row.write(rs_ptr_offset, record.rs_ptrs[i]);
        }

        row.write(offsetof(ConcreteAdapterCols, from_state.timestamp), record.from_timestamp);
        row.write(offsetof(ConcreteAdapterCols, from_state.pc), record.from_pc);
    }
};