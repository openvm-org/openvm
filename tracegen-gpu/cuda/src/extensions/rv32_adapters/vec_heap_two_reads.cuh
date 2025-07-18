#pragma once

#include "execution.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"
#include "trace_access.h"

using namespace riscv;

template <
    typename T,
    size_t BLOCKS_PER_READ1,
    size_t BLOCKS_PER_READ2,
    size_t BLOCKS_PER_WRITE,
    size_t READ_SIZE,
    size_t WRITE_SIZE>
struct Rv32VecHeapTwoReadsAdapterCols {
    ExecutionState<T> from_state;

    T rs1_ptr;
    T rs2_ptr;
    T rd_ptr;

    T rs1_val[RV32_REGISTER_NUM_LIMBS];
    T rs2_val[RV32_REGISTER_NUM_LIMBS];
    T rd_val[RV32_REGISTER_NUM_LIMBS];

    MemoryReadAuxCols<T> rs1_read_aux;
    MemoryReadAuxCols<T> rs2_read_aux;
    MemoryReadAuxCols<T> rd_read_aux;

    MemoryReadAuxCols<T> reads1_aux[BLOCKS_PER_READ1];
    MemoryReadAuxCols<T> reads2_aux[BLOCKS_PER_READ2];
    MemoryWriteAuxCols<T, WRITE_SIZE> writes_aux[BLOCKS_PER_WRITE];
};

template <
    size_t BLOCKS_PER_READ1,
    size_t BLOCKS_PER_READ2,
    size_t BLOCKS_PER_WRITE,
    size_t WRITE_SIZE>
struct Rv32VecHeapTwoReadsAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    uint32_t rs1_ptr;
    uint32_t rs2_ptr;
    uint32_t rd_ptr;

    uint32_t rs1_val;
    uint32_t rs2_val;
    uint32_t rd_val;

    MemoryReadAuxRecord rs1_read_aux;
    MemoryReadAuxRecord rs2_read_aux;
    MemoryReadAuxRecord rd_read_aux;

    MemoryReadAuxRecord reads1_aux[BLOCKS_PER_READ1];
    MemoryReadAuxRecord reads2_aux[BLOCKS_PER_READ2];
    MemoryWriteAuxRecord<uint8_t, WRITE_SIZE> writes_aux[BLOCKS_PER_WRITE];
};

template <
    size_t BLOCKS_PER_READ1,
    size_t BLOCKS_PER_READ2,
    size_t BLOCKS_PER_WRITE,
    size_t READ_SIZE,
    size_t WRITE_SIZE>
struct Rv32VecHeapTwoReadsAdapterStep {
    size_t pointer_max_bits;
    BitwiseOperationLookup bitwise_lookup;
    MemoryAuxColsFactory mem_helper;

    static constexpr size_t RV32_REGISTER_TOTAL_BITS = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS;
    static constexpr size_t MSL_SHIFT = RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);

    template <typename T>
    using Cols = Rv32VecHeapTwoReadsAdapterCols<
        T,
        BLOCKS_PER_READ1,
        BLOCKS_PER_READ2,
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE>;

    __device__ Rv32VecHeapTwoReadsAdapterStep(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        BitwiseOperationLookup bitwise_lookup
    )
        : pointer_max_bits(pointer_max_bits), bitwise_lookup(bitwise_lookup),
          mem_helper(range_checker) {}

    __device__ void fill_trace_row(
        RowSlice row,
        Rv32VecHeapTwoReadsAdapterRecord<
            BLOCKS_PER_READ1,
            BLOCKS_PER_READ2,
            BLOCKS_PER_WRITE,
            WRITE_SIZE> record
    ) {
        const size_t limb_shift_bits = RV32_REGISTER_TOTAL_BITS - pointer_max_bits;

        bitwise_lookup.add_range(
            (record.rs1_val >> MSL_SHIFT) << limb_shift_bits,
            (record.rs2_val >> MSL_SHIFT) << limb_shift_bits
        );
        bitwise_lookup.add_range(
            (record.rd_val >> MSL_SHIFT) << limb_shift_bits,
            (record.rd_val >> MSL_SHIFT) << limb_shift_bits
        );

        uint32_t timestamp =
            record.from_timestamp + 2 + BLOCKS_PER_READ1 + BLOCKS_PER_READ2 + BLOCKS_PER_WRITE;

        for (int i = BLOCKS_PER_WRITE - 1; i >= 0; i--) {
            timestamp--;
            RowSlice write_aux_row = row.slice_from(COL_INDEX(Cols, writes_aux[i]));

            COL_WRITE_ARRAY(
                write_aux_row, MemoryWriteAuxCols, prev_data, record.writes_aux[i].prev_data
            );
            mem_helper.fill(
                write_aux_row.slice_from(COL_INDEX(MemoryWriteAuxCols, base)),
                record.writes_aux[i].prev_timestamp,
                timestamp
            );
        }

        for (int i = BLOCKS_PER_READ2 - 1; i >= 0; i--) {
            timestamp--;
            mem_helper.fill(
                row.slice_from(COL_INDEX(Cols, reads2_aux[i])),
                record.reads2_aux[i].prev_timestamp,
                timestamp
            );
        }

        for (int i = BLOCKS_PER_READ1 - 1; i >= 0; i--) {
            timestamp--;
            mem_helper.fill(
                row.slice_from(COL_INDEX(Cols, reads1_aux[i])),
                record.reads1_aux[i].prev_timestamp,
                timestamp
            );
        }

        timestamp--;
        mem_helper.fill(
            row.slice_from(COL_INDEX(Cols, rd_read_aux)),
            record.rd_read_aux.prev_timestamp,
            timestamp
        );

        timestamp--;
        mem_helper.fill(
            row.slice_from(COL_INDEX(Cols, rs2_read_aux)),
            record.rs2_read_aux.prev_timestamp,
            timestamp
        );

        timestamp--;
        mem_helper.fill(
            row.slice_from(COL_INDEX(Cols, rs1_read_aux)),
            record.rs1_read_aux.prev_timestamp,
            timestamp
        );

        COL_WRITE_ARRAY(row, Cols, rd_val, (uint8_t *)&record.rd_val);
        COL_WRITE_ARRAY(row, Cols, rs2_val, (uint8_t *)&record.rs2_val);
        COL_WRITE_ARRAY(row, Cols, rs1_val, (uint8_t *)&record.rs1_val);

        COL_WRITE_VALUE(row, Cols, rd_ptr, record.rd_ptr);
        COL_WRITE_VALUE(row, Cols, rs2_ptr, record.rs2_ptr);
        COL_WRITE_VALUE(row, Cols, rs1_ptr, record.rs1_ptr);

        COL_WRITE_VALUE(row, Cols, from_state.timestamp, timestamp);
        COL_WRITE_VALUE(row, Cols, from_state.pc, record.from_pc);
    }
};
