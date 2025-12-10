#pragma once

#include "vec_heap.cuh"

using namespace riscv;

// Heap adapter with explicit READ_BLOCKS and WRITE_BLOCKS parameters.
// The adapter reads READ_SIZE bytes total using READ_BLOCKS blocks of (READ_SIZE/READ_BLOCKS) bytes each.
// Similarly for writes.
// For 256-bit (32-byte) operations with 4-byte block size: READ_SIZE=32, READ_BLOCKS=8

// Block size used for memory bus interactions (must match CONST_BLOCK_SIZE in Rust)
constexpr size_t HEAP_ADAPTER_BLOCK_SIZE = 4;

template <typename T, size_t NUM_READS, size_t READ_SIZE, size_t WRITE_SIZE, size_t READ_BLOCKS, size_t WRITE_BLOCKS>
struct Rv32HeapAdapterCols {
    ExecutionState<T> from_state;

    T rs_ptr[NUM_READS];
    T rd_ptr;

    T rs_val[NUM_READS][RV32_REGISTER_NUM_LIMBS];
    T rd_val[RV32_REGISTER_NUM_LIMBS];

    MemoryReadAuxCols<T> rs_read_aux[NUM_READS];
    MemoryReadAuxCols<T> rd_read_aux;

    // READ_BLOCKS reads per pointer (each of HEAP_ADAPTER_BLOCK_SIZE bytes)
    MemoryReadAuxCols<T> reads_aux[NUM_READS][READ_BLOCKS];
    // WRITE_BLOCKS writes (each of HEAP_ADAPTER_BLOCK_SIZE bytes)
    MemoryWriteAuxCols<T, HEAP_ADAPTER_BLOCK_SIZE> writes_aux[WRITE_BLOCKS];
};

template <size_t NUM_READS, size_t READ_SIZE, size_t WRITE_SIZE, size_t READ_BLOCKS, size_t WRITE_BLOCKS>
struct Rv32HeapAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    uint32_t rs_ptrs[NUM_READS];
    uint32_t rd_ptr;

    uint32_t rs_vals[NUM_READS];
    uint32_t rd_val;

    MemoryReadAuxRecord rs_read_aux[NUM_READS];
    MemoryReadAuxRecord rd_read_aux;

    MemoryReadAuxRecord reads_aux[NUM_READS][READ_BLOCKS];
    MemoryWriteAuxRecord<uint8_t, HEAP_ADAPTER_BLOCK_SIZE> writes_aux[WRITE_BLOCKS];
};

template <size_t NUM_READS, size_t READ_SIZE, size_t WRITE_SIZE, size_t READ_BLOCKS, size_t WRITE_BLOCKS>
struct Rv32HeapAdapterExecutor {
    size_t pointer_max_bits;
    BitwiseOperationLookup bitwise_lookup;
    MemoryAuxColsFactory mem_helper;

    static constexpr size_t RV32_REGISTER_TOTAL_BITS = RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS;
    static constexpr size_t MSL_SHIFT = RV32_CELL_BITS * (RV32_REGISTER_NUM_LIMBS - 1);

    __device__ Rv32HeapAdapterExecutor(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        BitwiseOperationLookup bitwise_lookup,
        uint32_t timestamp_max_bits
    )
        : pointer_max_bits(pointer_max_bits), bitwise_lookup(bitwise_lookup),
          mem_helper(range_checker, timestamp_max_bits) {}

    template <typename T>
    using Cols = Rv32HeapAdapterCols<T, NUM_READS, READ_SIZE, WRITE_SIZE, READ_BLOCKS, WRITE_BLOCKS>;

    __device__ void fill_trace_row(
        RowSlice row,
        Rv32HeapAdapterRecord<NUM_READS, READ_SIZE, WRITE_SIZE, READ_BLOCKS, WRITE_BLOCKS> record
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
            record.from_timestamp + NUM_READS + 1 + NUM_READS * READ_BLOCKS + WRITE_BLOCKS;

        for (int i = WRITE_BLOCKS - 1; i >= 0; i--) {
            timestamp--;
            COL_WRITE_ARRAY(row, Cols, writes_aux[i].prev_data, record.writes_aux[i].prev_data);
            mem_helper.fill(
                row.slice_from(COL_INDEX(Cols, writes_aux[i])),
                record.writes_aux[i].prev_timestamp,
                timestamp
            );
        }

        for (int i = NUM_READS - 1; i >= 0; i--) {
            for (int j = READ_BLOCKS - 1; j >= 0; j--) {
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