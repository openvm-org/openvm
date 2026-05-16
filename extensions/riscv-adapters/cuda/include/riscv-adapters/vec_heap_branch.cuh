#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

/// Number of u16 cells holding the low 32 bits of a register pointer.
constexpr size_t REG_PTR_U16S_BRANCH = RV64_WORD_NUM_LIMBS / 2;

template <typename T, size_t NUM_READS, size_t BLOCKS_PER_READ, size_t READ_SIZE>
struct Rv64VecHeapBranchAdapterCols {
    ExecutionState<T> from_state;

    T rs_ptr[NUM_READS];
    /// Low 32 bits of rs registers, packed as 2 u16 cells.
    T rs_val[NUM_READS][REG_PTR_U16S_BRANCH];
    MemoryReadAuxCols<T> rs_read_aux[NUM_READS];

    MemoryReadAuxCols<T> heap_read_aux[NUM_READS][BLOCKS_PER_READ];
};

template <size_t NUM_READS, size_t BLOCKS_PER_READ>
struct Rv64VecHeapBranchAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    uint32_t rs_ptr[NUM_READS];
    uint32_t rs_vals[NUM_READS];

    MemoryReadAuxRecord rs_read_aux[NUM_READS];
    MemoryReadAuxRecord heap_read_aux[NUM_READS][BLOCKS_PER_READ];
};

template <size_t NUM_READS, size_t BLOCKS_PER_READ, size_t READ_SIZE>
struct Rv64VecHeapBranchAdapter {
    size_t pointer_max_bits;
    VariableRangeChecker range_checker;
    MemoryAuxColsFactory mem_helper;

    static constexpr size_t U16_BITS = RV64_CELL_BITS * 2;
    static constexpr size_t RV64_WORD_TOTAL_BITS = U16_BITS * 2;

    __device__ Rv64VecHeapBranchAdapter(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : pointer_max_bits(pointer_max_bits), range_checker(range_checker),
          mem_helper(range_checker, timestamp_max_bits) {}

    template <typename T>
    using Cols = Rv64VecHeapBranchAdapterCols<T, NUM_READS, BLOCKS_PER_READ, READ_SIZE>;

    __device__ void fill_trace_row(
        RowSlice row,
        Rv64VecHeapBranchAdapterRecord<NUM_READS, BLOCKS_PER_READ> record
    ) {
        // Range-check the high u16 of each register pointer.
        const size_t limb_shift_bits = RV64_WORD_TOTAL_BITS - pointer_max_bits;
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
                    row.slice_from(COL_INDEX(Cols, heap_read_aux[i][j])),
                    record.heap_read_aux[i][j].prev_timestamp,
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
            auto bytes = reinterpret_cast<uint8_t *>(&record.rs_vals[i]);
            Fp rs_val_packed[REG_PTR_U16S_BRANCH];
#pragma unroll
            for (size_t k = 0; k < REG_PTR_U16S_BRANCH; k++) {
                rs_val_packed[k] =
                    Fp(uint32_t(bytes[2 * k]) + 256u * uint32_t(bytes[2 * k + 1]));
            }
            COL_WRITE_ARRAY(row, Cols, rs_val[i], rs_val_packed);
        }

        for (int i = NUM_READS - 1; i >= 0; i--) {
            COL_WRITE_VALUE(row, Cols, rs_ptr[i], record.rs_ptr[i]);
        }

        COL_WRITE_VALUE(row, Cols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Cols, from_state.pc, record.from_pc);
    }
};

// Type aliases for the simple case with BLOCKS_PER_READ=1
template <typename T, size_t NUM_READS, size_t READ_SIZE>
using Rv64HeapBranchAdapterCols = Rv64VecHeapBranchAdapterCols<T, NUM_READS, 1, READ_SIZE>;

template <size_t NUM_READS>
using Rv64HeapBranchAdapterRecord = Rv64VecHeapBranchAdapterRecord<NUM_READS, 1>;

template <size_t NUM_READS, size_t READ_SIZE>
using Rv64HeapBranchAdapter = Rv64VecHeapBranchAdapter<NUM_READS, 1, READ_SIZE>;
