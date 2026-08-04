#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "riscv-adapters/pointer_conv.cuh"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <typename T, size_t NUM_READS, size_t BLOCKS_PER_READ>
struct Rv64VecHeapBranchAdapterCols {
    ExecutionState<T> from_state;

    T rs_ptr[NUM_READS];
    T rs_val[NUM_READS][RV64_PTR_U16_LIMBS];

    // Carry for converting each base byte pointer to AS-native u16 *cell* pointer limbs.
    T rs_cell_carry[NUM_READS];
    // Per-block carry for adding the cell offset `j * (MEMORY_BLOCK_BYTES / U16_CELL_SIZE)` to each
    // base cell pointer (block `j`'s carry into the high cell limb).
    T reads_add_carry[NUM_READS][BLOCKS_PER_READ];

    MemoryReadAuxCols<T> rs_read_aux[NUM_READS];

    MemoryReadAuxCols<T> reads_aux[NUM_READS][BLOCKS_PER_READ];
};

template <size_t NUM_READS, size_t BLOCKS_PER_READ>
struct Rv64VecHeapBranchAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    uint32_t rs_ptr[NUM_READS];
    uint32_t rs_vals[NUM_READS];

    MemoryReadAuxRecord rs_read_aux[NUM_READS];
    MemoryReadAuxRecord reads_aux[NUM_READS][BLOCKS_PER_READ];
};

template <size_t NUM_READS, size_t BLOCKS_PER_READ>
struct Rv64VecHeapBranchAdapter {
    size_t pointer_max_bits;
    VariableRangeChecker range_checker;
    MemoryAuxColsFactory mem_helper;

    __device__ Rv64VecHeapBranchAdapter(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : pointer_max_bits(pointer_max_bits), range_checker(range_checker),
          mem_helper(range_checker, timestamp_max_bits) {}

    template <typename T>
    using Cols = Rv64VecHeapBranchAdapterCols<T, NUM_READS, BLOCKS_PER_READ>;

    __device__ void fill_trace_row(
        RowSlice row,
        Rv64VecHeapBranchAdapterRecord<NUM_READS, BLOCKS_PER_READ> record
    ) {
        static_assert(NUM_READS == 1 || NUM_READS == 2);

        // Byte -> cell pointer conversion carries and per-block cell-offset carries, plus matching
        // range-check counts. Mirrors the host filler in vec_heap_branch.rs.
        const uint32_t cell_stride = MEMORY_BLOCK_BYTES / U16_CELL_SIZE;

#pragma unroll
        for (size_t i = 0; i < NUM_READS; i++) {
            uint32_t add_carries[BLOCKS_PER_READ];
            uint32_t conv_carry = compute_pointer_carries(
                range_checker,
                record.rs_vals[i],
                pointer_max_bits,
                BLOCKS_PER_READ,
                cell_stride,
                add_carries
            );
            COL_WRITE_VALUE(row, Cols, rs_cell_carry[i], conv_carry);
#pragma unroll
            for (size_t j = 0; j < BLOCKS_PER_READ; j++) {
                COL_WRITE_VALUE(row, Cols, reads_add_carry[i][j], add_carries[j]);
            }
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
            Fp rs_val[RV64_PTR_U16_LIMBS];
            ptr_to_u16_limbs(rs_val, record.rs_vals[i]);
            COL_WRITE_ARRAY(row, Cols, rs_val[i], rs_val);
        }

        for (int i = NUM_READS - 1; i >= 0; i--) {
            COL_WRITE_VALUE(row, Cols, rs_ptr[i], record.rs_ptr[i]);
        }

        COL_WRITE_VALUE(row, Cols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Cols, from_state.pc, record.from_pc);
    }
};

// Type aliases for the simple case with BLOCKS_PER_READ=1
template <typename T, size_t NUM_READS>
using Rv64HeapBranchAdapterCols = Rv64VecHeapBranchAdapterCols<T, NUM_READS, 1>;

template <size_t NUM_READS>
using Rv64HeapBranchAdapterRecord = Rv64VecHeapBranchAdapterRecord<NUM_READS, 1>;

template <size_t NUM_READS>
using Rv64HeapBranchAdapter = Rv64VecHeapBranchAdapter<NUM_READS, 1>;
