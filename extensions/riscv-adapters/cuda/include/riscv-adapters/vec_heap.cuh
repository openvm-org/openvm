#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "riscv-adapters/pointer_conv.cuh"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <
    typename T,
    size_t NUM_READS,
    size_t BLOCKS_PER_READ,
    size_t BLOCKS_PER_WRITE>
struct Rv64VecHeapAdapterCols {
    ExecutionState<T> from_state;

    T rs_ptr[NUM_READS];
    T rd_ptr;

    T rs_val[NUM_READS][RV64_PTR_U16_LIMBS];
    T rd_val[RV64_PTR_U16_LIMBS];

    // Carry for converting each base byte pointer to AS-native u16 *cell* pointer limbs.
    T rs_cell_carry[NUM_READS];
    T rd_cell_carry;
    // Per-block carry for adding the cell offset `j * (MEMORY_BLOCK_BYTES / U16_CELL_SIZE)` to each
    // base cell pointer (block `j`'s carry into the high cell limb).
    T reads_add_carry[NUM_READS][BLOCKS_PER_READ];
    T writes_add_carry[BLOCKS_PER_WRITE];

    MemoryReadAuxCols<T> rs_read_aux[NUM_READS];
    MemoryReadAuxCols<T> rd_read_aux;

    MemoryReadAuxCols<T> reads_aux[NUM_READS][BLOCKS_PER_READ];
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> writes_aux[BLOCKS_PER_WRITE];
};

template <
    size_t NUM_READS,
    size_t BLOCKS_PER_READ,
    size_t BLOCKS_PER_WRITE>
struct Rv64VecHeapAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    uint32_t rs_ptrs[NUM_READS];
    uint32_t rd_ptr;

    uint32_t rs_vals[NUM_READS];
    uint32_t rd_val;

    MemoryReadAuxRecord rs_read_aux[NUM_READS];
    MemoryReadAuxRecord rd_read_aux;

    MemoryReadAuxRecord reads_aux[NUM_READS][BLOCKS_PER_READ];
    MemoryWriteBytesAuxRecord<MEMORY_BLOCK_BYTES> writes_aux[BLOCKS_PER_WRITE];
};

template <
    size_t NUM_READS,
    size_t BLOCKS_PER_READ,
    size_t BLOCKS_PER_WRITE>
struct Rv64VecHeapAdapter {
    size_t pointer_max_bits;
    VariableRangeChecker range_checker;
    MemoryAuxColsFactory mem_helper;

    __device__ Rv64VecHeapAdapter(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : pointer_max_bits(pointer_max_bits), range_checker(range_checker),
          mem_helper(range_checker, timestamp_max_bits) {}

    template <typename T>
    using Cols = Rv64VecHeapAdapterCols<
        T,
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE>;

    __device__ void fill_trace_row(
        RowSlice row,
        Rv64VecHeapAdapterRecord<
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE> record
    ) {
        static_assert(NUM_READS == 1 || NUM_READS == 2);

        // Byte -> cell pointer conversion carries and per-block cell-offset carries, plus matching
        // range-check counts. Mirrors the host filler in vec_heap.rs.
        uint32_t hi_bits = cell_ptr_hi_bits(pointer_max_bits);
        const uint32_t cell_stride = MEMORY_BLOCK_BYTES / U16_CELL_SIZE;

#pragma unroll
        for (size_t i = 0; i < NUM_READS; i++) {
            CellPtr conv = byte_ptr_limbs_to_cell_ptr_limbs_value(
                uint16_t(record.rs_vals[i]), uint16_t(record.rs_vals[i] >> U16_BITS)
            );
            range_checker.add_count(conv.limbs[1], hi_bits);
            COL_WRITE_VALUE(row, Cols, rs_cell_carry[i], conv.carry);
#pragma unroll
            for (size_t j = 0; j < BLOCKS_PER_READ; j++) {
                CellPtr add =
                    add_const_u16_limbs_value(conv.limbs[0], conv.limbs[1], j * cell_stride);
                range_checker.add_count(add.limbs[0], U16_BITS);
                COL_WRITE_VALUE(row, Cols, reads_add_carry[i][j], add.carry);
            }
        }
        {
            CellPtr conv = byte_ptr_limbs_to_cell_ptr_limbs_value(
                uint16_t(record.rd_val), uint16_t(record.rd_val >> U16_BITS)
            );
            range_checker.add_count(conv.limbs[1], hi_bits);
            COL_WRITE_VALUE(row, Cols, rd_cell_carry, conv.carry);
#pragma unroll
            for (size_t j = 0; j < BLOCKS_PER_WRITE; j++) {
                CellPtr add =
                    add_const_u16_limbs_value(conv.limbs[0], conv.limbs[1], j * cell_stride);
                range_checker.add_count(add.limbs[0], U16_BITS);
                COL_WRITE_VALUE(row, Cols, writes_add_carry[j], add.carry);
            }
        }

        uint32_t timestamp =
            record.from_timestamp + NUM_READS + 1 + NUM_READS * BLOCKS_PER_READ + BLOCKS_PER_WRITE;

        for (int i = BLOCKS_PER_WRITE - 1; i >= 0; i--) {
            timestamp--;
            Fp packed_prev[BLOCK_FE_WIDTH];
            pack_u8_block_bytes(packed_prev, record.writes_aux[i].prev_data);
            COL_WRITE_ARRAY(row, Cols, writes_aux[i].prev_data, packed_prev);
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

        Fp rd_val[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(rd_val, record.rd_val);
        COL_WRITE_ARRAY(row, Cols, rd_val, rd_val);

        for (int i = NUM_READS - 1; i >= 0; i--) {
            Fp rs_val[RV64_PTR_U16_LIMBS];
            ptr_to_u16_limbs(rs_val, record.rs_vals[i]);
            COL_WRITE_ARRAY(row, Cols, rs_val[i], rs_val);
        }

        COL_WRITE_VALUE(row, Cols, rd_ptr, record.rd_ptr);

        for (int i = NUM_READS - 1; i >= 0; i--) {
            COL_WRITE_VALUE(row, Cols, rs_ptr[i], record.rs_ptrs[i]);
        }

        COL_WRITE_VALUE(row, Cols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Cols, from_state.pc, record.from_pc);
    }
};

// Type aliases for the simple case with BLOCKS_PER_READ=1, BLOCKS_PER_WRITE=1
template <typename T, size_t NUM_READS>
using Rv64HeapAdapterCols = Rv64VecHeapAdapterCols<T, NUM_READS, 1, 1>;

template <size_t NUM_READS>
using Rv64HeapAdapterRecord = Rv64VecHeapAdapterRecord<NUM_READS, 1, 1>;

template <size_t NUM_READS>
using Rv64HeapAdapterExecutor = Rv64VecHeapAdapter<NUM_READS, 1, 1>;
