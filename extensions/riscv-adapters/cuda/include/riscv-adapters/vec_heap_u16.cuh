#pragma once

#include "primitives/execution.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/constants.h"
#include "primitives/utils.cuh"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <
    typename T,
    size_t NUM_READS,
    size_t BLOCKS_PER_READ,
    size_t BLOCKS_PER_WRITE>
struct Rv64VecHeapU16AdapterCols {
    ExecutionState<T> from_state;

    T rs_ptr[NUM_READS];
    T rd_ptr;

    // Low 32 bits of rs registers as u16 limbs.
    T rs_val[NUM_READS][RV64_PTR_U16_LIMBS];
    // Low 32 bits of rd register as u16 limbs.
    T rd_val[RV64_PTR_U16_LIMBS];

    MemoryReadAuxCols<T> rs_read_aux[NUM_READS];
    MemoryReadAuxCols<T> rd_read_aux;

    MemoryReadAuxCols<T> reads_aux[NUM_READS][BLOCKS_PER_READ];
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> writes_aux[BLOCKS_PER_WRITE];
};

template <
    size_t NUM_READS,
    size_t BLOCKS_PER_READ,
    size_t BLOCKS_PER_WRITE>
struct Rv64VecHeapU16AdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    uint32_t rs_ptrs[NUM_READS];
    uint32_t rd_ptr;

    uint32_t rs_vals[NUM_READS];
    uint32_t rd_val;

    MemoryReadAuxRecord rs_read_aux[NUM_READS];
    MemoryReadAuxRecord rd_read_aux;

    MemoryReadAuxRecord reads_aux[NUM_READS][BLOCKS_PER_READ];
    MemoryWriteU16AuxRecord<BLOCK_FE_WIDTH> writes_aux[BLOCKS_PER_WRITE];
};

template <
    size_t NUM_READS,
    size_t BLOCKS_PER_READ,
    size_t BLOCKS_PER_WRITE>
struct Rv64VecHeapU16Adapter {
    size_t pointer_max_bits;
    VariableRangeChecker range_checker;
    MemoryAuxColsFactory mem_helper;

    __device__ Rv64VecHeapU16Adapter(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : pointer_max_bits(pointer_max_bits), range_checker(range_checker),
          mem_helper(range_checker, timestamp_max_bits) {}

    template <typename T>
    using Cols = Rv64VecHeapU16AdapterCols<
        T,
        NUM_READS,
        BLOCKS_PER_READ,
        BLOCKS_PER_WRITE>;

    __device__ void fill_trace_row(
        RowSlice row,
        Rv64VecHeapU16AdapterRecord<
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE> record
    ) {
        // Bound each register pointer to pointer_max_bits by narrowing the high u16 limb.
        const size_t limb_shift_bits = RV64_PTR_BITS - pointer_max_bits;
        for (size_t i = 0; i < NUM_READS; i++) {
            range_checker.add_count(
                (record.rs_vals[i] >> U16_BITS) << limb_shift_bits, U16_BITS
            );
        }
        range_checker.add_count(
            (record.rd_val >> U16_BITS) << limb_shift_bits, U16_BITS
        );

        uint32_t timestamp =
            record.from_timestamp + NUM_READS + 1 + NUM_READS * BLOCKS_PER_READ + BLOCKS_PER_WRITE;

        for (int i = BLOCKS_PER_WRITE - 1; i >= 0; i--) {
            timestamp--;
            Fp prev[BLOCK_FE_WIDTH];
            copy_u16_cells(prev, record.writes_aux[i].prev_data);
            COL_WRITE_ARRAY(row, Cols, writes_aux[i].prev_data, prev);
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

        Fp rd_val_packed[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(rd_val_packed, record.rd_val);
        COL_WRITE_ARRAY(row, Cols, rd_val, rd_val_packed);

        for (int i = NUM_READS - 1; i >= 0; i--) {
            Fp rs_val_packed[RV64_PTR_U16_LIMBS];
            ptr_to_u16_limbs(rs_val_packed, record.rs_vals[i]);
            COL_WRITE_ARRAY(row, Cols, rs_val[i], rs_val_packed);
        }

        COL_WRITE_VALUE(row, Cols, rd_ptr, record.rd_ptr);

        for (int i = NUM_READS - 1; i >= 0; i--) {
            COL_WRITE_VALUE(row, Cols, rs_ptr[i], record.rs_ptrs[i]);
        }

        COL_WRITE_VALUE(row, Cols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Cols, from_state.pc, record.from_pc);
    }
};
