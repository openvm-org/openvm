#pragma once

#include "primitives/execution.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

/// Number of u16 cells holding the low 32 bits of a register pointer.
constexpr size_t REG_PTR_U16S_U16 = RV64_WORD_NUM_LIMBS / 2;

template <
    typename T,
    size_t NUM_READS,
    size_t BLOCKS_PER_READ,
    size_t BLOCKS_PER_WRITE,
    size_t READ_SIZE,
    size_t WRITE_SIZE>
struct Rv64VecHeapU16AdapterCols {
    ExecutionState<T> from_state;

    T rs_ptr[NUM_READS];
    T rd_ptr;

    /// Low 32 bits of rs registers, packed as 2 u16 cells.
    T rs_val[NUM_READS][REG_PTR_U16S_U16];
    /// Low 32 bits of rd register, packed as 2 u16 cells.
    T rd_val[REG_PTR_U16S_U16];

    MemoryReadAuxCols<T> rs_read_aux[NUM_READS];
    MemoryReadAuxCols<T> rd_read_aux;

    MemoryReadAuxCols<T> reads_aux[NUM_READS][BLOCKS_PER_READ];
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> writes_aux[BLOCKS_PER_WRITE];
};

template <size_t WRITE_SIZE> struct Rv64VecHeapU16WriteAuxRecord {
    uint32_t prev_timestamp;
    uint16_t prev_data[WRITE_SIZE];
};

template <
    size_t NUM_READS,
    size_t BLOCKS_PER_READ,
    size_t BLOCKS_PER_WRITE,
    size_t READ_SIZE,
    size_t WRITE_SIZE>
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
    Rv64VecHeapU16WriteAuxRecord<WRITE_SIZE> writes_aux[BLOCKS_PER_WRITE];
};

template <
    size_t NUM_READS,
    size_t BLOCKS_PER_READ,
    size_t BLOCKS_PER_WRITE,
    size_t READ_SIZE,
    size_t WRITE_SIZE>
struct Rv64VecHeapU16Adapter {
    size_t pointer_max_bits;
    VariableRangeChecker range_checker;
    MemoryAuxColsFactory mem_helper;

    static constexpr size_t U16_BITS = RV64_CELL_BITS * 2;
    static constexpr size_t RV64_WORD_TOTAL_BITS = U16_BITS * 2;

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
        BLOCKS_PER_WRITE,
        READ_SIZE,
        WRITE_SIZE>;

    __device__ void fill_trace_row(
        RowSlice row,
        Rv64VecHeapU16AdapterRecord<
            NUM_READS,
            BLOCKS_PER_READ,
            BLOCKS_PER_WRITE,
            READ_SIZE,
            WRITE_SIZE> record
    ) {
        // Range-check the high u16 of each register pointer: `(reg >> 16) << shift_bits < 2^16`.
        const size_t limb_shift_bits = RV64_WORD_TOTAL_BITS - pointer_max_bits;
        for (size_t i = 0; i < NUM_READS; i++) {
            range_checker.add_count(
                (record.rs_vals[i] >> U16_BITS) << limb_shift_bits, U16_BITS
            );
        }
        range_checker.add_count((record.rd_val >> U16_BITS) << limb_shift_bits, U16_BITS);

        uint32_t timestamp =
            record.from_timestamp + NUM_READS + 1 + NUM_READS * BLOCKS_PER_READ + BLOCKS_PER_WRITE;

        for (int i = BLOCKS_PER_WRITE - 1; i >= 0; i--) {
            timestamp--;
            // u16 path: prev_data is already u16-cell-shaped; no byte→u16 packing.
            // WRITE_SIZE == BLOCK_FE_WIDTH enforced by the AIR.
            Fp prev[BLOCK_FE_WIDTH];
#pragma unroll
            for (size_t k = 0; k < BLOCK_FE_WIDTH; k++) {
                prev[k] = Fp(uint32_t(record.writes_aux[i].prev_data[k]));
            }
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

        auto pack_u16s = [](uint32_t v, Fp out[REG_PTR_U16S_U16]) {
            auto bytes = reinterpret_cast<uint8_t *>(&v);
#pragma unroll
            for (size_t k = 0; k < REG_PTR_U16S_U16; k++) {
                out[k] = Fp(uint32_t(bytes[2 * k]) + 256u * uint32_t(bytes[2 * k + 1]));
            }
        };
        Fp rd_val_packed[REG_PTR_U16S_U16];
        pack_u16s(record.rd_val, rd_val_packed);
        COL_WRITE_ARRAY(row, Cols, rd_val, rd_val_packed);

        for (int i = NUM_READS - 1; i >= 0; i--) {
            Fp rs_val_packed[REG_PTR_U16S_U16];
            pack_u16s(record.rs_vals[i], rs_val_packed);
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
