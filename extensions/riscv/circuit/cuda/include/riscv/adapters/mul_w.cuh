#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <typename T> struct Rv64MultWAdapterCols {
    ExecutionState<T> from_state;
    T rd_ptr;
    T rs1_ptr;
    T rs2_ptr;
    /// Upper 4 bytes of rs1 register read (kept in adapter to satisfy full-width memory read).
    T rs1_high[RV64_REGISTER_NUM_LIMBS - RV64_WORD_NUM_LIMBS];
    /// Upper 4 bytes of rs2 register read.
    T rs2_high[RV64_REGISTER_NUM_LIMBS - RV64_WORD_NUM_LIMBS];
    /// Sign bit of the low-word core result used to build full-width sign-extended writes.
    T result_sign;
    MemoryReadAuxCols<T> reads_aux[2];
    MemoryWriteAuxCols<T, RV64_REGISTER_NUM_LIMBS> writes_aux;
};

struct Rv64MultWAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    uint32_t rd_ptr;
    uint32_t rs1_ptr;
    uint32_t rs2_ptr;
    uint8_t rs1_high[RV64_REGISTER_NUM_LIMBS - RV64_WORD_NUM_LIMBS];
    uint8_t rs2_high[RV64_REGISTER_NUM_LIMBS - RV64_WORD_NUM_LIMBS];
    uint8_t result_sign;
    uint8_t result_word_msl;

    MemoryReadAuxRecord reads_aux[2];
    MemoryWriteBytesAuxRecord<RV64_REGISTER_NUM_LIMBS> writes_aux;
};

struct Rv64MultWAdapter {
    MemoryAuxColsFactory mem_helper;
    BitwiseOperationLookup bitwise_lookup;

    __device__ Rv64MultWAdapter(
        VariableRangeChecker range_checker,
        BitwiseOperationLookup lookup,
        uint32_t timestamp_max_bits
    )
        : mem_helper(range_checker, timestamp_max_bits), bitwise_lookup(lookup) {}

    __device__ void fill_trace_row(RowSlice row, Rv64MultWAdapterRecord record) {
        COL_WRITE_ARRAY(
            row, Rv64MultWAdapterCols, writes_aux.prev_data, record.writes_aux.prev_data
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64MultWAdapterCols, writes_aux.base)),
            record.writes_aux.prev_timestamp,
            record.from_timestamp + 2
        );

#pragma unroll
        for (int i = 0; i < 2; i++) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Rv64MultWAdapterCols, reads_aux[i])),
                record.reads_aux[i].prev_timestamp,
                record.from_timestamp + i
            );
        }

        bitwise_lookup.add_xor(
            static_cast<uint32_t>(record.result_word_msl), 1u << (RV64_CELL_BITS - 1)
        );

        COL_WRITE_VALUE(row, Rv64MultWAdapterCols, result_sign, record.result_sign);
        COL_WRITE_ARRAY(row, Rv64MultWAdapterCols, rs2_high, record.rs2_high);
        COL_WRITE_ARRAY(row, Rv64MultWAdapterCols, rs1_high, record.rs1_high);
        COL_WRITE_VALUE(row, Rv64MultWAdapterCols, rs2_ptr, record.rs2_ptr);
        COL_WRITE_VALUE(row, Rv64MultWAdapterCols, rs1_ptr, record.rs1_ptr);
        COL_WRITE_VALUE(row, Rv64MultWAdapterCols, rd_ptr, record.rd_ptr);
        COL_WRITE_VALUE(row, Rv64MultWAdapterCols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Rv64MultWAdapterCols, from_state.pc, record.from_pc);
    }
};
