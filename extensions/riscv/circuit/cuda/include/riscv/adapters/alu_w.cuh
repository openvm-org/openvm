#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <typename T> struct Rv64BaseAluWAdapterCols {
    ExecutionState<T> from_state;
    T rd_ptr;
    T rs1_ptr;
    /// Upper 4 bytes of rs1 register read (kept in adapter to satisfy full-width memory read).
    T rs1_high[RV64_REGISTER_NUM_LIMBS - RV64_WORD_NUM_LIMBS];
    /// Pointer if rs2 was a read, immediate value otherwise
    T rs2;
    /// 1 if rs2 was a read, 0 if an immediate
    T rs2_as;
    /// Upper 4 bytes of rs2 register read (unused when rs2 is immediate).
    T rs2_high[RV64_REGISTER_NUM_LIMBS - RV64_WORD_NUM_LIMBS];
    /// Sign bit of the low-word core result used to build full-width sign-extended writes.
    T result_sign;
    MemoryReadAuxCols<T> reads_aux[2];
    MemoryWriteAuxCols<T, RV64_REGISTER_NUM_LIMBS> writes_aux;
};

struct Rv64BaseAluWAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    uint32_t rd_ptr;
    uint32_t rs1_ptr;
    uint8_t rs1_high[RV64_REGISTER_NUM_LIMBS - RV64_WORD_NUM_LIMBS];
    /// Pointer if rs2 was a read, immediate value otherwise
    uint32_t rs2;
    /// 1 if rs2 was a read, 0 if an immediate
    uint8_t rs2_as;
    uint8_t rs2_high[RV64_REGISTER_NUM_LIMBS - RV64_WORD_NUM_LIMBS];
    uint8_t result_sign;
    uint8_t result_word_msl;

    MemoryReadAuxRecord reads_aux[2];
    MemoryWriteBytesAuxRecord<RV64_REGISTER_NUM_LIMBS> writes_aux;
};

struct Rv64BaseAluWAdapter {
    MemoryAuxColsFactory mem_helper;
    BitwiseOperationLookup bitwise_lookup;

    __device__ Rv64BaseAluWAdapter(
        VariableRangeChecker range_checker,
        BitwiseOperationLookup lookup,
        uint32_t timestamp_max_bits
    )
        : mem_helper(range_checker, timestamp_max_bits), bitwise_lookup(lookup) {}

    __device__ void fill_trace_row(RowSlice row, Rv64BaseAluWAdapterRecord record) {
        // writes_aux at from_timestamp + 2 (full-width 8-limb register write).
        COL_WRITE_ARRAY(
            row, Rv64BaseAluWAdapterCols, writes_aux.prev_data, record.writes_aux.prev_data
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluWAdapterCols, writes_aux.base)),
            record.writes_aux.prev_timestamp,
            record.from_timestamp + 2
        );

        // reads_aux[1]: register read at from_timestamp + 1 when rs2 is a register;
        // zeroed + range-checked as an immediate otherwise.
        if (record.rs2_as != 0) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Rv64BaseAluWAdapterCols, reads_aux[1])),
                record.reads_aux[1].prev_timestamp,
                record.from_timestamp + 1
            );
        } else {
            RowSlice rs2_aux = row.slice_from(COL_INDEX(Rv64BaseAluWAdapterCols, reads_aux[1]));
#pragma unroll
            for (size_t i = 0; i < sizeof(MemoryReadAuxCols<uint8_t>); i++) {
                rs2_aux.write(i, 0);
            }
            uint32_t mask = (1u << RV64_CELL_BITS) - 1u;
            bitwise_lookup.add_range(record.rs2 & mask, (record.rs2 >> RV64_CELL_BITS) & mask);
        }

        // reads_aux[0] at from_timestamp.
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluWAdapterCols, reads_aux[0])),
            record.reads_aux[0].prev_timestamp,
            record.from_timestamp
        );

        // result_word_msl XOR with 0x80 provides the sign bit lookup for W-width writes.
        bitwise_lookup.add_xor(
            static_cast<uint32_t>(record.result_word_msl), 1u << (RV64_CELL_BITS - 1)
        );

        COL_WRITE_VALUE(row, Rv64BaseAluWAdapterCols, result_sign, record.result_sign);
        COL_WRITE_ARRAY(row, Rv64BaseAluWAdapterCols, rs2_high, record.rs2_high);
        COL_WRITE_VALUE(row, Rv64BaseAluWAdapterCols, rs2_as, record.rs2_as);
        COL_WRITE_VALUE(row, Rv64BaseAluWAdapterCols, rs2, record.rs2);
        COL_WRITE_ARRAY(row, Rv64BaseAluWAdapterCols, rs1_high, record.rs1_high);
        COL_WRITE_VALUE(row, Rv64BaseAluWAdapterCols, rs1_ptr, record.rs1_ptr);
        COL_WRITE_VALUE(row, Rv64BaseAluWAdapterCols, rd_ptr, record.rd_ptr);
        COL_WRITE_VALUE(row, Rv64BaseAluWAdapterCols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Rv64BaseAluWAdapterCols, from_state.pc, record.from_pc);
    }
};
