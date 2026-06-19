#pragma once

#include "primitives/constants.h"
#include "primitives/execution.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

// u16-celled counterpart to Rv64BaseAluWAdapter. Exposes only the low 32-bit word
// (RV64_WORD_U16_LIMBS u16 limbs) to the core; full-width RV64 writes are rebuilt by
// sign-extending the low-word result, and the result sign bit is extracted with the
// variable range checker instead of the bitwise lookup.
template <typename T> struct Rv64BaseAluWU16AdapterCols {
    ExecutionState<T> from_state;
    T rd_ptr;
    T rs1_ptr;
    /// Upper 32 bits of the rs1 register read, as two u16 cells.
    T rs1_high[RV64_WORD_U16_LIMBS];
    /// Pointer if rs2 was a read, immediate value otherwise.
    T rs2;
    /// 1 if rs2 was a read, 0 if an immediate.
    T rs2_as;
    /// Upper 32 bits of the rs2 register read, as two u16 cells (unused when rs2 is immediate).
    T rs2_high[RV64_WORD_U16_LIMBS];
    /// Sign bit of the immediate (0 or 1); 0 when rs2_as == RV64_REGISTER_AS.
    T rs2_imm_sign;
    /// Sign bit of the low-word result, used to build full-width sign-extended writes.
    T result_sign;
    MemoryReadAuxCols<T> reads_aux[2];
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> writes_aux;
};

struct Rv64BaseAluWU16AdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rd_ptr;
    uint32_t rs1_ptr;
    uint16_t rs1_high[RV64_WORD_U16_LIMBS];
    uint32_t rs2;
    uint8_t rs2_as;
    uint8_t rs2_imm_sign;
    uint16_t rs2_high[RV64_WORD_U16_LIMBS];
    uint16_t result_high;
    uint8_t result_sign;
    MemoryReadAuxRecord reads_aux[2];
    MemoryWriteU16AuxRecord<BLOCK_FE_WIDTH> writes_aux;
};

struct Rv64BaseAluWU16Adapter {
    MemoryAuxColsFactory mem_helper;
    VariableRangeChecker range_checker;

    __device__ Rv64BaseAluWU16Adapter(VariableRangeChecker rc, uint32_t timestamp_max_bits)
        : mem_helper(rc, timestamp_max_bits), range_checker(rc) {}

    __device__ void fill_trace_row(RowSlice row, Rv64BaseAluWU16AdapterRecord record) {
        // writes_aux at from_timestamp + 2 (full-width 64-bit register write, 4 u16 cells).
        Fp prev[BLOCK_FE_WIDTH];
        copy_u16_cells(prev, record.writes_aux.prev_data);
        COL_WRITE_ARRAY(row, Rv64BaseAluWU16AdapterCols, writes_aux.prev_data, prev);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluWU16AdapterCols, writes_aux)),
            record.writes_aux.prev_timestamp,
            record.from_timestamp + 2
        );

        // reads_aux[1]: register read at from_timestamp + 1 when rs2 is a register;
        // zeroed + low-limb range-checked as an immediate otherwise.
        if (record.rs2_as != 0) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Rv64BaseAluWU16AdapterCols, reads_aux[1])),
                record.reads_aux[1].prev_timestamp,
                record.from_timestamp + 1
            );
        } else {
            RowSlice rs2_aux =
                row.slice_from(COL_INDEX(Rv64BaseAluWU16AdapterCols, reads_aux[1]));
#pragma unroll
            for (size_t i = 0; i < sizeof(MemoryReadAuxCols<uint8_t>); i++) {
                rs2_aux.write(i, 0);
            }
            range_checker.add_count(record.rs2 & uint32_t(UINT16_MAX), U16_BITS);
        }

        // reads_aux[0] at from_timestamp.
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluWU16AdapterCols, reads_aux[0])),
            record.reads_aux[0].prev_timestamp,
            record.from_timestamp
        );

        // 15-bit range-check witness for the result sign decomposition (a[1] = low15 + sign*2^15).
        range_checker.add_count(
            static_cast<uint32_t>(record.result_high) & ((1u << (U16_BITS - 1)) - 1u),
            U16_BITS - 1
        );

        Fp rs2_high[RV64_WORD_U16_LIMBS];
        Fp rs1_high[RV64_WORD_U16_LIMBS];
        copy_u16_cells(rs2_high, record.rs2_high);
        copy_u16_cells(rs1_high, record.rs1_high);

        COL_WRITE_VALUE(row, Rv64BaseAluWU16AdapterCols, result_sign, record.result_sign);
        COL_WRITE_VALUE(row, Rv64BaseAluWU16AdapterCols, rs2_imm_sign, record.rs2_imm_sign);
        COL_WRITE_ARRAY(row, Rv64BaseAluWU16AdapterCols, rs2_high, rs2_high);
        COL_WRITE_VALUE(row, Rv64BaseAluWU16AdapterCols, rs2_as, record.rs2_as);
        COL_WRITE_VALUE(row, Rv64BaseAluWU16AdapterCols, rs2, record.rs2);
        COL_WRITE_ARRAY(row, Rv64BaseAluWU16AdapterCols, rs1_high, rs1_high);
        COL_WRITE_VALUE(row, Rv64BaseAluWU16AdapterCols, rs1_ptr, record.rs1_ptr);
        COL_WRITE_VALUE(row, Rv64BaseAluWU16AdapterCols, rd_ptr, record.rd_ptr);
        COL_WRITE_VALUE(
            row, Rv64BaseAluWU16AdapterCols, from_state.timestamp, record.from_timestamp
        );
        COL_WRITE_VALUE(row, Rv64BaseAluWU16AdapterCols, from_state.pc, record.from_pc);
    }
};
