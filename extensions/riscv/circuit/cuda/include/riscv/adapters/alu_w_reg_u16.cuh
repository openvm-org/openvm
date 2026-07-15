#pragma once

#include "primitives/constants.h"
#include "primitives/execution.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <typename T> struct Rv64BaseAluWRegU16AdapterCols {
    ExecutionState<T> from_state;
    T rd_ptr;
    T rs1_ptr;
    T rs1_high[RV64_WORD_U16_LIMBS];
    T rs2_ptr;
    T rs2_high[RV64_WORD_U16_LIMBS];
    T result_sign;
    MemoryReadAuxCols<T> reads_aux[2];
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> writes_aux;
};

static_assert(sizeof(Rv64BaseAluWRegU16AdapterCols<uint8_t>) == 23);

struct Rv64BaseAluWRegU16AdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rd_ptr;
    uint32_t rs1_ptr;
    uint16_t rs1_high[RV64_WORD_U16_LIMBS];
    uint32_t rs2_ptr;
    uint16_t rs2_high[RV64_WORD_U16_LIMBS];
    uint16_t result_high;
    uint8_t result_sign;
    MemoryReadAuxRecord reads_aux[2];
    MemoryWriteU16AuxRecord<BLOCK_FE_WIDTH> writes_aux;
};

struct Rv64BaseAluWRegU16Adapter {
    MemoryAuxColsFactory mem_helper;
    VariableRangeChecker range_checker;

    __device__ Rv64BaseAluWRegU16Adapter(VariableRangeChecker rc, uint32_t timestamp_max_bits)
        : mem_helper(rc, timestamp_max_bits), range_checker(rc) {}

    __device__ void fill_trace_row(RowSlice row, Rv64BaseAluWRegU16AdapterRecord record) {
        Fp prev[BLOCK_FE_WIDTH];
        copy_u16_cells(prev, record.writes_aux.prev_data);
        COL_WRITE_ARRAY(row, Rv64BaseAluWRegU16AdapterCols, writes_aux.prev_data, prev);
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluWRegU16AdapterCols, writes_aux)),
            record.writes_aux.prev_timestamp,
            record.from_timestamp + 2
        );

        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluWRegU16AdapterCols, reads_aux[1])),
            record.reads_aux[1].prev_timestamp,
            record.from_timestamp + 1
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64BaseAluWRegU16AdapterCols, reads_aux[0])),
            record.reads_aux[0].prev_timestamp,
            record.from_timestamp
        );

        range_checker.add_count(
            static_cast<uint32_t>(record.result_high) & ((1u << (U16_BITS - 1)) - 1u), U16_BITS - 1
        );

        Fp rs1_high[RV64_WORD_U16_LIMBS];
        Fp rs2_high[RV64_WORD_U16_LIMBS];
        copy_u16_cells(rs1_high, record.rs1_high);
        copy_u16_cells(rs2_high, record.rs2_high);

        COL_WRITE_VALUE(row, Rv64BaseAluWRegU16AdapterCols, result_sign, record.result_sign);
        COL_WRITE_ARRAY(row, Rv64BaseAluWRegU16AdapterCols, rs2_high, rs2_high);
        COL_WRITE_VALUE(row, Rv64BaseAluWRegU16AdapterCols, rs2_ptr, record.rs2_ptr);
        COL_WRITE_ARRAY(row, Rv64BaseAluWRegU16AdapterCols, rs1_high, rs1_high);
        COL_WRITE_VALUE(row, Rv64BaseAluWRegU16AdapterCols, rs1_ptr, record.rs1_ptr);
        COL_WRITE_VALUE(row, Rv64BaseAluWRegU16AdapterCols, rd_ptr, record.rd_ptr);
        COL_WRITE_VALUE(
            row, Rv64BaseAluWRegU16AdapterCols, from_state.timestamp, record.from_timestamp
        );
        COL_WRITE_VALUE(row, Rv64BaseAluWRegU16AdapterCols, from_state.pc, record.from_pc);
    }
};
