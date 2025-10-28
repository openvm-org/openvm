#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "system/memory/controller.cuh"

using namespace riscv;

template <typename T> struct Rv32LoadStoreAdapterCols {
    ExecutionState<T> from_state;
    T rs1_ptr;
    T rs1_data[RV32_REGISTER_NUM_LIMBS];
    MemoryReadAuxCols<T> rs1_aux_cols;

    /// Will write to rd when Load and read from rs2 when Store
    T rd_rs2_ptr;
    MemoryReadAuxCols<T> read_data_aux;
    T imm;
    T imm_sign;
    /// mem_ptr is the intermediate memory pointer limbs, needed to check the correct addition
    T mem_ptr_limbs[2];
    T mem_as;
    /// prev_data will be provided by the core chip to make a complete MemoryWriteAuxCols
    MemoryBaseAuxCols<T> write_base_aux;
    /// Only writes if `needs_write`.
    /// If the instruction is a Load:
    /// - Sets `needs_write` to 0 iff `rd == x0`
    ///
    /// Otherwise:
    /// - Sets `needs_write` to 1
    T needs_write;
};

struct Rv32LoadStoreAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    uint32_t rs1_ptr;
    uint32_t rs1_val;
    MemoryReadAuxRecord rs1_aux_record;

    uint32_t rd_rs2_ptr;
    MemoryReadAuxRecord read_data_aux;
    uint16_t imm;
    bool imm_sign;

    uint8_t mem_as;

    uint32_t write_prev_timestamp;
};

struct Rv32LoadStoreAdapter {
    size_t pointer_max_bits;
    VariableRangeChecker range_checker;
    MemoryAuxColsFactory mem_helper;

    __device__ Rv32LoadStoreAdapter(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : pointer_max_bits(pointer_max_bits), range_checker(range_checker),
          mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_trace_row(RowSlice row, Rv32LoadStoreAdapterRecord record) {
        COL_WRITE_VALUE(row, Rv32LoadStoreAdapterCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, Rv32LoadStoreAdapterCols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Rv32LoadStoreAdapterCols, rs1_ptr, record.rs1_ptr);

        auto rs1_data = reinterpret_cast<uint8_t *>(&record.rs1_val);
        COL_WRITE_ARRAY(row, Rv32LoadStoreAdapterCols, rs1_data, rs1_data);

        bool needs_write = record.rd_rs2_ptr != UINT32_MAX;

        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv32LoadStoreAdapterCols, rs1_aux_cols)),
            record.rs1_aux_record.prev_timestamp,
            record.from_timestamp
        );

        if (needs_write) {
            COL_WRITE_VALUE(row, Rv32LoadStoreAdapterCols, rd_rs2_ptr, record.rd_rs2_ptr);
        } else {
            COL_WRITE_VALUE(row, Rv32LoadStoreAdapterCols, rd_rs2_ptr, 0);
        }

        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv32LoadStoreAdapterCols, read_data_aux)),
            record.read_data_aux.prev_timestamp,
            record.from_timestamp + 1
        );

        COL_WRITE_VALUE(row, Rv32LoadStoreAdapterCols, imm, record.imm);
        COL_WRITE_VALUE(row, Rv32LoadStoreAdapterCols, imm_sign, record.imm_sign);

        uint32_t ptr = record.rs1_val + ((uint32_t)record.imm + record.imm_sign * 0xffff0000);
        auto ptr_limbs = reinterpret_cast<uint16_t *>(&ptr);
        COL_WRITE_ARRAY(row, Rv32LoadStoreAdapterCols, mem_ptr_limbs, ptr_limbs);
        COL_WRITE_VALUE(row, Rv32LoadStoreAdapterCols, mem_as, record.mem_as);

        range_checker.add_count((uint32_t)ptr_limbs[0] >> 2, RV32_CELL_BITS * 2 - 2);
        range_checker.add_count((uint32_t)ptr_limbs[1], pointer_max_bits - 16);

        COL_WRITE_VALUE(row, Rv32LoadStoreAdapterCols, needs_write, needs_write);
        if (needs_write) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Rv32LoadStoreAdapterCols, write_base_aux)),
                record.write_prev_timestamp,
                record.from_timestamp + 2
            );
        } else {
            mem_helper.fill_zero(
                row.slice_from(COL_INDEX(Rv32LoadStoreAdapterCols, write_base_aux))
            );
        }
    }
};