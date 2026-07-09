#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <typename T> struct Rv64LoadStoreAdapterCols {
    ExecutionState<T> from_state;
    T rs1_ptr;
    T rs1_data[RV64_PTR_U16_LIMBS];
    MemoryReadAuxCols<T> rs1_aux_cols;

    /// Will write to rd when Load and read from rs2 when Store
    T rd_rs2_ptr;
    MemoryReadAuxCols<T> read_data_aux;
    /// Timestamp aux for the second block read of block-spanning loads.
    MemoryReadAuxCols<T> read1_aux;
    T imm;
    T imm_sign;
    /// mem_ptr is the intermediate memory pointer limbs, needed to check the correct addition
    T mem_ptr_limbs[2];
    T mem_as;
    /// Timestamp aux for the write; previous data is provided by the core chip.
    MemoryBaseAuxCols<T> write_base_aux;
    /// Timestamp aux for the second block write of block-spanning stores; previous data is
    /// provided by the core chip.
    MemoryBaseAuxCols<T> write1_base_aux;
    /// Only writes if `needs_write`.
    /// If the instruction is a Load:
    /// - Sets `needs_write` to 0 iff `rd == x0`
    ///
    /// Otherwise:
    /// - Sets `needs_write` to 1
    T needs_write;
};

struct Rv64LoadStoreAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    uint32_t rs1_ptr;
    uint32_t rs1_val;
    MemoryReadAuxRecord rs1_aux_record;

    uint32_t rd_rs2_ptr;
    MemoryReadAuxRecord read_data_aux;
    /// Aux for the second block read; only meaningful for block-spanning loads.
    MemoryReadAuxRecord read1_aux;
    uint16_t imm;
    bool imm_sign;

    uint8_t local_opcode;
    uint8_t mem_as;

    uint32_t write_prev_timestamp;
    /// Prev timestamp of the second block write; only meaningful for block-spanning stores.
    uint32_t write1_prev_timestamp;
};

// Access width in bytes per Rv64LoadStoreOpcode (transpiler order):
// LOADD, LOADBU, LOADHU, LOADWU, STORED, STOREW, STOREH, STOREB, LOADB, LOADH, LOADW
__device__ constexpr uint32_t RV64_LOADSTORE_ACCESS_WIDTH[11] = {8, 1, 2, 4, 8, 4, 2, 1, 1, 2, 4};

__device__ __forceinline__ bool rv64_loadstore_is_load(uint8_t local_opcode) {
    // STORED..STOREB are opcodes 4..=7; everything else is a load.
    return local_opcode < 4 || local_opcode > 7;
}

__device__ __forceinline__ uint32_t rv64_loadstore_shift_amount(
    Rv64LoadStoreAdapterRecord const &record
) {
    uint32_t ptr = record.rs1_val + uint32_t(record.imm) +
                   uint32_t(record.imm_sign) * (uint32_t(UINT16_MAX) << U16_BITS);
    return ptr & uint32_t(RV64_REGISTER_NUM_LIMBS - 1);
}

__device__ __forceinline__ bool rv64_loadstore_crosses_block(
    Rv64LoadStoreAdapterRecord const &record
) {
    return rv64_loadstore_shift_amount(record) +
               RV64_LOADSTORE_ACCESS_WIDTH[record.local_opcode] >
           uint32_t(RV64_REGISTER_NUM_LIMBS);
}

struct Rv64LoadStoreAdapter {
    size_t pointer_max_bits;
    VariableRangeChecker range_checker;
    MemoryAuxColsFactory mem_helper;

    __device__ Rv64LoadStoreAdapter(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : pointer_max_bits(pointer_max_bits), range_checker(range_checker),
          mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_trace_row(RowSlice row, Rv64LoadStoreAdapterRecord record) {
        bool is_load = rv64_loadstore_is_load(record.local_opcode);
        bool crosses = rv64_loadstore_crosses_block(record);

        COL_WRITE_VALUE(row, Rv64LoadStoreAdapterCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, Rv64LoadStoreAdapterCols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Rv64LoadStoreAdapterCols, rs1_ptr, record.rs1_ptr);

        Fp rs1_data[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(rs1_data, record.rs1_val);
        COL_WRITE_ARRAY(row, Rv64LoadStoreAdapterCols, rs1_data, rs1_data);

        bool needs_write = record.rd_rs2_ptr != UINT32_MAX;

        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64LoadStoreAdapterCols, rs1_aux_cols)),
            record.rs1_aux_record.prev_timestamp,
            record.from_timestamp
        );

        if (needs_write) {
            COL_WRITE_VALUE(row, Rv64LoadStoreAdapterCols, rd_rs2_ptr, record.rd_rs2_ptr);
        } else {
            COL_WRITE_VALUE(row, Rv64LoadStoreAdapterCols, rd_rs2_ptr, 0);
        }

        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64LoadStoreAdapterCols, read_data_aux)),
            record.read_data_aux.prev_timestamp,
            record.from_timestamp + 1
        );

        // Second block read aux (slot t + 2), block-spanning loads only.
        if (is_load && crosses) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Rv64LoadStoreAdapterCols, read1_aux)),
                record.read1_aux.prev_timestamp,
                record.from_timestamp + 2
            );
        } else {
            mem_helper.fill_zero(row.slice_from(COL_INDEX(Rv64LoadStoreAdapterCols, read1_aux)));
        }

        COL_WRITE_VALUE(row, Rv64LoadStoreAdapterCols, imm, record.imm);
        COL_WRITE_VALUE(row, Rv64LoadStoreAdapterCols, imm_sign, record.imm_sign);

        uint32_t ptr = record.rs1_val + uint32_t(record.imm) +
                       uint32_t(record.imm_sign) * (uint32_t(UINT16_MAX) << U16_BITS);
        uint32_t ptr_limbs[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(ptr_limbs, ptr);
        COL_WRITE_ARRAY(row, Rv64LoadStoreAdapterCols, mem_ptr_limbs, ptr_limbs);
        COL_WRITE_VALUE(row, Rv64LoadStoreAdapterCols, mem_as, record.mem_as);

        range_checker.add_count((ptr_limbs[0] >> 3) + uint32_t(crosses), U16_BITS - 3);
        range_checker.add_count(ptr_limbs[1], pointer_max_bits - U16_BITS);

        COL_WRITE_VALUE(row, Rv64LoadStoreAdapterCols, needs_write, needs_write);
        // First write (slot t + 3).
        if (needs_write) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Rv64LoadStoreAdapterCols, write_base_aux)),
                record.write_prev_timestamp,
                record.from_timestamp + 3
            );
        } else {
            mem_helper.fill_zero(
                row.slice_from(COL_INDEX(Rv64LoadStoreAdapterCols, write_base_aux))
            );
        }
        // Second block write aux (slot t + 4), block-spanning stores only.
        if (!is_load && crosses) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Rv64LoadStoreAdapterCols, write1_base_aux)),
                record.write1_prev_timestamp,
                record.from_timestamp + 4
            );
        } else {
            mem_helper.fill_zero(
                row.slice_from(COL_INDEX(Rv64LoadStoreAdapterCols, write1_base_aux))
            );
        }
    }
};
