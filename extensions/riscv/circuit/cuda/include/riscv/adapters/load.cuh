#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <typename T> struct Rv64LoadAdapterCols {
    ExecutionState<T> from_state;
    T rs1_ptr;
    T rs1_data[RV64_PTR_U16_LIMBS];
    MemoryReadAuxCols<T> rs1_aux_cols;
    T rd_ptr;
    MemoryReadAuxCols<T> read_data_aux;
    MemoryReadAuxCols<T> read_data1_aux;
    T imm;
    T imm_sign;
    T mem_ptr_limbs[2];
    T mem_ptr_carry;
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> write_aux;
    T needs_write;
};

struct Rv64LoadAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    uint32_t rs1_ptr;
    uint32_t rs1_val;
    MemoryReadAuxRecord rs1_aux_record;

    uint32_t rd_ptr;
    MemoryReadAuxRecord read_data_aux;
    // prev_timestamp == UINT32_MAX means the access does not cross a block boundary.
    MemoryReadAuxRecord read_data1_aux;
    uint16_t imm;
    bool imm_sign;

    uint32_t write_prev_timestamp;
    uint16_t write_prev_data[BLOCK_FE_WIDTH];
};

static __device__ __forceinline__ uint32_t
rv64_load_effective_ptr(Rv64LoadAdapterRecord record) {
    return record.rs1_val + uint32_t(record.imm) +
           uint32_t(record.imm_sign) * (uint32_t(UINT16_MAX) << U16_BITS);
}

static __device__ __forceinline__ uint32_t rv64_load_shift_amount(Rv64LoadAdapterRecord record) {
    return rv64_load_effective_ptr(record) & (RV64_REGISTER_NUM_LIMBS - 1);
}

struct Rv64LoadAdapter {
    size_t pointer_max_bits;
    VariableRangeChecker range_checker;
    MemoryAuxColsFactory mem_helper;

    __device__ Rv64LoadAdapter(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : pointer_max_bits(pointer_max_bits), range_checker(range_checker),
          mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_trace_row(RowSlice row, Rv64LoadAdapterRecord record) {
        COL_WRITE_VALUE(row, Rv64LoadAdapterCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, Rv64LoadAdapterCols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Rv64LoadAdapterCols, rs1_ptr, record.rs1_ptr);

        Fp rs1_data[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(rs1_data, record.rs1_val);
        COL_WRITE_ARRAY(row, Rv64LoadAdapterCols, rs1_data, rs1_data);

        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64LoadAdapterCols, rs1_aux_cols)),
            record.rs1_aux_record.prev_timestamp,
            record.from_timestamp
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64LoadAdapterCols, read_data_aux)),
            record.read_data_aux.prev_timestamp,
            record.from_timestamp + 1
        );
        bool crosses = record.read_data1_aux.prev_timestamp != UINT32_MAX;
        if (crosses) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Rv64LoadAdapterCols, read_data1_aux)),
                record.read_data1_aux.prev_timestamp,
                record.from_timestamp + 2
            );
        } else {
            mem_helper.fill_zero(row.slice_from(COL_INDEX(Rv64LoadAdapterCols, read_data1_aux)));
        }

        bool needs_write = record.rd_ptr != UINT32_MAX;
        COL_WRITE_VALUE(row, Rv64LoadAdapterCols, rd_ptr, needs_write ? record.rd_ptr : 0);
        COL_WRITE_VALUE(row, Rv64LoadAdapterCols, needs_write, needs_write);
        if (needs_write) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Rv64LoadAdapterCols, write_aux.base)),
                record.write_prev_timestamp,
                record.from_timestamp + 3
            );
            Fp prev_data[BLOCK_FE_WIDTH];
            copy_u16_cells(prev_data, record.write_prev_data);
            COL_WRITE_ARRAY(row, Rv64LoadAdapterCols, write_aux.prev_data, prev_data);
        } else {
            mem_helper.fill_zero(row.slice_from(COL_INDEX(Rv64LoadAdapterCols, write_aux.base)));
            row.fill_zero(COL_INDEX(Rv64LoadAdapterCols, write_aux.prev_data), BLOCK_FE_WIDTH);
        }

        COL_WRITE_VALUE(row, Rv64LoadAdapterCols, imm, record.imm);
        COL_WRITE_VALUE(row, Rv64LoadAdapterCols, imm_sign, record.imm_sign);

        uint32_t ptr = rv64_load_effective_ptr(record);
        uint32_t ptr_limbs[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(ptr_limbs, ptr);
        COL_WRITE_ARRAY(row, Rv64LoadAdapterCols, mem_ptr_limbs, ptr_limbs);

        uint32_t shift_amount = rv64_load_shift_amount(record);
        uint32_t aligned_limb0 = ptr_limbs[0] - shift_amount;
        range_checker.add_count(aligned_limb0 >> 3, U16_BITS - 3);
        range_checker.add_count(ptr_limbs[1], pointer_max_bits - U16_BITS);

        bool carry = crosses && (aligned_limb0 + 8 == (1u << U16_BITS));
        COL_WRITE_VALUE(row, Rv64LoadAdapterCols, mem_ptr_carry, carry);
        if (crosses) {
            range_checker.add_count(
                (aligned_limb0 + 8 - (uint32_t(carry) << U16_BITS)) >> 3, U16_BITS - 3
            );
            range_checker.add_count(ptr_limbs[1] + carry, pointer_max_bits - U16_BITS);
        }
    }
};

// Lean byte-load adapter for `lb`/`lbu`, which never cross a block boundary. Drops the crossing
// columns (`read_data1_aux`, `mem_ptr_carry`) and the second read's timestamp slot. Reuses
// `Rv64LoadAdapterRecord`; its crossing fields are ignored here.
template <typename T> struct Rv64LoadByteAdapterCols {
    ExecutionState<T> from_state;
    T rs1_ptr;
    T rs1_data[RV64_PTR_U16_LIMBS];
    MemoryReadAuxCols<T> rs1_aux_cols;
    T rd_ptr;
    MemoryReadAuxCols<T> read_data_aux;
    T imm;
    T imm_sign;
    T mem_ptr_limbs[2];
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> write_aux;
    T needs_write;
};

struct Rv64LoadByteAdapter {
    size_t pointer_max_bits;
    VariableRangeChecker range_checker;
    MemoryAuxColsFactory mem_helper;

    __device__ Rv64LoadByteAdapter(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : pointer_max_bits(pointer_max_bits), range_checker(range_checker),
          mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_trace_row(RowSlice row, Rv64LoadAdapterRecord record) {
        COL_WRITE_VALUE(row, Rv64LoadByteAdapterCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(
            row, Rv64LoadByteAdapterCols, from_state.timestamp, record.from_timestamp
        );
        COL_WRITE_VALUE(row, Rv64LoadByteAdapterCols, rs1_ptr, record.rs1_ptr);

        Fp rs1_data[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(rs1_data, record.rs1_val);
        COL_WRITE_ARRAY(row, Rv64LoadByteAdapterCols, rs1_data, rs1_data);

        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64LoadByteAdapterCols, rs1_aux_cols)),
            record.rs1_aux_record.prev_timestamp,
            record.from_timestamp
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64LoadByteAdapterCols, read_data_aux)),
            record.read_data_aux.prev_timestamp,
            record.from_timestamp + 1
        );

        bool needs_write = record.rd_ptr != UINT32_MAX;
        COL_WRITE_VALUE(row, Rv64LoadByteAdapterCols, rd_ptr, needs_write ? record.rd_ptr : 0);
        COL_WRITE_VALUE(row, Rv64LoadByteAdapterCols, needs_write, needs_write);
        if (needs_write) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Rv64LoadByteAdapterCols, write_aux.base)),
                record.write_prev_timestamp,
                record.from_timestamp + 2
            );
            Fp prev_data[BLOCK_FE_WIDTH];
            copy_u16_cells(prev_data, record.write_prev_data);
            COL_WRITE_ARRAY(row, Rv64LoadByteAdapterCols, write_aux.prev_data, prev_data);
        } else {
            mem_helper.fill_zero(
                row.slice_from(COL_INDEX(Rv64LoadByteAdapterCols, write_aux.base))
            );
            row.fill_zero(COL_INDEX(Rv64LoadByteAdapterCols, write_aux.prev_data), BLOCK_FE_WIDTH);
        }

        COL_WRITE_VALUE(row, Rv64LoadByteAdapterCols, imm, record.imm);
        COL_WRITE_VALUE(row, Rv64LoadByteAdapterCols, imm_sign, record.imm_sign);

        uint32_t ptr = rv64_load_effective_ptr(record);
        uint32_t ptr_limbs[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(ptr_limbs, ptr);
        COL_WRITE_ARRAY(row, Rv64LoadByteAdapterCols, mem_ptr_limbs, ptr_limbs);

        uint32_t shift_amount = rv64_load_shift_amount(record);
        range_checker.add_count((ptr_limbs[0] - shift_amount) >> 3, U16_BITS - 3);
        range_checker.add_count(ptr_limbs[1], pointer_max_bits - U16_BITS);
    }
};
