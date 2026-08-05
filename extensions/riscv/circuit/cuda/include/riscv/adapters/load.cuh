#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <typename T> struct LoadMultiByteAdapterCols {
    ExecutionState<T> from_state;
    T rs1_ptr;
    T rs1_data[PTR_U16_LIMBS];
    MemoryReadAuxCols<T> rs1_aux_cols;
    T rd_ptr;
    MemoryReadAuxCols<T> read_data_aux[2];
    T imm;
    T imm_sign;
    T mem_ptr_low_limb;
    T mem_ptr_carry;
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> write_aux;
    T needs_write;
};

struct LoadAdapter {
    size_t pointer_max_bits;
    VariableRangeChecker range_checker;
    MemoryAuxColsFactory mem_helper;

    __device__ LoadAdapter(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : pointer_max_bits(pointer_max_bits), range_checker(range_checker),
          mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_trace_row(
        RowSlice row,
        uint32_t from_pc,
        uint32_t from_timestamp,
        uint32_t rs1_ptr,
        uint32_t rd_ptr,
        bool needs_write,
        uint32_t rs1_val,
        uint32_t rs1_prev_timestamp,
        uint32_t block0_prev_timestamp,
        uint32_t block1_prev_timestamp,
        bool crosses,
        uint32_t write_prev_timestamp,
        uint16_t const (&write_prev_data)[BLOCK_FE_WIDTH],
        uint16_t imm,
        bool imm_sign
    ) {
        COL_WRITE_VALUE(row, LoadMultiByteAdapterCols, from_state.pc, from_pc);
        COL_WRITE_VALUE(row, LoadMultiByteAdapterCols, from_state.timestamp, from_timestamp);
        COL_WRITE_VALUE(row, LoadMultiByteAdapterCols, rs1_ptr, rs1_ptr);

        Fp rs1_data[PTR_U16_LIMBS];
        ptr_to_u16_limbs(rs1_data, rs1_val);
        COL_WRITE_ARRAY(row, LoadMultiByteAdapterCols, rs1_data, rs1_data);

        mem_helper.fill(
            row.slice_from(COL_INDEX(LoadMultiByteAdapterCols, rs1_aux_cols)),
            rs1_prev_timestamp,
            from_timestamp
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(LoadMultiByteAdapterCols, read_data_aux[0])),
            block0_prev_timestamp,
            from_timestamp + 1
        );
        if (crosses) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(LoadMultiByteAdapterCols, read_data_aux[1])),
                block1_prev_timestamp,
                from_timestamp + 2
            );
        } else {
            mem_helper.fill_zero(row.slice_from(COL_INDEX(LoadMultiByteAdapterCols, read_data_aux[1])));
        }

        COL_WRITE_VALUE(row, LoadMultiByteAdapterCols, rd_ptr, needs_write ? rd_ptr : 0);
        COL_WRITE_VALUE(row, LoadMultiByteAdapterCols, needs_write, needs_write);
        if (needs_write) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(LoadMultiByteAdapterCols, write_aux.base)),
                write_prev_timestamp,
                from_timestamp + 3
            );
            Fp prev_data[BLOCK_FE_WIDTH];
            copy_u16_cells(prev_data, write_prev_data);
            COL_WRITE_ARRAY(row, LoadMultiByteAdapterCols, write_aux.prev_data, prev_data);
        } else {
            mem_helper.fill_zero(row.slice_from(COL_INDEX(LoadMultiByteAdapterCols, write_aux.base)));
            row.fill_zero(COL_INDEX(LoadMultiByteAdapterCols, write_aux.prev_data), BLOCK_FE_WIDTH);
        }

        COL_WRITE_VALUE(row, LoadMultiByteAdapterCols, imm, imm);
        COL_WRITE_VALUE(row, LoadMultiByteAdapterCols, imm_sign, imm_sign);

        uint32_t ptr = rs1_val + uint32_t(imm) +
                       uint32_t(imm_sign) * (uint32_t(UINT16_MAX) << U16_BITS);
        uint32_t ptr_limbs[PTR_U16_LIMBS];
        ptr_to_u16_limbs(ptr_limbs, ptr);
        COL_WRITE_VALUE(row, LoadMultiByteAdapterCols, mem_ptr_low_limb, ptr_limbs[0]);

        uint32_t shift_amount = ptr & (MEMORY_BLOCK_BYTES - 1);
        uint32_t aligned_limb = ptr_limbs[0] - shift_amount;
        range_checker.add_count(aligned_limb >> 3, U16_BITS - 3);
        range_checker.add_count(ptr_limbs[1], pointer_max_bits - U16_BITS);

        uint32_t block1_low_sum = aligned_limb + uint32_t(MEMORY_BLOCK_BYTES);
        bool carry = crosses && block1_low_sum == (1u << U16_BITS);
        COL_WRITE_VALUE(row, LoadMultiByteAdapterCols, mem_ptr_carry, carry);
        if (crosses) {
            range_checker.add_count(
                (block1_low_sum - (uint32_t(carry) << U16_BITS)) >> 3,
                U16_BITS - 3
            );
        }
        if (carry) {
            range_checker.add_count(ptr_limbs[1] + carry, pointer_max_bits - U16_BITS);
        }
    }
};

// Byte loads use one memory block and need no crossing-related trace columns.
template <typename T> struct LoadByteAdapterCols {
    ExecutionState<T> from_state;
    T rs1_ptr;
    T rs1_data[PTR_U16_LIMBS];
    MemoryReadAuxCols<T> rs1_aux_cols;
    T rd_ptr;
    MemoryReadAuxCols<T> read_data_aux;
    T imm;
    T imm_sign;
    T mem_ptr_low_limb;
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> write_aux;
    T needs_write;
};

struct LoadByteAdapter {
    size_t pointer_max_bits;
    VariableRangeChecker range_checker;
    MemoryAuxColsFactory mem_helper;

    __device__ LoadByteAdapter(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : pointer_max_bits(pointer_max_bits), range_checker(range_checker),
          mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_trace_row(
        RowSlice row,
        uint32_t from_pc,
        uint32_t from_timestamp,
        uint32_t rs1_ptr,
        uint32_t rd_ptr,
        bool needs_write,
        uint32_t rs1_val,
        uint32_t rs1_prev_timestamp,
        uint32_t read_prev_timestamp,
        uint32_t write_prev_timestamp,
        uint16_t const (&write_prev_data)[BLOCK_FE_WIDTH],
        uint16_t imm,
        bool imm_sign
    ) {
        COL_WRITE_VALUE(row, LoadByteAdapterCols, from_state.pc, from_pc);
        COL_WRITE_VALUE(
            row, LoadByteAdapterCols, from_state.timestamp, from_timestamp
        );
        COL_WRITE_VALUE(row, LoadByteAdapterCols, rs1_ptr, rs1_ptr);

        Fp rs1_data[PTR_U16_LIMBS];
        ptr_to_u16_limbs(rs1_data, rs1_val);
        COL_WRITE_ARRAY(row, LoadByteAdapterCols, rs1_data, rs1_data);

        mem_helper.fill(
            row.slice_from(COL_INDEX(LoadByteAdapterCols, rs1_aux_cols)),
            rs1_prev_timestamp,
            from_timestamp
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(LoadByteAdapterCols, read_data_aux)),
            read_prev_timestamp,
            from_timestamp + 1
        );

        COL_WRITE_VALUE(row, LoadByteAdapterCols, rd_ptr, needs_write ? rd_ptr : 0);
        COL_WRITE_VALUE(row, LoadByteAdapterCols, needs_write, needs_write);
        if (needs_write) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(LoadByteAdapterCols, write_aux.base)),
                write_prev_timestamp,
                from_timestamp + 2
            );
            Fp prev_data[BLOCK_FE_WIDTH];
            copy_u16_cells(prev_data, write_prev_data);
            COL_WRITE_ARRAY(row, LoadByteAdapterCols, write_aux.prev_data, prev_data);
        } else {
            mem_helper.fill_zero(
                row.slice_from(COL_INDEX(LoadByteAdapterCols, write_aux.base))
            );
            row.fill_zero(COL_INDEX(LoadByteAdapterCols, write_aux.prev_data), BLOCK_FE_WIDTH);
        }

        COL_WRITE_VALUE(row, LoadByteAdapterCols, imm, imm);
        COL_WRITE_VALUE(row, LoadByteAdapterCols, imm_sign, imm_sign);

        uint32_t ptr = rs1_val + uint32_t(imm) +
                       uint32_t(imm_sign) * (uint32_t(UINT16_MAX) << U16_BITS);
        uint32_t ptr_limbs[PTR_U16_LIMBS];
        ptr_to_u16_limbs(ptr_limbs, ptr);
        COL_WRITE_VALUE(row, LoadByteAdapterCols, mem_ptr_low_limb, ptr_limbs[0]);

        uint32_t shift_amount = ptr & (MEMORY_BLOCK_BYTES - 1);
        range_checker.add_count((ptr_limbs[0] - shift_amount) >> 3, U16_BITS - 3);
        range_checker.add_count(ptr_limbs[1], pointer_max_bits - U16_BITS);
    }
};
