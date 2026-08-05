#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <typename T> struct StoreMultiByteAdapterCols {
    ExecutionState<T> from_state;
    T rs1_ptr;
    T rs1_data[PTR_U16_LIMBS];
    MemoryReadAuxCols<T> rs1_aux_cols;
    T rs2_ptr;
    MemoryReadAuxCols<T> read_data_aux;
    T imm;
    T imm_sign;
    T mem_ptr_low_limb;
    T mem_as;
    T mem_ptr_carry;
    MemoryBaseAuxCols<T> write_base_aux[2];
};

struct StoreAdapter {
    size_t pointer_max_bits;
    VariableRangeChecker range_checker;
    MemoryAuxColsFactory mem_helper;

    __device__ StoreAdapter(
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
        uint32_t rs2_ptr,
        uint32_t rs1_val,
        uint32_t rs1_prev_timestamp,
        uint32_t rs2_prev_timestamp,
        uint32_t write0_prev_timestamp,
        uint32_t write1_prev_timestamp,
        uint16_t imm,
        uint8_t imm_sign,
        uint32_t mem_as
    ) {
        COL_WRITE_VALUE(row, StoreMultiByteAdapterCols, from_state.pc, from_pc);
        COL_WRITE_VALUE(row, StoreMultiByteAdapterCols, from_state.timestamp, from_timestamp);
        COL_WRITE_VALUE(row, StoreMultiByteAdapterCols, rs1_ptr, rs1_ptr);

        Fp rs1_data[PTR_U16_LIMBS];
        ptr_to_u16_limbs(rs1_data, rs1_val);
        COL_WRITE_ARRAY(row, StoreMultiByteAdapterCols, rs1_data, rs1_data);

        mem_helper.fill(
            row.slice_from(COL_INDEX(StoreMultiByteAdapterCols, rs1_aux_cols)),
            rs1_prev_timestamp,
            from_timestamp
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(StoreMultiByteAdapterCols, read_data_aux)),
            rs2_prev_timestamp,
            from_timestamp + 1
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(StoreMultiByteAdapterCols, write_base_aux[0])),
            write0_prev_timestamp,
            from_timestamp + 2
        );
        bool crosses = write1_prev_timestamp != UINT32_MAX;
        if (crosses) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(StoreMultiByteAdapterCols, write_base_aux[1])),
                write1_prev_timestamp,
                from_timestamp + 3
            );
        } else {
            mem_helper.fill_zero(
                row.slice_from(COL_INDEX(StoreMultiByteAdapterCols, write_base_aux[1]))
            );
        }

        COL_WRITE_VALUE(row, StoreMultiByteAdapterCols, rs2_ptr, rs2_ptr);
        COL_WRITE_VALUE(row, StoreMultiByteAdapterCols, imm, imm);
        COL_WRITE_VALUE(row, StoreMultiByteAdapterCols, imm_sign, imm_sign);
        COL_WRITE_VALUE(row, StoreMultiByteAdapterCols, mem_as, mem_as);

        uint32_t ptr = rs1_val + uint32_t(imm) +
                       uint32_t(imm_sign) * (uint32_t(UINT16_MAX) << U16_BITS);
        uint32_t ptr_limbs[PTR_U16_LIMBS];
        ptr_to_u16_limbs(ptr_limbs, ptr);
        COL_WRITE_VALUE(row, StoreMultiByteAdapterCols, mem_ptr_low_limb, ptr_limbs[0]);

        uint32_t shift_amount = ptr & (MEMORY_BLOCK_BYTES - 1);
        uint32_t aligned_limb = ptr_limbs[0] - shift_amount;
        range_checker.add_count(aligned_limb >> 3, U16_BITS - 3);
        range_checker.add_count(ptr_limbs[1], pointer_max_bits - U16_BITS);

        uint32_t block1_low_sum = aligned_limb + uint32_t(MEMORY_BLOCK_BYTES);
        bool carry = crosses && block1_low_sum == (1u << U16_BITS);
        COL_WRITE_VALUE(row, StoreMultiByteAdapterCols, mem_ptr_carry, carry);
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

// Byte stores use one memory block and need no crossing-related trace columns.
template <typename T> struct StoreByteAdapterCols {
    ExecutionState<T> from_state;
    T rs1_ptr;
    T rs1_data[PTR_U16_LIMBS];
    MemoryReadAuxCols<T> rs1_aux_cols;
    T rs2_ptr;
    MemoryReadAuxCols<T> read_data_aux;
    T imm;
    T imm_sign;
    T mem_ptr_low_limb;
    T mem_as;
    MemoryBaseAuxCols<T> write_base_aux;
};

struct StoreByteAdapter {
    size_t pointer_max_bits;
    VariableRangeChecker range_checker;
    MemoryAuxColsFactory mem_helper;

    __device__ StoreByteAdapter(
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
        uint32_t rs2_ptr,
        uint32_t rs1_val,
        uint32_t rs1_prev_timestamp,
        uint32_t rs2_prev_timestamp,
        uint32_t write_prev_timestamp,
        uint16_t imm,
        uint8_t imm_sign,
        uint32_t mem_as
    ) {
        COL_WRITE_VALUE(row, StoreByteAdapterCols, from_state.pc, from_pc);
        COL_WRITE_VALUE(
            row, StoreByteAdapterCols, from_state.timestamp, from_timestamp
        );
        COL_WRITE_VALUE(row, StoreByteAdapterCols, rs1_ptr, rs1_ptr);

        Fp rs1_data[PTR_U16_LIMBS];
        ptr_to_u16_limbs(rs1_data, rs1_val);
        COL_WRITE_ARRAY(row, StoreByteAdapterCols, rs1_data, rs1_data);

        mem_helper.fill(
            row.slice_from(COL_INDEX(StoreByteAdapterCols, rs1_aux_cols)),
            rs1_prev_timestamp,
            from_timestamp
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(StoreByteAdapterCols, read_data_aux)),
            rs2_prev_timestamp,
            from_timestamp + 1
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(StoreByteAdapterCols, write_base_aux)),
            write_prev_timestamp,
            from_timestamp + 2
        );

        COL_WRITE_VALUE(row, StoreByteAdapterCols, rs2_ptr, rs2_ptr);
        COL_WRITE_VALUE(row, StoreByteAdapterCols, imm, imm);
        COL_WRITE_VALUE(row, StoreByteAdapterCols, imm_sign, imm_sign);
        COL_WRITE_VALUE(row, StoreByteAdapterCols, mem_as, mem_as);

        uint32_t ptr = rs1_val + uint32_t(imm) +
                       uint32_t(imm_sign) * (uint32_t(UINT16_MAX) << U16_BITS);
        uint32_t ptr_limbs[PTR_U16_LIMBS];
        ptr_to_u16_limbs(ptr_limbs, ptr);
        COL_WRITE_VALUE(row, StoreByteAdapterCols, mem_ptr_low_limb, ptr_limbs[0]);

        uint32_t shift_amount = ptr & (MEMORY_BLOCK_BYTES - 1);
        range_checker.add_count((ptr_limbs[0] - shift_amount) >> 3, U16_BITS - 3);
        range_checker.add_count(ptr_limbs[1], pointer_max_bits - U16_BITS);
    }
};
