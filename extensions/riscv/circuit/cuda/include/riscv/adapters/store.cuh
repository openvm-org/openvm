#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <typename T> struct Rv64StoreMultiByteAdapterCols {
    ExecutionState<T> from_state;
    T rs1_ptr;
    T rs1_data[RV64_PTR_U16_LIMBS];
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

struct Rv64StoreMultiByteAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    uint32_t rs1_val;
    MemoryReadAuxRecord rs1_aux_record;

    MemoryReadAuxRecord read_data_aux;
    // The second timestamp is UINT32_MAX when the access does not cross a block boundary.
    uint32_t write_prev_timestamps[2];
    uint16_t imm;
    uint8_t rs1_ptr;
    uint8_t rs2_ptr;
    bool imm_sign;
    uint8_t mem_as;
};

template <typename Record>
static __device__ __forceinline__ uint32_t rv64_store_effective_ptr(Record record) {
    return record.rs1_val + uint32_t(record.imm) +
           uint32_t(record.imm_sign) * (uint32_t(UINT16_MAX) << U16_BITS);
}

template <typename Record>
static __device__ __forceinline__ uint32_t rv64_store_shift_amount(Record record) {
    return rv64_store_effective_ptr(record) & (MEMORY_BLOCK_BYTES - 1);
}

struct Rv64StoreAdapter {
    size_t pointer_max_bits;
    VariableRangeChecker range_checker;
    MemoryAuxColsFactory mem_helper;

    __device__ Rv64StoreAdapter(
        size_t pointer_max_bits,
        VariableRangeChecker range_checker,
        uint32_t timestamp_max_bits
    )
        : pointer_max_bits(pointer_max_bits), range_checker(range_checker),
          mem_helper(range_checker, timestamp_max_bits) {}

    __device__ void fill_trace_row(RowSlice row, Rv64StoreMultiByteAdapterRecord record) {
        COL_WRITE_VALUE(row, Rv64StoreMultiByteAdapterCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, Rv64StoreMultiByteAdapterCols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Rv64StoreMultiByteAdapterCols, rs1_ptr, record.rs1_ptr);

        Fp rs1_data[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(rs1_data, record.rs1_val);
        COL_WRITE_ARRAY(row, Rv64StoreMultiByteAdapterCols, rs1_data, rs1_data);

        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64StoreMultiByteAdapterCols, rs1_aux_cols)),
            record.rs1_aux_record.prev_timestamp,
            record.from_timestamp
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64StoreMultiByteAdapterCols, read_data_aux)),
            record.read_data_aux.prev_timestamp,
            record.from_timestamp + 1
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64StoreMultiByteAdapterCols, write_base_aux[0])),
            record.write_prev_timestamps[0],
            record.from_timestamp + 2
        );
        bool crosses = record.write_prev_timestamps[1] != UINT32_MAX;
        if (crosses) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Rv64StoreMultiByteAdapterCols, write_base_aux[1])),
                record.write_prev_timestamps[1],
                record.from_timestamp + 3
            );
        } else {
            mem_helper.fill_zero(
                row.slice_from(COL_INDEX(Rv64StoreMultiByteAdapterCols, write_base_aux[1]))
            );
        }

        COL_WRITE_VALUE(row, Rv64StoreMultiByteAdapterCols, rs2_ptr, record.rs2_ptr);
        COL_WRITE_VALUE(row, Rv64StoreMultiByteAdapterCols, imm, record.imm);
        COL_WRITE_VALUE(row, Rv64StoreMultiByteAdapterCols, imm_sign, record.imm_sign);
        COL_WRITE_VALUE(row, Rv64StoreMultiByteAdapterCols, mem_as, record.mem_as);

        uint32_t ptr = rv64_store_effective_ptr(record);
        uint32_t ptr_limbs[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(ptr_limbs, ptr);
        COL_WRITE_VALUE(row, Rv64StoreMultiByteAdapterCols, mem_ptr_low_limb, ptr_limbs[0]);

        uint32_t shift_amount = rv64_store_shift_amount(record);
        uint32_t aligned_limb = ptr_limbs[0] - shift_amount;
        range_checker.add_count(aligned_limb >> 3, U16_BITS - 3);
        range_checker.add_count(ptr_limbs[1], pointer_max_bits - U16_BITS);

        uint32_t block1_low_sum = aligned_limb + uint32_t(MEMORY_BLOCK_BYTES);
        bool carry = crosses && block1_low_sum == (1u << U16_BITS);
        COL_WRITE_VALUE(row, Rv64StoreMultiByteAdapterCols, mem_ptr_carry, carry);
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
template <typename T> struct Rv64StoreByteAdapterCols {
    ExecutionState<T> from_state;
    T rs1_ptr;
    T rs1_data[RV64_PTR_U16_LIMBS];
    MemoryReadAuxCols<T> rs1_aux_cols;
    T rs2_ptr;
    MemoryReadAuxCols<T> read_data_aux;
    T imm;
    T imm_sign;
    T mem_ptr_low_limb;
    T mem_as;
    MemoryBaseAuxCols<T> write_base_aux;
};

struct Rv64StoreByteAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rs1_val;
    MemoryReadAuxRecord rs1_aux_record;
    MemoryReadAuxRecord read_data_aux;
    uint32_t write_prev_timestamp;
    uint16_t imm;
    uint8_t rs1_ptr;
    uint8_t rs2_ptr;
    bool imm_sign;
    uint8_t mem_as;
};

struct Rv64StoreByteAdapter {
    size_t pointer_max_bits;
    VariableRangeChecker range_checker;
    MemoryAuxColsFactory mem_helper;

    __device__ Rv64StoreByteAdapter(
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
        COL_WRITE_VALUE(row, Rv64StoreByteAdapterCols, from_state.pc, from_pc);
        COL_WRITE_VALUE(
            row, Rv64StoreByteAdapterCols, from_state.timestamp, from_timestamp
        );
        COL_WRITE_VALUE(row, Rv64StoreByteAdapterCols, rs1_ptr, rs1_ptr);

        Fp rs1_data[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(rs1_data, rs1_val);
        COL_WRITE_ARRAY(row, Rv64StoreByteAdapterCols, rs1_data, rs1_data);

        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64StoreByteAdapterCols, rs1_aux_cols)),
            rs1_prev_timestamp,
            from_timestamp
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64StoreByteAdapterCols, read_data_aux)),
            rs2_prev_timestamp,
            from_timestamp + 1
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64StoreByteAdapterCols, write_base_aux)),
            write_prev_timestamp,
            from_timestamp + 2
        );

        COL_WRITE_VALUE(row, Rv64StoreByteAdapterCols, rs2_ptr, rs2_ptr);
        COL_WRITE_VALUE(row, Rv64StoreByteAdapterCols, imm, imm);
        COL_WRITE_VALUE(row, Rv64StoreByteAdapterCols, imm_sign, imm_sign);
        COL_WRITE_VALUE(row, Rv64StoreByteAdapterCols, mem_as, mem_as);

        uint32_t ptr = rs1_val + uint32_t(imm) +
                       uint32_t(imm_sign) * (uint32_t(UINT16_MAX) << U16_BITS);
        uint32_t ptr_limbs[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(ptr_limbs, ptr);
        COL_WRITE_VALUE(row, Rv64StoreByteAdapterCols, mem_ptr_low_limb, ptr_limbs[0]);

        uint32_t shift_amount = ptr & (MEMORY_BLOCK_BYTES - 1);
        range_checker.add_count((ptr_limbs[0] - shift_amount) >> 3, U16_BITS - 3);
        range_checker.add_count(ptr_limbs[1], pointer_max_bits - U16_BITS);
    }

    __device__ void fill_trace_row(RowSlice row, Rv64StoreByteAdapterRecord record) {
        fill_trace_row(
            row,
            record.from_pc,
            record.from_timestamp,
            record.rs1_ptr,
            record.rs2_ptr,
            record.rs1_val,
            record.rs1_aux_record.prev_timestamp,
            record.read_data_aux.prev_timestamp,
            record.write_prev_timestamp,
            record.imm,
            record.imm_sign,
            record.mem_as
        );
    }
};
