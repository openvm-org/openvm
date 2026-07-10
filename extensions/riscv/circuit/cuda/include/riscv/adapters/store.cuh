#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <typename T> struct Rv64StoreAdapterCols {
    ExecutionState<T> from_state;
    T rs1_ptr;
    T rs1_data[RV64_PTR_U16_LIMBS];
    MemoryReadAuxCols<T> rs1_aux_cols;
    T rs2_ptr;
    MemoryReadAuxCols<T> read_data_aux;
    T imm;
    T imm_sign;
    T mem_ptr_limbs[2];
    T mem_as;
    T mem_ptr_carry;
    MemoryBaseAuxCols<T> write_base_aux;
    MemoryBaseAuxCols<T> write1_base_aux;
};

struct Rv64StoreAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    uint32_t rs1_ptr;
    uint32_t rs1_val;
    MemoryReadAuxRecord rs1_aux_record;

    uint32_t rs2_ptr;
    MemoryReadAuxRecord read_data_aux;
    uint16_t imm;
    bool imm_sign;
    uint8_t mem_as;

    uint32_t write_prev_timestamp;
    // UINT32_MAX means the access does not cross a block boundary.
    uint32_t write1_prev_timestamp;
};

static __device__ __forceinline__ uint32_t
rv64_store_effective_ptr(Rv64StoreAdapterRecord record) {
    return record.rs1_val + uint32_t(record.imm) +
           uint32_t(record.imm_sign) * (uint32_t(UINT16_MAX) << U16_BITS);
}

static __device__ __forceinline__ uint32_t
rv64_store_shift_amount(Rv64StoreAdapterRecord record) {
    return rv64_store_effective_ptr(record) & (RV64_REGISTER_NUM_LIMBS - 1);
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

    __device__ void fill_trace_row(RowSlice row, Rv64StoreAdapterRecord record) {
        COL_WRITE_VALUE(row, Rv64StoreAdapterCols, from_state.pc, record.from_pc);
        COL_WRITE_VALUE(row, Rv64StoreAdapterCols, from_state.timestamp, record.from_timestamp);
        COL_WRITE_VALUE(row, Rv64StoreAdapterCols, rs1_ptr, record.rs1_ptr);

        Fp rs1_data[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(rs1_data, record.rs1_val);
        COL_WRITE_ARRAY(row, Rv64StoreAdapterCols, rs1_data, rs1_data);

        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64StoreAdapterCols, rs1_aux_cols)),
            record.rs1_aux_record.prev_timestamp,
            record.from_timestamp
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64StoreAdapterCols, read_data_aux)),
            record.read_data_aux.prev_timestamp,
            record.from_timestamp + 1
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64StoreAdapterCols, write_base_aux)),
            record.write_prev_timestamp,
            record.from_timestamp + 2
        );
        bool crosses = record.write1_prev_timestamp != UINT32_MAX;
        if (crosses) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Rv64StoreAdapterCols, write1_base_aux)),
                record.write1_prev_timestamp,
                record.from_timestamp + 3
            );
        } else {
            mem_helper.fill_zero(row.slice_from(COL_INDEX(Rv64StoreAdapterCols, write1_base_aux))
            );
        }

        COL_WRITE_VALUE(row, Rv64StoreAdapterCols, rs2_ptr, record.rs2_ptr);
        COL_WRITE_VALUE(row, Rv64StoreAdapterCols, imm, record.imm);
        COL_WRITE_VALUE(row, Rv64StoreAdapterCols, imm_sign, record.imm_sign);
        COL_WRITE_VALUE(row, Rv64StoreAdapterCols, mem_as, record.mem_as);

        uint32_t ptr = rv64_store_effective_ptr(record);
        uint32_t ptr_limbs[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(ptr_limbs, ptr);
        COL_WRITE_ARRAY(row, Rv64StoreAdapterCols, mem_ptr_limbs, ptr_limbs);

        uint32_t shift_amount = rv64_store_shift_amount(record);
        uint32_t aligned_limb0 = ptr_limbs[0] - shift_amount;
        range_checker.add_count(aligned_limb0 >> 3, U16_BITS - 3);
        range_checker.add_count(ptr_limbs[1], pointer_max_bits - U16_BITS);

        bool carry = crosses && (aligned_limb0 + 8 == (1u << U16_BITS));
        COL_WRITE_VALUE(row, Rv64StoreAdapterCols, mem_ptr_carry, carry);
        if (crosses) {
            range_checker.add_count(
                (aligned_limb0 + 8 - (uint32_t(carry) << U16_BITS)) >> 3, U16_BITS - 3
            );
            range_checker.add_count(ptr_limbs[1] + carry, pointer_max_bits - U16_BITS);
        }
    }
};
