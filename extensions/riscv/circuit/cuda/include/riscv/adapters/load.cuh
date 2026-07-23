#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "system/memory/controller.cuh"
#include "system/memory/offline_checker.cuh"

using namespace riscv;

template <typename T> struct Rv64LoadMultiByteAdapterCols {
    ExecutionState<T> from_state;
    T rs1_ptr;
    T rs1_data[RV64_PTR_U16_LIMBS];
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

struct Rv64LoadMultiByteAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;

    uint32_t rs1_val;
    MemoryReadAuxRecord rs1_aux_record;

    // The second timestamp is UINT32_MAX when the access does not cross a block boundary.
    MemoryReadAuxRecord read_data_aux[2];
    uint16_t imm;
    bool imm_sign;

    uint32_t write_prev_timestamp;
    uint16_t write_prev_data[BLOCK_FE_WIDTH];
    uint8_t rs1_ptr;
    // UINT8_MAX means the load does not write a register.
    uint8_t rd_ptr;
};

template <typename Record>
static __device__ __forceinline__ uint32_t rv64_load_effective_ptr(Record record) {
    return record.rs1_val + uint32_t(record.imm) +
           uint32_t(record.imm_sign) * (uint32_t(UINT16_MAX) << U16_BITS);
}

template <typename Record>
static __device__ __forceinline__ uint32_t rv64_load_shift_amount(Record record) {
    return rv64_load_effective_ptr(record) & (MEMORY_BLOCK_BYTES - 1);
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
        COL_WRITE_VALUE(row, Rv64LoadMultiByteAdapterCols, from_state.pc, from_pc);
        COL_WRITE_VALUE(row, Rv64LoadMultiByteAdapterCols, from_state.timestamp, from_timestamp);
        COL_WRITE_VALUE(row, Rv64LoadMultiByteAdapterCols, rs1_ptr, rs1_ptr);

        Fp rs1_data[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(rs1_data, rs1_val);
        COL_WRITE_ARRAY(row, Rv64LoadMultiByteAdapterCols, rs1_data, rs1_data);

        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64LoadMultiByteAdapterCols, rs1_aux_cols)),
            rs1_prev_timestamp,
            from_timestamp
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64LoadMultiByteAdapterCols, read_data_aux[0])),
            block0_prev_timestamp,
            from_timestamp + 1
        );
        if (crosses) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Rv64LoadMultiByteAdapterCols, read_data_aux[1])),
                block1_prev_timestamp,
                from_timestamp + 2
            );
        } else {
            mem_helper.fill_zero(row.slice_from(COL_INDEX(Rv64LoadMultiByteAdapterCols, read_data_aux[1])));
        }

        COL_WRITE_VALUE(row, Rv64LoadMultiByteAdapterCols, rd_ptr, needs_write ? rd_ptr : 0);
        COL_WRITE_VALUE(row, Rv64LoadMultiByteAdapterCols, needs_write, needs_write);
        if (needs_write) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Rv64LoadMultiByteAdapterCols, write_aux.base)),
                write_prev_timestamp,
                from_timestamp + 3
            );
            Fp prev_data[BLOCK_FE_WIDTH];
            copy_u16_cells(prev_data, write_prev_data);
            COL_WRITE_ARRAY(row, Rv64LoadMultiByteAdapterCols, write_aux.prev_data, prev_data);
        } else {
            mem_helper.fill_zero(row.slice_from(COL_INDEX(Rv64LoadMultiByteAdapterCols, write_aux.base)));
            row.fill_zero(COL_INDEX(Rv64LoadMultiByteAdapterCols, write_aux.prev_data), BLOCK_FE_WIDTH);
        }

        COL_WRITE_VALUE(row, Rv64LoadMultiByteAdapterCols, imm, imm);
        COL_WRITE_VALUE(row, Rv64LoadMultiByteAdapterCols, imm_sign, imm_sign);

        uint32_t ptr = rs1_val + uint32_t(imm) +
                       uint32_t(imm_sign) * (uint32_t(UINT16_MAX) << U16_BITS);
        uint32_t ptr_limbs[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(ptr_limbs, ptr);
        COL_WRITE_VALUE(row, Rv64LoadMultiByteAdapterCols, mem_ptr_low_limb, ptr_limbs[0]);

        uint32_t shift_amount = ptr & (MEMORY_BLOCK_BYTES - 1);
        uint32_t aligned_limb = ptr_limbs[0] - shift_amount;
        range_checker.add_count(aligned_limb >> 3, U16_BITS - 3);
        range_checker.add_count(ptr_limbs[1], pointer_max_bits - U16_BITS);

        uint32_t block1_low_sum = aligned_limb + uint32_t(MEMORY_BLOCK_BYTES);
        bool carry = crosses && block1_low_sum == (1u << U16_BITS);
        COL_WRITE_VALUE(row, Rv64LoadMultiByteAdapterCols, mem_ptr_carry, carry);
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

    __device__ void fill_trace_row(RowSlice row, Rv64LoadMultiByteAdapterRecord record) {
        bool crosses = record.read_data_aux[1].prev_timestamp != UINT32_MAX;
        bool needs_write = record.rd_ptr != UINT8_MAX;
        fill_trace_row(
            row,
            record.from_pc,
            record.from_timestamp,
            record.rs1_ptr,
            needs_write ? record.rd_ptr : 0,
            needs_write,
            record.rs1_val,
            record.rs1_aux_record.prev_timestamp,
            record.read_data_aux[0].prev_timestamp,
            crosses ? record.read_data_aux[1].prev_timestamp : 0,
            crosses,
            needs_write ? record.write_prev_timestamp : 0,
            record.write_prev_data,
            record.imm,
            record.imm_sign
        );
    }
};

// Byte loads use one memory block and need no crossing-related trace columns.
template <typename T> struct Rv64LoadByteAdapterCols {
    ExecutionState<T> from_state;
    T rs1_ptr;
    T rs1_data[RV64_PTR_U16_LIMBS];
    MemoryReadAuxCols<T> rs1_aux_cols;
    T rd_ptr;
    MemoryReadAuxCols<T> read_data_aux;
    T imm;
    T imm_sign;
    T mem_ptr_low_limb;
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> write_aux;
    T needs_write;
};

struct Rv64LoadByteAdapterRecord {
    uint32_t from_pc;
    uint32_t from_timestamp;
    uint32_t rs1_val;
    MemoryReadAuxRecord rs1_aux_record;
    MemoryReadAuxRecord read_data_aux;
    uint16_t imm;
    bool imm_sign;
    uint32_t write_prev_timestamp;
    uint16_t write_prev_data[BLOCK_FE_WIDTH];
    uint8_t rs1_ptr;
    // UINT8_MAX means the load does not write a register.
    uint8_t rd_ptr;
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
        COL_WRITE_VALUE(row, Rv64LoadByteAdapterCols, from_state.pc, from_pc);
        COL_WRITE_VALUE(
            row, Rv64LoadByteAdapterCols, from_state.timestamp, from_timestamp
        );
        COL_WRITE_VALUE(row, Rv64LoadByteAdapterCols, rs1_ptr, rs1_ptr);

        Fp rs1_data[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(rs1_data, rs1_val);
        COL_WRITE_ARRAY(row, Rv64LoadByteAdapterCols, rs1_data, rs1_data);

        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64LoadByteAdapterCols, rs1_aux_cols)),
            rs1_prev_timestamp,
            from_timestamp
        );
        mem_helper.fill(
            row.slice_from(COL_INDEX(Rv64LoadByteAdapterCols, read_data_aux)),
            read_prev_timestamp,
            from_timestamp + 1
        );

        COL_WRITE_VALUE(row, Rv64LoadByteAdapterCols, rd_ptr, needs_write ? rd_ptr : 0);
        COL_WRITE_VALUE(row, Rv64LoadByteAdapterCols, needs_write, needs_write);
        if (needs_write) {
            mem_helper.fill(
                row.slice_from(COL_INDEX(Rv64LoadByteAdapterCols, write_aux.base)),
                write_prev_timestamp,
                from_timestamp + 2
            );
            Fp prev_data[BLOCK_FE_WIDTH];
            copy_u16_cells(prev_data, write_prev_data);
            COL_WRITE_ARRAY(row, Rv64LoadByteAdapterCols, write_aux.prev_data, prev_data);
        } else {
            mem_helper.fill_zero(
                row.slice_from(COL_INDEX(Rv64LoadByteAdapterCols, write_aux.base))
            );
            row.fill_zero(COL_INDEX(Rv64LoadByteAdapterCols, write_aux.prev_data), BLOCK_FE_WIDTH);
        }

        COL_WRITE_VALUE(row, Rv64LoadByteAdapterCols, imm, imm);
        COL_WRITE_VALUE(row, Rv64LoadByteAdapterCols, imm_sign, imm_sign);

        uint32_t ptr = rs1_val + uint32_t(imm) +
                       uint32_t(imm_sign) * (uint32_t(UINT16_MAX) << U16_BITS);
        uint32_t ptr_limbs[RV64_PTR_U16_LIMBS];
        ptr_to_u16_limbs(ptr_limbs, ptr);
        COL_WRITE_VALUE(row, Rv64LoadByteAdapterCols, mem_ptr_low_limb, ptr_limbs[0]);

        uint32_t shift_amount = ptr & (MEMORY_BLOCK_BYTES - 1);
        range_checker.add_count((ptr_limbs[0] - shift_amount) >> 3, U16_BITS - 3);
        range_checker.add_count(ptr_limbs[1], pointer_max_bits - U16_BITS);
    }

    __device__ void fill_trace_row(RowSlice row, Rv64LoadByteAdapterRecord record) {
        bool needs_write = record.rd_ptr != UINT8_MAX;
        fill_trace_row(
            row,
            record.from_pc,
            record.from_timestamp,
            record.rs1_ptr,
            needs_write ? record.rd_ptr : 0,
            needs_write,
            record.rs1_val,
            record.rs1_aux_record.prev_timestamp,
            record.read_data_aux.prev_timestamp,
            needs_write ? record.write_prev_timestamp : 0,
            record.write_prev_data,
            record.imm,
            record.imm_sign
        );
    }
};
