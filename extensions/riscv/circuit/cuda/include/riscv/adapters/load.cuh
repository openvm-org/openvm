#pragma once

#include "primitives/execution.h"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "riscv-adapters/pointer_conv.cuh"
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
    // Carry (`byte_hi & 1`) for converting the aligned heap *byte* pointer into AS-native u16
    // *cell* pointer limbs.
    T mem_ptr_carry;
    // Carry into the high cell limb when adding the block stride (in u16 cells) to the first
    // block's cell pointer to address the second block.
    T block1_add_carry;
    MemoryWriteAuxCols<T, BLOCK_FE_WIDTH> write_aux;
    T needs_write;
};

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
        // Alignment check on the aligned low byte limb: `aligned_limb / 8 < 2^13`.
        range_checker.add_count(aligned_limb >> 3, U16_BITS - 3);
        // Byte -> cell pointer conversion for the first block; the AIR range-checks `cell_hi`
        // with `enabled = is_valid`.
        CellPtr mem_cell = byte_ptr_limbs_to_cell_ptr_limbs_value(aligned_limb, ptr_limbs[1]);
        COL_WRITE_VALUE(row, Rv64LoadMultiByteAdapterCols, mem_ptr_carry, mem_cell.carry);
        range_checker.add_count(mem_cell.limbs[1], cell_ptr_hi_bits(pointer_max_bits));
        // Second-block cell pointer carry and low-limb range check (AIR `enabled = cross`).
        if (crosses) {
            CellPtr block1_cell = add_const_u16_limbs_value(
                mem_cell.limbs[0],
                mem_cell.limbs[1],
                uint32_t(MEMORY_BLOCK_BYTES / U16_CELL_SIZE)
            );
            COL_WRITE_VALUE(
                row, Rv64LoadMultiByteAdapterCols, block1_add_carry, block1_cell.carry
            );
            range_checker.add_count(block1_cell.limbs[0], U16_BITS);
        } else {
            COL_WRITE_VALUE(row, Rv64LoadMultiByteAdapterCols, block1_add_carry, 0);
        }
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
    // Carry (`byte_hi & 1`) for converting the aligned heap *byte* pointer into AS-native u16
    // *cell* pointer limbs.
    T mem_ptr_carry;
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
        uint32_t aligned_limb = ptr_limbs[0] - shift_amount;
        // Alignment check on the aligned low byte limb: `aligned_limb / 8 < 2^13`.
        range_checker.add_count(aligned_limb >> 3, U16_BITS - 3);
        // Byte -> cell pointer conversion for the heap block; the AIR range-checks `cell_hi`
        // with `enabled = is_valid`.
        CellPtr mem_cell = byte_ptr_limbs_to_cell_ptr_limbs_value(aligned_limb, ptr_limbs[1]);
        COL_WRITE_VALUE(row, Rv64LoadByteAdapterCols, mem_ptr_carry, mem_cell.carry);
        range_checker.add_count(mem_cell.limbs[1], cell_ptr_hi_bits(pointer_max_bits));
    }
};
