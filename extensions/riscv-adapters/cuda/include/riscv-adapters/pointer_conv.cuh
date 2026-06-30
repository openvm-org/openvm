#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "system/memory/params.cuh"

// CUDA mirrors of the host-side pointer-conversion value helpers in
// `openvm_riscv_circuit::adapters` (see `extensions/riscv/circuit/src/adapters/mod.rs`).
// They convert a guest *byte* pointer into AS-native u16 *cell* pointer limbs and add a small
// per-block cell offset, returning the witness carries that the heap adapters store in their
// `*_cell_carry` / `*_add_carry` columns.

// Cell high-limb range-check bit width corresponding to a guest `byte_ptr_max_bits`.
__device__ __forceinline__ uint32_t cell_ptr_hi_bits(size_t byte_ptr_max_bits) {
    return uint32_t(byte_ptr_max_bits) - openvm::U16_CELL_SIZE_BITS - openvm::U16_BITS;
}

struct CellPtr {
    // Witness boolean carry, equal to `byte_hi & 1`.
    uint32_t carry;
    // AS-native u16 cell pointer limbs `[cell_lo, cell_hi]`.
    uint32_t limbs[2];
};

// Value form of `byte_ptr_limbs_to_cell_ptr_limbs_value`: given an aligned byte pointer's
// little-endian 16-bit limbs, returns `(carry, [cell_lo, cell_hi])`. The caller is responsible for
// range-checking `cell_hi` to `cell_ptr_hi_bits(...)`.
__device__ __forceinline__ CellPtr byte_ptr_limbs_to_cell_ptr_limbs_value(
    uint32_t byte_lo,
    uint32_t byte_hi
) {
    uint32_t carry = byte_hi & 1u;
    uint32_t cell_lo = (byte_lo + (carry << openvm::U16_BITS)) >> 1;
    uint32_t cell_hi = byte_hi >> 1;
    return CellPtr{carry, {cell_lo, cell_hi}};
}

// Value form of `add_const_u16_limbs_value`: adds a small `constant` (`< 2^16`) to a pointer given
// as little-endian 16-bit limbs `[lo, hi]`, carrying into the high limb. Returns `(carry, [new_lo,
// new_hi])`. The caller is responsible for range-checking `new_lo` to `U16_BITS`.
__device__ __forceinline__ CellPtr add_const_u16_limbs_value(
    uint32_t lo,
    uint32_t hi,
    uint32_t constant
) {
    uint32_t sum_lo = lo + constant;
    uint32_t carry = sum_lo >> openvm::U16_BITS;
    return CellPtr{carry, {sum_lo & 0xffffu, hi + carry}};
}

__device__ __forceinline__ void compute_block_add_carries(
    VariableRangeChecker &range_checker,
    uint32_t base_cell_lo,
    uint32_t num_blocks,
    uint32_t cell_stride,
    uint32_t *add_carry_out
) {
    for (uint32_t i = 0; i < num_blocks; i++) {
        uint32_t sum_lo = base_cell_lo + i * cell_stride;
        range_checker.add_count(sum_lo & 0xffffu, openvm::U16_BITS);
        add_carry_out[i] = sum_lo >> openvm::U16_BITS;
    }
}

// Returns the conversion carry; writes one add-carry per block into add_carry_out.
__device__ __forceinline__ uint32_t compute_pointer_carries(
    VariableRangeChecker &range_checker,
    uint32_t byte_ptr,
    size_t byte_ptr_max_bits,
    uint32_t num_blocks,
    uint32_t cell_stride,
    uint32_t *add_carry_out
) {
    CellPtr conv = byte_ptr_limbs_to_cell_ptr_limbs_value(
        byte_ptr & 0xffffu, byte_ptr >> openvm::U16_BITS
    );
    range_checker.add_count(conv.limbs[1], cell_ptr_hi_bits(byte_ptr_max_bits));
    compute_block_add_carries(
        range_checker, conv.limbs[0], num_blocks, cell_stride, add_carry_out
    );
    return conv.carry;
}
