#pragma once

#include "primitives/constants.h"
#include "system/memory/params.cuh"

using namespace riscv;

// CUDA mirrors of the host-side pointer-conversion value helpers in
// `openvm_riscv_circuit::adapters` (see `extensions/riscv/circuit/src/adapters/mod.rs`).
// They convert a guest *byte* pointer into AS-native u16 *cell* pointer limbs and add a small
// per-block cell offset, returning the witness carries that the heap adapters store in their
// `*_cell_carry` / `*_add_carry` columns.

// Cell high-limb range-check bit width corresponding to a guest `byte_ptr_max_bits`.
__device__ __forceinline__ uint32_t cell_ptr_hi_bits(size_t byte_ptr_max_bits) {
    return uint32_t(byte_ptr_max_bits) - U16_CELL_SIZE_BITS - U16_BITS;
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
    uint32_t cell_lo = (byte_lo + (carry << U16_BITS)) >> 1;
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
    uint32_t carry = sum_lo >> U16_BITS;
    return CellPtr{carry, {sum_lo & 0xffffu, hi + carry}};
}
