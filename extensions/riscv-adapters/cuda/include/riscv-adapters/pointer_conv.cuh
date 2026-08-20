#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "system/memory/params.cuh"

// CUDA mirrors of the host-side pointer-conversion value helpers in
// `openvm_riscv_circuit::adapters` (see `extensions/riscv/circuit/src/adapters/mod.rs`).
// They convert a guest *byte* pointer into AS-native u16 *cell* pointer limbs, returning the
// witness carry that the heap adapters store in their `*_cell_carry` columns.

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

// Bit width of the block-alignment range check on a byte pointer's low byte. Mirrors
// `BYTE_PTR_ALIGN_BITS` in `openvm_riscv_circuit::adapters`.
inline constexpr size_t BYTE_PTR_ALIGN_BITS = openvm::BYTE_BITS - 3;
static_assert(MEMORY_BLOCK_BYTES == 1 << 3);

// Mirrors `compute_aligned_pointer_carry` in the deferral circuit: returns the conversion
// carry, registering the block-alignment and high-limb range-check counts.
__device__ __forceinline__ uint32_t compute_aligned_pointer_carry(
    VariableRangeChecker &range_checker,
    uint32_t byte_ptr,
    size_t byte_ptr_max_bits
) {
    range_checker.add_count((byte_ptr & 0xffu) / MEMORY_BLOCK_BYTES, BYTE_PTR_ALIGN_BITS);
    CellPtr conv = byte_ptr_limbs_to_cell_ptr_limbs_value(
        byte_ptr & 0xffffu, byte_ptr >> openvm::U16_BITS
    );
    range_checker.add_count(conv.limbs[1], cell_ptr_hi_bits(byte_ptr_max_bits));
    return conv.carry;
}

// Returns the conversion carry, registering the high-limb range-check count.
__device__ __forceinline__ uint32_t compute_pointer_carry(
    VariableRangeChecker &range_checker,
    uint32_t byte_ptr,
    size_t byte_ptr_max_bits
) {
    CellPtr conv = byte_ptr_limbs_to_cell_ptr_limbs_value(
        byte_ptr & 0xffffu, byte_ptr >> openvm::U16_BITS
    );
    range_checker.add_count(conv.limbs[1], cell_ptr_hi_bits(byte_ptr_max_bits));
    return conv.carry;
}
