#pragma once

#include <cstddef>
#include <cstdint>

#include "fp.h"
#include "primitives/constants.h"
#include "system/memory/params.cuh"

namespace deferral {

using openvm::U16_BITS;

inline constexpr size_t DIGEST_SIZE = 8;
inline constexpr size_t F_NUM_BYTES = 4;
inline constexpr size_t COMMIT_NUM_BYTES = DIGEST_SIZE * F_NUM_BYTES;
inline constexpr size_t OUTPUT_LEN_NUM_BYTES = 8;
inline constexpr size_t OUTPUT_TOTAL_BYTES = COMMIT_NUM_BYTES + OUTPUT_LEN_NUM_BYTES;

// Memory-bus message counts for heap byte chunks.
inline constexpr size_t DIGEST_BYTE_MEMORY_OPS = DIGEST_SIZE / MEMORY_BLOCK_BYTES;
inline constexpr size_t COMMIT_MEMORY_OPS = COMMIT_NUM_BYTES / MEMORY_BLOCK_BYTES;
inline constexpr size_t OUTPUT_TOTAL_MEMORY_OPS = OUTPUT_TOTAL_BYTES / MEMORY_BLOCK_BYTES;

// Memory-bus message count for DEFERRAL_AS cell chunks.
inline constexpr size_t DIGEST_F_MEMORY_OPS = DIGEST_SIZE / BLOCK_FE_WIDTH;

inline constexpr uint32_t BABY_BEAR_ORDER = Fp::P;
inline constexpr uint8_t BABY_BEAR_ORDER_BE[F_NUM_BYTES] = {
    static_cast<uint8_t>((BABY_BEAR_ORDER >> 24) & 0xff),
    static_cast<uint8_t>((BABY_BEAR_ORDER >> 16) & 0xff),
    static_cast<uint8_t>((BABY_BEAR_ORDER >> 8) & 0xff),
    static_cast<uint8_t>(BABY_BEAR_ORDER & 0xff),
};

// log2(U16_CELL_SIZE); a u16 cell spans `U16_CELL_SIZE` bytes.
inline constexpr size_t U16_CELL_SIZE_BITS = 1;
static_assert(
    U16_CELL_SIZE == (size_t(1) << U16_CELL_SIZE_BITS),
    "U16_CELL_SIZE_BITS must be log2(U16_CELL_SIZE)"
);

// Maximum AS-native cell-pointer width; mirrors
// `openvm_circuit::system::memory::POINTER_MAX_BITS`.
inline constexpr size_t POINTER_MAX_BITS = 31;

// AS-native pointer helpers mirroring the host value-form helpers in
// `openvm_riscv_circuit::adapters`. Every memory-bus pointer is two little-endian 16-bit
// AS-native cell-pointer limbs `[lo16, hi16]`; these convert RV64 *byte* pointers (read from
// registers) into cell-pointer limbs without composing a full pointer into one field element.

// Converts an aligned RV64 byte pointer given as little-endian 16-bit limbs `[byte_lo, byte_hi]`
// into AS-native u16 *cell* pointer limbs (cell = byte / 2). Returns the conversion carry
// (= `byte_hi & 1`) and writes `cell_lo` / `cell_hi`. Mirrors
// `byte_ptr_limbs_to_cell_ptr_limbs_value`.
__device__ __forceinline__ uint32_t byte_ptr_limbs_to_cell_ptr_limbs(
    uint32_t byte_lo,
    uint32_t byte_hi,
    uint32_t &cell_lo,
    uint32_t &cell_hi
) {
    const uint32_t carry = byte_hi & 1u;
    cell_lo = (byte_lo + (carry << U16_BITS)) >> 1;
    cell_hi = byte_hi >> 1;
    return carry;
}

// Adds a small constant (`< 2^16`) to a pointer given as little-endian 16-bit limbs `[lo, hi]`,
// carrying into the high limb. Returns the carry and writes the new limbs. Mirrors
// `add_const_u16_limbs_value`.
__device__ __forceinline__ uint32_t add_const_u16_limbs(
    uint32_t lo,
    uint32_t hi,
    uint32_t constant,
    uint32_t &new_lo,
    uint32_t &new_hi
) {
    const uint32_t sum_lo = lo + constant;
    const uint32_t carry = sum_lo >> U16_BITS;
    new_lo = sum_lo & 0xffffu;
    new_hi = hi + carry;
    return carry;
}

// Cell high-limb range-check bit width for a guest `byte_ptr_max_bits`. Mirrors `cell_ptr_hi_bits`.
__device__ __forceinline__ size_t cell_ptr_hi_bits(size_t byte_ptr_max_bits) {
    return byte_ptr_max_bits - U16_CELL_SIZE_BITS - U16_BITS;
}

} // namespace deferral
