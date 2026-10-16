#pragma once

#include <cstdint>

// RV64 base-register arithmetic shared by trace generation and replay.

// Sum of the low-u16 and bit-32 carries; matches the Rust adapter filler.
__device__ __forceinline__ uint32_t
base_address_carries(uint32_t base_low, uint16_t imm, bool imm_sign) {
    uint32_t carry0 = ((base_low & UINT16_MAX) + uint32_t(imm)) >> 16;
    uint32_t carry1 = ((base_low >> 16) + uint32_t(imm_sign) * UINT16_MAX + carry0) >> 16;
    return carry0 + carry1;
}

// RV64I adds the sign-extended immediate to the full register modulo 2^64.
__device__ __forceinline__ uint64_t base_register_add_imm(uint64_t base, int64_t signed_imm) {
    return base + static_cast<uint64_t>(signed_imm);
}

__device__ __forceinline__ uint64_t u16_block_to_u64(uint16_t const (&block)[4]) {
    return static_cast<uint64_t>(block[0]) | (static_cast<uint64_t>(block[1]) << 16) |
           (static_cast<uint64_t>(block[2]) << 32) | (static_cast<uint64_t>(block[3]) << 48);
}
