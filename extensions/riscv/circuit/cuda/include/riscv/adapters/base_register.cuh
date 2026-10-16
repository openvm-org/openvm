#pragma once

#include <cstdint>

// CUDA mirror of base-register validation and addition in
// `openvm_riscv_circuit::adapters` (see `extensions/riscv/circuit/src/adapters/mod.rs`).
//
// RV64I JALR, loads and stores accept any base register as long as
// `rs1 + sign_extend(imm)` (mod 2^64) is a valid address. With a 16-bit signed immediate and a
// 32-bit address space, the only upper words that can reach a valid address are 0, all ones (a
// small negative base) and one (a base just above 2^32).

static constexpr uint32_t BASE_HIGH_NEG = UINT32_MAX;
static constexpr uint32_t BASE_HIGH_ONE = 1u;

__device__ __forceinline__ bool valid_base_high(uint32_t base_high) {
    return base_high == 0 || base_high == BASE_HIGH_NEG || base_high == BASE_HIGH_ONE;
}

// Sum of the low-u16 and bit-32 carries; matches the Rust adapter filler.
__device__ __forceinline__ uint32_t
base_address_carries(uint32_t base_low, uint16_t imm, bool imm_sign) {
    uint32_t carry0 = ((base_low & UINT16_MAX) + uint32_t(imm)) >> 16;
    uint32_t carry1 = ((base_low >> 16) + uint32_t(imm_sign) * UINT16_MAX + carry0) >> 16;
    return carry0 + carry1;
}

__device__ __forceinline__ uint64_t base_register_add_imm(uint64_t base, int64_t signed_imm) {
    return base + static_cast<uint64_t>(signed_imm);
}

__device__ __forceinline__ uint64_t u16_block_to_u64(uint16_t const (&block)[4]) {
    return static_cast<uint64_t>(block[0]) | (static_cast<uint64_t>(block[1]) << 16) |
           (static_cast<uint64_t>(block[2]) << 32) | (static_cast<uint64_t>(block[3]) << 48);
}
