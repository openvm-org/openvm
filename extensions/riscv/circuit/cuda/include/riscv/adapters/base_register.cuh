#pragma once

#include <cstdint>

// CUDA mirror of `base_high_flags` and `base_register_add_imm` in
// `openvm_riscv_circuit::adapters` (see `extensions/riscv/circuit/src/adapters/mod.rs`).
//
// RV64I JALR, loads and stores accept any base register as long as
// `rs1 + sign_extend(imm)` (mod 2^64) is a valid address. With a 16-bit signed immediate and a
// 32-bit address space, the only upper words that can reach a valid address are 0, all ones (a
// small negative base) and one (a base just above 2^32).

static constexpr uint32_t BASE_HIGH_NEG = UINT32_MAX;
static constexpr uint32_t BASE_HIGH_ONE = 1u;

// Returns whether `base_high` is a reachable upper word and sets the AIR witnesses.
__device__ __forceinline__ bool base_high_flags(uint32_t base_high, bool &hi_neg, bool &hi_one) {
    hi_neg = base_high == BASE_HIGH_NEG;
    hi_one = base_high == BASE_HIGH_ONE;
    return base_high == 0 || hi_neg || hi_one;
}

__device__ __forceinline__ uint64_t base_register_add_imm(uint64_t base, int64_t signed_imm) {
    return base + static_cast<uint64_t>(signed_imm);
}

__device__ __forceinline__ uint64_t u16_block_to_u64(uint16_t const (&block)[4]) {
    return static_cast<uint64_t>(block[0]) | (static_cast<uint64_t>(block[1]) << 16) |
           (static_cast<uint64_t>(block[2]) << 32) | (static_cast<uint64_t>(block[3]) << 48);
}
