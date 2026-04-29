#pragma once

#include <cstddef>

namespace riscv {
inline constexpr size_t RV64_REGISTER_NUM_LIMBS = 8;
inline constexpr size_t RV64_WORD_NUM_LIMBS = 4;
inline constexpr size_t RV64_CELL_BITS = 8;
} // namespace riscv

namespace program {
inline constexpr size_t PC_BITS = 30;
inline constexpr size_t DEFAULT_PC_STEP = 4;
inline constexpr size_t DEFAULT_BLOCK_SIZE = 8;
} // namespace program

namespace p3_keccak_air {
inline constexpr size_t NUM_ROUNDS = 24;
inline constexpr size_t BITS_PER_LIMB = 16;
inline constexpr size_t U64_LIMBS = 64 / BITS_PER_LIMB;
} // namespace p3_keccak_air

namespace keccak256 {
/// Total number of sponge bytes: number of rate bytes + number of capacity bytes.
inline constexpr size_t KECCAK_WIDTH_BYTES = 200;
/// Number of rate bytes.
inline constexpr size_t KECCAK_RATE_BYTES = 136;
/// Memory reads for the full state per row
inline constexpr size_t KECCAK_WIDTH_MEM_OPS = KECCAK_WIDTH_BYTES / program::DEFAULT_BLOCK_SIZE;
/// Memory reads for absorb per row
inline constexpr size_t KECCAK_RATE_MEM_OPS = KECCAK_RATE_BYTES / program::DEFAULT_BLOCK_SIZE;
} // namespace keccak256

namespace hintstore {
// Must match MAX_HINT_BUFFER_DWORDS_BITS in openvm_riscv_guest::lib.rs
inline constexpr size_t MAX_HINT_BUFFER_DWORDS_BITS = 10;
inline constexpr size_t MAX_HINT_BUFFER_DWORDS = (1 << MAX_HINT_BUFFER_DWORDS_BITS) - 1;
} // namespace hintstore
