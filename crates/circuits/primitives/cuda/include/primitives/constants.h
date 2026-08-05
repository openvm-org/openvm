#pragma once

#include <cstddef>
#include <cstdint>

namespace openvm {
inline constexpr size_t BYTE_BITS = 8;
inline constexpr size_t U16_BITS = 16;
// A u16 storage cell is 2 bytes wide, so byte pointers and u16-cell pointers convert by a
// shift: `byte_ptr = cell_ptr << U16_CELL_SIZE_BITS`.
inline constexpr size_t U16_CELL_SIZE_BITS = 1;
} // namespace openvm

namespace riscv {
using openvm::BYTE_BITS;
using openvm::U16_BITS;
using openvm::U16_CELL_SIZE_BITS;

inline constexpr size_t REGISTER_NUM_LIMBS = 8;
inline constexpr size_t WORD_NUM_LIMBS = 4;
inline constexpr uint32_t BYTE_MASK = (1u << BYTE_BITS) - 1;
inline constexpr size_t PTR_U16_LIMBS = WORD_NUM_LIMBS / 2;
inline constexpr size_t PTR_BITS = U16_BITS * PTR_U16_LIMBS;
inline constexpr size_t WORD_U16_LIMBS = WORD_NUM_LIMBS / 2;
inline constexpr size_t RV_IS_TYPE_IMM_BITS = 12;
} // namespace riscv

namespace program {
// Number of bits of a pc index (`pc / DEFAULT_PC_STEP`), the circuit representation of the
// program counter. Byte pcs span PC_BITS + PC_STEP_BITS = 32 bits.
inline constexpr size_t PC_BITS = 30;
inline constexpr size_t DEFAULT_PC_STEP = 4;
// log2 of DEFAULT_PC_STEP.
inline constexpr size_t PC_STEP_BITS = 2;
// Maximum allowed byte pc: the last DEFAULT_PC_STEP-aligned 32-bit address.
inline constexpr uint32_t MAX_ALLOWED_PC = UINT32_MAX - (DEFAULT_PC_STEP - 1);
// Low bits of a pc index packed into the low u16 limb of the corresponding byte pc.
inline constexpr size_t PC_IDX_LOW_BITS = openvm::U16_BITS - PC_STEP_BITS;
inline constexpr size_t DEFAULT_BLOCK_SIZE = 8;

// Converts a DEFAULT_PC_STEP-aligned byte pc into the pc index used by circuits.
__device__ __host__ inline constexpr uint32_t pc_to_idx(uint32_t pc) { return pc >> PC_STEP_BITS; }
} // namespace program

namespace p3_keccak_air {
inline constexpr size_t NUM_ROUNDS = 24;
inline constexpr size_t BITS_PER_LIMB = 16;
inline constexpr size_t U64_LIMBS = 64 / BITS_PER_LIMB;
} // namespace p3_keccak_air

namespace keccak256 {
/// Total number of sponge bytes: number of rate bytes + number of capacity bytes.
inline constexpr size_t KECCAK_WIDTH_BYTES = 200;
/// Total number of 16-bit limbs in the sponge.
inline constexpr size_t KECCAK_WIDTH_U16S = KECCAK_WIDTH_BYTES / 2;
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
