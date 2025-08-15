#pragma once

#include <cstddef>

namespace riscv {
static const size_t RV32_REGISTER_NUM_LIMBS = 4;
static const size_t RV32_CELL_BITS = 8;
static const size_t RV_J_TYPE_IMM_BITS = 21;

static const size_t RV32_IMM_AS = 0;
} // namespace riscv

namespace program {
static const size_t PC_BITS = 30;
static const size_t DEFAULT_PC_STEP = 4;
} // namespace program

namespace native {
static const size_t AS_IMMEDIATE = 0;
static const size_t AS_NATIVE = 4;
static const size_t EXT_DEG = 4;
static const size_t BETA = 11;
} // namespace native

namespace poseidon2 {
static const size_t CHUNK = 8;
} // namespace poseidon2

namespace p3_keccak_air {
static const size_t NUM_ROUNDS = 24;
static const size_t BITS_PER_LIMB = 16;
static const size_t U64_LIMBS = 64 / BITS_PER_LIMB;
static const size_t RATE_BITS = 1088;
static const size_t RATE_LIMBS = RATE_BITS / BITS_PER_LIMB;
} // namespace p3_keccak_air

namespace keccak256 {
/// Total number of sponge bytes: number of rate bytes + number of capacity bytes.
static const size_t KECCAK_WIDTH_BYTES = 200;
/// Total number of 16-bit limbs in the sponge.
static const size_t KECCAK_WIDTH_U16S = KECCAK_WIDTH_BYTES / 2;
/// Number of rate bytes.
static const size_t KECCAK_RATE_BYTES = 136;
/// Number of 16-bit rate limbs.
static const size_t KECCAK_RATE_U16S = KECCAK_RATE_BYTES / 2;
/// Number of absorb rounds, equal to rate in u64s.
static const size_t NUM_ABSORB_ROUNDS = KECCAK_RATE_BYTES / 8;
/// Number of capacity bytes.
static const size_t KECCAK_CAPACITY_BYTES = 64;
/// Number of 16-bit capacity limbs.
static const size_t KECCAK_CAPACITY_U16S = KECCAK_CAPACITY_BYTES / 2;
/// Number of output digest bytes used during the squeezing phase.
static const size_t KECCAK_DIGEST_BYTES = 32;
/// Number of 64-bit digest limbs.
static const size_t KECCAK_DIGEST_U64S = KECCAK_DIGEST_BYTES / 8;

// ==== Constants for register/memory adapter ====
/// Register reads to get dst, src, len
static const size_t KECCAK_REGISTER_READS = 3;
/// Number of cells to read/write in a single memory access
static const size_t KECCAK_WORD_SIZE = 4;
/// Memory reads for absorb per row
static const size_t KECCAK_ABSORB_READS = KECCAK_RATE_BYTES / KECCAK_WORD_SIZE;
/// Memory writes for digest per row
static const size_t KECCAK_DIGEST_WRITES = KECCAK_DIGEST_BYTES / KECCAK_WORD_SIZE;
/// keccakf parameters
static const size_t KECCAK_ROUND = 24;
static const size_t KECCAK_STATE_SIZE = 25;
static const size_t KECCAK_Q_SIZE = 192;
/// From memory config
static const size_t KECCAK_POINTER_MAX_BITS = 29;
} // namespace keccak256

namespace mod_builder {
static const size_t MAX_LIMBS = 97;
} // namespace mod_builder

namespace sha256 {
static const size_t SHA256_BLOCK_BITS = 512;
static const size_t SHA256_BLOCK_U8S = 64;
static const size_t SHA256_BLOCK_WORDS = 16;
static const size_t SHA256_WORD_U8S = 4;
static const size_t SHA256_WORD_BITS = 32;
static const size_t SHA256_WORD_U16S = 2;
static const size_t SHA256_HASH_WORDS = 8;
static const size_t SHA256_NUM_READ_ROWS = 4;
static const size_t SHA256_ROWS_PER_BLOCK = 17;
static const size_t SHA256_ROUNDS_PER_ROW = 4;
static const size_t SHA256_ROW_VAR_CNT = 5;
static const size_t SHA256_REGISTER_READS = 3;
static const size_t SHA256_READ_SIZE = 16;
static const size_t SHA256_WRITE_SIZE = 32;
} // namespace sha256