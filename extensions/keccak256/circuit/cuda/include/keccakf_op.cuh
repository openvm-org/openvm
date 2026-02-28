#pragma once

#include "p3_keccakf.cuh"
#include "primitives/constants.h"
#include "system/memory/offline_checker.cuh"

#include <cstddef>
#include <cstdint>

namespace keccakf_op {
using namespace riscv;
using namespace keccak256;

inline constexpr size_t KECCAK_WIDTH_WORDS = KECCAK_WIDTH_BYTES / RV32_REGISTER_NUM_LIMBS;
inline constexpr size_t NUM_OP_ROWS_PER_INS = 1; // 1 row per instruction

// Record structure matching Rust KeccakfRecord (from trace.rs)
// Must match exact layout with #[repr(C)]
struct KeccakfOpRecord {
    uint32_t pc;
    uint32_t timestamp;
    uint32_t rd_ptr;
    uint32_t buffer_ptr;
    MemoryReadAuxRecord rd_aux;
    MemoryReadAuxRecord buffer_word_aux[KECCAK_WIDTH_WORDS];
    uint8_t preimage_buffer_bytes[KECCAK_WIDTH_BYTES];
};

// Column structure matching Rust KeccakfOpCols (from columns.rs)
template <typename T> struct KeccakfOpCols {
    T pc;
    T is_valid;
    T timestamp;
    T rd_ptr;
    T buffer_ptr_limbs[RV32_REGISTER_NUM_LIMBS]; // 4 limbs
    T preimage[KECCAK_WIDTH_BYTES];              // 200 bytes
    T postimage[KECCAK_WIDTH_BYTES];             // 200 bytes
    MemoryReadAuxCols<T> rd_aux;
    MemoryBaseAuxCols<T> buffer_word_aux[KECCAK_WIDTH_WORDS]; // 50 words
};

inline constexpr size_t NUM_KECCAKF_OP_COLS = sizeof(KeccakfOpCols<uint8_t>);

// Helper to rotate left a 64-bit value
__device__ __forceinline__ uint64_t rotl64(uint64_t x, int n) { return (x << n) | (x >> (64 - n)); }

// Compute keccak-f permutation on a 200-byte state
__device__ __forceinline__ void keccakf_permutation(uint64_t state[25]) {
    for (int round = 0; round < 24; round++) {
        // Theta
        uint64_t c[5];
        for (int x = 0; x < 5; x++) {
            c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        for (int x = 0; x < 5; x++) {
            uint64_t d = c[(x + 4) % 5] ^ rotl64(c[(x + 1) % 5], 1);
            for (int y = 0; y < 5; y++) {
                state[x + 5 * y] ^= d;
            }
        }

        // Rho and Pi
        uint64_t temp[25];
        for (int y = 0; y < 5; y++) {
            for (int x = 0; x < 5; x++) {
                temp[y + 5 * ((2 * x + 3 * y) % 5)] = rotl64(state[x + 5 * y], R[x][y]);
            }
        }

        // Chi
        for (int y = 0; y < 5; y++) {
            for (int x = 0; x < 5; x++) {
                state[x + 5 * y] =
                    temp[x + 5 * y] ^ ((~temp[(x + 1) % 5 + 5 * y]) & temp[(x + 2) % 5 + 5 * y]);
            }
        }

        // Iota
        state[0] ^= RC[round];
    }
}

} // namespace keccakf_op
