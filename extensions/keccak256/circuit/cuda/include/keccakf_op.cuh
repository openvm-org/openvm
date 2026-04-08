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

// Helper to rotate left a 64-bit value.
// Guard: when n == 0, (x >> 64) is undefined behavior per C++ standard.
// R[0][0] == 0 in the Keccak rho step, so this path is reachable.
__device__ __forceinline__ uint64_t rotl64(uint64_t x, int n) {
    n &= 63;
    return n ? ((x << n) | (x >> (64 - n))) : x;
}

// Compute keccak-f permutation on a 200-byte state.
//
// Uses in-place rho/pi via the 24-element permutation cycle and in-place chi
// with two temporaries per row, so no scratch array is needed.
__device__ __forceinline__ void keccakf_permutation(uint64_t state[25]) {
    using keccak256::RHO_PI_CYCLE_IDX;
    using keccak256::RHO_PI_CYCLE_ROT;

    for (int round = 0; round < 24; round++) {
        // Theta: C[x] = xor(A[x, 0..4])
        uint64_t c[5];
#pragma unroll 5
        for (int x = 0; x < 5; x++) {
            c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        // A'[x, y] = A[x, y] ^ D[x] where D[x] = C[x-1] ^ ROTL(C[x+1], 1).
        // Use a scalar `d` instead of materializing D[5].
        for (int x = 0; x < 5; x++) {
            uint64_t d = c[(x + 4) % 5] ^ rotl64(c[(x + 1) % 5], 1);
#pragma unroll 5
            for (int y = 0; y < 5; y++) {
                state[x + 5 * y] ^= d;
            }
        }

        // Rho/Pi in place via the 24-element permutation cycle (one temp).
        uint64_t temp = rotl64(state[RHO_PI_CYCLE_IDX[0]], RHO_PI_CYCLE_ROT[23]);
        // Prevent unrolling to avoid 23 simultaneous rotations in flight.
#pragma unroll 1
        for (int i = 0; i < 23; i++) {
            state[RHO_PI_CYCLE_IDX[i]] =
                rotl64(state[RHO_PI_CYCLE_IDX[i + 1]], RHO_PI_CYCLE_ROT[i]);
        }
        state[RHO_PI_CYCLE_IDX[23]] = temp;

        // Chi in place with 2 temps per row.
        for (int y = 0; y < 5; y++) {
            uint64_t *row_state = &state[5 * y];
            uint64_t t0 = row_state[0];
            uint64_t t1 = row_state[1];
            row_state[0] = t0 ^ ((~t1) & row_state[2]);
            row_state[1] = t1 ^ ((~row_state[2]) & row_state[3]);
            row_state[2] ^= (~row_state[3]) & row_state[4];
            row_state[3] ^= (~row_state[4]) & t0;
            row_state[4] ^= (~t0) & t1;
        }

        // Iota
        state[0] ^= RC[round];
    }
}

} // namespace keccakf_op
