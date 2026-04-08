#pragma once

#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include <cstddef>
#include <cstdint>

namespace keccak256 {
using p3_keccak_air::NUM_ROUNDS;

__device__ __constant__ inline uint8_t R[5][5] = {
    {0, 36, 3, 41, 18},
    {1, 44, 10, 45, 2},
    {62, 6, 43, 15, 61},
    {28, 55, 25, 21, 56},
    {27, 20, 39, 8, 14},
};

__device__ __constant__ inline uint64_t RC[NUM_ROUNDS] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// In-place rho/pi permutation cycle (24 elements, flat index y*5+x).
// Follows (x,y) -> (y, 2x+3y mod 5); (0,0) is a fixed point.
// cycle[i] receives its value from cycle[(i+1) % 24] with the listed rotation.
__device__ __constant__ inline int RHO_PI_CYCLE_IDX[24] = {1,  6,  9,  22, 14, 20, 2,  12,
                                                           13, 19, 23, 15, 4,  24, 21, 8,
                                                           16, 5,  3,  18, 17, 11, 7,  10};
__device__ __constant__ inline uint8_t RHO_PI_CYCLE_ROT[24] = {44, 20, 61, 39, 18, 62, 43, 25,
                                                               8,  56, 41, 27, 14, 2,  55, 45,
                                                               36, 28, 21, 15, 10, 6,  3,  1};

// Single-round keccak-f body, operating on a flat 25-element state array.
// Marked __forceinline__ so callers control whether the round gets its own
// stack frame (__noinline__ wrapper) or folds into a multi-round loop.
__device__ __forceinline__ void keccakf_round_body(uint64_t *state, int round) {
    // Theta: C[x] = xor(A[x, 0..4])
    uint64_t c[5];
#pragma unroll 5
    for (int x = 0; x < 5; x++) {
        c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
    }
    // A'[x, y] = A[x, y] ^ D[x] where D[x] = C[x-1] ^ ROTL(C[x+1], 1).
    for (int x = 0; x < 5; x++) {
        uint64_t d = c[(x + 4) % 5] ^ rotl64(c[(x + 1) % 5], 1);
#pragma unroll 5
        for (int y = 0; y < 5; y++) {
            state[x + 5 * y] ^= d;
        }
    }

    // Rho/Pi in place via the 24-element permutation cycle (one temp).
    uint64_t temp = rotl64(state[RHO_PI_CYCLE_IDX[0]], RHO_PI_CYCLE_ROT[23]);
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

} // namespace keccak256

namespace p3_keccak_air {
using keccak256::R;
using keccak256::RC;
using keccak256::RHO_PI_CYCLE_IDX;
using keccak256::RHO_PI_CYCLE_ROT;

// Plonky3 KeccakCols structure (from p3_keccak_air)
// Must match exactly for trace compatibility
template <typename T> struct KeccakCols {
    // The `i`th value is set to 1 if we are in the `i`th round, otherwise 0.
    T step_flags[NUM_ROUNDS];

    // A register which indicates if a row should be exported, i.e. included in a multiset equality
    // argument. Should be 1 only for certain rows which are final steps, i.e. with
    // `step_flags[23] = 1`.
    T _export;

    // Permutation inputs, stored in y-major order.
    T preimage[5][5][U64_LIMBS];

    T a[5][5][U64_LIMBS];

    // C[x] = xor(A[x, 0], A[x, 1], A[x, 2], A[x, 3], A[x, 4])
    T c[5][64];

    // C'[x, z] = xor(C[x, z], C[x - 1, z], C[x + 1, z - 1])
    T c_prime[5][64];

    // A'[x, y] = xor(A[x, y], D[x])
    //          = xor(A[x, y], C[x - 1], ROT(C[x + 1], 1))
    T a_prime[5][5][64];

    // A''[x, y] = xor(B[x, y], andn(B[x + 1, y], B[x + 2, y])).
    T a_prime_prime[5][5][U64_LIMBS];

    // The bits of `A''[0, 0]`.
    T a_prime_prime_0_0_bits[64];

    // A'''[0, 0, z] = A''[0, 0, z] ^ RC[k, z]
    T a_prime_prime_prime_0_0_limbs[U64_LIMBS];
};

inline constexpr size_t NUM_KECCAK_COLS = sizeof(KeccakCols<uint8_t>);

// Apply one keccak-f round in-place without trace writes.
// Used by phase 1 to advance state between rounds.
// Thin __noinline__ wrapper around the shared round body so each round gets
// its own stack frame (keeps register pressure manageable in the caller's loop).
static __device__ __noinline__ void apply_round_in_place(
    uint32_t round,
    uint64_t current_state[5][5]
) {
    keccak256::keccakf_round_body(&current_state[0][0], round);
}

// tracegen matching plonky3
// `row` must have first NUM_KECCAK_COLS columns matching KeccakCols
static __device__ __noinline__ void generate_trace_row_for_round(
    RowSlice row,
    uint32_t round,
    uint64_t current_state[5][5]
) {
    COL_FILL_ZERO(row, KeccakCols, step_flags);
    COL_WRITE_VALUE(row, KeccakCols, step_flags[round], 1);

    // Populate C[x] = xor(A[x, 0], A[x, 1], A[x, 2], A[x, 3], A[x, 4]).
    uint64_t state_c[5];
#pragma unroll 5
    for (auto x = 0; x < 5; x++) {
        state_c[x] = current_state[0][x] ^ current_state[1][x] ^ current_state[2][x] ^
                     current_state[3][x] ^ current_state[4][x];
        COL_WRITE_BITS(row, KeccakCols, c[x], state_c[x]);
    }

    // Populate C'[x, z] and A'[x, y] using scalar d = C[x-1] ^ ROTL(C[x+1], 1).
    // Avoids materializing state_c_prime[5] array (~10 regs saved).
    for (int x = 0; x < 5; x++) {
        uint64_t d = state_c[(x + 4) % 5] ^ rotl64(state_c[(x + 1) % 5], 1);
        COL_WRITE_BITS(row, KeccakCols, c_prime[x], state_c[x] ^ d);

#pragma unroll 5
        for (int y = 0; y < 5; y++) {
            current_state[y][x] ^= d;
            COL_WRITE_BITS(row, KeccakCols, a_prime[y][x], current_state[y][x]);
        }
    }

    // In-place rho/pi using the 24-element permutation cycle.
    // Avoids allocating state_b[5][5] (~50 regs saved).
    uint64_t *flat_state = &current_state[0][0];
    uint64_t temp = rotl64(flat_state[RHO_PI_CYCLE_IDX[0]], RHO_PI_CYCLE_ROT[23]);
    // Prevent unrolling to avoid code bloat and register pressure from 23 simultaneous rotations.
#pragma unroll 1
    for (int i = 0; i < 23; i++) {
        flat_state[RHO_PI_CYCLE_IDX[i]] =
            rotl64(flat_state[RHO_PI_CYCLE_IDX[i + 1]], RHO_PI_CYCLE_ROT[i]);
    }
    flat_state[RHO_PI_CYCLE_IDX[23]] = temp;

    // Populate A'' = B[x,y] ^ (~B[x+1,y] & B[x+2,y]), in-place chi with 2 temps per row.
    for (int y = 0; y < 5; y++) {
        uint64_t t0 = current_state[y][0];
        uint64_t t1 = current_state[y][1];
        current_state[y][0] = t0 ^ ((~t1) & current_state[y][2]);
        current_state[y][1] = t1 ^ ((~current_state[y][2]) & current_state[y][3]);
        current_state[y][2] ^= (~current_state[y][3]) & current_state[y][4];
        current_state[y][3] ^= (~current_state[y][4]) & t0;
        current_state[y][4] ^= (~t0) & t1;
    }

    uint16_t *state_limbs = reinterpret_cast<uint16_t *>(&current_state[0][0]);
    COL_WRITE_ARRAY(row, KeccakCols, a_prime_prime, state_limbs);

    COL_WRITE_BITS(row, KeccakCols, a_prime_prime_0_0_bits, current_state[0][0]);

    // A''[0, 0] is additionally xor'd with RC.
    current_state[0][0] ^= RC[round];

    state_limbs = reinterpret_cast<uint16_t *>(&current_state[0][0]);
    COL_WRITE_ARRAY(row, KeccakCols, a_prime_prime_prime_0_0_limbs, state_limbs);
}
} // namespace p3_keccak_air
