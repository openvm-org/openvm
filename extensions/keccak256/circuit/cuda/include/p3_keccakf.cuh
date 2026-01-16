#pragma once

#include "primitives/constants.h"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include <cstddef>
#include <cstdint>

namespace p3_keccak_air {

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

__device__ __constant__ uint8_t R[5][5] = {
    {0, 36, 3, 41, 18},
    {1, 44, 10, 45, 2},
    {62, 6, 43, 15, 61},
    {28, 55, 25, 21, 56},
    {27, 20, 39, 8, 14},
};

__device__ __constant__ uint64_t RC[NUM_ROUNDS] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

// tracegen matching plonky3
// `row` must have first NUM_KECCAK_COLS columns matching KeccakCols
__device__ void generate_trace_row_for_round(
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

    // Populate C'[x, z] = xor(C[x, z], C[x - 1, z], ROTL1(C[x + 1, z - 1])).
    uint64_t state_c_prime[5];
#pragma unroll 5
    for (auto x = 0; x < 5; x++) {
        state_c_prime[x] = state_c[x] ^ state_c[(x + 4) % 5] ^ ROTL64(state_c[(x + 1) % 5], 1);
        COL_WRITE_BITS(row, KeccakCols, c_prime[x], state_c_prime[x]);
    }

    // Populate A'. To avoid shifting indices, we rewrite
    //     A'[x, y, z] = xor(A[x, y, z], C[x - 1, z], C[x + 1, z - 1])
    // as
    //     A'[x, y, z] = xor(A[x, y, z], C[x, z], C'[x, z]).
    for (int x = 0; x < 5; x++) {
#pragma unroll 5
        for (int y = 0; y < 5; y++) {
            current_state[y][x] ^= state_c[x] ^ state_c_prime[x];
            COL_WRITE_BITS(row, KeccakCols, a_prime[y][x], current_state[y][x]);
        }
    }

    // Rotate the current state to get the B array.
    uint64_t state_b[5][5];
    for (auto i = 0; i < 5; i++) {
#pragma unroll 5
        for (auto j = 0; j < 5; j++) {
            auto new_i = (i + 3 * j) % 5;
            auto new_j = i;
            state_b[j][i] = ROTL64(current_state[new_j][new_i], R[new_i][new_j]);
        }
    }

    // Populate A'' as A''[x, y] = xor(B[x, y], andn(B[x + 1, y], B[x + 2, y])).
    for (int i = 0; i < 5; i++) {
#pragma unroll 5
        for (int j = 0; j < 5; j++) {
            current_state[i][j] =
                state_b[i][j] ^ ((~state_b[i][(j + 1) % 5]) & state_b[i][(j + 2) % 5]);
        }
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
