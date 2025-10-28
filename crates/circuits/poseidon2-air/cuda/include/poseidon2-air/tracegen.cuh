#pragma once

#include "columns.cuh"
#include "linearlayer.cuh"
#include "poseidon2.cuh"
#include "primitives/trace_access.h"

namespace poseidon2 {

//
// S-box application, unrolled on DEGREE and SBOX_REGS
//
template <size_t DEGREE, size_t SBOX_REGS>
__device__ __forceinline__ void apply_sbox(Fp &x, RowSlice sbox_regs) {
    if constexpr (DEGREE == 3 && SBOX_REGS == 0) {
        x = x * x * x;
    } else if constexpr (DEGREE == 5 && SBOX_REGS == 0) {
        x = pow(x, 5);
    } else if constexpr (DEGREE == 7 && SBOX_REGS == 0) {
        x = pow(x, 7);
    } else if constexpr (DEGREE == 5 && SBOX_REGS == 1) {
        Fp x2 = x * x;
        Fp x3 = x2 * x;
        if (sbox_regs.is_valid()) {
            sbox_regs[0] = x3;
        }
        x = x3 * x2;
    } else if constexpr (DEGREE == 7 && SBOX_REGS == 1) {
        Fp x3 = x * x * x;
        if (sbox_regs.is_valid()) {
            sbox_regs[0] = x3;
        }
        x = x3 * x3 * x;
    } else if constexpr (DEGREE == 11 && SBOX_REGS == 2) {
        Fp x2 = x * x;
        Fp x3 = x2 * x;
        Fp x9 = x3 * x3 * x3;
        if (sbox_regs.is_valid()) {
            sbox_regs[0] = x3;
            sbox_regs[1] = x9;
        }
        x = x9 * x2;
    } else {
        asm("unexpected (DEGREE, REGISTERS);");
    }
}

//
// Full‐round step: add constants, S-box on every lane, external mix
//
template <
    size_t WIDTH,
    size_t SBOX_DEGREE,
    size_t SBOX_REGS,
    size_t HALF_FULL_ROUNDS,
    size_t PARTIAL_ROUNDS>
__device__ __forceinline__ void full_round(
    RowSlice state,
    Poseidon2Row<WIDTH, SBOX_DEGREE, SBOX_REGS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS> perm,
    size_t round,
    bool is_end
) {
    // add round constants
#pragma unroll
    for (size_t i = 0; i < WIDTH; ++i) {
        if (is_end) {
            state[i] += TERMINAL_ROUND_CONSTANTS[round * WIDTH + i];
        } else {
            state[i] += INITIAL_ROUND_CONSTANTS[round * WIDTH + i];
        }
    }
    // apply S-box per lane
#pragma unroll
    for (size_t i = 0; i < WIDTH; ++i) {
        if (is_end) {
            apply_sbox<SBOX_DEGREE, SBOX_REGS>(state[i], perm.ending_full_sbox(round, i));
        } else {
            apply_sbox<SBOX_DEGREE, SBOX_REGS>(state[i], perm.beginning_full_sbox(round, i));
        }
    }
    // external linear layer (full MDS)
    LinearLayers<WIDTH>::external_linear_layer(state);
    // update post-state
    if (perm.is_valid()) {
#pragma unroll
        for (size_t i = 0; i < WIDTH; ++i) {
            if (is_end) {
                perm.ending_full_post(round)[i] = state[i];
            } else {
                perm.beginning_full_post(round)[i] = state[i];
            }
        }
    }
}

//
// Partial‐round step: add one constant, S-box on lane 0, internal mix
//
template <
    size_t WIDTH,
    size_t SBOX_DEGREE,
    size_t SBOX_REGS,
    size_t HALF_FULL_ROUNDS,
    size_t PARTIAL_ROUNDS>
__device__ __forceinline__ void partial_round(
    RowSlice state,
    Poseidon2Row<WIDTH, SBOX_DEGREE, SBOX_REGS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS> perm,
    size_t round
) {
    state[0] += INTERNAL_ROUND_CONSTANTS[round];
    apply_sbox<SBOX_DEGREE, SBOX_REGS>(state[0], perm.partial_sbox(round));
    // update post
    if (perm.is_valid()) {
        perm.partial_post(round)[0] = state[0];
    }
    LinearLayers<WIDTH>::internal_linear_layer(state);
}

//
// Public API: generate one Poseidon2 trace‐row for a single permutation. Since we
// sometimes need to compute Poseidon2 permutation without recording the trace row.
//
template <
    size_t WIDTH,
    size_t SBOX_DEGREE,
    size_t SBOX_REGS,
    size_t HALF_FULL_ROUNDS,
    size_t PARTIAL_ROUNDS>
__device__ __forceinline__ void generate_trace_row_for_perm(
    Poseidon2Row<WIDTH, SBOX_DEGREE, SBOX_REGS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS> perm,
    RowSlice state
) {
    if (perm.is_valid()) {
        perm.export_col()[0] = Fp::one();

        // initial inputs
#pragma unroll
        for (size_t i = 0; i < WIDTH; ++i) {
            perm.inputs()[i] = state[i];
        }
    }

    // initial external mix
    LinearLayers<WIDTH>::external_linear_layer(state);

    // beginning full rounds
#pragma unroll
    for (size_t r = 0; r < HALF_FULL_ROUNDS; ++r) {
        full_round<WIDTH, SBOX_DEGREE, SBOX_REGS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>(
            state, perm, r, false
        );
    }

    // partial rounds
#pragma unroll
    for (size_t r = 0; r < PARTIAL_ROUNDS; ++r) {
        partial_round<WIDTH, SBOX_DEGREE, SBOX_REGS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>(
            state, perm, r
        );
    }

    // ending full rounds
#pragma unroll
    for (size_t r = 0; r < HALF_FULL_ROUNDS; ++r) {
        full_round<WIDTH, SBOX_DEGREE, SBOX_REGS, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>(
            state, perm, r, true
        );
    }
}

} // namespace poseidon2
