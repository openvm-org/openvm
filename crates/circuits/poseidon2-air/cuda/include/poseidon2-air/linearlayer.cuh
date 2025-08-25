#pragma once

#include "fp.h"
#include "poseidon2.cuh"
#include "primitives/trace_access.h"

namespace poseidon2 {

//-----------------------------------------------------------------------------
// "Fast" 4×4 MDS and light‐circulant extension
//-----------------------------------------------------------------------------

__device__ __forceinline__ void apply_mat4(RowSlice x) {
    Fp t01 = x[0] + x[1];
    Fp t23 = x[2] + x[3];
    Fp t0123 = t01 + t23;
    Fp t01123 = t0123 + x[1];
    Fp t01233 = t0123 + x[3];
    x[3] = t01233 + x[0].doubled();
    x[1] = t01123 + x[2].doubled();
    x[0] = t01123 + t01;
    x[2] = t01233 + t23;
}

template <size_t WIDTH> __device__ __forceinline__ void mds_light_permutation(RowSlice state) {
    if constexpr (WIDTH == 2) {
        Fp s = state[0] + state[1];
        state[0] += s;
        state[1] += s;
    } else if constexpr (WIDTH == 3) {
        Fp s = state[0] + state[1] + state[2];
        state[0] += s;
        state[1] += s;
        state[2] += s;
    } else {
#pragma unroll
        for (size_t i = 0; i < WIDTH; i += 4) {
            apply_mat4(state.slice_from(i));
        }

        Fp sums[4] = {Fp::zero(), Fp::zero(), Fp::zero(), Fp::zero()};
#pragma unroll
        for (size_t i = 0; i < WIDTH; i++) {
            sums[i & 3] += state[i];
        }
#pragma unroll
        for (size_t i = 0; i < WIDTH; i++) {
            state[i] += sums[i & 3];
        }
    }
}

//-----------------------------------------------------------------------------
// Poseidon2 linear layers
//-----------------------------------------------------------------------------

template <size_t WIDTH> struct LinearLayers {
    /// External = full "light" MDS
    __device__ __forceinline__ static void external_linear_layer(RowSlice state) {
        mds_light_permutation<WIDTH>(state);
    }

    /// Internal = Rust's generic_internal_linear_layer for WIDTH=16
    /// (plus hard-coded 2×2/3×3 for small widths)
    __device__ __forceinline__ static void internal_linear_layer(RowSlice state) {
        if constexpr (WIDTH == 2) {
            Fp s = state[0] + state[1];
            state[0] += s;
            state[1] = state[1].doubled() + s;
        } else if constexpr (WIDTH == 3) {
            Fp s = state[0] + state[1] + state[2];
            state[0] += s;
            state[1] += s;
            state[2] = state[2].doubled() + s;
        } else if constexpr (WIDTH == 16) {
            // 1) Compute part_sum and full_sum
            Fp part_sum = Fp::zero();
#pragma unroll
            for (size_t i = 1; i < 16; ++i)
                part_sum += state[i];
            Fp full_sum = part_sum + state[0];

            // 2) Custom for first three lanes
            state[0] = part_sum - state[0];
            state[1] = full_sum + state[1];
            state[2] = full_sum + state[2].doubled();

// 3) Remaining lanes via diag-vector
#pragma unroll
            for (size_t i = 3; i < 16; ++i) {
                state[i] = full_sum + state[i] * internal_diag16[i];
            }
        } else {
            static_assert(
                WIDTH == 2 || WIDTH == 3 || WIDTH == 16,
                "internal_linear_layer only implemented for WIDTH=2,3,16"
            );
        }
    }
};

} // namespace poseidon2
