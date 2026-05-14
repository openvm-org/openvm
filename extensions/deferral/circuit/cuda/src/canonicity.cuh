#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "def_types.h"
#include "fp.h"

namespace canonicity {

using namespace deferral;

// Number of limbs / bits per limb the canonicity sub-AIR walks. After the u16
// migration these match the CPU-side `CANONICITY_NUM_LIMBS` /
// `CANONICITY_LIMB_BITS` constants.
inline constexpr size_t CANONICITY_NUM_LIMBS = F_NUM_U16S;
inline constexpr size_t CANONICITY_LIMB_BITS = 16;

template <typename T> struct CanonicityAuxCols {
    T diff_marker[CANONICITY_NUM_LIMBS];
    T diff_val;
};

__device__ __forceinline__ uint32_t
generate_subrow(const Fp x_le[CANONICITY_NUM_LIMBS], CanonicityAuxCols<Fp> &aux) {
#pragma unroll
    for (size_t i = 0; i < CANONICITY_NUM_LIMBS; ++i) {
        aux.diff_marker[i] = Fp::zero();
    }
    aux.diff_val = Fp::zero();

    Fp x_be[CANONICITY_NUM_LIMBS];
#pragma unroll
    for (size_t i = 0; i < CANONICITY_NUM_LIMBS; ++i) {
        x_be[i] = x_le[CANONICITY_NUM_LIMBS - 1 - i];
    }

    bool found = false;
    uint32_t to_range_check = 0;

#pragma unroll
    for (size_t i = 0; i < CANONICITY_NUM_LIMBS; ++i) {
        const uint32_t x_u32 = x_be[i].asUInt32();
        const uint32_t y =
            (BABY_BEAR_ORDER >> (CANONICITY_LIMB_BITS * (CANONICITY_NUM_LIMBS - 1 - i))) &
            ((1u << CANONICITY_LIMB_BITS) - 1);

        if (!found && x_u32 != y) {
#ifdef CUDA_DEBUG
            assert(x_u32 < (1u << CANONICITY_LIMB_BITS));
            assert(y > x_u32);
#endif
            const uint32_t diff = y - x_u32;
            aux.diff_marker[i] = Fp::one();
            aux.diff_val = Fp(diff);
            to_range_check = diff - 1;
            found = true;
        }
    }

#ifdef CUDA_DEBUG
    assert(found);
#endif

    return to_range_check;
}

} // namespace canonicity
