#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "def_types.h"
#include "fp.h"

namespace canonicity {

using namespace deferral;

template <typename T> struct CanonicityAuxCols {
    T diff_marker[F_NUM_BYTES];
    T diff_val;
};

__device__ __forceinline__ uint32_t
generate_subrow(const Fp x_le[F_NUM_BYTES], CanonicityAuxCols<Fp> &aux) {
#pragma unroll
    for (size_t i = 0; i < F_NUM_BYTES; ++i) {
        aux.diff_marker[i] = Fp::zero();
    }
    aux.diff_val = Fp::zero();

    Fp x_be[F_NUM_BYTES];
#pragma unroll
    for (size_t i = 0; i < F_NUM_BYTES; ++i) {
        x_be[i] = x_le[F_NUM_BYTES - 1 - i];
    }

    bool found = false;
    uint32_t to_range_check = 0;

#pragma unroll
    for (size_t i = 0; i < F_NUM_BYTES; ++i) {
        const uint32_t x_u32 = x_be[i].asUInt32();
        const uint32_t y =
            (BABY_BEAR_ORDER >> (8 * (F_NUM_BYTES - 1 - i))) & static_cast<uint32_t>(0xff);

        if (!found && x_u32 != y) {
#ifdef CUDA_DEBUG
            assert(x_u32 < 256);
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
