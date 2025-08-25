#pragma once

#include "fp.h"

namespace IsZero {
/**
 * @brief Generates columns needed to constrain out == (x == 0).
 * 
 * @section Trace Context Parameters
 * @param x Value that is being constrained to be (not) equal to 0
 * 
 * @section Mutable Column Parameters
 * @param inv_ptr Field inverse of x if x != 0, else 0
 * @param out_ptr Boolean value equal to x == 0
 */
__device__ __forceinline__ void generate_subrow(Fp x, Fp *inv_ptr, Fp *out_ptr) {
    *inv_ptr = x == Fp::zero() ? Fp::zero() : inv(x);
    *out_ptr = x == Fp::zero() ? Fp::one() : Fp::zero();
}
} // namespace IsZero
