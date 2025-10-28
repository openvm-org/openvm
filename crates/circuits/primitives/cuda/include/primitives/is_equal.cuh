#pragma once

#include "trace_access.h"

namespace IsEqual {
/**
 * @brief Generates columns needed to constrain out == (x == y).
 * 
 * @section Trace Context Parameters
 * @param x First value to compare
 * @param y Second value to compare
 * 
 * @section Mutable Column Parameters
 * @param inv_diff Field inverse of x - y if x != y, else 0
 * @param out_ptr Boolean value equal to x == y
 */
__device__ __forceinline__ void generate_subrow(Fp x, Fp y, Fp *inv_diff, Fp *out) {
    *inv_diff = (x == y) ? Fp::zero() : inv(x - y);
    *out = (x == y) ? Fp::one() : Fp::zero();
}
} // namespace IsEqual

namespace IsEqualArray {
/**
 * @brief Generates columns needed to constrain out == (x == y) where
 *        x and y are represented by arr_length limbs.
 * 
 * @section Trace Context Parameters
 * @param arr_length Number of limbs to represent x and y
 * @param x First value to compare
 * @param y Second value to compare
 * 
 * @section Mutable Column Parameters
 * @param diff_inv_marker Marks most significant limb difference between x and y
 * @param out_ptr Boolean value equal to x == y
 */
__device__ __forceinline__ void generate_subrow(
    uint32_t arr_length,
    RowSlice x,
    RowSlice y,
    RowSlice diff_inv_marker,
    Fp *out
) {
    bool equal = true;
    for (uint32_t i = 0; i < arr_length; ++i) {
        Fp diff = x[i] - y[i];
        if (diff != Fp::zero() && equal) {
            diff_inv_marker[i] = inv(diff);
            equal = false;
        } else {
            diff_inv_marker[i] = Fp::zero();
        }
    }
    *out = Fp(equal);
}
} // namespace IsEqualArray