#pragma once

#include "histogram.cuh"
#include "fp_array.cuh"

static const size_t AUX_LEN = 2;

template <typename T, size_t AUX_LEN = AUX_LEN> struct LessThanAuxCols {
    T lower_decomp[AUX_LEN];
};

template <typename T, size_t NUM, size_t AUX_LEN = AUX_LEN> struct LessThanArrayAuxCols {
    T diff_marker[NUM];
    T diff_inv;
    LessThanAuxCols<T, AUX_LEN> lt_decomp;
};

/// Generators aka subairs
namespace AssertLessThan {
/**
 * @brief Generates columns needed to constrain that x < y
 * 
 * @section Trace Context Parameters
 * @param rc Range checker histogram reference
 * @param max_bits Maximum number of bits the respresntation of x and y can be
 * @param x First value to compare (must be strictly less than y)
 * @param y Second value to compare
 * @param lower_decomp_len Number of columns needed to constrain x < y
 * 
 * @section Mutable Column Parameters
 * @param lower_decomp Columns used to constrain x < y
 */
__device__ __forceinline__ void generate_subrow(
    VariableRangeChecker &rc,
    const uint32_t max_bits,
    uint32_t x,
    uint32_t y,
    const size_t lower_decomp_len,
    RowSlice lower_decomp
) {
    rc.decompose(y - x - 1, max_bits, lower_decomp, lower_decomp_len);
}
} // namespace AssertLessThan

namespace IsLessThan {
/**
 * @brief Generates columns needed to constrain that out_flag == (x < y)
 * 
 * @section Trace Context Parameters
 * @param rc Range checker histogram reference
 * @param max_bits Maximum number of bits the respresntation of x and y can be
 * @param x First value to compare
 * @param y Second value to compare
 * @param lower_decomp_len Number of columns needed to constrain out_flag == (x < y)
 * 
 * @section Mutable Column Parameters
 * @param lower_decomp Columns used to constrain out_flag == (x < y)
 * @param out_flag Boolean value equal to x < y
 */
__device__ __forceinline__ void generate_subrow(
    VariableRangeChecker &rc,
    const uint32_t max_bits,
    uint32_t x,
    uint32_t y,
    const size_t lower_decomp_len,
    RowSlice lower_decomp,
    Fp *out_flag
) {
    *out_flag = Fp(x < y);
    uint32_t check_less_than = (1u << max_bits) + y - x - 1;
    uint32_t lower = check_less_than & ((1u << max_bits) - 1);
    rc.decompose(lower, max_bits, lower_decomp, lower_decomp_len);
}
} // namespace IsLessThan

namespace IsLessThanArray {
/**
 * @brief Generates columns needed to constrain that out_flag == (x < y),
 *        where x and y are represented by array_len limbs.
 * 
 * @section Trace Context Parameters
 * @param rc Range checker histogram reference
 * @param max_bits Maximum number of bits each limb of x and y can be
 * @param x First value to compare
 * @param y Second value to compare
 * @param array_len Number of limbs to represent x and y
 * @param aux_len Number of additional columns needed to constrain outflag == (x < y)
 * 
 * @section Mutable Column Parameters
 * @param diff_marker Array that marks the most significant limb difference in x and y
 * @param diff_inv Field inverse of the first differing y[i] - x[i], or 0
 * @param lt_decomp Columns used to constrain outflag == (x < y)
 * @param out_flag Boolean value equal to x < y
 */
__device__ __forceinline__ void generate_subrow(
    VariableRangeChecker &rc,
    const uint32_t max_bits,
    const RowSlice x,
    const RowSlice y,
    const size_t array_len,
    const size_t aux_len,
    RowSlice diff_marker,
    Fp *diff_inv,
    RowSlice lt_decomp,
    Fp *out_flag
) {
    bool is_eq = true;
    Fp diff_val = Fp::zero();
    *diff_inv = Fp::zero();

    for (size_t i = 0; i < array_len; i++) {
        if (x[i] != y[i] && is_eq) {
            is_eq = false;
            diff_val = y[i] - x[i];
            *diff_inv = inv(diff_val);
            diff_marker[i] = Fp::one();
        } else {
            diff_marker[i] = Fp::zero();
        }
    }

    Fp shifted_diff_fp = (diff_val + Fp((1 << max_bits) - 1));
    uint32_t shifted_diff = shifted_diff_fp.asUInt32();
    uint32_t lower = shifted_diff & ((1 << max_bits) - 1);
    if (out_flag) {
        *out_flag = Fp(shifted_diff != lower);
    }
    rc.decompose(lower, max_bits, lt_decomp, aux_len);
}

template <size_t N>
__device__ __forceinline__ void generate_subrow(
    VariableRangeChecker &rc,
    const uint32_t max_bits,
    FpArray<N> x,
    FpArray<N> y,
    const size_t aux_len,
    RowSlice diff_marker,
    Fp *diff_inv,
    RowSlice lt_decomp,
    Fp *out_flag
) {
    generate_subrow(
        rc, max_bits, x.as_row(), y.as_row(), N, aux_len, diff_marker, diff_inv, lt_decomp, out_flag
    );
}
} // namespace IsLessThanArray
