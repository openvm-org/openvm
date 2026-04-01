#pragma once

#include "fp.h"
#include "mod-builder/bigint_ops.cuh"
#include "mod-builder/overflow_ops.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include <cstddef>
#include <cstdint>

template <size_t NUM_LIMBS>
__device__ __forceinline__ void write_biguint(RowSlice row, size_t &col, const BigUintGpu &value) {
    for (size_t limb = 0; limb < NUM_LIMBS; limb++) {
        uint32_t v = limb < value.num_limbs ? value.limbs[limb] : 0;
        row[col++] = Fp(v);
    }
}

template <size_t NUM_LIMBS>
__device__ __forceinline__ BigUintGpu read_biguint(RowSlice row, size_t col) {
    uint32_t limbs[NUM_LIMBS];
    for (size_t limb = 0; limb < NUM_LIMBS; limb++) {
        limbs[limb] = row[col + limb].asUInt32();
    }
    BigUintGpu value(limbs, NUM_LIMBS, 8);
    value.normalize();
    return value;
}

__device__ __forceinline__ void write_signed_bigint_limb(
    RowSlice row,
    size_t &col,
    const BigIntGpu &value,
    size_t limb
) {
    uint32_t q_limb = limb < value.mag.num_limbs ? value.mag.limbs[limb] : 0;
    row[col++] = value.is_negative ? (Fp::zero() - Fp(q_limb)) : Fp(q_limb);
}

__device__ __forceinline__ int64_t biguint_limb_or_zero(const BigUintGpu &value, uint32_t limb) {
    return limb < value.num_limbs ? (int64_t)value.limbs[limb] : 0;
}

__device__ __forceinline__ int64_t biguint_product_limb(
    const BigUintGpu &lhs,
    const BigUintGpu &rhs,
    uint32_t limb
) {
    int64_t coeff = 0;
    for (uint32_t i = 0; i < lhs.num_limbs && i <= limb; i++) {
        uint32_t j = limb - i;
        if (j < rhs.num_limbs) {
            coeff += (int64_t)lhs.limbs[i] * (int64_t)rhs.limbs[j];
        }
    }
    return coeff;
}

__device__ __forceinline__ void read_signed_bigint_limbs(
    const BigIntGpu &value,
    int32_t out_limbs[MAX_LIMBS]
) {
    for (uint32_t limb = 0; limb < MAX_LIMBS; limb++) {
        int32_t signed_limb = limb < value.mag.num_limbs ? (int32_t)value.mag.limbs[limb] : 0;
        out_limbs[limb] = value.is_negative ? -signed_limb : signed_limb;
    }
}

__device__ __forceinline__ int64_t subtract_signed_limb_product_limb(
    const int32_t *lhs_limbs,
    uint32_t lhs_limb_count,
    const uint8_t *rhs_limbs,
    uint32_t rhs_limb_count,
    uint32_t limb
) {
    int64_t coeff = 0;
    for (uint32_t lhs_limb = 0; lhs_limb < lhs_limb_count && lhs_limb <= limb; lhs_limb++) {
        uint32_t rhs_limb = limb - lhs_limb;
        if (rhs_limb < rhs_limb_count) {
            coeff -= (int64_t)lhs_limbs[lhs_limb] * (int64_t)rhs_limbs[rhs_limb];
        }
    }
    return coeff;
}

__device__ __forceinline__ void write_carry_limb(
    RowSlice core_row,
    size_t &col,
    VariableRangeChecker &range_checker,
    bool track_range,
    int64_t carry,
    uint32_t carry_min_abs,
    uint32_t carry_bits
) {
    int32_t carry_i32 = (int32_t)carry;
    core_row[col++] =
        carry_i32 >= 0 ? Fp((uint32_t)carry_i32) : (Fp::zero() - Fp((uint32_t)(-carry_i32)));
    if (track_range) {
        range_checker.add_count((uint32_t)(carry_i32 + (int32_t)carry_min_abs), carry_bits);
    }
}

__device__ __forceinline__ void write_quotient_limbs(
    RowSlice core_row,
    size_t quotient_col,
    VariableRangeChecker &range_checker,
    bool track_range,
    const BigIntGpu &quotient,
    uint32_t quotient_limb_count
) {
    size_t col = quotient_col;
    for (uint32_t limb = 0; limb < quotient_limb_count; limb++) {
        int32_t q_signed = limb < quotient.mag.num_limbs ? (int32_t)quotient.mag.limbs[limb] : 0;
        if (quotient.is_negative) {
            q_signed = -q_signed;
        }
        write_signed_bigint_limb(core_row, col, quotient, limb);
        if (track_range) {
            range_checker.add_count((uint32_t)(q_signed + (1 << 8)), 9);
        }
    }
}

template <size_t NUM_LIMBS>
__device__ __forceinline__ OverflowInt make_double_lambda_den_overflow(
    bool is_setup,
    const BigUintGpu &y1
) {
    OverflowInt value;
    if (is_setup) {
        value = OverflowInt(1, 8);
        value.num_limbs = NUM_LIMBS;
        value.limb_max_abs = ((1u << value.limb_bits) - 1) * 2;
        value.max_overflow_bits = log2_ceil_usize(value.limb_max_abs);
    } else {
        value = OverflowInt(y1, NUM_LIMBS);
        value *= 2;
    }
    return value;
}
