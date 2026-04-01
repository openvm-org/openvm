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

__device__ __forceinline__ void clear_result_limbs(int64_t result_limbs[MAX_LIMBS]) {
    for (uint32_t limb = 0; limb < MAX_LIMBS; limb++) {
        result_limbs[limb] = 0;
    }
}

__device__ __forceinline__ void add_biguint_limbs(
    int64_t result_limbs[MAX_LIMBS],
    const BigUintGpu &value
) {
    for (uint32_t limb = 0; limb < value.num_limbs && limb < MAX_LIMBS; limb++) {
        result_limbs[limb] += value.limbs[limb];
    }
}

__device__ __forceinline__ void subtract_biguint_limbs(
    int64_t result_limbs[MAX_LIMBS],
    const BigUintGpu &value
) {
    for (uint32_t limb = 0; limb < value.num_limbs && limb < MAX_LIMBS; limb++) {
        result_limbs[limb] -= value.limbs[limb];
    }
}

__device__ __forceinline__ void add_biguint_product(
    int64_t result_limbs[MAX_LIMBS],
    const BigUintGpu &lhs,
    const BigUintGpu &rhs
) {
    for (uint32_t i = 0; i < lhs.num_limbs; i++) {
        for (uint32_t j = 0; j < rhs.num_limbs && i + j < MAX_LIMBS; j++) {
            result_limbs[i + j] += (int64_t)lhs.limbs[i] * (int64_t)rhs.limbs[j];
        }
    }
}

__device__ __forceinline__ void add_bigint_biguint_product(
    int64_t result_limbs[MAX_LIMBS],
    const BigIntGpu &lhs,
    const BigUintGpu &rhs
) {
    int64_t lhs_signed_limbs[MAX_LIMBS];
    lhs.to_signed_limbs(lhs_signed_limbs);
    for (uint32_t i = 0; i < lhs.mag.num_limbs && i < MAX_LIMBS; i++) {
        for (uint32_t j = 0; j < rhs.num_limbs && i + j < MAX_LIMBS; j++) {
            result_limbs[i + j] += lhs_signed_limbs[i] * (int64_t)rhs.limbs[j];
        }
    }
}

__device__ __forceinline__ void subtract_signed_limbs_times_prime(
    int64_t result_limbs[MAX_LIMBS],
    const int64_t quotient_signed_limbs[MAX_LIMBS],
    uint32_t quotient_limbs,
    const uint8_t *prime_limbs,
    uint32_t prime_num_limbs
) {
    for (uint32_t q_limb = 0; q_limb < quotient_limbs; q_limb++) {
        int64_t q_signed = quotient_signed_limbs[q_limb];
        for (uint32_t p_limb = 0; p_limb < prime_num_limbs && q_limb + p_limb < MAX_LIMBS;
             p_limb++) {
            result_limbs[q_limb + p_limb] -= q_signed * (int64_t)prime_limbs[p_limb];
        }
    }
}

__device__ __forceinline__ void write_manual_carries(
    RowSlice core_row,
    size_t &carry_col,
    VariableRangeChecker &range_checker,
    bool track_range,
    const int64_t result_limbs[MAX_LIMBS],
    uint32_t carry_count,
    uint32_t carry_min_abs,
    uint32_t carry_bits
) {
    int64_t carry = 0;
    for (uint32_t limb = 0; limb < carry_count; limb++) {
        carry = (carry + result_limbs[limb]) >> 8;
        int32_t carry_i32 = (int32_t)carry;
        core_row[carry_col++] =
            carry_i32 >= 0 ? Fp((uint32_t)carry_i32) : (Fp::zero() - Fp((uint32_t)(-carry_i32)));
        if (track_range) {
            range_checker.add_count((uint32_t)(carry_i32 + (int32_t)carry_min_abs), carry_bits);
        }
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
