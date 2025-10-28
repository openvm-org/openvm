#pragma once

#include "bigint_ops.cuh"
#include "primitives/constants.h"
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

using namespace mod_builder;

__device__ inline uint32_t log2_ceil_usize(uint32_t n) {
    if (n == 0)
        return 0;
    if (n == 1)
        return 0;
    return 32 - __clz(n - 1);
}

struct OverflowInt {
    int32_t limbs[MAX_LIMBS];
    uint32_t num_limbs;
    uint32_t limb_bits;
    uint32_t limb_max_abs;
    uint32_t max_overflow_bits;

    __device__ OverflowInt(uint32_t value, uint32_t bits) : num_limbs(1), limb_bits(bits) {
        limbs[0] = value;
        for (uint32_t i = 1; i < MAX_LIMBS; i++) {
            limbs[i] = 0;
        }
        limb_max_abs = value;
        max_overflow_bits = log2_ceil_usize(limb_max_abs);
    }

    __device__ OverflowInt() : OverflowInt(0, 8) {}

    __device__ OverflowInt(uint32_t bits) : OverflowInt(0, bits) {}

    __device__ OverflowInt(const BigUintGpu &biguint, uint32_t n_limbs)
        : num_limbs(n_limbs), limb_bits(biguint.limb_bits) {
        for (uint32_t i = 0; i < biguint.num_limbs; i++) {
            limbs[i] = (int32_t)biguint.limbs[i];
        }
        for (uint32_t i = biguint.num_limbs; i < MAX_LIMBS; i++) {
            limbs[i] = 0;
        }

        limb_max_abs = (1 << limb_bits) - 1;
        max_overflow_bits = limb_bits;
    }

    __device__ OverflowInt(const BigIntGpu &bigint, uint32_t n_limbs)
        : num_limbs(n_limbs), limb_bits(bigint.mag.limb_bits) {
        for (uint32_t i = 0; i < bigint.mag.num_limbs; i++) {
            limbs[i] = (int32_t)bigint.mag.limbs[i];
            if (bigint.is_negative) {
                limbs[i] = -limbs[i];
            }
        }
        for (uint32_t i = bigint.mag.num_limbs; i < MAX_LIMBS; i++) {
            limbs[i] = 0;
        }

        limb_max_abs = 1 << limb_bits;
        max_overflow_bits = limb_bits + 1;
    }

    __device__ OverflowInt(const uint8_t *data, uint32_t n, uint32_t bits)
        : num_limbs(n), limb_bits(bits) {
        for (uint32_t i = 0; i < MAX_LIMBS; i++) {
            limbs[i] = (i < n) ? (int64_t)data[i] : 0;
        }

        limb_max_abs = (1 << limb_bits) - 1;
        max_overflow_bits = limb_bits;
    }

    __device__ OverflowInt(const uint32_t *data, uint32_t n, uint32_t bits)
        : num_limbs(n), limb_bits(bits) {
        for (uint32_t i = 0; i < MAX_LIMBS; i++) {
            limbs[i] = (i < n) ? (int64_t)data[i] : 0;
        }

        limb_max_abs = (1 << limb_bits) - 1;
        max_overflow_bits = limb_bits;
    }

    __device__ OverflowInt(const int64_t *signed_limbs, uint32_t count, uint32_t bits)
        : num_limbs(count), limb_bits(bits) {

        for (uint32_t i = 0; i < count; i++) {
            limbs[i] = signed_limbs[i];
        }
        for (uint32_t i = count; i < MAX_LIMBS; i++) {
            limbs[i] = 0;
        }

        limb_max_abs = 1 << limb_bits;
        max_overflow_bits = limb_bits + 1;
    }

    __device__ OverflowInt(const OverflowInt &other)
        : num_limbs(other.num_limbs), limb_bits(other.limb_bits), limb_max_abs(other.limb_max_abs),
          max_overflow_bits(other.max_overflow_bits) {
        for (uint32_t i = 0; i < MAX_LIMBS; i++) {
            limbs[i] = other.limbs[i];
        }
    }

    __device__ OverflowInt &operator=(const OverflowInt &other) {
        if (this != &other) {
            num_limbs = other.num_limbs;
            limb_bits = other.limb_bits;
            limb_max_abs = other.limb_max_abs;
            max_overflow_bits = other.max_overflow_bits;
            for (uint32_t i = 0; i < MAX_LIMBS; i++) {
                limbs[i] = other.limbs[i];
            }
        }
        return *this;
    }

    __device__ void add_in_place(const OverflowInt &other) {
        uint32_t new_num_limbs = max(num_limbs, other.num_limbs);

        for (uint32_t i = 0; i < new_num_limbs; i++) {
            int32_t ai = (i < num_limbs) ? limbs[i] : 0;
            int32_t bi = (i < other.num_limbs) ? other.limbs[i] : 0;
            limbs[i] = ai + bi;
        }

        for (uint32_t i = new_num_limbs; i < MAX_LIMBS; i++) {
            limbs[i] = 0;
        }

        num_limbs = new_num_limbs;
        limb_max_abs = limb_max_abs + other.limb_max_abs;
        max_overflow_bits = log2_ceil_usize(limb_max_abs);
    }

    __device__ OverflowInt add(const OverflowInt &other) const {
        OverflowInt result(*this);
        result.add_in_place(other);
        return result;
    }

    __device__ void sub_in_place(const OverflowInt &other) {
        uint32_t new_num_limbs = max(num_limbs, other.num_limbs);

        for (uint32_t i = 0; i < new_num_limbs; i++) {
            int32_t ai = (i < num_limbs) ? limbs[i] : 0;
            int32_t bi = (i < other.num_limbs) ? other.limbs[i] : 0;
            limbs[i] = ai - bi;
        }

        for (uint32_t i = new_num_limbs; i < MAX_LIMBS; i++) {
            limbs[i] = 0;
        }

        num_limbs = new_num_limbs;
        limb_max_abs = limb_max_abs + other.limb_max_abs;
        max_overflow_bits = log2_ceil_usize(limb_max_abs);
    }

    __device__ OverflowInt sub(const OverflowInt &other) const {
        OverflowInt result(*this); // Copy constructor
        result.sub_in_place(other);
        return result;
    }

    __device__ OverflowInt mul(const OverflowInt &other) const {
        OverflowInt result(limb_bits);
        result.num_limbs = num_limbs + other.num_limbs - 1;

        // Initialize all limbs to zero
        for (uint32_t i = 0; i < MAX_LIMBS; i++) {
            result.limbs[i] = 0;
        }

        for (uint32_t i = 0; i < num_limbs; i++) {
            for (uint32_t j = 0; j < other.num_limbs; j++) {
                if (i + j < result.num_limbs) {
                    int64_t prod = limbs[i] * other.limbs[j];
                    result.limbs[i + j] += prod;
                }
            }
        }

        result.limb_max_abs = limb_max_abs * other.limb_max_abs * min(num_limbs, other.num_limbs);
        result.max_overflow_bits = log2_ceil_usize(result.limb_max_abs);
        return result;
    }

    __device__ OverflowInt add_scalar(int32_t scalar) const {
        OverflowInt result = *this;
        result.limbs[0] += scalar;
        result.limb_max_abs = limb_max_abs + abs(scalar);
        result.max_overflow_bits = log2_ceil_usize(result.limb_max_abs);
        return result;
    }

    __device__ OverflowInt mul_scalar(int32_t scalar) const {
        OverflowInt result = *this;
        for (uint32_t i = 0; i < num_limbs; i++) {
            result.limbs[i] *= scalar;
        }
        result.limb_max_abs = limb_max_abs * abs(scalar);
        result.max_overflow_bits = log2_ceil_usize(result.limb_max_abs);
        return result;
    }

    __device__ OverflowInt operator+(const OverflowInt &other) const { return add(other); }
    __device__ OverflowInt operator-(const OverflowInt &other) const { return sub(other); }
    __device__ OverflowInt operator*(const OverflowInt &other) const { return mul(other); }
    __device__ OverflowInt &operator+=(const OverflowInt &other) {
        add_in_place(other);
        return *this;
    }
    __device__ OverflowInt &operator-=(const OverflowInt &other) {
        sub_in_place(other);
        return *this;
    }
    __device__ OverflowInt &operator*=(const OverflowInt &other) {
        *this = mul(other);
        return *this;
    }

    __device__ OverflowInt operator+(int32_t scalar) const { return add_scalar(scalar); }
    __device__ OverflowInt operator*(int32_t scalar) const { return mul_scalar(scalar); }
    __device__ OverflowInt &operator+=(int32_t scalar) {
        *this = add_scalar(scalar);
        return *this;
    }
    __device__ OverflowInt &operator*=(int32_t scalar) {
        *this = mul_scalar(scalar);
        return *this;
    }

    __device__ OverflowInt carry_limbs(uint32_t carry_count) const {
        OverflowInt carries(limb_bits);
        carries.num_limbs = carry_count;

        int32_t carry = 0;
        for (uint32_t i = 0; i < carry_count; ++i) {
            carry = (carry + limbs[i]) >> limb_bits;
            carries.limbs[i] = carry;
        }

        int32_t max_carry_abs = 0;
        for (uint32_t i = 0; i < carry_count; i++) {
            int32_t abs_val = carries.limbs[i] < 0 ? -carries.limbs[i] : carries.limbs[i];
            if (abs_val > max_carry_abs) {
                max_carry_abs = abs_val;
            }
        }
        carries.limb_max_abs = max_carry_abs;
        carries.max_overflow_bits = log2_ceil_usize(carries.limb_max_abs);

        return carries;
    }
};
