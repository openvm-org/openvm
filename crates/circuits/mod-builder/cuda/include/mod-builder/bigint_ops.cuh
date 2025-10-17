#pragma once

#include "primitives/constants.h"
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

using namespace mod_builder;

__device__ inline uint32_t get_limb_mask(uint32_t limb_bits) { return (1ULL << limb_bits) - 1; }

struct BigUintGpu {
    uint8_t limbs[MAX_LIMBS];
    uint32_t num_limbs;
    uint32_t limb_bits;

    __device__ BigUintGpu(uint8_t value, uint32_t bits) : num_limbs(1), limb_bits(bits) {
        limbs[0] = value & get_limb_mask(bits);
        assert(limbs[0] == value);
        for (uint32_t i = 1; i < MAX_LIMBS; i++) {
            limbs[i] = 0;
        }
    }

    __device__ BigUintGpu() : BigUintGpu(0, 8) {}

    __device__ BigUintGpu(uint32_t bits) : BigUintGpu(0, bits) {}

    __device__ BigUintGpu(const uint32_t *data, uint32_t n, uint32_t bits)
        : num_limbs(n), limb_bits(bits) {
        for (uint32_t i = 0; i < MAX_LIMBS; i++) {
            limbs[i] = (i < n) ? data[i] : 0;
        }
    }

    __device__ BigUintGpu(const uint8_t *data, uint32_t n, uint32_t bits)
        : num_limbs(n), limb_bits(bits) {
        for (uint32_t i = 0; i < MAX_LIMBS; i++) {
            limbs[i] = (i < n) ? data[i] : 0;
        }
    }

    __device__ void normalize() {
        while (num_limbs > 1 && limbs[num_limbs - 1] == 0) {
            num_limbs--;
        }
    }

    __device__ BigUintGpu(const BigUintGpu &other) : BigUintGpu(0, other.limb_bits) {
        num_limbs = other.num_limbs;
        for (uint32_t i = 0; i < MAX_LIMBS; i++) {
            limbs[i] = other.limbs[i];
        }
    }

    __device__ BigUintGpu &operator=(const BigUintGpu &other) {
        if (this != &other) {
            num_limbs = other.num_limbs;
            limb_bits = other.limb_bits;
            for (uint32_t i = 0; i < MAX_LIMBS; i++) {
                limbs[i] = other.limbs[i];
            }
        }
        return *this;
    }

    __device__ int compare(const BigUintGpu &other) const {
        for (int i = max(num_limbs, other.num_limbs) - 1; i >= 0; i--) {
            uint32_t ai = (i < num_limbs) ? limbs[i] : 0;
            uint32_t bi = (i < other.num_limbs) ? other.limbs[i] : 0;
            if (ai < bi)
                return -1;
            if (ai > bi)
                return 1;
        }
        return 0;
    }

    __device__ void add_in_place(const BigUintGpu &other) {
        uint32_t mask = get_limb_mask(limb_bits);
        uint32_t max_limbs = max(num_limbs, other.num_limbs) + 1;
        uint64_t carry = 0;

        for (uint32_t i = 0; i < max_limbs; i++) {
            uint32_t ai = (i < num_limbs) ? limbs[i] : 0;
            uint32_t bi = (i < other.num_limbs) ? other.limbs[i] : 0;
            uint64_t sum = ai + bi + carry;
            limbs[i] = sum & mask;
            carry = sum >> limb_bits;
        }

        num_limbs = max_limbs;
        if (carry > 0) {
            assert(max_limbs < MAX_LIMBS);
            limbs[max_limbs] = carry;
            num_limbs++;
        }

        for (uint32_t i = num_limbs; i < MAX_LIMBS; i++) {
            limbs[i] = 0;
        }

        normalize();
    }

    __device__ BigUintGpu add(const BigUintGpu &other) const {
        BigUintGpu result(*this);
        result.add_in_place(other);
        return result;
    }

    __device__ void sub_in_place(const BigUintGpu &other) {
        uint32_t mask = get_limb_mask(limb_bits);
        int32_t borrow = 0;

        for (uint32_t i = 0; i < num_limbs; i++) {
            int32_t ai = limbs[i];
            int32_t bi = (i < other.num_limbs) ? other.limbs[i] : 0;
            int32_t diff = ai - bi - borrow;

            if (diff < 0) {
                limbs[i] = (diff + (1LL << limb_bits)) & mask;
                borrow = 1;
            } else {
                limbs[i] = diff & mask;
                borrow = 0;
            }
        }

        normalize();
    }

    __device__ BigUintGpu sub(const BigUintGpu &other) const {
        BigUintGpu result(*this);
        result.sub_in_place(other);
        return result;
    }

    __device__ BigUintGpu mul(const BigUintGpu &other) const {
        BigUintGpu result(limb_bits);
        uint32_t mask = get_limb_mask(limb_bits);

        result.limb_bits = limb_bits;
        result.num_limbs = num_limbs + other.num_limbs;

        for (uint32_t i = 0; i < MAX_LIMBS; i++) {
            result.limbs[i] = 0;
        }

        for (uint32_t i = 0; i < num_limbs; i++) {
            uint64_t carry = 0;
            for (uint32_t j = 0; j < other.num_limbs; j++) {
                if (i + j < MAX_LIMBS) {
                    uint64_t prod =
                        (uint64_t)limbs[i] * other.limbs[j] + result.limbs[i + j] + carry;
                    result.limbs[i + j] = prod & mask;
                    carry = prod >> limb_bits;
                }
            }
            if (i + other.num_limbs < MAX_LIMBS && carry > 0) {
                result.limbs[i + other.num_limbs] = carry;
            }
        }

        result.normalize();
        return result;
    }

    __device__ BigUintGpu mod_sub(const BigUintGpu &other, const BigUintGpu &prime) const {
        BigUintGpu result(limb_bits);
        int32_t borrow = 0;
        uint32_t mask = get_limb_mask(limb_bits);
        uint32_t n = max(prime.num_limbs, max(num_limbs, other.num_limbs));
        result.num_limbs = prime.num_limbs;

        for (uint32_t i = 0; i < n; i++) {
            uint8_t ai = (i < num_limbs) ? limbs[i] : 0;
            uint8_t bi = (i < other.num_limbs) ? other.limbs[i] : 0;
            int32_t diff = (int32_t)ai - bi - borrow;
            if (diff < 0) {
                result.limbs[i] = (diff + (1LL << limb_bits)) & mask;
                borrow = 1;
            } else {
                result.limbs[i] = diff & mask;
                borrow = 0;
            }
        }

        while (borrow != 0) {
            uint32_t carry = 0;
            for (uint32_t i = 0; i < n; i++) {
                uint32_t sum = (uint32_t)result.limbs[i] + prime.limbs[i] + carry;
                result.limbs[i] = sum & mask;
                carry = sum >> limb_bits;
            }
            borrow -= carry;
        }

        result.normalize();
        return result;
    }

    __device__ BigUintGpu rem(const BigUintGpu &divisor) const {
        uint32_t mask = get_limb_mask(limb_bits);
        BigUintGpu zero(limb_bits);

        if (divisor == zero)
            return zero;
        if (*this < divisor)
            return *this;

        int msb_pos = -1;
        for (int limb = num_limbs - 1; limb >= 0; limb--) {
            uint32_t v = limbs[limb] & mask;
            if (v != 0) {
                int leading = __clz(v);
                int bit_index = 31 - leading;
                msb_pos = limb * limb_bits + bit_index;
                break;
            }
        }

        assert(msb_pos != -1);

        BigUintGpu temp_rem(limb_bits);
        temp_rem.num_limbs = 1;

        for (int bit_pos = msb_pos; bit_pos >= 0; bit_pos--) {
            uint32_t carry = 0;
            for (uint32_t i = 0; i < temp_rem.num_limbs; i++) {
                uint64_t shifted = ((uint64_t)temp_rem.limbs[i] << 1) | carry;
                temp_rem.limbs[i] = shifted & mask;
                carry = shifted >> limb_bits;
            }
            if (carry > 0 && temp_rem.num_limbs < MAX_LIMBS) {
                temp_rem.limbs[temp_rem.num_limbs] = carry;
                temp_rem.num_limbs++;
            }

            uint32_t limb_idx = bit_pos / limb_bits;
            uint32_t bit_idx = bit_pos % limb_bits;
            uint32_t bit = (limbs[limb_idx] >> bit_idx) & 1;
            temp_rem.limbs[0] = (temp_rem.limbs[0] & ~1U) | bit;

            temp_rem.normalize();

            if (temp_rem >= divisor) {
                temp_rem -= divisor;
            }
        }

        return temp_rem;
    }

    __device__ void divrem(
        BigUintGpu &quotient,
        BigUintGpu &remainder,
        const BigUintGpu &divisor
    ) const {
        uint32_t mask = get_limb_mask(limb_bits);

        quotient.limb_bits = limb_bits;
        quotient.num_limbs = 1;
        for (uint32_t i = 0; i < MAX_LIMBS; i++) {
            quotient.limbs[i] = 0;
        }

        remainder = *this;

        BigUintGpu zero(limb_bits);
        if (divisor == zero)
            return;
        if (*this < divisor)
            return;

        int msb_pos = -1;
        for (int limb = num_limbs - 1; limb >= 0; limb--) {
            uint32_t v = limbs[limb] & mask;
            if (v != 0) {
                int leading = __clz(v);
                int bit_index = 31 - leading;
                msb_pos = limb * limb_bits + bit_index;
                break;
            }
        }

        if (msb_pos == -1)
            return;

        BigUintGpu temp_rem(limb_bits);
        temp_rem.num_limbs = 1;

        for (int bit_pos = msb_pos; bit_pos >= 0; bit_pos--) {
            uint32_t carry = 0;
            for (uint32_t i = 0; i < temp_rem.num_limbs; i++) {
                uint64_t shifted = ((uint64_t)temp_rem.limbs[i] << 1) | carry;
                temp_rem.limbs[i] = shifted & mask;
                carry = shifted >> limb_bits;
            }
            if (carry > 0 && temp_rem.num_limbs < MAX_LIMBS) {
                temp_rem.limbs[temp_rem.num_limbs] = carry;
                temp_rem.num_limbs++;
            }

            uint32_t limb_idx = bit_pos / limb_bits;
            uint32_t bit_idx = bit_pos % limb_bits;
            uint32_t bit = (limbs[limb_idx] >> bit_idx) & 1;
            temp_rem.limbs[0] = (temp_rem.limbs[0] & ~1U) | bit;

            temp_rem.normalize();

            if (temp_rem >= divisor) {
                temp_rem = temp_rem - divisor;

                uint32_t q_limb_idx = bit_pos / limb_bits;
                uint32_t q_bit_idx = bit_pos % limb_bits;
                if (q_limb_idx < MAX_LIMBS) {
                    quotient.limbs[q_limb_idx] |= (1U << q_bit_idx);
                }
            }
        }

        quotient.num_limbs = 1;
        for (int i = MAX_LIMBS - 1; i >= 0; i--) {
            if (quotient.limbs[i] != 0) {
                quotient.num_limbs = i + 1;
                break;
            }
        }

        remainder = temp_rem;
    }

    __device__ BigUintGpu mod_reduce(const BigUintGpu &modulus, const uint8_t *barrett_mu) const {
        BigUintGpu result(limb_bits);
        const uint32_t mask = get_limb_mask(limb_bits);
        uint32_t n = modulus.num_limbs;

        if (*this < modulus) {
            return *this;
        }

        // q1 = value >> ((num_limbs - 1) * limb_bits)
        BigUintGpu q1(limb_bits);
        q1.num_limbs = min(num_limbs - (n - 1), n + 1);
        for (uint32_t i = 0; i < q1.num_limbs && i + n - 1 < num_limbs; i++) {
            q1.limbs[i] = limbs[i + n - 1];
        }
        q1.normalize();

        BigUintGpu mu(barrett_mu, 2 * n, limb_bits);
        mu.normalize();

        BigUintGpu q2 = q1.mul(mu);

        BigUintGpu q3(limb_bits);
        if (q2.num_limbs > n + 1) {
            q3.num_limbs = min(q2.num_limbs - (n + 1), n);
            for (uint32_t i = 0; i < q3.num_limbs && i + n + 1 < q2.num_limbs; i++) {
                q3.limbs[i] = q2.limbs[i + n + 1];
            }
        }
        q3.normalize();

        BigUintGpu r2 = q3 * modulus;

        if (*this >= r2) {
            result = *this;
            result -= r2;
        } else {
            result = *this;
            result += modulus;
            result -= r2;
        }

        result = result % modulus;
        return result;
    }

    __device__ BigUintGpu mod_inverse(const BigUintGpu &modulus, const uint8_t *barrett_mu) const {
        // Check if modulus is zero
        bool modulus_is_zero = true;
        for (uint32_t i = 0; i < modulus.num_limbs; i++) {
            if (modulus.limbs[i] != 0) {
                modulus_is_zero = false;
                break;
            }
        }
        if (modulus_is_zero) {
            return BigUintGpu(limb_bits); // Return zero
        }

        // Check if modulus is one
        bool modulus_is_one = (modulus.limbs[0] == 1);
        for (uint32_t i = 1; i < modulus.num_limbs; i++) {
            if (modulus.limbs[i] != 0) {
                modulus_is_one = false;
                break;
            }
        }
        if (modulus_is_one) {
            return BigUintGpu(limb_bits); // Return zero
        }

        // Reduce a modulo modulus first
        BigUintGpu a_mod = mod_reduce(modulus, barrett_mu);

        // Check if a_mod is zero
        bool a_is_zero = true;
        for (uint32_t i = 0; i < a_mod.num_limbs; i++) {
            if (a_mod.limbs[i] != 0) {
                a_is_zero = false;
                break;
            }
        }
        if (a_is_zero) {
            return BigUintGpu(limb_bits); // Return zero (no inverse)
        }

        // Check if a_mod is one
        bool a_is_one = (a_mod.limbs[0] == 1);
        for (uint32_t i = 1; i < a_mod.num_limbs; i++) {
            if (a_mod.limbs[i] != 0) {
                a_is_one = false;
                break;
            }
        }
        if (a_is_one) {
            return BigUintGpu(1, limb_bits); // Return one
        }

        // Extended Euclidean Algorithm
        BigUintGpu r0, r1, t0, t1, q, r2, qt1, t2;

        // Initial values
        r1 = a_mod;

        // First iteration outside the loop
        BigUintGpu temp_quotient, temp_remainder;
        modulus.divrem(temp_quotient, temp_remainder, r1);
        q = temp_quotient;
        r2 = temp_remainder;

        // Check if r2 is zero (gcd(a, modulus) != 1)
        bool r2_is_zero = true;
        for (uint32_t i = 0; i < r2.num_limbs; i++) {
            if (r2.limbs[i] != 0) {
                r2_is_zero = false;
                break;
            }
        }
        if (r2_is_zero) {
            return BigUintGpu(limb_bits); // Return zero (no inverse)
        }

        // Update for next iteration
        r0 = r1;
        r1 = r2;

        // Initialize t values after first iteration
        // t0 = 1
        t0 = BigUintGpu(1, limb_bits);

        // t1 = modulus - q
        t1 = modulus - q;

        // Main loop
        while (true) {
            // Check if r1 is zero
            bool r1_zero = true;
            for (uint32_t i = 0; i < r1.num_limbs; i++) {
                if (r1.limbs[i] != 0) {
                    r1_zero = false;
                    break;
                }
            }
            if (r1_zero)
                break;

            // (q, r2) = divrem(r0, r1)
            r0.divrem(q, r2, r1);
            r0 = r1;
            r1 = r2;

            // qt1 = q * t1 % modulus
            qt1 = q.mul(t1).mod_reduce(modulus, barrett_mu);

            // t2 = (t0 - qt1) % modulus
            if (t0 >= qt1) {
                t2 = t0 - qt1;
            } else {
                // t0 < qt1, so compute t0 + modulus - qt1
                BigUintGpu modulus_minus_qt1 = modulus - qt1;
                t2 = t0 + modulus_minus_qt1;
            }

            t0 = t1;
            t1 = t2;
        }

        // Check if gcd is 1 (r0 should be 1)
        bool r0_is_one = (r0.limbs[0] == 1);
        for (uint32_t i = 1; i < r0.num_limbs; i++) {
            if (r0.limbs[i] != 0) {
                r0_is_one = false;
                break;
            }
        }
        if (r0_is_one) {
            return t0; // t0 is the modular inverse
        } else {
            return BigUintGpu(limb_bits); // Return zero (no inverse)
        }
    }

    __device__ BigUintGpu
    mod_div(const BigUintGpu &b, const BigUintGpu &modulus, const uint8_t *barrett_mu) const {
        BigUintGpu b_inv = b.mod_inverse(modulus, barrett_mu);
        BigUintGpu prod = mul(b_inv);

        // Reduce the product modulo modulus
        return prod.mod_reduce(modulus, barrett_mu);
    }

    // Comparison operators
    __device__ bool operator==(const BigUintGpu &other) const { return compare(other) == 0; }
    __device__ bool operator!=(const BigUintGpu &other) const { return compare(other) != 0; }
    __device__ bool operator<(const BigUintGpu &other) const { return compare(other) < 0; }
    __device__ bool operator<=(const BigUintGpu &other) const { return compare(other) <= 0; }
    __device__ bool operator>(const BigUintGpu &other) const { return compare(other) > 0; }
    __device__ bool operator>=(const BigUintGpu &other) const { return compare(other) >= 0; }

    // Arithmetic operators
    __device__ BigUintGpu operator+(const BigUintGpu &other) const { return add(other); }
    __device__ BigUintGpu operator-(const BigUintGpu &other) const { return sub(other); }
    __device__ BigUintGpu operator*(const BigUintGpu &other) const { return mul(other); }
    __device__ BigUintGpu operator%(const BigUintGpu &other) const { return rem(other); }
    __device__ BigUintGpu &operator+=(const BigUintGpu &other) {
        add_in_place(other);
        return *this;
    }
    __device__ BigUintGpu &operator-=(const BigUintGpu &other) {
        sub_in_place(other);
        return *this;
    }

    // Division returns both quotient and remainder via divrem method
    __device__ std::pair<BigUintGpu, BigUintGpu> operator/(const BigUintGpu &divisor) const {
        BigUintGpu quotient(limb_bits), remainder(limb_bits);
        divrem(quotient, remainder, divisor);
        return std::make_pair(quotient, remainder);
    }
};

struct BigIntGpu {
    BigUintGpu mag; // Actual magnitude limbs
    bool is_negative;

    __device__ BigIntGpu() : mag(), is_negative(false) {}

    __device__ BigIntGpu(uint32_t bits) : mag(bits), is_negative(false) {}

    __device__ BigIntGpu(int32_t value, uint32_t bits)
        : mag((uint32_t)(value < 0 ? -value : value), bits), is_negative(value < 0) {}

    __device__ BigIntGpu(const BigUintGpu &magnitude, bool negative = false)
        : mag(magnitude),
          is_negative(negative && !(magnitude.num_limbs == 1 && magnitude.limbs[0] == 0)) {}

    __device__ BigIntGpu(const uint32_t *data, uint32_t n, uint32_t bits, bool negative = false)
        : mag(data, n, bits), is_negative(negative && !(mag.num_limbs == 1 && mag.limbs[0] == 0)) {}

    __device__ void normalize() { mag.normalize(); }

    // Comparison method
    __device__ int compare(const BigIntGpu &other) const {
        if (is_negative != other.is_negative)
            return 1 - 2 * is_negative;
        int mag_cmp = mag.compare(other.mag);
        return (1 - 2 * is_negative) * mag_cmp;
    }

    // Arithmetic methods that return values
    __device__ BigIntGpu add(const BigIntGpu &other) const {
        BigIntGpu result(mag.limb_bits);
        if (is_negative == other.is_negative) {
            result.mag = mag.add(other.mag);
            result.is_negative = is_negative;
        } else {
            if (mag >= other.mag) {
                result.mag = mag - other.mag;
                result.is_negative = is_negative;
            } else {
                result.mag = other.mag - mag;
                result.is_negative = other.is_negative;
            }
        }
        return result;
    }

    __device__ BigIntGpu sub(const BigIntGpu &other) const {
        BigIntGpu neg_other = other;
        neg_other.is_negative = !other.is_negative;
        return add(neg_other);
    }

    __device__ BigIntGpu mul(const BigIntGpu &other) const {
        return BigIntGpu(mag.mul(other.mag), is_negative != other.is_negative);
    }

    __device__ BigIntGpu div_biguint(const BigUintGpu &divisor) const {
        BigIntGpu result(mag.limb_bits);
        BigUintGpu remainder_mag;
        mag.divrem(result.mag, remainder_mag, divisor);
        result.is_negative = is_negative;
        return result;
    }

    __device__ BigIntGpu mod_reduce(const BigUintGpu &modulus, const uint8_t *barrett_mu) const {
        return BigIntGpu(mag.mod_reduce(modulus, barrett_mu), is_negative);
    }

    __device__ BigIntGpu negate() const {
        BigIntGpu result(mag.limb_bits);
        result.mag = mag;
        result.is_negative = !is_negative && !is_zero();
        return result;
    }

    __device__ BigUintGpu abs() const { return mag; }

    __device__ void to_signed_limbs(int64_t *limbs) const {
        if (is_negative) {
            for (uint32_t i = 0; i < mag.num_limbs; i++) {
                limbs[i] = -(int64_t)mag.limbs[i];
            }
        } else {
            for (uint32_t i = 0; i < mag.num_limbs; i++) {
                limbs[i] = (int64_t)mag.limbs[i];
            }
        }
        for (uint32_t i = mag.num_limbs; i < MAX_LIMBS; i++) {
            limbs[i] = 0;
        }
    }

    // Comparison operators
    __device__ bool operator==(const BigIntGpu &other) const { return compare(other) == 0; }
    __device__ bool operator!=(const BigIntGpu &other) const { return compare(other) != 0; }
    __device__ bool operator<(const BigIntGpu &other) const { return compare(other) < 0; }
    __device__ bool operator<=(const BigIntGpu &other) const { return compare(other) <= 0; }
    __device__ bool operator>(const BigIntGpu &other) const { return compare(other) > 0; }
    __device__ bool operator>=(const BigIntGpu &other) const { return compare(other) >= 0; }

    // Arithmetic operators
    __device__ BigIntGpu operator+(const BigIntGpu &other) const { return add(other); }
    __device__ BigIntGpu operator-(const BigIntGpu &other) const { return sub(other); }
    __device__ BigIntGpu operator*(const BigIntGpu &other) const { return mul(other); }
    __device__ BigIntGpu operator/(const BigUintGpu &divisor) const { return div_biguint(divisor); }
    __device__ BigIntGpu &operator+=(const BigIntGpu &other) {
        *this = add(other);
        return *this;
    }
    __device__ BigIntGpu &operator-=(const BigIntGpu &other) {
        *this = sub(other);
        return *this;
    }
    __device__ BigIntGpu &operator*=(const BigIntGpu &other) {
        *this = mul(other);
        return *this;
    }
    __device__ BigIntGpu &operator/=(const BigUintGpu &divisor) {
        *this = div_biguint(divisor);
        return *this;
    }
    __device__ BigIntGpu operator-() const { return negate(); }

    __device__ bool is_zero() const { return mag.num_limbs == 1 && mag.limbs[0] == 0; }
};
