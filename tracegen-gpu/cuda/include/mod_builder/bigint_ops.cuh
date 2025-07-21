#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "constants.h"

using namespace mod_builder;

__device__ inline uint32_t get_limb_mask(uint32_t limb_bits) {
    return (1ULL << limb_bits) - 1;
}

struct BigUintGpu {
    uint8_t limbs[MAX_LIMBS];
    uint32_t num_limbs;
    uint32_t limb_bits;
    
    __device__ BigUintGpu() : num_limbs(1), limb_bits(8) {
        for (uint32_t i = 0; i < MAX_LIMBS; i++) {
            limbs[i] = 0;
        }
    }
    
    __device__ BigUintGpu(uint8_t value, uint32_t bits) : num_limbs(1), limb_bits(bits) {
        limbs[0] = value & get_limb_mask(bits);
        assert(limbs[0] == value);
        for (uint32_t i = 1; i < MAX_LIMBS; i++) {
            limbs[i] = 0;
        }
    }
    
    __device__ BigUintGpu(uint32_t bits) : BigUintGpu(0, bits) {}
    
    __device__ BigUintGpu(const uint32_t* data, uint32_t n, uint32_t bits) : num_limbs(n), limb_bits(bits) {
        for (uint32_t i = 0; i < MAX_LIMBS; i++) {
            limbs[i] = (i < n) ? data[i] : 0;
        }
    }
    
    __device__ BigUintGpu(const uint8_t* data, uint32_t n, uint32_t bits) : num_limbs(n), limb_bits(bits) {
        for (uint32_t i = 0; i < MAX_LIMBS; i++) {
            limbs[i] = (i < n) ? data[i] : 0;
        }
    }

    __device__ void normalize() {
        while (num_limbs > 1 && limbs[num_limbs - 1] == 0) {
            num_limbs--;
        }
    }
};

struct BigIntGpu {
    BigUintGpu mag; // Actual magnitude limbs
    bool is_negative;
    
    __device__ BigIntGpu() : mag(), is_negative(false) {}
    
    __device__ BigIntGpu(uint32_t bits) : mag(bits), is_negative(false) {}
    
    __device__ BigIntGpu(int32_t value, uint32_t bits) 
        : mag((uint32_t)(value < 0 ? -value : value), bits), 
          is_negative(value < 0) {}
    
    __device__ BigIntGpu(const BigUintGpu& magnitude, bool negative = false) 
        : mag(magnitude), is_negative(negative && !(magnitude.num_limbs == 1 && magnitude.limbs[0] == 0)) {}
    
    __device__ BigIntGpu(const uint32_t* data, uint32_t n, uint32_t bits, bool negative = false)
        : mag(data, n, bits), is_negative(negative && !(mag.num_limbs == 1 && mag.limbs[0] == 0)) {}

    __device__ void normalize() {
        mag.normalize();
    }
};

__device__ inline int biguint_compare(const BigUintGpu* a, const BigUintGpu* b) {
    for (int i = max(a->num_limbs, b->num_limbs) - 1; i >= 0; i--) {
        uint32_t ai = (i < a->num_limbs) ? a->limbs[i] : 0;
        uint32_t bi = (i < b->num_limbs) ? b->limbs[i] : 0;
        if (ai < bi) return -1;
        if (ai > bi) return 1;
    }
    return 0;
}

__device__ inline void biguint_add(BigUintGpu* result, const BigUintGpu* a, const BigUintGpu* b) {
    uint32_t mask = get_limb_mask(a->limb_bits);
    uint32_t max_limbs = max(a->num_limbs, b->num_limbs);
    uint64_t carry = 0;
    
    result->limb_bits = a->limb_bits;
    result->num_limbs = max_limbs;
    
    for (uint32_t i = 0; i < max_limbs; i++) {
        uint32_t ai = (i < a->num_limbs) ? a->limbs[i] : 0;
        uint32_t bi = (i < b->num_limbs) ? b->limbs[i] : 0;
        uint64_t sum = ai + bi + carry;
        result->limbs[i] = sum & mask;
        carry = sum >> a->limb_bits;
    }
    
    if (carry > 0 && max_limbs < MAX_LIMBS) {
        result->limbs[max_limbs] = carry;
        result->num_limbs++;
    }
    
    for (uint32_t i = result->num_limbs; i < MAX_LIMBS; i++) {
        result->limbs[i] = 0;
    }
    
    result->normalize();
}

__device__ inline void biguint_sub(BigUintGpu* result, const BigUintGpu* a, const BigUintGpu* b) {
    uint32_t mask = get_limb_mask(a->limb_bits);
    int32_t borrow = 0;
    
    result->limb_bits = a->limb_bits;
    result->num_limbs = a->num_limbs;
    
    for (uint32_t i = 0; i < a->num_limbs; i++) {
        int32_t ai = a->limbs[i];
        int32_t bi = (i < b->num_limbs) ? b->limbs[i] : 0;
        int32_t diff = ai - bi - borrow;
        
        if (diff < 0) {
            result->limbs[i] = (diff + (1LL << a->limb_bits)) & mask;
            borrow = 1;
        } else {
            result->limbs[i] = diff & mask;
            borrow = 0;
        }
    }
    
    for (uint32_t i = result->num_limbs; i < MAX_LIMBS; i++) {
        result->limbs[i] = 0;
    }
    
    result->normalize();
}

__device__ inline void biguint_mul(BigUintGpu* result, const BigUintGpu* a, const BigUintGpu* b) {
    uint32_t mask = get_limb_mask(a->limb_bits);
    
    result->limb_bits = a->limb_bits;
    result->num_limbs = a->num_limbs + b->num_limbs;
    
    for (uint32_t i = 0; i < MAX_LIMBS; i++) {
        result->limbs[i] = 0;
    }
    
    for (uint32_t i = 0; i < a->num_limbs; i++) {
        uint64_t carry = 0;
        for (uint32_t j = 0; j < b->num_limbs; j++) {
            if (i + j < MAX_LIMBS) {
                uint64_t prod = (uint64_t)a->limbs[i] * b->limbs[j] + result->limbs[i + j] + carry;
                result->limbs[i + j] = prod & mask;
                carry = prod >> a->limb_bits;
            }
        }
        if (i + b->num_limbs < MAX_LIMBS && carry > 0) {
            result->limbs[i + b->num_limbs] = carry;
        }
    }
    
    result->normalize();
}

__device__ inline void biguint_divrem(BigUintGpu* quotient, BigUintGpu* remainder, 
                                      const BigUintGpu* dividend, const BigUintGpu* divisor) {
    uint32_t mask = get_limb_mask(dividend->limb_bits);
    
    quotient->limb_bits = dividend->limb_bits;
    quotient->num_limbs = 1;
    for (uint32_t i = 0; i < MAX_LIMBS; i++) {
        quotient->limbs[i] = 0;
    }
    
    *remainder = *dividend;

    BigUintGpu zero(divisor->limb_bits);
    if (biguint_compare(divisor, &zero) == 0) return;
    
    if (biguint_compare(dividend, divisor) < 0) return;
    
    int msb_pos = -1;
    for (int limb = dividend->num_limbs - 1; limb >= 0; limb--) {
        uint32_t v = dividend->limbs[limb] & mask;
        if (v != 0) {
            int leading = __clz(v);
            int bit_index = 31 - leading;
            msb_pos = limb * dividend->limb_bits + bit_index;
            break;
        }
    }
    
    if (msb_pos == -1) return;
    
    BigUintGpu temp_rem;
    temp_rem.limb_bits = dividend->limb_bits;
    temp_rem.num_limbs = 1;
    for (uint32_t i = 0; i < MAX_LIMBS; i++) {
        temp_rem.limbs[i] = 0;
    }
    
    for (int bit_pos = msb_pos; bit_pos >= 0; bit_pos--) {
        uint32_t carry = 0;
        for (uint32_t i = 0; i < temp_rem.num_limbs; i++) {
            uint64_t shifted = ((uint64_t)temp_rem.limbs[i] << 1) | carry;
            temp_rem.limbs[i] = shifted & mask;
            carry = shifted >> dividend->limb_bits;
        }
        if (carry > 0 && temp_rem.num_limbs < MAX_LIMBS) {
            temp_rem.limbs[temp_rem.num_limbs] = carry;
            temp_rem.num_limbs++;
        }
        
        uint32_t limb_idx = bit_pos / dividend->limb_bits;
        uint32_t bit_idx = bit_pos % dividend->limb_bits;
        uint32_t bit = (dividend->limbs[limb_idx] >> bit_idx) & 1;
        temp_rem.limbs[0] = (temp_rem.limbs[0] & ~1U) | bit;
        
        temp_rem.normalize();
        
        if (biguint_compare(&temp_rem, divisor) >= 0) {
            biguint_sub(&temp_rem, &temp_rem, divisor);
            
            uint32_t q_limb_idx = bit_pos / dividend->limb_bits;
            uint32_t q_bit_idx = bit_pos % dividend->limb_bits;
            if (q_limb_idx < MAX_LIMBS) {
                quotient->limbs[q_limb_idx] |= (1U << q_bit_idx);
            }
        }
    }
    
    quotient->num_limbs = 1;
    for (int i = MAX_LIMBS - 1; i >= 0; i--) {
        if (quotient->limbs[i] != 0) {
            quotient->num_limbs = i + 1;
            break;
        }
    }
    
    *remainder = temp_rem;
}

__device__ inline void bigint_add(BigIntGpu* result, const BigIntGpu* a, const BigIntGpu* b) {
    if (a->is_negative == b->is_negative) {
        biguint_add(&result->mag, &a->mag, &b->mag);
        result->is_negative = a->is_negative;
    } else {
        int cmp = biguint_compare(&a->mag, &b->mag);
        if (cmp >= 0) {
            biguint_sub(&result->mag, &a->mag, &b->mag);
            result->is_negative = a->is_negative;
        } else {
            biguint_sub(&result->mag, &b->mag, &a->mag);
            result->is_negative = b->is_negative;
        }
    }
}

__device__ inline void bigint_sub(BigIntGpu* result, const BigIntGpu* a, const BigIntGpu* b) {
    BigIntGpu neg_b = *b;
    neg_b.is_negative = !b->is_negative;
    bigint_add(result, a, &neg_b);
}

__device__ inline void bigint_mul(BigIntGpu* result, const BigIntGpu* a, const BigIntGpu* b) {
    biguint_mul(&result->mag, &a->mag, &b->mag);
    result->is_negative = (a->is_negative != b->is_negative);
}

__device__ inline void bigint_div_biguint(BigIntGpu* quotient, const BigIntGpu* dividend, const BigUintGpu* divisor) {
    BigUintGpu remainder_mag;
    biguint_divrem(&quotient->mag, &remainder_mag, &dividend->mag, divisor);
    quotient->is_negative = dividend->is_negative;
}

__device__ inline void bigint_to_signed_limbs(const BigIntGpu* num, int64_t* limbs) {
    if (num->is_negative) {
        for (uint32_t i = 0; i < num->mag.num_limbs; i++) {
            limbs[i] = -(int64_t)num->mag.limbs[i];
        }
    } else {
        for (uint32_t i = 0; i < num->mag.num_limbs; i++) {
            limbs[i] = (int64_t)num->mag.limbs[i];
        }
    }
    for (uint32_t i = num->mag.num_limbs; i < MAX_LIMBS; i++) {
        limbs[i] = 0;
    }
}

__device__ inline const BigUintGpu* bigint_mag(const BigIntGpu* num) {
    return &num->mag;
}

__device__ inline bool bigint_is_zero(const BigIntGpu* num) {
    return num->mag.num_limbs == 1 && num->mag.limbs[0] == 0;
}

__device__ inline void bigint_negate(BigIntGpu* result, const BigIntGpu* num) {
    result->mag = num->mag;
    result->is_negative = !num->is_negative && !bigint_is_zero(num);
}

__device__ inline int bigint_compare(const BigIntGpu* a, const BigIntGpu* b) {
    if (a->is_negative != b->is_negative) return 1 - 2 * a->is_negative;
    
    int mag_cmp = biguint_compare(&a->mag, &b->mag);
    
    return (1 - 2 * a->is_negative) * mag_cmp;
}

__device__ inline void bigint_abs(BigUintGpu* result, const BigIntGpu* num) {
    *result = num->mag;
}

__device__ inline void biguint_mod_reduce(BigUintGpu* result, const BigUintGpu* value, 
                                         const BigUintGpu* modulus, const uint8_t* barrett_mu) {
    uint32_t limb_bits = value->limb_bits;
    uint32_t num_limbs = modulus->num_limbs;
    const uint32_t mask = get_limb_mask(limb_bits);
    
    if (value->num_limbs < modulus->num_limbs || 
        (value->num_limbs == modulus->num_limbs && biguint_compare(value, modulus) < 0)) {
        *result = *value;
        return;
    }
    
    // q1 = value >> ((num_limbs - 1) * limb_bits)
    BigUintGpu q1(limb_bits);
    q1.num_limbs = min(value->num_limbs - (num_limbs - 1), num_limbs + 1);
    for (uint32_t i = 0; i < q1.num_limbs && i + num_limbs - 1 < value->num_limbs; i++) {
        q1.limbs[i] = value->limbs[i + num_limbs - 1];
    }
    q1.normalize();
    
    BigUintGpu mu(barrett_mu, 2 * num_limbs, limb_bits);
    mu.normalize();
    
    BigUintGpu q2(limb_bits);
    biguint_mul(&q2, &q1, &mu);
    
    BigUintGpu q3(limb_bits);
    if (q2.num_limbs > num_limbs + 1) {
        q3.num_limbs = min(q2.num_limbs - (num_limbs + 1), num_limbs);
        for (uint32_t i = 0; i < q3.num_limbs && i + num_limbs + 1 < q2.num_limbs; i++) {
            q3.limbs[i] = q2.limbs[i + num_limbs + 1];
        }
    }
    q3.normalize();
    
    BigUintGpu r2;
    biguint_mul(&r2, &q3, modulus);
    
    if (biguint_compare(value, &r2) >= 0) {
        biguint_sub(result, value, &r2);
    } else {
        BigUintGpu temp;
        biguint_add(&temp, value, modulus);
        biguint_sub(result, &temp, &r2);
    }
}