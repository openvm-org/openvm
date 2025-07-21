#pragma once
#include <stdint.h>
#include "bigint_ops.cuh"

__device__ void limb_add(uint32_t* result, const uint32_t* a, const uint32_t* b, 
                        uint32_t num_limbs, uint32_t limb_bits) {
    uint64_t carry = 0;
    uint32_t mask = get_limb_mask(limb_bits);
    for (uint32_t i = 0; i < num_limbs; i++) {
        uint64_t sum = (uint64_t)a[i] + b[i] + carry;
        result[i] = sum & mask;
        carry = sum >> limb_bits;
    }
    result[num_limbs] = carry;
    for (uint32_t i = num_limbs + 1; i < 2 * num_limbs; i++) {
        result[i] = 0;
    }
}

__device__ void limb_sub(uint32_t* result, const uint32_t* a, const uint32_t* b, 
                        uint32_t num_limbs, uint32_t limb_bits) {
    int64_t borrow = 0;
    uint32_t mask = get_limb_mask(limb_bits);
    for (uint32_t i = 0; i < num_limbs; i++) {
        int64_t diff = (int64_t)a[i] - b[i] - borrow;
        if (diff < 0) {
            result[i] = (diff + (1LL << limb_bits)) & mask;
            borrow = 1;
        } else {
            result[i] = diff & mask;
            borrow = 0;
        }
    }
}

__device__ void limb_mul(uint32_t* result, const uint32_t* a, const uint32_t* b, 
                        uint32_t num_limbs, uint32_t limb_bits) {
    for (uint32_t i = 0; i < 2 * num_limbs; i++) {
        result[i] = 0;
    }
    
    uint32_t mask = get_limb_mask(limb_bits);
    for (uint32_t i = 0; i < num_limbs; i++) {
        for (uint32_t j = 0; j < num_limbs; j++) {
            uint64_t prod = (uint64_t)a[i] * b[j];
            uint32_t pos = i + j;
            
            uint64_t sum = (uint64_t)result[pos] + (prod & mask);
            result[pos] = sum & mask;
            uint64_t carry = (sum >> limb_bits) + (prod >> limb_bits);
            
            pos++;
            while (carry > 0 && pos < 2 * num_limbs) {
                sum = (uint64_t)result[pos] + carry;
                result[pos] = sum & mask;
                carry = sum >> limb_bits;
                pos++;
            }
        }
    }
}

__device__ void limb_add_raw(uint32_t* result, const uint32_t* a, const uint32_t* b, uint32_t num_limbs) {
    for (uint32_t i = 0; i < 2 * num_limbs; i++) {
        result[i] = a[i] + b[i];
    }
}

__device__ void limb_sub_raw(uint32_t* result, const uint32_t* a, const uint32_t* b, uint32_t num_limbs) {
    for (uint32_t i = 0; i < 2 * num_limbs; i++) {
        result[i] = a[i] - b[i];
    }
}

__device__ void limb_mul_raw(uint32_t* result, const uint32_t* a, const uint32_t* b, uint32_t num_limbs) {
    for (uint32_t i = 0; i < 2 * num_limbs; i++) {
        result[i] = 0;
    }
    for (uint32_t i = 0; i < num_limbs; i++) {
        for (uint32_t j = 0; j < num_limbs; j++) {
            uint64_t prod = (uint64_t)a[i] * (uint64_t)b[j];
            uint32_t pos = i + j;
            
            uint64_t sum = (uint64_t)result[pos] + prod;
            result[pos] = (uint32_t)sum;
        }
    }
}

__device__ inline int limb_compare(const uint32_t* a, const uint32_t* b, uint32_t num_limbs) {
    for (int i = num_limbs - 1; i >= 0; i--) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

__device__ void limb_mul_mixed(uint32_t* result, const uint32_t* a, const uint32_t* b2, uint32_t num_limbs, uint32_t limb_bits) {
    const uint64_t mask = ((uint64_t)1 << limb_bits) - 1;

    for (int i = 0; i < 3*(int)num_limbs; i++) result[i] = 0;
  
    for (int i = 0; i < (int)num_limbs; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 2*(int)num_limbs; j++) {
            uint64_t acc = (uint64_t)result[i+j] + (uint64_t)a[i] * b2[j] + carry;
            result[i+j] = acc & mask;
            carry = acc >> limb_bits;
        }
        result[i + 2*num_limbs] = (uint32_t)carry;
    }
}

__device__ inline void limb_mod_reduce(uint32_t* result, const uint32_t* value, const uint32_t* prime, 
                                       uint32_t num_limbs, uint32_t limb_bits, const uint8_t* barrett_mu) {
    BigUintGpu value_big(value, 2 * num_limbs, limb_bits);
    value_big.normalize();
    
    BigUintGpu prime_big(prime, num_limbs, limb_bits);

    BigUintGpu result_big;
    biguint_mod_reduce(&result_big, &value_big, &prime_big, barrett_mu);
    
    for (uint32_t i = 0; i < num_limbs; i++) {
        result[i] = (i < result_big.num_limbs) ? result_big.limbs[i] : 0;
    }
}

__device__ inline void limb_mod_sub(uint32_t* result, const uint32_t* a, const uint32_t* b, 
                                   const uint32_t* prime, uint32_t num_limbs, uint32_t limb_bits) {
    int64_t borrow = 0;
    uint32_t mask = get_limb_mask(limb_bits);
    
    for (uint32_t i = 0; i < num_limbs; i++) {
        int64_t diff = (int64_t)a[i] - b[i] - borrow;
        if (diff < 0) {
            result[i] = (diff + (1LL << limb_bits)) & mask;
            borrow = 1;
        } else {
            result[i] = diff & mask;
            borrow = 0;
        }
    }
    
    if (borrow != 0) {
        uint64_t carry = 0;
        for (uint32_t i = 0; i < num_limbs; i++) {
            uint64_t sum = (uint64_t)result[i] + prime[i] + carry;
            result[i] = sum & mask;
            carry = sum >> limb_bits;
        }
    }
}

__device__ inline void limb_mod_inverse(uint32_t* result, const uint32_t* a, const uint32_t* prime, 
                                       uint32_t num_limbs, uint32_t limb_bits, const uint8_t* barrett_mu) {
    bool modulus_is_zero = true;
    for (uint32_t i = 0; i < num_limbs; i++) {
        if (prime[i] != 0) {
            modulus_is_zero = false;
            break;
        }
    }
    if (modulus_is_zero) {
        for (uint32_t i = 0; i < num_limbs; i++) {
            result[i] = 0;
        }
        return;
    }
    
    // Check if modulus is one
    bool modulus_is_one = (prime[0] == 1);
    for (uint32_t i = 1; i < num_limbs; i++) {
        if (prime[i] != 0) {
            modulus_is_one = false;
            break;
        }
    }
    if (modulus_is_one) {
        for (uint32_t i = 0; i < num_limbs; i++) {
            result[i] = 0;
        }
        return;
    }
    
    BigUintGpu r0, r1, t0, t1, q, r2, qt1, t2;
    
    BigUintGpu modulus_val(prime, num_limbs, limb_bits);
    
    BigUintGpu a_mod(a, num_limbs, limb_bits);
    biguint_mod_reduce(&a_mod, &a_mod, &modulus_val, barrett_mu);
    
    r1 = a_mod;
    
    bool r1_is_zero = true;
    for (uint32_t i = 0; i < r1.num_limbs; i++) {
        if (r1.limbs[i] != 0) {
            r1_is_zero = false;
            break;
        }
    }
    if (r1_is_zero) {
        for (uint32_t i = 0; i < num_limbs; i++) {
            result[i] = 0;
        }
        return;
    }
    
    bool r1_is_one = (r1.limbs[0] == 1);
    for (uint32_t i = 1; i < r1.num_limbs; i++) {
        if (r1.limbs[i] != 0) {
            r1_is_one = false;
            break;
        }
    }
    if (r1_is_one) {
        result[0] = 1;
        for (uint32_t i = 1; i < num_limbs; i++) {
            result[i] = 0;
        }
        return;
    }
    
    BigUintGpu temp_quotient, temp_remainder;
    biguint_divrem(&temp_quotient, &temp_remainder, &modulus_val, &r1);
    q = temp_quotient;
    r2 = temp_remainder;
    
    bool r2_is_zero = true;
    for (uint32_t i = 0; i < r2.num_limbs; i++) {
        if (r2.limbs[i] != 0) {
            r2_is_zero = false;
            break;
        }
    }
    if (r2_is_zero) {
        for (uint32_t i = 0; i < num_limbs; i++) {
            result[i] = 0;
        }
        return;
    }
    
    r0 = r1;
    r1 = r2;
    
    t0 = BigUintGpu(1, limb_bits);
    
    biguint_sub(&t1, &modulus_val, &q);
    
    while (true) {
        bool r1_zero = true;
        for (uint32_t i = 0; i < r1.num_limbs; i++) {
            if (r1.limbs[i] != 0) {
                r1_zero = false;
                break;
            }
        }
        if (r1_zero) break;
        
        // (q, r2) = r0.div_rem(&r1)
        biguint_divrem(&q, &r2, &r0, &r1);
        r0 = r1;
        r1 = r2;
        
        // qt1 = q * t1 % modulus
        biguint_mul(&qt1, &q, &t1);
        biguint_mod_reduce(&qt1, &qt1, &modulus_val, barrett_mu);
        
        // t2 = if t0 < qt1 { t0 + (modulus - qt1) } else { t0 - qt1 }
        if (biguint_compare(&t0, &qt1) < 0) {
            BigUintGpu modulus_minus_qt1;
            biguint_sub(&modulus_minus_qt1, &modulus_val, &qt1);
            biguint_add(&t2, &t0, &modulus_minus_qt1);
        } else {
            biguint_sub(&t2, &t0, &qt1);
        }
        
        t0 = t1;
        t1 = t2;
    }
    
    bool r0_is_one = (r0.limbs[0] == 1 && r0.num_limbs == 1);
    if (r0_is_one) {
        for (uint32_t i = 0; i < num_limbs; i++) {
            if (i < t0.num_limbs) {
                result[i] = t0.limbs[i];
            } else {
                result[i] = 0;
            }
        }
    } else {
        for (uint32_t i = 0; i < num_limbs; i++) {
            result[i] = 0;
        }
    }
}

__device__ inline void limb_mod_div(uint32_t* result, const uint32_t* a, const uint32_t* b, 
                                   const uint32_t* prime, uint32_t num_limbs, uint32_t limb_bits, 
                                   const uint8_t* barrett_mu, uint32_t* tmp) {
    uint32_t* b_inv = tmp;
    uint32_t* prod = tmp + num_limbs;
    
    limb_mod_inverse(b_inv, b, prime, num_limbs, limb_bits, barrett_mu);
    limb_mul(prod, a, b_inv, num_limbs, limb_bits);
    limb_mod_reduce(result, prod, prime, num_limbs, limb_bits, barrett_mu);
}

__device__ inline void limb_int_add(uint32_t* result, const uint32_t* a, int32_t scalar, 
                                   uint32_t num_limbs, uint32_t limb_bits) {
    uint32_t mask = get_limb_mask(limb_bits);
    
    if (scalar >= 0) {
        uint64_t carry = scalar;
        for (uint32_t i = 0; i < num_limbs; i++) {
            uint64_t sum = (uint64_t)a[i] + carry;
            result[i] = sum & mask;
            carry = sum >> limb_bits;
        }
    } else {
        uint32_t abs_scalar = (uint32_t)(-scalar);
        int64_t borrow = abs_scalar;
        for (uint32_t i = 0; i < num_limbs; i++) {
            int64_t diff = (int64_t)a[i] - borrow;
            if (diff < 0) {
                result[i] = (diff + (1LL << limb_bits)) & mask;
                borrow = 1;
            } else {
                result[i] = diff & mask;
                borrow = 0;
            }
        }
    }
}

__device__ inline void limb_int_mul(uint32_t* result, const uint32_t* a, int32_t scalar, 
                                   uint32_t num_limbs, uint32_t limb_bits, const uint32_t* prime_limbs) {
    // IMPORTANT: result must have space for 2 * num_limbs to handle overflow
    
    if (scalar >= 0) {
        // Positive scalar: simple multiplication
        uint64_t carry = 0;
        uint32_t mask = get_limb_mask(limb_bits);
        
        for (uint32_t i = 0; i < num_limbs; i++) {
            uint64_t prod = (uint64_t)a[i] * (uint32_t)scalar + carry;
            result[i] = prod & mask;
            carry = prod >> limb_bits;
        }
        
        // Store remaining carry in higher limbs
        for (uint32_t i = num_limbs; i < 2 * num_limbs; i++) {
            result[i] = carry & mask;
            carry = carry >> limb_bits;
            if (carry == 0) {
                // Zero out remaining limbs
                for (uint32_t j = i + 1; j < 2 * num_limbs; j++) {
                    result[j] = 0;
                }
                break;
            }
        }
    } else {
        // Negative scalar: compute (a * (prime - abs(scalar)))
        uint32_t abs_scalar = (uint32_t)(-scalar);
        
        // First compute prime - abs(scalar)
        uint32_t scalar_limbs[MAX_LIMBS];
        for (uint32_t i = 0; i < num_limbs; i++) {
            scalar_limbs[i] = 0;
        }
        scalar_limbs[0] = abs_scalar;
        
        // scalar_field = prime - abs_scalar  
        uint32_t scalar_field[MAX_LIMBS];
        limb_mod_sub(scalar_field, prime_limbs, scalar_limbs, prime_limbs, num_limbs, limb_bits);
        
        // Now multiply a by scalar_field
        // limb_mul writes to 2*num_limbs result
        limb_mul(result, a, scalar_field, num_limbs, limb_bits);
    }
}

__device__ inline void carry_limbs_overflow(
    const int64_t* overflow_limbs,
    uint32_t* carry_buf,
    uint32_t num_limbs,
    uint32_t limb_bits
) {
    int64_t carry = 0;
    
    for (uint32_t i = 0; i < num_limbs; ++i) {
        carry = (carry + overflow_limbs[i]) >> limb_bits;
        carry_buf[i] = carry;
    }
}