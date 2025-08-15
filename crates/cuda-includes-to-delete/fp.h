// Copyright 2024 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

/// \file
/// Defines the core finite field data type, Fp, and some free functions on the type.

#include <cassert>
#include <cstdint>

/// The Fp class is an element of the finite field F_p, where P is the prime number 15*2^27 + 1.
/// Put another way, Fp is basically integer arithmetic modulo P.
///
/// The 'Fp' datatype is the core type of all of the operations done within the zero knowledge
/// proofs, and is smallest 'addressable' datatype, and the base type of which all composite types
/// are built.  In many ways, one can imagine it as the word size of a very strange architecture.
///
/// This specific prime P was chosen to:
/// - Be less than 2^31 so that it fits within a 32 bit word and doesn't overflow on addition.
/// - Otherwise have as large a power of 2 in the factors of P-1 as possible.
///
/// This last property is useful for number theoretical transforms (the fast fourier transform
/// equivalent on finite fields).  See NTT.h for details.
///
/// The Fp class wraps all the standard arithmetic operations to make the finite field elements look
/// basically like ordinary numbers (which they mostly are).
class Fp {
  public:
    /// The value of P, the modulus of Fp.
    static constexpr uint32_t P = 15 * (uint32_t(1) << 27) + 1;
    static constexpr uint32_t M = 0x88000001;
    static constexpr uint32_t R2 = 1172168163;
    static constexpr uint32_t MONTY_BITS = 32;
    static constexpr uint32_t MONTY_MASK = 0xffffffff; // uint32_t((uint64_t(1) << MONTY_BITS) - 1);
    static constexpr uint32_t HALF_P_PLUS_1 = (P + 1) >> 1;
    static constexpr uint32_t TWO_ADICITY = 27;
    // Precomputed inverse of 2^(2*FIELD_BITS - 2) mod P for FIELD_BITS = 31
    // FIELD_BITS = 31, so k = 2*FIELD_BITS - 2 = 60
    // INV_2EXP_K = (2^k)^(-1) mod P
    static constexpr uint32_t INV_2EXP_K = 975175684u;

  private:
    // The actual value, always < P.
    uint32_t val;

    // We make 'impls' of the core ops which all the other uses call.  This is done to allow for
    // tweaking of the implementation later, for example switching to montgomery representation or
    // doing inline assembly or some crazy CUDA stuff.

    // Add two numbers
    static __device__ constexpr uint32_t add(uint32_t a, uint32_t b) {
        uint32_t r = a + b;
        return (r >= P ? r - P : r);
    }

    // Subtract two numbers
    static __device__ constexpr uint32_t sub(uint32_t a, uint32_t b) {
        uint32_t r = a - b;
        return (r > P ? r + P : r);
    }

    // Multiply two numbers
    static __device__ constexpr uint32_t mul(uint32_t a, uint32_t b) {
        uint64_t o64 = uint64_t(a) * uint64_t(b);
        uint32_t low = -uint32_t(o64);
        uint32_t red = M * low;
        o64 += uint64_t(red) * uint64_t(P);
        uint32_t ret = o64 >> 32;
        return (ret >= P ? ret - P : ret);
    }

    // Encode / Decode
    static __device__ constexpr uint32_t encode(uint32_t a) { return mul(R2, a); }

    static __device__ constexpr uint32_t decode(uint32_t a) { return mul(1, a); }

    // A private constructor that take the 'internal' form.
    __device__ constexpr Fp(uint32_t val, [[maybe_unused]] bool ignore) : val(val) {}

  public:
    /// Default constructor, sets value to 0.
    __device__ constexpr Fp() : val(0) {}

    /// Construct an FP from a uint32_t, wrap if needed
    __device__ constexpr Fp(uint32_t val) : val(encode(val)) {}

    /// Construct an Fp from an already-encoded raw value
    __device__ static constexpr Fp fromRaw(uint32_t val) { return Fp(val, true); }

    /// Fp with zero value
    __device__ static constexpr Fp zero() { return Fp(0); }

    /// Fp with one value
    __device__ static constexpr Fp one() { return Fp(1); }

    /// Fp with negative one value
    __device__ static constexpr Fp neg_one() { return maxVal(); }

    /// Convert to a uint32_t
    __device__ constexpr uint32_t asUInt32() const { return decode(val); }

    /// Return the raw underlying word
    __device__ constexpr uint32_t asRaw() const { return val; }

    /// Get the largest value, basically P - 1.
    __device__ static constexpr Fp maxVal() { return P - 1; }

    /// Get an 'invalid' Fp value
    __device__ static constexpr Fp invalid() { return Fp(0xfffffffful, true); }

    /// get value
    __device__ constexpr uint32_t get() const { return val; }

    /// force set the value
    __device__ constexpr void set(uint32_t input) { val = input; }

    /// force set self as a decoded val
    __device__ constexpr void set_raw() { val = decode(val); }

    // Implement all the various overloads
    __device__ constexpr void operator=(uint32_t rhs) { val = encode(rhs); }

    __device__ constexpr Fp operator+(Fp rhs) const { return Fp(add(val, rhs.val), true); }

    __device__ constexpr Fp operator-() const { return Fp(sub(0, val), true); }

    __device__ constexpr Fp operator-(Fp rhs) const { return Fp(sub(val, rhs.val), true); }

    __device__ constexpr Fp operator*(Fp rhs) const { return Fp(mul(val, rhs.val), true); }

    __device__ constexpr Fp operator+=(Fp rhs) {
        val = add(val, rhs.val);
        return *this;
    }

    __device__ constexpr Fp operator-=(Fp rhs) {
        val = sub(val, rhs.val);
        return *this;
    }

    __device__ constexpr Fp operator*=(Fp rhs) {
        val = mul(val, rhs.val);
        return *this;
    }

    __device__ constexpr bool operator==(Fp rhs) const { return val == rhs.val; }

    __device__ constexpr bool operator!=(Fp rhs) const { return val != rhs.val; }

    __device__ constexpr bool operator<(Fp rhs) const { return decode(val) < decode(rhs.val); }

    __device__ constexpr bool operator<=(Fp rhs) const { return decode(val) <= decode(rhs.val); }

    __device__ constexpr bool operator>(Fp rhs) const { return decode(val) > decode(rhs.val); }

    __device__ constexpr bool operator>=(Fp rhs) const { return decode(val) >= decode(rhs.val); }

    // Post-inc/dec
    __device__ constexpr Fp operator++(int) {
        Fp r = *this;
        val = add(val, encode(1));
        return r;
    }

    __device__ constexpr Fp operator--(int) {
        Fp r = *this;
        val = sub(val, encode(1));
        return r;
    }

    // Pre-inc/dec
    __device__ constexpr Fp operator++() {
        val = add(val, encode(1));
        return *this;
    }

    __device__ constexpr Fp operator--() {
        val = sub(val, encode(1));
        return *this;
    }

    /// Given an element x from a 31 bit field F_P compute x/2.
    /// The input must be in [0, P).
    /// The output will also be in [0, P).
    static __device__ constexpr inline uint32_t halve_u32(uint32_t input) {
        uint32_t shr = input >> 1;
        uint32_t lo_bit = input & 1;
        return shr + (lo_bit * HALF_P_PLUS_1); // optimized for cuda
                                               // uint32_t shr_corr = shr + HALF_P_PLUS_1;
                                               // if (lo_bit == 0) {
                                               //   return shr;
                                               // } else {
                                               //   return shr_corr;
                                               // }
    }

    /// monty_reduce from Plonky3: monty-31/src/utils.rs
    static __device__ constexpr uint32_t monty_reduce(uint64_t x) {
        uint64_t t = (x * uint64_t(Fp::M)) & uint64_t(Fp::MONTY_MASK);
        uint64_t u = t * uint64_t(Fp::P);

        uint64_t x_sub_u = x - u;
        bool overflow = x < u; // overflowing_sub

        uint32_t x_sub_u_hi = uint32_t(x_sub_u >> Fp::MONTY_BITS);
        uint32_t corr = overflow ? (Fp::P) : 0;
        return x_sub_u_hi + corr;
    }

  public:
    /// Return a new Fp that is double of this value
    /// We use `doubled` here since 'double' is a C++ reserved keyword
    __device__ constexpr Fp doubled() const { return *this + *this; }

    /// Return a new Fp that is half of this value
    __device__ constexpr Fp halve() const { return Fp::fromRaw(halve_u32(val)); }

    /// Multiply the given MontyField31 element by `2^{-n}`.
    ///
    /// This makes use of the fact that, as the monty constant is `2^32`,
    /// the monty form of `2^{-n}` is `2^{32 - n}`. Monty reduction works
    /// provided the input is `< 2^32P` so this works for `0 <= n <= 32`.
    __device__ constexpr Fp mul_2exp_neg_n(uint32_t n) const {
        assert(n < 33 && "n must be less than 33");
        uint64_t value_mul_2exp_neg_n = static_cast<uint64_t>(val) << (32 - n);
        return Fp::fromRaw(monty_reduce(value_mul_2exp_neg_n));
    }
};

/// Raise an value to a power
__device__ constexpr inline Fp pow(Fp x, size_t n) {
    Fp tot = 1;
    while (n != 0) {
        if (n % 2 == 1) {
            tot *= x;
        }
        n = n / 2;
        x *= x;
    }
    return tot;
}

/// Helper: gcd-based inversion for 32-bit prime fields with FIELD_BITS <= 32.
/// Returns v = 2^{2*FIELD_BITS - 2} * a^{-1} mod P where FIELD_BITS = 31 and P is odd prime.
/// Copied from Plonky3: https://github.com/Plonky3/Plonky3/pull/921
static __device__ constexpr inline int64_t gcd_inversion_prime_field_32(uint32_t a, uint32_t b) {
    int64_t u = 1;
    int64_t v = 0;
    // FIELD_BITS = 31 => iterate 2*FIELD_BITS - 2 = 60 times
    for (uint32_t i = 0; i < 60; i++) {
        if ((a & 1u) != 0u) {
            if (a < b) {
                // temp variable swap for a, b and u, v
                uint32_t tmp_ab = a;
                a = b;
                b = tmp_ab;
                int64_t tmp_uv = u;
                u = v;
                v = tmp_uv;
            }
            a -= b;
            u -= v;
        }
        a >>= 1;
        v <<= 1;
    }
    return v;
}

/// Compute the multiplicative inverse of x, or `1/x` in finite field terms, using a fast
/// gcd-based algorithm specialized for 31-bit prime P.
/// For x = 0, returns 0.
__device__ constexpr inline Fp inv(Fp x) {
    if (x.asRaw() == 0u) {
        return Fp::zero();
    }

    uint32_t a = x.asUInt32();
    int64_t v = gcd_inversion_prime_field_32(a, Fp::P);
    // Bring v into [0, P)
    int64_t v_mod = v % static_cast<int64_t>(Fp::P);
    if (v_mod < 0) {
        v_mod += static_cast<int64_t>(Fp::P);
    }

    return Fp(static_cast<uint32_t>(v_mod)) * Fp(Fp::INV_2EXP_K);
}

constexpr __device__ Fp TWO_ADIC_GENERATORS[Fp::TWO_ADICITY + 1] = {
    Fp(0x1),        // Fp(0x0ffffffeu),
    Fp(0x78000000), // Fp(0x68000003u),
    Fp(0x67055c21), // Fp(0x1c38d511u),
    Fp(0x5ee99486), // Fp(0x3d85298fu),
    Fp(0xbb4c4e4),  // Fp(0x5f06e481u),
    Fp(0x2d4cc4da), // Fp(0x3f5c39ecu),
    Fp(0x669d6090), // Fp(0x5516a97au),
    Fp(0x17b56c64), // Fp(0x3d6be592u),
    Fp(0x67456167), // Fp(0x5bb04149u),
    Fp(0x688442f9), // Fp(0x4907f9abu),
    Fp(0x145e952d), // Fp(0x548b8e90u),
    Fp(0x4fe61226), // Fp(0x1d8ca617u),
    Fp(0x4c734715), // Fp(0x2ce7f0e6u),
    Fp(0x11c33e2a), // Fp(0x621b371fu),
    Fp(0x62c3d2b1), // Fp(0x6d4d2d78u),
    Fp(0x77cad399), // Fp(0x18716fcdu),
    Fp(0x54c131f4), // Fp(0x3b30a682u),
    Fp(0x4cabd6a6), // Fp(0x1c6f4728u),
    Fp(0x5cf5713f), // Fp(0x59b01f7cu),
    Fp(0x3e9430e8), // Fp(0x1a7f97acu),
    Fp(0xba067a3),  // Fp(0x0732561cu),
    Fp(0x18adc27d), // Fp(0x2b5a1cd4u),
    Fp(0x21fd55bc), // Fp(0x6f7d26f9u),
    Fp(0x4b859b3d), // Fp(0x16e2f919u),
    Fp(0x3bd57996), // Fp(0x285ab85bu),
    Fp(0x4483d85a), // Fp(0x0dd5a9ecu),
    Fp(0x3a26eef8), // Fp(0x43f13568u),
    Fp(0x1a427a41)  // Fp(0x57fab6eeu)
};