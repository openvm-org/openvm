#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"

using namespace riscv;

template <size_t NUM_LIMBS>
__device__ __forceinline__ uint64_t limbs_to_u64(const uint8_t (&limbs)[NUM_LIMBS]) {
    uint64_t val = 0;
#pragma unroll
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        val |= (uint64_t)limbs[i] << (i * RV64_CELL_BITS);
    }
    return val;
}

template <size_t NUM_LIMBS>
__device__ __forceinline__ void u64_to_limbs(uint64_t val, uint8_t (&limbs)[NUM_LIMBS]) {
#pragma unroll
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        limbs[i] = (uint8_t)(val >> (i * RV64_CELL_BITS));
    }
}

template <typename T, size_t NUM_LIMBS> struct DivRemCoreCols {
    // b = c * q + r for some 0 <= |r| < |c | and sign(r) = sign(b) or r = 0.
    T b[NUM_LIMBS];
    T c[NUM_LIMBS];
    T q[NUM_LIMBS];
    T r[NUM_LIMBS];

    // Flags to indicate special cases.
    T zero_divisor;
    T r_zero;

    // Sign of b and c respectively, while q_sign = b_sign ^ c_sign if q is non-zero
    // and is 0 otherwise. sign_xor = b_sign ^ c_sign always.
    T b_sign;
    T c_sign;
    T q_sign;
    T sign_xor;

    // Auxiliary columns to constrain that zero_divisor = 1 if and only if c = 0.
    T c_sum_inv;
    // Auxiliary columns to constrain that r_zero = 1 if and only if r = 0 and zero_divisor = 0.
    T r_sum_inv;

    // Auxiliary columns to constrain that 0 <= |r| < |c|. When sign_xor == 1 we have
    // r_prime = -r, and when sign_xor == 0 we have r_prime = r. Each r_inv[i] is the
    // field inverse of r_prime[i] - 2^RV64_CELL_BITS, ensures each r_prime[i] is in range.
    T r_prime[NUM_LIMBS];
    T r_inv[NUM_LIMBS];
    T lt_marker[NUM_LIMBS];
    T lt_diff;

    // Opcode flags
    T opcode_div_flag;
    T opcode_divu_flag;
    T opcode_rem_flag;
    T opcode_remu_flag;
};

template <size_t NUM_LIMBS> struct DivRemCoreRecords {
    uint8_t b[NUM_LIMBS];
    uint8_t c[NUM_LIMBS];
    uint8_t local_opcode;
};

enum DivRemOpcode {
    DIV,
    DIVU,
    REM,
    REMU,
};

template <size_t NUM_LIMBS> struct DivRemCore {
    BitwiseOperationLookup bitwise_lookup;
    RangeTupleChecker<2> range_tuple_checker;

    template <typename T> using Cols = DivRemCoreCols<T, NUM_LIMBS>;

    static constexpr size_t TOTAL_BITS = NUM_LIMBS * RV64_CELL_BITS;
    static_assert(TOTAL_BITS <= 64, "DivRemCore supports up to 64 total bits");
    static constexpr uint64_t VALUE_MASK =
        (TOTAL_BITS == 64) ? ~uint64_t(0) : ((uint64_t(1) << TOTAL_BITS) - 1);
    static constexpr uint64_t SIGN_BIT = uint64_t(1) << (TOTAL_BITS - 1);

    __device__ static uint64_t abs_mod(uint64_t x, bool is_neg) {
        return is_neg ? ((~x + 1) & VALUE_MASK) : x;
    }

    __device__ DivRemCore(
        BitwiseOperationLookup bitwise_lookup,
        RangeTupleChecker<2> range_tuple_checker
    )
        : bitwise_lookup(bitwise_lookup), range_tuple_checker(range_tuple_checker) {}

    __device__ void fill_trace_row(RowSlice row, DivRemCoreRecords<NUM_LIMBS> const &record) {
        DivRemOpcode opcode = static_cast<DivRemOpcode>(record.local_opcode);

        bool is_signed = opcode == DIV || opcode == REM;
        bool b_sign = is_signed && (record.b[NUM_LIMBS - 1] >> (RV64_CELL_BITS - 1));
        bool c_sign = is_signed && (record.c[NUM_LIMBS - 1] >> (RV64_CELL_BITS - 1));
        bool q_sign = false;
        bool case_none = false;

        uint64_t b_val = limbs_to_u64(record.b);
        uint64_t c_val = limbs_to_u64(record.c);
        uint64_t q_val = 0;
        uint64_t r_val = 0;

        if (c_val == 0) {
            q_val = VALUE_MASK;
            r_val = b_val;
            q_sign = is_signed;
        } else if ((b_val == SIGN_BIT) && (c_val == VALUE_MASK) && b_sign && c_sign) {
            q_val = b_val;
            r_val = 0;
            q_sign = false;
        } else {
            uint64_t b_abs = abs_mod(b_val, b_sign);
            uint64_t c_abs = abs_mod(c_val, c_sign);
            q_val = abs_mod(b_abs / c_abs, b_sign != c_sign);
            r_val = abs_mod(b_abs % c_abs, b_sign);
            q_sign = is_signed && (q_val >> (TOTAL_BITS - 1));
            case_none = true;
        }

        uint8_t q[NUM_LIMBS];
        uint8_t r[NUM_LIMBS];
        uint8_t r_prime[NUM_LIMBS];
        uint64_t r_prime_val = abs_mod(r_val, b_sign ^ c_sign);
        u64_to_limbs(q_val, q);
        u64_to_limbs(r_val, r);
        u64_to_limbs(r_prime_val, r_prime);
        bool r_zero = (r_val == 0) && (c_val != 0);

        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, c, record.c);
        COL_WRITE_ARRAY(row, Cols, q, q);
        COL_WRITE_ARRAY(row, Cols, r, r);
        COL_WRITE_VALUE(row, Cols, zero_divisor, c_val == 0);
        COL_WRITE_VALUE(row, Cols, r_zero, r_zero);
        COL_WRITE_VALUE(row, Cols, b_sign, b_sign);
        COL_WRITE_VALUE(row, Cols, c_sign, c_sign);
        COL_WRITE_VALUE(row, Cols, q_sign, q_sign);
        COL_WRITE_VALUE(row, Cols, sign_xor, b_sign ^ c_sign);

        uint64_t c_sum = 0;
        uint64_t r_sum = 0;
#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            c_sum += record.c[i];
            r_sum += r[i];
        }
        if (c_sum == 0) {
            COL_WRITE_VALUE(row, Cols, c_sum_inv, 0);
        } else {
            COL_WRITE_VALUE(row, Cols, c_sum_inv, inv(Fp(c_sum)));
        }

        if (r_sum == 0) {
            COL_WRITE_VALUE(row, Cols, r_sum_inv, 0);
        } else {
            COL_WRITE_VALUE(row, Cols, r_sum_inv, inv(Fp(r_sum)));
        }

        COL_WRITE_ARRAY(row, Cols, r_prime, r_prime);
#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            Fp r_inv = inv(Fp(r_prime[i]) - Fp(256));
            COL_WRITE_VALUE(row, Cols, r_inv[i], r_inv);
        }

        COL_FILL_ZERO(row, Cols, lt_marker);
        if (case_none && !r_zero) {
            uint32_t idx = NUM_LIMBS;
#pragma unroll
            for (int i = NUM_LIMBS - 1; i >= 0; i--) {
                if (record.c[i] != r_prime[i]) {
                    idx = i;
                    break;
                }
            }
            uint8_t val = 0;
            if (c_sign) {
                val = r_prime[idx] - record.c[idx];
            } else {
                val = record.c[idx] - r_prime[idx];
            }
            bitwise_lookup.add_range(val - 1, 0);
            COL_WRITE_VALUE(row, Cols, lt_marker[idx], 1);
            COL_WRITE_VALUE(row, Cols, lt_diff, val);
        } else {
            COL_WRITE_VALUE(row, Cols, lt_diff, 0);
        }

        COL_WRITE_VALUE(row, Cols, opcode_div_flag, opcode == DIV);
        COL_WRITE_VALUE(row, Cols, opcode_divu_flag, opcode == DIVU);
        COL_WRITE_VALUE(row, Cols, opcode_rem_flag, opcode == REM);
        COL_WRITE_VALUE(row, Cols, opcode_remu_flag, opcode == REMU);

        if (is_signed) {
            bitwise_lookup.add_range(
                (record.b[NUM_LIMBS - 1] & 0x7f) << 1, (record.c[NUM_LIMBS - 1] & 0x7f) << 1
            );
        }

        // range tuple check carries
        uint32_t carry = 0;
#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            carry += r[i];
#pragma unroll
            for (size_t j = 0; j <= i; j++) {
                carry += (uint32_t)q[j] * (uint32_t)record.c[i - j];
            }
            carry = carry >> RV64_CELL_BITS;
            range_tuple_checker.add_count((uint32_t[2]){(uint32_t)q[i], carry});
        }
        bool r_sign = is_signed && (r[NUM_LIMBS - 1] >> (RV64_CELL_BITS - 1));

        uint32_t q_ext = (q_sign && is_signed) * ((1 << RV64_CELL_BITS) - 1);
        uint32_t c_ext = (c_sign << RV64_CELL_BITS) - c_sign;
        uint32_t r_ext = (r_sign << RV64_CELL_BITS) - r_sign;

        uint32_t c_pref = 0;
        uint32_t q_pref = 0;
#pragma unroll
        for (size_t i = 0; i < NUM_LIMBS; i++) {
            c_pref += record.c[i];
            q_pref += q[i];
            carry += c_pref * q_ext + q_pref * c_ext + r_ext;
#pragma unroll
            for (size_t j = i + 1; j < NUM_LIMBS; j++) {
                carry += (uint32_t)record.c[j] * (uint32_t)q[NUM_LIMBS + i - j];
            }
            carry = carry >> RV64_CELL_BITS;
            range_tuple_checker.add_count((uint32_t[2]){(uint32_t)r[i], carry});
        }
    }
};
