#pragma once

#include "riscv/cores/divrem.cuh"

template <size_t NUM_LIMBS>
static __device__ void replay_divrem_values(
    uint8_t const (&b)[8], uint8_t const (&c)[8], DivRemOpcode opcode,
    uint8_t (&quotient)[NUM_LIMBS], uint8_t (&remainder)[NUM_LIMBS]
) {
    constexpr size_t TOTAL_BITS = NUM_LIMBS * BYTE_BITS;
    constexpr uint64_t MASK = TOTAL_BITS == 64 ? ~uint64_t(0) : ((uint64_t(1) << TOTAL_BITS) - 1);
    constexpr uint64_t SIGN = uint64_t(1) << (TOTAL_BITS - 1);
    uint64_t b_value = 0;
    uint64_t c_value = 0;
#pragma unroll
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        b_value |= uint64_t(b[i]) << (i * BYTE_BITS);
        c_value |= uint64_t(c[i]) << (i * BYTE_BITS);
    }
    bool is_signed = opcode == DIV || opcode == REM;
    bool b_negative = is_signed && (b_value & SIGN) != 0;
    bool c_negative = is_signed && (c_value & SIGN) != 0;
    uint64_t q_value;
    uint64_t r_value;
    if (c_value == 0) {
        q_value = MASK;
        r_value = b_value;
    } else if (b_negative && c_negative && b_value == SIGN && c_value == MASK) {
        q_value = b_value;
        r_value = 0;
    } else {
        uint64_t b_abs = b_negative ? ((~b_value + 1) & MASK) : b_value;
        uint64_t c_abs = c_negative ? ((~c_value + 1) & MASK) : c_value;
        uint64_t q_abs = b_abs / c_abs;
        uint64_t r_abs = b_abs % c_abs;
        q_value = b_negative != c_negative ? ((~q_abs + 1) & MASK) : q_abs;
        r_value = b_negative ? ((~r_abs + 1) & MASK) : r_abs;
    }
#pragma unroll
    for (size_t i = 0; i < NUM_LIMBS; i++) {
        quotient[i] = static_cast<uint8_t>(q_value >> (i * BYTE_BITS));
        remainder[i] = static_cast<uint8_t>(r_value >> (i * BYTE_BITS));
    }
}
