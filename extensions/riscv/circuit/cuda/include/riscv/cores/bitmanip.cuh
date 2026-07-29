#pragma once

#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"

using namespace riscv;

constexpr size_t BITMANIP_NUM_LIMBS = BLOCK_FE_WIDTH;
constexpr size_t BITMANIP_NUM_BITS = BITMANIP_NUM_LIMBS * U16_BITS;
constexpr size_t BITMANIP_SHADD_OP_COUNT = 7;
constexpr size_t BITMANIP_REG_OP_COUNT = 15;
constexpr size_t BITMANIP_IMM_OP_COUNT = 17;

constexpr uint8_t SH1ADD = 0;
constexpr uint8_t SH2ADD = 1;
constexpr uint8_t SH3ADD = 2;
constexpr uint8_t ADD_UW = 3;
constexpr uint8_t SH1ADD_UW = 4;
constexpr uint8_t SH2ADD_UW = 5;
constexpr uint8_t SH3ADD_UW = 6;
constexpr uint8_t SLLI_UW = 7;
constexpr uint8_t ANDN = 8;
constexpr uint8_t ORN = 9;
constexpr uint8_t XNOR = 10;
constexpr uint8_t ROL = 11;
constexpr uint8_t ROR = 12;
constexpr uint8_t RORI = 13;
constexpr uint8_t ROLW = 14;
constexpr uint8_t RORW = 15;
constexpr uint8_t RORIW = 16;
constexpr uint8_t CLZ = 17;
constexpr uint8_t CTZ = 18;
constexpr uint8_t CLZW = 19;
constexpr uint8_t CTZW = 20;
constexpr uint8_t CPOP = 21;
constexpr uint8_t CPOPW = 22;
constexpr uint8_t OP_MIN = 23;
constexpr uint8_t MINU = 24;
constexpr uint8_t OP_MAX = 25;
constexpr uint8_t MAXU = 26;
constexpr uint8_t SEXT_B = 27;
constexpr uint8_t SEXT_H = 28;
constexpr uint8_t ZEXT_H = 29;
constexpr uint8_t ORC_B = 30;
constexpr uint8_t REV8 = 31;
constexpr uint8_t BCLR = 32;
constexpr uint8_t BSET = 33;
constexpr uint8_t BINV = 34;
constexpr uint8_t BEXT = 35;
constexpr uint8_t BCLRI = 36;
constexpr uint8_t BSETI = 37;
constexpr uint8_t BINVI = 38;
constexpr uint8_t BEXTI = 39;

__device__ __forceinline__ int bitmanip_shadd_flag_pos(uint8_t op) {
    constexpr uint8_t OPS[BITMANIP_SHADD_OP_COUNT] = {
        SH1ADD, SH2ADD, SH3ADD, ADD_UW, SH1ADD_UW, SH2ADD_UW, SH3ADD_UW,
    };
    for (size_t i = 0; i < BITMANIP_SHADD_OP_COUNT; i++) {
        if (OPS[i] == op) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

__device__ __forceinline__ int bitmanip_reg_flag_pos(uint8_t op) {
    constexpr uint8_t OPS[BITMANIP_REG_OP_COUNT] = {
        ANDN, ORN, XNOR, ROL, ROR, ROLW, RORW, OP_MIN, MINU, OP_MAX, MAXU, BCLR, BSET, BINV, BEXT,
    };
    for (size_t i = 0; i < BITMANIP_REG_OP_COUNT; i++) {
        if (OPS[i] == op) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

__device__ __forceinline__ int bitmanip_imm_flag_pos(uint8_t op) {
    constexpr uint8_t OPS[BITMANIP_IMM_OP_COUNT] = {
        RORI, RORIW, CLZ, CTZ, CLZW, CTZW, CPOP, CPOPW, SEXT_B, SEXT_H, ZEXT_H, ORC_B,
        REV8, BCLRI, BSETI, BINVI, BEXTI,
    };
    for (size_t i = 0; i < BITMANIP_IMM_OP_COUNT; i++) {
        if (OPS[i] == op) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

__device__ __forceinline__ uint64_t bitmanip_limbs_to_u64(const uint16_t limbs[BITMANIP_NUM_LIMBS]) {
    uint64_t value = 0;
    for (size_t i = 0; i < BITMANIP_NUM_LIMBS; i++) {
        value |= static_cast<uint64_t>(limbs[i]) << (U16_BITS * i);
    }
    return value;
}

__device__ __forceinline__ void
bitmanip_u64_to_limbs(uint64_t value, uint16_t limbs[BITMANIP_NUM_LIMBS]) {
    for (size_t i = 0; i < BITMANIP_NUM_LIMBS; i++) {
        limbs[i] = static_cast<uint16_t>((value >> (U16_BITS * i)) & 0xffffull);
    }
}

__device__ __forceinline__ void bitmanip_bits_u64(uint64_t value, uint8_t bits[BITMANIP_NUM_BITS]) {
    for (size_t i = 0; i < BITMANIP_NUM_BITS; i++) {
        bits[i] = static_cast<uint8_t>((value >> i) & 1ull);
    }
}

__device__ __forceinline__ uint64_t bitmanip_rol64(uint64_t value, uint32_t shamt) {
    shamt &= 63u;
    return shamt == 0 ? value : ((value << shamt) | (value >> (64u - shamt)));
}

__device__ __forceinline__ uint64_t bitmanip_ror64(uint64_t value, uint32_t shamt) {
    shamt &= 63u;
    return shamt == 0 ? value : ((value >> shamt) | (value << (64u - shamt)));
}

__device__ __forceinline__ uint32_t bitmanip_rol32(uint32_t value, uint32_t shamt) {
    shamt &= 31u;
    return shamt == 0 ? value : ((value << shamt) | (value >> (32u - shamt)));
}

__device__ __forceinline__ uint32_t bitmanip_ror32(uint32_t value, uint32_t shamt) {
    shamt &= 31u;
    return shamt == 0 ? value : ((value >> shamt) | (value << (32u - shamt)));
}

__device__ __forceinline__ uint64_t bitmanip_sext32(uint32_t value) {
    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(value)));
}

__device__ __forceinline__ uint64_t bitmanip_sext16(uint16_t value) {
    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int16_t>(value)));
}

__device__ __forceinline__ uint64_t bitmanip_sext8(uint8_t value) {
    return static_cast<uint64_t>(static_cast<int64_t>(static_cast<int8_t>(value)));
}

__device__ __forceinline__ bool bitmanip_signed_lt(uint64_t lhs, uint64_t rhs) {
    return (lhs ^ (1ull << 63)) < (rhs ^ (1ull << 63));
}

__device__ __forceinline__ uint32_t bitmanip_clz64(uint64_t value) {
    if (value == 0) {
        return 64;
    }
    uint32_t count = 0;
    for (int i = 63; i >= 0 && ((value >> i) & 1ull) == 0; i--) {
        count++;
    }
    return count;
}

__device__ __forceinline__ uint32_t bitmanip_ctz64(uint64_t value) {
    if (value == 0) {
        return 64;
    }
    uint32_t count = 0;
    for (size_t i = 0; i < 64 && ((value >> i) & 1ull) == 0; i++) {
        count++;
    }
    return count;
}

__device__ __forceinline__ uint32_t bitmanip_clz32(uint32_t value) {
    if (value == 0) {
        return 32;
    }
    uint32_t count = 0;
    for (int i = 31; i >= 0 && ((value >> i) & 1u) == 0; i--) {
        count++;
    }
    return count;
}

__device__ __forceinline__ uint32_t bitmanip_ctz32(uint32_t value) {
    if (value == 0) {
        return 32;
    }
    uint32_t count = 0;
    for (size_t i = 0; i < 32 && ((value >> i) & 1u) == 0; i++) {
        count++;
    }
    return count;
}

__device__ __forceinline__ uint32_t bitmanip_cpop64(uint64_t value) {
    uint32_t count = 0;
    for (size_t i = 0; i < 64; i++) {
        count += static_cast<uint32_t>((value >> i) & 1ull);
    }
    return count;
}

__device__ __forceinline__ uint32_t bitmanip_cpop32(uint32_t value) {
    uint32_t count = 0;
    for (size_t i = 0; i < 32; i++) {
        count += (value >> i) & 1u;
    }
    return count;
}

__device__ __forceinline__ uint64_t bitmanip_orc_b(uint64_t value) {
    uint64_t out = 0;
    for (size_t byte = 0; byte < 8; byte++) {
        if (((value >> (byte * 8)) & 0xffull) != 0) {
            out |= 0xffull << (byte * 8);
        }
    }
    return out;
}

__device__ __forceinline__ uint64_t bitmanip_rev8(uint64_t value) {
    uint64_t out = 0;
    for (size_t byte = 0; byte < 8; byte++) {
        out |= ((value >> (byte * 8)) & 0xffull) << ((7 - byte) * 8);
    }
    return out;
}

__device__ __forceinline__ uint64_t
bitmanip_run_reg(uint8_t local_opcode, uint64_t rs1, uint64_t rs2) {
    uint32_t shamt64 = static_cast<uint32_t>(rs2 & 63ull);
    uint32_t shamt32 = static_cast<uint32_t>(rs2 & 31ull);
    switch (local_opcode) {
    case SH1ADD:
        return (rs1 << 1) + rs2;
    case SH2ADD:
        return (rs1 << 2) + rs2;
    case SH3ADD:
        return (rs1 << 3) + rs2;
    case ADD_UW:
        return static_cast<uint64_t>(static_cast<uint32_t>(rs1)) + rs2;
    case SH1ADD_UW:
        return (static_cast<uint64_t>(static_cast<uint32_t>(rs1)) << 1) + rs2;
    case SH2ADD_UW:
        return (static_cast<uint64_t>(static_cast<uint32_t>(rs1)) << 2) + rs2;
    case SH3ADD_UW:
        return (static_cast<uint64_t>(static_cast<uint32_t>(rs1)) << 3) + rs2;
    case ANDN:
        return rs1 & ~rs2;
    case ORN:
        return rs1 | ~rs2;
    case XNOR:
        return ~(rs1 ^ rs2);
    case ROL:
        return bitmanip_rol64(rs1, shamt64);
    case ROR:
        return bitmanip_ror64(rs1, shamt64);
    case ROLW:
        return bitmanip_sext32(bitmanip_rol32(static_cast<uint32_t>(rs1), shamt32));
    case RORW:
        return bitmanip_sext32(bitmanip_ror32(static_cast<uint32_t>(rs1), shamt32));
    case OP_MIN:
        return bitmanip_signed_lt(rs1, rs2) ? rs1 : rs2;
    case MINU:
        return rs1 < rs2 ? rs1 : rs2;
    case OP_MAX:
        return bitmanip_signed_lt(rs1, rs2) ? rs2 : rs1;
    case MAXU:
        return rs1 < rs2 ? rs2 : rs1;
    case BCLR:
        return rs1 & ~(1ull << shamt64);
    case BSET:
        return rs1 | (1ull << shamt64);
    case BINV:
        return rs1 ^ (1ull << shamt64);
    case BEXT:
        return (rs1 >> shamt64) & 1ull;
    default:
        return 0;
    }
}

__device__ __forceinline__ uint64_t
bitmanip_run_imm(uint8_t local_opcode, uint64_t rs1, uint32_t imm) {
    switch (local_opcode) {
    case SLLI_UW:
        return static_cast<uint64_t>(static_cast<uint32_t>(rs1)) << imm;
    case RORI:
        return bitmanip_ror64(rs1, imm);
    case RORIW:
        return bitmanip_sext32(bitmanip_ror32(static_cast<uint32_t>(rs1), imm));
    case CLZ:
        return bitmanip_clz64(rs1);
    case CTZ:
        return bitmanip_ctz64(rs1);
    case CLZW:
        return bitmanip_clz32(static_cast<uint32_t>(rs1));
    case CTZW:
        return bitmanip_ctz32(static_cast<uint32_t>(rs1));
    case CPOP:
        return bitmanip_cpop64(rs1);
    case CPOPW:
        return bitmanip_cpop32(static_cast<uint32_t>(rs1));
    case SEXT_B:
        return bitmanip_sext8(static_cast<uint8_t>(rs1));
    case SEXT_H:
        return bitmanip_sext16(static_cast<uint16_t>(rs1));
    case ZEXT_H:
        return static_cast<uint64_t>(static_cast<uint16_t>(rs1));
    case ORC_B:
        return bitmanip_orc_b(rs1);
    case REV8:
        return bitmanip_rev8(rs1);
    case BCLRI:
        return rs1 & ~(1ull << imm);
    case BSETI:
        return rs1 | (1ull << imm);
    case BINVI:
        return rs1 ^ (1ull << imm);
    case BEXTI:
        return (rs1 >> imm) & 1ull;
    default:
        return 0;
    }
}

struct BitManipShAddCoreRecord {
    uint16_t b[BITMANIP_NUM_LIMBS];
    uint16_t c[BITMANIP_NUM_LIMBS];
    uint8_t local_opcode;
};

static_assert(sizeof(BitManipShAddCoreRecord) == 18);

template <typename T> struct BitManipShAddCoreCols {
    T a[BITMANIP_NUM_LIMBS];
    T b[BITMANIP_NUM_LIMBS];
    T c[BITMANIP_NUM_LIMBS];
    T opcode_flags[BITMANIP_SHADD_OP_COUNT];
    T bit_shift_carry[BITMANIP_NUM_LIMBS];
    T bit_shift_aux[BITMANIP_NUM_LIMBS];
    T add_carry[BITMANIP_NUM_LIMBS + 1];
};

__device__ __forceinline__ void bitmanip_shadd_shift_uw(
    uint8_t local_opcode,
    uint32_t *shift,
    bool *uw
) {
    *shift = 0;
    *uw = false;
    switch (local_opcode) {
    case SH1ADD:
        *shift = 1;
        break;
    case SH2ADD:
        *shift = 2;
        break;
    case SH3ADD:
        *shift = 3;
        break;
    case ADD_UW:
        *uw = true;
        break;
    case SH1ADD_UW:
        *shift = 1;
        *uw = true;
        break;
    case SH2ADD_UW:
        *shift = 2;
        *uw = true;
        break;
    case SH3ADD_UW:
        *shift = 3;
        *uw = true;
        break;
    }
}

struct BitManipShAddCore {
    VariableRangeChecker range_checker;

    template <typename T> using Cols = BitManipShAddCoreCols<T>;

    __device__ BitManipShAddCore(VariableRangeChecker rc) : range_checker(rc) {}

    __device__ void fill_trace_row(RowSlice row, BitManipShAddCoreRecord record) {
        uint64_t b_u64 = bitmanip_limbs_to_u64(record.b);
        uint64_t c_u64 = bitmanip_limbs_to_u64(record.c);
        uint64_t a_u64 = bitmanip_run_reg(record.local_opcode, b_u64, c_u64);

        uint16_t a[BITMANIP_NUM_LIMBS];
        uint8_t opcode_flags[BITMANIP_SHADD_OP_COUNT] = {0};
        uint16_t bit_shift_carry[BITMANIP_NUM_LIMBS] = {0};
        uint16_t bit_shift_aux[BITMANIP_NUM_LIMBS] = {0};
        uint8_t add_carry[BITMANIP_NUM_LIMBS + 1] = {0};

        bitmanip_u64_to_limbs(a_u64, a);

        int flag_pos = bitmanip_shadd_flag_pos(record.local_opcode);
        if (flag_pos >= 0) {
            opcode_flags[flag_pos] = 1;
        }

        uint32_t shift;
        bool uw;
        bitmanip_shadd_shift_uw(record.local_opcode, &shift, &uw);
        uint32_t carry_mask = (1u << shift) - 1;
        uint32_t aux_mask = (1u << (U16_BITS - shift)) - 1;
        for (size_t limb = 0; limb < BITMANIP_NUM_LIMBS; limb++) {
            uint32_t source = (uw && limb >= 2) ? 0 : record.b[limb];
            bit_shift_aux[limb] = source & aux_mask;
            bit_shift_carry[limb] = (source >> (U16_BITS - shift)) & carry_mask;
            range_checker.add_count(bit_shift_carry[limb], shift);
            range_checker.add_count(bit_shift_aux[limb], U16_BITS - shift);
        }

        uint64_t shifted =
            uw ? (static_cast<uint64_t>(static_cast<uint32_t>(b_u64)) << shift)
               : (b_u64 << shift);
        uint8_t carry = 0;
        for (size_t bit = 0; bit < BITMANIP_NUM_BITS; bit++) {
            if (bit % U16_BITS == 0) {
                add_carry[bit / U16_BITS] = carry;
            }
            uint8_t total = static_cast<uint8_t>(
                ((shifted >> bit) & 1ull) + ((c_u64 >> bit) & 1ull) + carry
            );
            carry = total >> 1;
        }
        add_carry[BITMANIP_NUM_LIMBS] = carry;

        COL_WRITE_ARRAY(row, Cols, a, a);
        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, c, record.c);
        COL_WRITE_ARRAY(row, Cols, opcode_flags, opcode_flags);
        COL_WRITE_ARRAY(row, Cols, bit_shift_carry, bit_shift_carry);
        COL_WRITE_ARRAY(row, Cols, bit_shift_aux, bit_shift_aux);
        COL_WRITE_ARRAY(row, Cols, add_carry, add_carry);
    }
};

struct BitManipSlliUwCoreRecord {
    uint16_t b[BITMANIP_NUM_LIMBS];
    uint8_t imm;
};

static_assert(sizeof(BitManipSlliUwCoreRecord) == 10);

template <typename T> struct BitManipSlliUwCoreCols {
    T a[BITMANIP_NUM_LIMBS];
    T b[BITMANIP_NUM_LIMBS];
    T bit_shift_marker[U16_BITS];
    T limb_shift_marker[BITMANIP_NUM_LIMBS];
    T bit_shift_carry[BITMANIP_NUM_LIMBS];
    T bit_shift_aux[BITMANIP_NUM_LIMBS];
};

struct BitManipSlliUwCore {
    VariableRangeChecker range_checker;

    template <typename T> using Cols = BitManipSlliUwCoreCols<T>;

    __device__ BitManipSlliUwCore(VariableRangeChecker rc) : range_checker(rc) {}

    __device__ void fill_trace_row(RowSlice row, BitManipSlliUwCoreRecord record) {
        uint64_t b_u64 = bitmanip_limbs_to_u64(record.b);
        uint64_t a_u64 = bitmanip_run_imm(SLLI_UW, b_u64, record.imm);
        uint32_t bit_shift = record.imm % U16_BITS;
        uint32_t limb_shift = record.imm / U16_BITS;

        uint16_t a[BITMANIP_NUM_LIMBS];
        uint8_t bit_shift_marker[U16_BITS] = {0};
        uint8_t limb_shift_marker[BITMANIP_NUM_LIMBS] = {0};
        uint16_t bit_shift_carry[BITMANIP_NUM_LIMBS] = {0};
        uint16_t bit_shift_aux[BITMANIP_NUM_LIMBS] = {0};

        bitmanip_u64_to_limbs(a_u64, a);
        bit_shift_marker[bit_shift] = 1;
        limb_shift_marker[limb_shift] = 1;

        uint32_t carry_mask = (1u << bit_shift) - 1;
        uint32_t aux_mask = (1u << (U16_BITS - bit_shift)) - 1;
        for (size_t limb = 0; limb < BITMANIP_NUM_LIMBS; limb++) {
            uint32_t source = limb >= 2 ? 0 : record.b[limb];
            bit_shift_aux[limb] = source & aux_mask;
            bit_shift_carry[limb] = (source >> (U16_BITS - bit_shift)) & carry_mask;
            range_checker.add_count(bit_shift_carry[limb], bit_shift);
            range_checker.add_count(bit_shift_aux[limb], U16_BITS - bit_shift);
        }

        COL_WRITE_ARRAY(row, Cols, a, a);
        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, bit_shift_marker, bit_shift_marker);
        COL_WRITE_ARRAY(row, Cols, limb_shift_marker, limb_shift_marker);
        COL_WRITE_ARRAY(row, Cols, bit_shift_carry, bit_shift_carry);
        COL_WRITE_ARRAY(row, Cols, bit_shift_aux, bit_shift_aux);
    }
};

struct BitManipRegCoreRecord {
    uint16_t b[BITMANIP_NUM_LIMBS];
    uint16_t c[BITMANIP_NUM_LIMBS];
    uint8_t local_opcode;
};

static_assert(sizeof(BitManipRegCoreRecord) == 18);

template <typename T> struct BitManipRegCoreCols {
    T a[BITMANIP_NUM_LIMBS];
    T b[BITMANIP_NUM_LIMBS];
    T c[BITMANIP_NUM_LIMBS];
    T a_bits[BITMANIP_NUM_BITS];
    T b_bits[BITMANIP_NUM_BITS];
    T c_bits[BITMANIP_NUM_BITS];
    T opcode_flags[BITMANIP_REG_OP_COUNT];
    T index_marker[BITMANIP_NUM_BITS];
    T minmax_lt;
    T minmax_diff_marker[BITMANIP_NUM_BITS];
};

struct BitManipRegCore {
    template <typename T> using Cols = BitManipRegCoreCols<T>;

    __device__ void fill_trace_row(RowSlice row, BitManipRegCoreRecord record) {
        uint64_t b_u64 = bitmanip_limbs_to_u64(record.b);
        uint64_t c_u64 = bitmanip_limbs_to_u64(record.c);
        uint64_t a_u64 = bitmanip_run_reg(record.local_opcode, b_u64, c_u64);

        uint16_t a[BITMANIP_NUM_LIMBS];
        uint8_t a_bits[BITMANIP_NUM_BITS];
        uint8_t b_bits[BITMANIP_NUM_BITS];
        uint8_t c_bits[BITMANIP_NUM_BITS];
        uint8_t opcode_flags[BITMANIP_REG_OP_COUNT] = {0};
        uint8_t index_marker[BITMANIP_NUM_BITS] = {0};
        uint8_t minmax_diff_marker[BITMANIP_NUM_BITS] = {0};
        uint8_t minmax_lt = 0;

        bitmanip_u64_to_limbs(a_u64, a);
        bitmanip_bits_u64(a_u64, a_bits);
        bitmanip_bits_u64(b_u64, b_bits);
        bitmanip_bits_u64(c_u64, c_bits);

        int flag_pos = bitmanip_reg_flag_pos(record.local_opcode);
        if (flag_pos >= 0) {
            opcode_flags[flag_pos] = 1;
        }

        if (record.local_opcode == ROL || record.local_opcode == ROR || record.local_opcode == BCLR ||
            record.local_opcode == BSET || record.local_opcode == BINV || record.local_opcode == BEXT) {
            index_marker[c_u64 & 63ull] = 1;
        } else if (record.local_opcode == ROLW || record.local_opcode == RORW) {
            index_marker[c_u64 & 31ull] = 1;
        }

        if (record.local_opcode == OP_MIN || record.local_opcode == MINU ||
            record.local_opcode == OP_MAX || record.local_opcode == MAXU) {
            bool lt = (record.local_opcode == OP_MIN || record.local_opcode == OP_MAX)
                          ? bitmanip_signed_lt(b_u64, c_u64)
                          : (b_u64 < c_u64);
            minmax_lt = lt ? 1 : 0;
            uint64_t diff = b_u64 ^ c_u64;
            if (diff != 0) {
                minmax_diff_marker[63 - bitmanip_clz64(diff)] = 1;
            }
        }

        COL_WRITE_ARRAY(row, Cols, a, a);
        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, c, record.c);
        COL_WRITE_ARRAY(row, Cols, a_bits, a_bits);
        COL_WRITE_ARRAY(row, Cols, b_bits, b_bits);
        COL_WRITE_ARRAY(row, Cols, c_bits, c_bits);
        COL_WRITE_ARRAY(row, Cols, opcode_flags, opcode_flags);
        COL_WRITE_ARRAY(row, Cols, index_marker, index_marker);
        COL_WRITE_VALUE(row, Cols, minmax_lt, minmax_lt);
        COL_WRITE_ARRAY(row, Cols, minmax_diff_marker, minmax_diff_marker);
    }
};

struct BitManipImmCoreRecord {
    uint16_t b[BITMANIP_NUM_LIMBS];
    uint8_t imm;
    uint8_t local_opcode;
};

static_assert(sizeof(BitManipImmCoreRecord) == 10);

template <typename T> struct BitManipImmCoreCols {
    T a[BITMANIP_NUM_LIMBS];
    T b[BITMANIP_NUM_LIMBS];
    T a_bits[BITMANIP_NUM_BITS];
    T b_bits[BITMANIP_NUM_BITS];
    T opcode_flags[BITMANIP_IMM_OP_COUNT];
    T index_marker[BITMANIP_NUM_BITS];
    T count_marker[BITMANIP_NUM_BITS + 1];
    T byte_nonzero[8];
    T byte_nonzero_inv[8];
};

struct BitManipImmCore {
    template <typename T> using Cols = BitManipImmCoreCols<T>;

    __device__ void fill_trace_row(RowSlice row, BitManipImmCoreRecord record) {
        uint64_t b_u64 = bitmanip_limbs_to_u64(record.b);
        uint64_t a_u64 = bitmanip_run_imm(record.local_opcode, b_u64, record.imm);

        uint16_t a[BITMANIP_NUM_LIMBS];
        uint8_t a_bits[BITMANIP_NUM_BITS];
        uint8_t b_bits[BITMANIP_NUM_BITS];
        uint8_t opcode_flags[BITMANIP_IMM_OP_COUNT] = {0};
        uint8_t index_marker[BITMANIP_NUM_BITS] = {0};
        uint8_t count_marker[BITMANIP_NUM_BITS + 1] = {0};
        uint8_t byte_nonzero[8] = {0};
        Fp byte_nonzero_inv[8];
        for (size_t byte = 0; byte < 8; byte++) {
            byte_nonzero_inv[byte] = Fp::zero();
        }

        bitmanip_u64_to_limbs(a_u64, a);
        bitmanip_bits_u64(a_u64, a_bits);
        bitmanip_bits_u64(b_u64, b_bits);

        int flag_pos = bitmanip_imm_flag_pos(record.local_opcode);
        if (flag_pos >= 0) {
            opcode_flags[flag_pos] = 1;
        }

        if (record.local_opcode == RORI || record.local_opcode == RORIW ||
            record.local_opcode == BCLRI || record.local_opcode == BSETI ||
            record.local_opcode == BINVI || record.local_opcode == BEXTI) {
            index_marker[record.imm] = 1;
        }

        if (record.local_opcode == CLZ || record.local_opcode == CTZ ||
            record.local_opcode == CLZW || record.local_opcode == CTZW) {
            count_marker[a_u64] = 1;
        }

        if (record.local_opcode == ORC_B) {
            for (size_t byte = 0; byte < 8; byte++) {
                uint32_t count = 0;
                for (size_t bit = 0; bit < 8; bit++) {
                    count += b_bits[byte * 8 + bit];
                }
                if (count != 0) {
                    byte_nonzero[byte] = 1;
                    byte_nonzero_inv[byte] = inv(Fp(count));
                }
            }
        }

        COL_WRITE_ARRAY(row, Cols, a, a);
        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, a_bits, a_bits);
        COL_WRITE_ARRAY(row, Cols, b_bits, b_bits);
        COL_WRITE_ARRAY(row, Cols, opcode_flags, opcode_flags);
        COL_WRITE_ARRAY(row, Cols, index_marker, index_marker);
        COL_WRITE_ARRAY(row, Cols, count_marker, count_marker);
        COL_WRITE_ARRAY(row, Cols, byte_nonzero, byte_nonzero);
        COL_WRITE_ARRAY(row, Cols, byte_nonzero_inv, byte_nonzero_inv);
    }
};
