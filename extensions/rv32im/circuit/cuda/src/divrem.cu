#include "launcher.cuh"
#include "primitives/buffer_view.cuh"
#include "primitives/constants.h"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "primitives/utils.cuh"
#include "rv32im/adapters/mul.cuh"

using namespace riscv;

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
    // field inverse of r_prime[i] - 2^RV32_CELL_BITS, ensures each r_prime[i] is in range.
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

__device__ __forceinline__ uint32_t abs_u32(uint32_t x, bool is_neg) {
    if (is_neg) {
        return UINT32_MAX - x + 1;
    } else {
        return x;
    }
}

template <size_t NUM_LIMBS> struct DivRemCore {
    BitwiseOperationLookup bitwise_lookup;
    RangeTupleChecker<2> range_tuple_checker;

    template <typename T> using Cols = DivRemCoreCols<T, NUM_LIMBS>;

    __device__ DivRemCore(
        BitwiseOperationLookup bitwise_lookup,
        RangeTupleChecker<2> range_tuple_checker
    )
        : bitwise_lookup(bitwise_lookup), range_tuple_checker(range_tuple_checker) {}

    __device__ void fill_trace_row(RowSlice row, DivRemCoreRecords<NUM_LIMBS> const &record) {
        DivRemOpcode opcode = static_cast<DivRemOpcode>(record.local_opcode);

        bool is_signed = opcode == DIV || opcode == REM;
        bool b_sign = is_signed && (record.b[3] >> 7);
        bool c_sign = is_signed && (record.c[3] >> 7);
        bool q_sign = false;
        bool case_none = false;
        uint32_t b_u32 = u32_from_bytes_le(record.b);
        uint32_t c_u32 = u32_from_bytes_le(record.c);
        uint32_t q_u32 = 0;
        uint32_t r_u32 = 0;

        if (c_u32 == 0) {
            q_u32 = UINT32_MAX;
            r_u32 = b_u32;
            q_sign = is_signed;
        } else if ((b_u32 == (1U << 31)) && (c_u32 == UINT32_MAX) && b_sign && c_sign) {
            q_u32 = b_u32;
            r_u32 = 0;
            q_sign = false;
        } else {
            uint32_t b_abs = abs_u32(b_u32, b_sign);
            uint32_t c_abs = abs_u32(c_u32, c_sign);
            q_u32 = abs_u32(b_abs / c_abs, b_sign != c_sign);
            r_u32 = abs_u32(b_abs % c_abs, b_sign);
            q_sign = is_signed && (q_u32 >> 31);
            case_none = true;
        }

        uint8_t *q = reinterpret_cast<uint8_t *>(&q_u32);
        uint8_t *r = reinterpret_cast<uint8_t *>(&r_u32);
        uint32_t r_prime_u32 = abs_u32(r_u32, b_sign ^ c_sign);
        uint8_t *r_prime = reinterpret_cast<uint8_t *>(&r_prime_u32);
        bool r_zero = (r_u32 == 0) && (c_u32 != 0);

        COL_WRITE_ARRAY(row, Cols, b, record.b);
        COL_WRITE_ARRAY(row, Cols, c, record.c);
        COL_WRITE_ARRAY(row, Cols, q, q);
        COL_WRITE_ARRAY(row, Cols, r, r);
        COL_WRITE_VALUE(row, Cols, zero_divisor, c_u32 == 0);
        COL_WRITE_VALUE(row, Cols, r_zero, r_zero);
        COL_WRITE_VALUE(row, Cols, b_sign, b_sign);
        COL_WRITE_VALUE(row, Cols, c_sign, c_sign);
        COL_WRITE_VALUE(row, Cols, q_sign, q_sign);
        COL_WRITE_VALUE(row, Cols, sign_xor, b_sign ^ c_sign);

        uint32_t c_sum = 0;
        uint32_t r_sum = 0;
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
            carry = carry >> RV32_CELL_BITS;
            range_tuple_checker.add_count((uint32_t[2]){(uint32_t)q[i], carry});
        }
        bool r_sign = is_signed && (r[NUM_LIMBS - 1] >> (RV32_CELL_BITS - 1));

        uint32_t q_ext = (q_sign && is_signed) * ((1 << RV32_CELL_BITS) - 1);
        uint32_t c_ext = (c_sign << RV32_CELL_BITS) - c_sign;
        uint32_t r_ext = (r_sign << RV32_CELL_BITS) - r_sign;

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
            carry = carry >> RV32_CELL_BITS;
            range_tuple_checker.add_count((uint32_t[2]){(uint32_t)r[i], carry});
        }
    }
};

// Below is `rv32` specific code.
template <typename T> struct Rv32DivRemCols {
    Rv32MultAdapterCols<T> adapter;
    DivRemCoreCols<T, RV32_REGISTER_NUM_LIMBS> core;
};

struct Rv32DivRemRecord {
    Rv32MultAdapterRecord adapter;
    DivRemCoreRecords<RV32_REGISTER_NUM_LIMBS> core;
};

__global__ void rv32_div_rem_tracegen(
    Fp *d_trace,
    size_t height,
    DeviceBufferConstView<Rv32DivRemRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_bits,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t bitwise_lookup_bits,
    uint32_t *d_range_tuple_checker_ptr,
    uint2 range_tuple_checker_sizes,
    uint32_t timestamp_max_bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    RowSlice row(d_trace + idx, height);

    if (idx < d_records.len()) {
        auto const &record = d_records[idx];

        Rv32MultAdapter adapter(
            VariableRangeChecker(d_range_checker_ptr, range_checker_bits), timestamp_max_bits
        );
        adapter.fill_trace_row(row, record.adapter);

        DivRemCore<RV32_REGISTER_NUM_LIMBS> core(
            BitwiseOperationLookup(d_bitwise_lookup_ptr, bitwise_lookup_bits),
            RangeTupleChecker<2>(
                d_range_tuple_checker_ptr,
                (uint32_t[2]){range_tuple_checker_sizes.x, range_tuple_checker_sizes.y}
            )
        );
        core.fill_trace_row(row.slice_from(COL_INDEX(Rv32DivRemCols, core)), record.core);
    } else {
        row.fill_zero(0, sizeof(Rv32DivRemCols<uint8_t>));
    }
}

extern "C" int _rv32_div_rem_tracegen(
    Fp *d_trace,
    size_t height,
    size_t width,
    DeviceBufferConstView<Rv32DivRemRecord> d_records,
    uint32_t *d_range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
    uint32_t *d_range_tuple_checker_ptr,
    uint2 range_tuple_checker_sizes,
    uint32_t timestamp_max_bits
) {
    assert((height & (height - 1)) == 0);
    assert(height >= d_records.len());
    assert(width == sizeof(Rv32DivRemCols<uint8_t>));
    auto [grid, block] = kernel_launch_params(height);

    rv32_div_rem_tracegen<<<grid, block>>>(
        d_trace,
        height,
        d_records,
        d_range_checker_ptr,
        range_checker_num_bins,
        d_bitwise_lookup_ptr,
        bitwise_num_bits,
        d_range_tuple_checker_ptr,
        range_tuple_checker_sizes,
        timestamp_max_bits
    );
    return CHECK_KERNEL();
}