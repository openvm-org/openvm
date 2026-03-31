#include "fp.h"
#include "launcher.cuh"
#include "mod-builder/bigint_ops.cuh"
#include "mod-builder/overflow_ops.cuh"
#include "primitives/histogram.cuh"
#include "primitives/trace_access.h"
#include "rv32-adapters/vec_heap.cuh"
#include <cstddef>
#include <cstdint>

using namespace mod_builder;
using namespace riscv;

template <size_t NUM_LIMBS> struct EcDoubleCoreRecord {
    uint8_t opcode;
    uint8_t input_limbs[2 * NUM_LIMBS];
};

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS> struct EcDoubleRecord {
    Rv32VecHeapAdapterRecord<1, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE> adapter;
    EcDoubleCoreRecord<NUM_LIMBS> core;
};

template <size_t NUM_LIMBS>
__device__ inline void write_biguint(RowSlice row, size_t &col, const BigUintGpu &value) {
    for (size_t limb = 0; limb < NUM_LIMBS; limb++) {
        uint32_t v = limb < value.num_limbs ? value.limbs[limb] : 0;
        row[col++] = Fp(v);
    }
}

template <size_t NUM_LIMBS> __device__ inline BigUintGpu read_biguint(RowSlice row, size_t col) {
    uint32_t limbs[NUM_LIMBS];
    for (size_t limb = 0; limb < NUM_LIMBS; limb++) {
        limbs[limb] = row[col + limb].asUInt32();
    }
    BigUintGpu value(limbs, NUM_LIMBS, 8);
    value.normalize();
    return value;
}

__device__ inline void write_signed_bigint_limb(
    RowSlice row,
    size_t &col,
    const BigIntGpu &value,
    size_t limb
) {
    uint32_t q_limb = limb < value.mag.num_limbs ? value.mag.limbs[limb] : 0;
    row[col++] = value.is_negative ? (Fp::zero() - Fp(q_limb)) : Fp(q_limb);
}

__device__ inline void clear_result_limbs(int64_t result_limbs[MAX_LIMBS]) {
    for (uint32_t limb = 0; limb < MAX_LIMBS; limb++) {
        result_limbs[limb] = 0;
    }
}

__device__ inline void add_biguint_limbs(
    int64_t result_limbs[MAX_LIMBS],
    const BigUintGpu &value
) {
    for (uint32_t limb = 0; limb < value.num_limbs && limb < MAX_LIMBS; limb++) {
        result_limbs[limb] += value.limbs[limb];
    }
}

__device__ inline void subtract_biguint_limbs(
    int64_t result_limbs[MAX_LIMBS],
    const BigUintGpu &value
) {
    for (uint32_t limb = 0; limb < value.num_limbs && limb < MAX_LIMBS; limb++) {
        result_limbs[limb] -= value.limbs[limb];
    }
}

__device__ inline void add_biguint_product(
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

__device__ inline void add_bigint_biguint_product(
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

__device__ inline void subtract_signed_limbs_times_prime(
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

__device__ inline void write_manual_carries(
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
        core_row[carry_col++] = carry_i32 >= 0 ? Fp((uint32_t)carry_i32)
                                               : (Fp::zero() - Fp((uint32_t)(-carry_i32)));
        if (track_range) {
            range_checker.add_count((uint32_t)(carry_i32 + (int32_t)carry_min_abs), carry_bits);
        }
    }
}

__device__ inline uint32_t sum_counts4(const uint32_t counts[4]) {
    return counts[0] + counts[1] + counts[2] + counts[3];
}

template <size_t NUM_LIMBS>
__device__ inline OverflowInt make_double_lambda_den_overflow(bool is_setup, const BigUintGpu &y1) {
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

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS>
__global__ void ec_double_adapter_tracegen_kernel(
    Fp *trace,
    size_t height,
    const uint8_t *records,
    size_t num_records,
    size_t record_stride,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins,
    uint32_t *bitwise_lookup_ptr,
    uint32_t bitwise_num_bits,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height) {
        return;
    }

    RowSlice row(trace + idx, height);
    VariableRangeChecker range_checker(range_checker_ptr, range_checker_num_bins);
    BitwiseOperationLookup bitwise_lookup(bitwise_lookup_ptr, bitwise_num_bits);

    constexpr size_t ADAPTER_WIDTH =
        sizeof(Rv32VecHeapAdapterCols<uint8_t, 1, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>);
    if (idx >= num_records) {
        row.fill_zero(0, ADAPTER_WIDTH);
        return;
    }

    const auto *record = reinterpret_cast<const EcDoubleRecord<BLOCKS, BLOCK_SIZE, NUM_LIMBS> *>(
        records + idx * record_stride
    );
    Rv32VecHeapAdapter<1, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE> adapter(
        pointer_max_bits, range_checker, bitwise_lookup, timestamp_max_bits
    );
    adapter.fill_trace_row(row, record->adapter);
}

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS>
__global__ void ec_double_compute_tracegen_kernel(
    Fp *trace,
    size_t height,
    size_t core_width,
    const uint8_t *records,
    size_t num_records,
    size_t record_stride,
    const uint8_t *prime_limbs,
    uint32_t prime_limb_count,
    const uint8_t *a_limbs,
    uint32_t a_limb_count,
    const uint8_t *barrett_mu,
    uint32_t num_variables,
    uint32_t setup_opcode,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height) {
        return;
    }

    constexpr size_t ADAPTER_WIDTH =
        sizeof(Rv32VecHeapAdapterCols<uint8_t, 1, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>);
    RowSlice row(trace + idx, height);
    RowSlice core_row = row.slice_from(ADAPTER_WIDTH);
    VariableRangeChecker range_checker(range_checker_ptr, range_checker_num_bins);
    const bool is_real = idx < num_records;
    const bool track_range = is_real;
    const auto *record =
        is_real ? reinterpret_cast<const EcDoubleRecord<BLOCKS, BLOCK_SIZE, NUM_LIMBS> *>(
                      records + idx * record_stride
                  )
                : nullptr;
    const bool is_setup = is_real && record->core.opcode == setup_opcode;
    const bool use_setup_arithmetic = !is_real || is_setup;

    BigUintGpu prime(prime_limbs, prime_limb_count, 8);
    prime.normalize();
    BigUintGpu a(a_limbs, a_limb_count, 8);
    a.normalize();

    uint8_t zero_input_limbs[2 * NUM_LIMBS] = {0};
    const uint8_t *input_limbs = is_real ? record->core.input_limbs : zero_input_limbs;
    BigUintGpu x1(input_limbs + 0 * NUM_LIMBS, NUM_LIMBS, 8);
    BigUintGpu y1(input_limbs + 1 * NUM_LIMBS, NUM_LIMBS, 8);
    x1.normalize();
    y1.normalize();
    x1 = x1.rem(prime);
    y1 = y1.rem(prime);

    BigUintGpu one(1, 8);
    BigUintGpu two(2, 8);
    BigUintGpu three(3, 8);

    BigUintGpu tmp0 = x1.mul(x1);
    tmp0 = tmp0.mod_reduce(prime, barrett_mu);
    BigUintGpu lambda_num = tmp0.mul(three);
    lambda_num = lambda_num.mod_reduce(prime, barrett_mu);
    lambda_num.add_in_place(a);
    lambda_num = lambda_num.mod_reduce(prime, barrett_mu);
    BigUintGpu lambda_den = one;
    if (!use_setup_arithmetic) {
        lambda_den = y1.mul(two);
        lambda_den = lambda_den.mod_reduce(prime, barrett_mu);
    }
    BigUintGpu lambda = lambda_num.mod_div(lambda_den, prime, barrett_mu);

    tmp0 = lambda.mul(lambda);
    tmp0 = tmp0.mod_reduce(prime, barrett_mu);
    BigUintGpu double_x1 = x1.mul(two);
    double_x1 = double_x1.mod_reduce(prime, barrett_mu);
    BigUintGpu x3 = tmp0.mod_sub(double_x1, prime);
    tmp0 = x1.mod_sub(x3, prime);
    BigUintGpu y3 = lambda.mul(tmp0);
    y3 = y3.mod_reduce(prime, barrett_mu);
    y3 = y3.mod_sub(y1, prime);

    size_t col = 0;
    core_row[col++] = is_real ? Fp::one() : Fp::zero();
    for (size_t i = 0; i < 2 * NUM_LIMBS; i++) {
        core_row[col++] = Fp(input_limbs[i]);
    }

    if (num_variables == 4) {
        write_biguint<NUM_LIMBS>(core_row, col, lambda_num);
    }
    write_biguint<NUM_LIMBS>(core_row, col, lambda);
    write_biguint<NUM_LIMBS>(core_row, col, x3);
    write_biguint<NUM_LIMBS>(core_row, col, y3);

    for (size_t i = 0; i < num_variables * NUM_LIMBS; i++) {
        if (track_range) {
            range_checker.add_count(core_row[1 + 2 * NUM_LIMBS + i].asUInt32(), 8);
        }
    }
}

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS>
__global__ void ec_double_constraint_tracegen_kernel(
    Fp *trace,
    size_t height,
    size_t core_width,
    const uint8_t *records,
    size_t num_records,
    size_t record_stride,
    const uint8_t *prime_limbs,
    uint32_t prime_limb_count,
    const uint8_t *a_limbs,
    uint32_t a_limb_count,
    const uint8_t *barrett_mu,
    uint32_t q0_limbs,
    uint32_t q1_limbs,
    uint32_t q2_limbs,
    uint32_t q3_limbs,
    uint32_t c0_limbs,
    uint32_t c1_limbs,
    uint32_t c2_limbs,
    uint32_t c3_limbs,
    uint32_t num_variables,
    uint32_t setup_opcode,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height) {
        return;
    }

    constexpr size_t ADAPTER_WIDTH =
        sizeof(Rv32VecHeapAdapterCols<uint8_t, 1, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>);
    RowSlice row(trace + idx, height);
    RowSlice core_row = row.slice_from(ADAPTER_WIDTH);
    VariableRangeChecker range_checker(range_checker_ptr, range_checker_num_bins);
    const bool is_real = idx < num_records;
    const bool track_range = is_real;
    const auto *record =
        is_real ? reinterpret_cast<const EcDoubleRecord<BLOCKS, BLOCK_SIZE, NUM_LIMBS> *>(
                      records + idx * record_stride
                  )
                : nullptr;
    const bool is_setup = is_real && record->core.opcode == setup_opcode;
    const bool use_setup_arithmetic = !is_real || is_setup;

    BigUintGpu prime(prime_limbs, prime_limb_count, 8);
    prime.normalize();
    BigUintGpu a(a_limbs, a_limb_count, 8);
    a.normalize();
    OverflowInt prime_overflow(prime, prime.num_limbs);

    uint8_t zero_input_limbs[2 * NUM_LIMBS] = {0};
    const uint8_t *input_limbs = is_real ? record->core.input_limbs : zero_input_limbs;
    BigUintGpu x1(input_limbs + 0 * NUM_LIMBS, NUM_LIMBS, 8);
    BigUintGpu y1(input_limbs + 1 * NUM_LIMBS, NUM_LIMBS, 8);
    x1.normalize();
    y1.normalize();

    BigUintGpu one(1, 8);
    BigUintGpu two(2, 8);
    BigUintGpu three(3, 8);

    const size_t vars_col = 1 + 2 * NUM_LIMBS;
    size_t var_col = vars_col;
    BigUintGpu lambda_num_var(prime.limb_bits);
    if (num_variables == 4) {
        lambda_num_var = read_biguint<NUM_LIMBS>(core_row, var_col);
        var_col += NUM_LIMBS;
    }
    BigUintGpu lambda = read_biguint<NUM_LIMBS>(core_row, var_col);
    var_col += NUM_LIMBS;
    BigUintGpu x3 = read_biguint<NUM_LIMBS>(core_row, var_col);
    var_col += NUM_LIMBS;
    BigUintGpu y3 = read_biguint<NUM_LIMBS>(core_row, var_col);
    BigIntGpu x1_big(x1);
    BigIntGpu y1_big(y1);
    BigIntGpu lambda_num_var_big(lambda_num_var);
    BigIntGpu lambda_big(lambda);
    BigIntGpu x3_big(x3);
    BigIntGpu y3_big(y3);
    BigIntGpu x1_minus_x3_big(x1);
    x1_minus_x3_big -= x3_big;
    BigUintGpu x1_square = x1.mul(x1);
    BigUintGpu lambda_num = x1_square.mul(three);
    lambda_num.add_in_place(a);
    BigUintGpu lambda_den = one;
    if (!use_setup_arithmetic) {
        lambda_den = y1.mul(two);
    }
    BigUintGpu lambda_sq = lambda.mul(lambda);
    BigUintGpu double_x1 = x1.mul(two);

    const uint32_t q_counts[4] = {q0_limbs, q1_limbs, q2_limbs, q3_limbs};
    const uint32_t c_counts[4] = {c0_limbs, c1_limbs, c2_limbs, c3_limbs};

    size_t col = 1 + 2 * NUM_LIMBS + num_variables * NUM_LIMBS;
    size_t carry_col = col + sum_counts4(q_counts);
    uint32_t block = 0;

    if (num_variables == 4) {
        BigIntGpu constraint_big(lambda_num);
        constraint_big -= lambda_num_var_big;
        BigIntGpu quotient(prime.limb_bits);
        quotient = constraint_big.div_biguint(prime);
        quotient.normalize();
        for (uint32_t limb = 0; limb < q_counts[block]; limb++) {
            int32_t q_signed =
                limb < quotient.mag.num_limbs ? (int32_t)quotient.mag.limbs[limb] : 0;
            if (quotient.is_negative) {
                q_signed = -q_signed;
            }
            write_signed_bigint_limb(core_row, col, quotient, limb);
            if (track_range) {
                range_checker.add_count((uint32_t)(q_signed + (1 << 8)), 9);
            }
        }
        OverflowInt constraint_ov(x1, NUM_LIMBS);
        OverflowInt tmp_ov(x1, NUM_LIMBS);
        constraint_ov *= tmp_ov;
        constraint_ov *= 3;
        tmp_ov = OverflowInt(a, NUM_LIMBS);
        constraint_ov += tmp_ov;
        tmp_ov = OverflowInt(lambda_num_var, NUM_LIMBS);
        constraint_ov -= tmp_ov;
        OverflowInt result = constraint_ov;
        OverflowInt q_ov(quotient, q_counts[block]);
        tmp_ov = q_ov;
        tmp_ov *= prime_overflow;
        result -= tmp_ov;
        uint32_t carry_bits = result.max_overflow_bits - 8;
        uint32_t carry_min_abs = 1u << carry_bits;
        carry_bits++;
        int64_t result_limbs[MAX_LIMBS];
        int64_t quotient_signed_limbs[MAX_LIMBS];
        clear_result_limbs(result_limbs);
        add_biguint_product(result_limbs, x1, x1);
        for (uint32_t limb = 0; limb < MAX_LIMBS; limb++) {
            result_limbs[limb] *= 3;
        }
        add_biguint_limbs(result_limbs, a);
        subtract_biguint_limbs(result_limbs, lambda_num_var);
        quotient.to_signed_limbs(quotient_signed_limbs);
        subtract_signed_limbs_times_prime(
            result_limbs, quotient_signed_limbs, q_counts[block], prime_limbs, prime.num_limbs
        );
        write_manual_carries(
            core_row,
            carry_col,
            range_checker,
            track_range,
            result_limbs,
            c_counts[block],
            carry_min_abs,
            carry_bits
        );
        block++;
    }

    {
        BigIntGpu constraint_big(lambda_den);
        BigIntGpu lambda_num_big(num_variables == 4 ? lambda_num_var : lambda_num);
        constraint_big *= lambda_big;
        constraint_big -= lambda_num_big;
        BigIntGpu quotient(prime.limb_bits);
        quotient = constraint_big.div_biguint(prime);
        quotient.normalize();
        for (uint32_t limb = 0; limb < q_counts[block]; limb++) {
            int32_t q_signed =
                limb < quotient.mag.num_limbs ? (int32_t)quotient.mag.limbs[limb] : 0;
            if (quotient.is_negative) {
                q_signed = -q_signed;
            }
            write_signed_bigint_limb(core_row, col, quotient, limb);
            if (track_range) {
                range_checker.add_count((uint32_t)(q_signed + (1 << 8)), 9);
            }
        }
        OverflowInt constraint_ov =
            make_double_lambda_den_overflow<NUM_LIMBS>(use_setup_arithmetic, y1);
        OverflowInt tmp_ov(lambda, NUM_LIMBS);
        constraint_ov *= tmp_ov;
        if (num_variables == 4) {
            tmp_ov = OverflowInt(lambda_num_var, NUM_LIMBS);
            constraint_ov -= tmp_ov;
        } else {
            tmp_ov = OverflowInt(x1, NUM_LIMBS);
            tmp_ov *= tmp_ov;
            tmp_ov *= 3;
            constraint_ov -= tmp_ov;
            tmp_ov = OverflowInt(a, NUM_LIMBS);
            constraint_ov -= tmp_ov;
        }
        OverflowInt result = constraint_ov;
        OverflowInt q_ov(quotient, q_counts[block]);
        tmp_ov = q_ov;
        tmp_ov *= prime_overflow;
        result -= tmp_ov;
        uint32_t carry_bits = result.max_overflow_bits - 8;
        uint32_t carry_min_abs = 1u << carry_bits;
        carry_bits++;
        int64_t result_limbs[MAX_LIMBS];
        int64_t quotient_signed_limbs[MAX_LIMBS];
        clear_result_limbs(result_limbs);
        if (num_variables == 4) {
            if (use_setup_arithmetic) {
                add_biguint_limbs(result_limbs, lambda);
            } else {
                add_biguint_product(result_limbs, y1, lambda);
                for (uint32_t limb = 0; limb < MAX_LIMBS; limb++) {
                    result_limbs[limb] *= 2;
                }
            }
            subtract_biguint_limbs(result_limbs, lambda_num_var);
        } else {
            if (use_setup_arithmetic) {
                add_biguint_limbs(result_limbs, lambda);
            } else {
                add_biguint_product(result_limbs, y1, lambda);
                for (uint32_t limb = 0; limb < MAX_LIMBS; limb++) {
                    result_limbs[limb] *= 2;
                }
            }
            for (uint32_t i = 0; i < x1.num_limbs && i < MAX_LIMBS; i++) {
                for (uint32_t j = 0; j < x1.num_limbs && i + j < MAX_LIMBS; j++) {
                    result_limbs[i + j] -= 3 * (int64_t)x1.limbs[i] * (int64_t)x1.limbs[j];
                }
            }
            subtract_biguint_limbs(result_limbs, a);
        }
        quotient.to_signed_limbs(quotient_signed_limbs);
        subtract_signed_limbs_times_prime(
            result_limbs, quotient_signed_limbs, q_counts[block], prime_limbs, prime.num_limbs
        );
        write_manual_carries(
            core_row,
            carry_col,
            range_checker,
            track_range,
            result_limbs,
            c_counts[block],
            carry_min_abs,
            carry_bits
        );
        block++;
    }

    {
        BigIntGpu constraint_big(lambda_sq);
        BigIntGpu double_x1_big(double_x1);
        BigIntGpu x3_big(x3);
        constraint_big -= double_x1_big;
        constraint_big -= x3_big;
        BigIntGpu quotient(prime.limb_bits);
        quotient = constraint_big.div_biguint(prime);
        quotient.normalize();
        for (uint32_t limb = 0; limb < q_counts[block]; limb++) {
            int32_t q_signed =
                limb < quotient.mag.num_limbs ? (int32_t)quotient.mag.limbs[limb] : 0;
            if (quotient.is_negative) {
                q_signed = -q_signed;
            }
            write_signed_bigint_limb(core_row, col, quotient, limb);
            if (track_range) {
                range_checker.add_count((uint32_t)(q_signed + (1 << 8)), 9);
            }
        }
        OverflowInt constraint_ov(lambda, NUM_LIMBS);
        OverflowInt tmp_ov(lambda, NUM_LIMBS);
        constraint_ov *= tmp_ov;
        tmp_ov = OverflowInt(x1, NUM_LIMBS);
        tmp_ov *= 2;
        constraint_ov -= tmp_ov;
        tmp_ov = OverflowInt(x3, NUM_LIMBS);
        constraint_ov -= tmp_ov;
        OverflowInt result = constraint_ov;
        OverflowInt q_ov(quotient, q_counts[block]);
        tmp_ov = q_ov;
        tmp_ov *= prime_overflow;
        result -= tmp_ov;
        uint32_t carry_bits = result.max_overflow_bits - 8;
        uint32_t carry_min_abs = 1u << carry_bits;
        carry_bits++;
        int64_t result_limbs[MAX_LIMBS];
        int64_t quotient_signed_limbs[MAX_LIMBS];
        clear_result_limbs(result_limbs);
        add_biguint_product(result_limbs, lambda, lambda);
        for (uint32_t limb = 0; limb < x1.num_limbs && limb < MAX_LIMBS; limb++) {
            result_limbs[limb] -= 2 * (int64_t)x1.limbs[limb];
        }
        subtract_biguint_limbs(result_limbs, x3);
        quotient.to_signed_limbs(quotient_signed_limbs);
        subtract_signed_limbs_times_prime(
            result_limbs, quotient_signed_limbs, q_counts[block], prime_limbs, prime.num_limbs
        );
        write_manual_carries(
            core_row,
            carry_col,
            range_checker,
            track_range,
            result_limbs,
            c_counts[block],
            carry_min_abs,
            carry_bits
        );
        block++;
    }

    {
        BigIntGpu constraint_big(prime.limb_bits);
        constraint_big = x1_minus_x3_big.mul(lambda_big);
        constraint_big -= y1_big;
        constraint_big -= y3_big;
        BigIntGpu quotient(prime.limb_bits);
        quotient = constraint_big.div_biguint(prime);
        quotient.normalize();
        for (uint32_t limb = 0; limb < q_counts[block]; limb++) {
            int32_t q_signed =
                limb < quotient.mag.num_limbs ? (int32_t)quotient.mag.limbs[limb] : 0;
            if (quotient.is_negative) {
                q_signed = -q_signed;
            }
            write_signed_bigint_limb(core_row, col, quotient, limb);
            if (track_range) {
                range_checker.add_count((uint32_t)(q_signed + (1 << 8)), 9);
            }
        }
        OverflowInt constraint_ov(x1, NUM_LIMBS);
        OverflowInt tmp_ov(x3, NUM_LIMBS);
        constraint_ov -= tmp_ov;
        tmp_ov = OverflowInt(lambda, NUM_LIMBS);
        constraint_ov *= tmp_ov;
        tmp_ov = OverflowInt(y1, NUM_LIMBS);
        constraint_ov -= tmp_ov;
        tmp_ov = OverflowInt(y3, NUM_LIMBS);
        constraint_ov -= tmp_ov;
        OverflowInt result = constraint_ov;
        OverflowInt q_ov(quotient, q_counts[block]);
        tmp_ov = q_ov;
        tmp_ov *= prime_overflow;
        result -= tmp_ov;
        uint32_t carry_bits = result.max_overflow_bits - 8;
        uint32_t carry_min_abs = 1u << carry_bits;
        carry_bits++;
        int64_t result_limbs[MAX_LIMBS];
        int64_t quotient_signed_limbs[MAX_LIMBS];
        clear_result_limbs(result_limbs);
        add_biguint_product(result_limbs, x1, lambda);
        for (uint32_t i = 0; i < x3.num_limbs && i < MAX_LIMBS; i++) {
            for (uint32_t j = 0; j < lambda.num_limbs && i + j < MAX_LIMBS; j++) {
                result_limbs[i + j] -= (int64_t)x3.limbs[i] * (int64_t)lambda.limbs[j];
            }
        }
        subtract_biguint_limbs(result_limbs, y1);
        subtract_biguint_limbs(result_limbs, y3);
        quotient.to_signed_limbs(quotient_signed_limbs);
        subtract_signed_limbs_times_prime(
            result_limbs, quotient_signed_limbs, q_counts[block], prime_limbs, prime.num_limbs
        );
        write_manual_carries(
            core_row,
            carry_col,
            range_checker,
            track_range,
            result_limbs,
            c_counts[block],
            carry_min_abs,
            carry_bits
        );
    }

    core_row[carry_col++] = (is_real && !is_setup) ? Fp::one() : Fp::zero();
    while (carry_col < core_width) {
        core_row[carry_col++] = Fp::zero();
    }
}

extern "C" int launch_ec_double_tracegen(
    Fp *d_trace,
    size_t trace_height,
    const uint8_t *d_records,
    size_t num_records,
    size_t record_stride,
    const uint8_t *d_prime,
    uint32_t prime_limb_count,
    const uint8_t *d_a,
    uint32_t a_limb_count,
    const uint8_t *d_barrett_mu,
    uint32_t q0_limbs,
    uint32_t q1_limbs,
    uint32_t q2_limbs,
    uint32_t q3_limbs,
    uint32_t c0_limbs,
    uint32_t c1_limbs,
    uint32_t c2_limbs,
    uint32_t c3_limbs,
    uint32_t num_variables,
    uint32_t setup_opcode,
    uint32_t *d_range_checker,
    uint32_t range_checker_num_bins,
    uint32_t *d_bitwise_lookup,
    uint32_t bitwise_num_bits,
    uint32_t pointer_max_bits,
    uint32_t timestamp_max_bits
) {
    auto [main_grid, main_block] = kernel_launch_params(trace_height, 256);
    auto [constraint_grid, constraint_block] = kernel_launch_params(trace_height, 128);

    if (prime_limb_count <= 32) {
        constexpr size_t BLOCKS = 16;
        constexpr size_t BLOCK_SIZE = 4;
        constexpr size_t NUM_LIMBS = 32;
        const size_t core_width = 1 + 2 * NUM_LIMBS + num_variables * NUM_LIMBS + q0_limbs +
                                  q1_limbs + q2_limbs + q3_limbs + c0_limbs + c1_limbs +
                                  c2_limbs + c3_limbs + 1;
        ec_double_adapter_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS><<<main_grid, main_block>>>(
            d_trace,
            trace_height,
            d_records,
            num_records,
            record_stride,
            d_range_checker,
            range_checker_num_bins,
            d_bitwise_lookup,
            bitwise_num_bits,
            pointer_max_bits,
            timestamp_max_bits
        );
        int ret = CHECK_KERNEL();
        if (ret) {
            return ret;
        }

        ec_double_compute_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS><<<main_grid, main_block>>>(
            d_trace,
            trace_height,
            core_width,
            d_records,
            num_records,
            record_stride,
            d_prime,
            prime_limb_count,
            d_a,
            a_limb_count,
            d_barrett_mu,
            num_variables,
            setup_opcode,
            d_range_checker,
            range_checker_num_bins
        );
        ret = CHECK_KERNEL();
        if (ret) {
            return ret;
        }

        ec_double_constraint_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
            <<<constraint_grid, constraint_block>>>(
                d_trace,
                trace_height,
                core_width,
                d_records,
                num_records,
                record_stride,
                d_prime,
                prime_limb_count,
                d_a,
                a_limb_count,
                d_barrett_mu,
                q0_limbs,
                q1_limbs,
                q2_limbs,
                q3_limbs,
                c0_limbs,
                c1_limbs,
                c2_limbs,
                c3_limbs,
                num_variables,
                setup_opcode,
                d_range_checker,
                range_checker_num_bins
            );
        ret = CHECK_KERNEL();
        if (ret) {
            return ret;
        }
    } else {
        constexpr size_t BLOCKS = 24;
        constexpr size_t BLOCK_SIZE = 4;
        constexpr size_t NUM_LIMBS = 48;
        const size_t core_width = 1 + 2 * NUM_LIMBS + num_variables * NUM_LIMBS + q0_limbs +
                                  q1_limbs + q2_limbs + q3_limbs + c0_limbs + c1_limbs +
                                  c2_limbs + c3_limbs + 1;

        ec_double_adapter_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS><<<main_grid, main_block>>>(
            d_trace,
            trace_height,
            d_records,
            num_records,
            record_stride,
            d_range_checker,
            range_checker_num_bins,
            d_bitwise_lookup,
            bitwise_num_bits,
            pointer_max_bits,
            timestamp_max_bits
        );
        int ret = CHECK_KERNEL();
        if (ret) {
            return ret;
        }

        ec_double_compute_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS><<<main_grid, main_block>>>(
            d_trace,
            trace_height,
            core_width,
            d_records,
            num_records,
            record_stride,
            d_prime,
            prime_limb_count,
            d_a,
            a_limb_count,
            d_barrett_mu,
            num_variables,
            setup_opcode,
            d_range_checker,
            range_checker_num_bins
        );
        ret = CHECK_KERNEL();
        if (ret) {
            return ret;
        }

        ec_double_constraint_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
            <<<constraint_grid, constraint_block>>>(
                d_trace,
                trace_height,
                core_width,
                d_records,
                num_records,
                record_stride,
                d_prime,
                prime_limb_count,
                d_a,
                a_limb_count,
                d_barrett_mu,
                q0_limbs,
                q1_limbs,
                q2_limbs,
                q3_limbs,
                c0_limbs,
                c1_limbs,
                c2_limbs,
                c3_limbs,
                num_variables,
                setup_opcode,
                d_range_checker,
                range_checker_num_bins
            );
        ret = CHECK_KERNEL();
        if (ret) {
            return ret;
        }
    }

    return CHECK_KERNEL();
}
