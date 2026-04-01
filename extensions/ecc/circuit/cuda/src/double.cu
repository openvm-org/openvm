#include "double_helpers.cuh"
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

    BigUintGpu tmp0 = x1.mul(x1);
    tmp0 = tmp0.mod_reduce(prime, barrett_mu);
    BigUintGpu lambda_num = tmp0.mul_scalar(3);
    lambda_num = lambda_num.mod_reduce(prime, barrett_mu);
    lambda_num.add_in_place(a);
    lambda_num = lambda_num.mod_reduce(prime, barrett_mu);
    BigUintGpu lambda_den(1, 8);
    if (!use_setup_arithmetic) {
        lambda_den = y1.mul_scalar(2);
        lambda_den = lambda_den.mod_reduce(prime, barrett_mu);
    }
    BigUintGpu lambda = lambda_num.mod_div(lambda_den, prime, barrett_mu);

    tmp0 = lambda.mul(lambda);
    tmp0 = tmp0.mod_reduce(prime, barrett_mu);
    BigUintGpu double_x1 = x1.mul_scalar(2);
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
__global__ void ec_double_lambda_num_constraint_tracegen_kernel(
    Fp *trace,
    size_t height,
    const uint8_t *records,
    size_t num_records,
    size_t record_stride,
    const uint8_t *prime_limbs,
    uint32_t prime_limb_count,
    const uint8_t *a_limbs,
    uint32_t a_limb_count,
    size_t lambda_num_var_col,
    size_t quotient_col,
    size_t carry_col,
    uint32_t q_limb_count,
    uint32_t carry_count,
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

    BigUintGpu prime(prime_limbs, prime_limb_count, 8);
    prime.normalize();

    uint8_t zero_input_limbs[2 * NUM_LIMBS] = {0};
    const uint8_t *input_limbs =
        is_real ? reinterpret_cast<const EcDoubleRecord<BLOCKS, BLOCK_SIZE, NUM_LIMBS> *>(
                      records + idx * record_stride
                  )
                      ->core.input_limbs
                : zero_input_limbs;
    BigUintGpu x1(input_limbs + 0 * NUM_LIMBS, NUM_LIMBS, 8);
    x1.normalize();

    BigUintGpu a(a_limbs, a_limb_count, 8);
    a.normalize();
    BigUintGpu lambda_num_var = read_biguint<NUM_LIMBS>(core_row, lambda_num_var_col);
    BigUintGpu lambda_num = x1.mul(x1);
    lambda_num = lambda_num.mul_scalar(3);
    lambda_num.add_in_place(a);

    BigIntGpu constraint_big(lambda_num);
    constraint_big -= BigIntGpu(lambda_num_var);
    BigIntGpu quotient(prime.limb_bits);
    quotient = constraint_big.div_biguint(prime);
    quotient.normalize();
    write_quotient_limbs(
        core_row, quotient_col, range_checker, track_range, quotient, q_limb_count
    );

    OverflowInt constraint_ov(x1, NUM_LIMBS);
    OverflowInt tmp_ov(x1, NUM_LIMBS);
    constraint_ov *= tmp_ov;
    constraint_ov *= 3;
    tmp_ov = OverflowInt(a, NUM_LIMBS);
    constraint_ov += tmp_ov;
    tmp_ov = OverflowInt(lambda_num_var, NUM_LIMBS);
    constraint_ov -= tmp_ov;
    OverflowInt result = constraint_ov;
    OverflowInt q_ov(quotient, q_limb_count);
    OverflowInt prime_overflow(prime, prime.num_limbs);
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
        result_limbs, quotient_signed_limbs, q_limb_count, prime_limbs, prime.num_limbs
    );
    size_t carry_write_col = carry_col;
    write_manual_carries(
        core_row,
        carry_write_col,
        range_checker,
        track_range,
        result_limbs,
        carry_count,
        carry_min_abs,
        carry_bits
    );
}

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS>
__global__ void ec_double_lambda_den_constraint_tracegen_kernel(
    Fp *trace,
    size_t height,
    const uint8_t *records,
    size_t num_records,
    size_t record_stride,
    const uint8_t *prime_limbs,
    uint32_t prime_limb_count,
    const uint8_t *a_limbs,
    uint32_t a_limb_count,
    size_t lambda_num_var_col,
    size_t lambda_col,
    size_t quotient_col,
    size_t carry_col,
    uint32_t q_limb_count,
    uint32_t carry_count,
    uint32_t has_lambda_num_var,
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
    uint8_t zero_input_limbs[2 * NUM_LIMBS] = {0};
    const uint8_t *input_limbs = is_real ? record->core.input_limbs : zero_input_limbs;
    BigUintGpu x1(input_limbs + 0 * NUM_LIMBS, NUM_LIMBS, 8);
    BigUintGpu y1(input_limbs + 1 * NUM_LIMBS, NUM_LIMBS, 8);
    x1.normalize();
    y1.normalize();

    BigUintGpu lambda = read_biguint<NUM_LIMBS>(core_row, lambda_col);
    BigUintGpu lambda_num_var(prime.limb_bits);
    BigUintGpu a(a_limbs, a_limb_count, 8);
    a.normalize();
    BigUintGpu lambda_den(1, 8);
    if (!use_setup_arithmetic) {
        lambda_den = y1.mul_scalar(2);
    }
    BigUintGpu x1_square = x1;
    x1_square = x1_square.mul(x1);
    BigUintGpu lambda_num_value = x1_square.mul_scalar(3);
    lambda_num_value.add_in_place(a);
    BigIntGpu lambda_big(lambda);
    BigIntGpu constraint_big(lambda_den);
    if (has_lambda_num_var != 0) {
        lambda_num_var = read_biguint<NUM_LIMBS>(core_row, lambda_num_var_col);
        lambda_num_value = lambda_num_var;
    }
    BigIntGpu lambda_num_big(lambda_num_value);
    constraint_big *= lambda_big;
    constraint_big -= lambda_num_big;
    BigIntGpu quotient(prime.limb_bits);
    quotient = constraint_big.div_biguint(prime);
    quotient.normalize();
    write_quotient_limbs(
        core_row, quotient_col, range_checker, track_range, quotient, q_limb_count
    );

    OverflowInt constraint_ov =
        make_double_lambda_den_overflow<NUM_LIMBS>(use_setup_arithmetic, y1);
    OverflowInt tmp_ov(lambda, NUM_LIMBS);
    constraint_ov *= tmp_ov;
    if (has_lambda_num_var != 0) {
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
    OverflowInt q_ov(quotient, q_limb_count);
    OverflowInt prime_overflow(prime, prime.num_limbs);
    tmp_ov = q_ov;
    tmp_ov *= prime_overflow;
    result -= tmp_ov;

    uint32_t carry_bits = result.max_overflow_bits - 8;
    uint32_t carry_min_abs = 1u << carry_bits;
    carry_bits++;
    int64_t result_limbs[MAX_LIMBS];
    int64_t quotient_signed_limbs[MAX_LIMBS];
    clear_result_limbs(result_limbs);
    if (has_lambda_num_var != 0) {
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
        result_limbs, quotient_signed_limbs, q_limb_count, prime_limbs, prime.num_limbs
    );
    size_t carry_write_col = carry_col;
    write_manual_carries(
        core_row,
        carry_write_col,
        range_checker,
        track_range,
        result_limbs,
        carry_count,
        carry_min_abs,
        carry_bits
    );
}

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS>
__global__ void ec_double_x3_constraint_tracegen_kernel(
    Fp *trace,
    size_t height,
    const uint8_t *records,
    size_t num_records,
    size_t record_stride,
    const uint8_t *prime_limbs,
    uint32_t prime_limb_count,
    size_t lambda_col,
    size_t x3_col,
    size_t quotient_col,
    size_t carry_col,
    uint32_t q_limb_count,
    uint32_t carry_count,
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

    BigUintGpu prime(prime_limbs, prime_limb_count, 8);
    prime.normalize();
    uint8_t zero_input_limbs[2 * NUM_LIMBS] = {0};
    const uint8_t *input_limbs =
        is_real ? reinterpret_cast<const EcDoubleRecord<BLOCKS, BLOCK_SIZE, NUM_LIMBS> *>(
                      records + idx * record_stride
                  )
                      ->core.input_limbs
                : zero_input_limbs;
    BigUintGpu x1(input_limbs + 0 * NUM_LIMBS, NUM_LIMBS, 8);
    x1.normalize();

    BigUintGpu lambda = read_biguint<NUM_LIMBS>(core_row, lambda_col);
    BigUintGpu x3 = read_biguint<NUM_LIMBS>(core_row, x3_col);
    BigUintGpu lambda_sq = lambda.mul(lambda);
    BigIntGpu constraint_big(lambda_sq);
    constraint_big -= BigIntGpu(x1.mul_scalar(2));
    constraint_big -= BigIntGpu(x3);
    BigIntGpu quotient(prime.limb_bits);
    quotient = constraint_big.div_biguint(prime);
    quotient.normalize();
    write_quotient_limbs(
        core_row, quotient_col, range_checker, track_range, quotient, q_limb_count
    );

    OverflowInt constraint_ov(lambda, NUM_LIMBS);
    OverflowInt tmp_ov(lambda, NUM_LIMBS);
    constraint_ov *= tmp_ov;
    tmp_ov = OverflowInt(x1, NUM_LIMBS);
    tmp_ov *= 2;
    constraint_ov -= tmp_ov;
    tmp_ov = OverflowInt(x3, NUM_LIMBS);
    constraint_ov -= tmp_ov;
    OverflowInt result = constraint_ov;
    OverflowInt q_ov(quotient, q_limb_count);
    OverflowInt prime_overflow(prime, prime.num_limbs);
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
        result_limbs, quotient_signed_limbs, q_limb_count, prime_limbs, prime.num_limbs
    );
    size_t carry_write_col = carry_col;
    write_manual_carries(
        core_row,
        carry_write_col,
        range_checker,
        track_range,
        result_limbs,
        carry_count,
        carry_min_abs,
        carry_bits
    );
}

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS>
__global__ void ec_double_y3_constraint_tracegen_kernel(
    Fp *trace,
    size_t height,
    const uint8_t *records,
    size_t num_records,
    size_t record_stride,
    const uint8_t *prime_limbs,
    uint32_t prime_limb_count,
    size_t lambda_col,
    size_t x3_col,
    size_t y3_col,
    size_t quotient_col,
    size_t carry_col,
    size_t final_flag_col,
    uint32_t q_limb_count,
    uint32_t carry_count,
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

    BigUintGpu prime(prime_limbs, prime_limb_count, 8);
    prime.normalize();
    uint8_t zero_input_limbs[2 * NUM_LIMBS] = {0};
    const uint8_t *input_limbs = is_real ? record->core.input_limbs : zero_input_limbs;
    BigUintGpu x1(input_limbs + 0 * NUM_LIMBS, NUM_LIMBS, 8);
    BigUintGpu y1(input_limbs + 1 * NUM_LIMBS, NUM_LIMBS, 8);
    x1.normalize();
    y1.normalize();

    BigUintGpu lambda = read_biguint<NUM_LIMBS>(core_row, lambda_col);
    BigUintGpu x3 = read_biguint<NUM_LIMBS>(core_row, x3_col);
    BigUintGpu y3 = read_biguint<NUM_LIMBS>(core_row, y3_col);
    BigIntGpu x1_minus_x3_big(x1);
    x1_minus_x3_big -= BigIntGpu(x3);
    BigIntGpu constraint_big = x1_minus_x3_big.mul(BigIntGpu(lambda));
    constraint_big -= BigIntGpu(y1);
    constraint_big -= BigIntGpu(y3);
    BigIntGpu quotient(prime.limb_bits);
    quotient = constraint_big.div_biguint(prime);
    quotient.normalize();
    write_quotient_limbs(
        core_row, quotient_col, range_checker, track_range, quotient, q_limb_count
    );

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
    OverflowInt q_ov(quotient, q_limb_count);
    OverflowInt prime_overflow(prime, prime.num_limbs);
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
        result_limbs, quotient_signed_limbs, q_limb_count, prime_limbs, prime.num_limbs
    );
    size_t carry_write_col = carry_col;
    write_manual_carries(
        core_row,
        carry_write_col,
        range_checker,
        track_range,
        result_limbs,
        carry_count,
        carry_min_abs,
        carry_bits
    );

    core_row[final_flag_col] = (is_real && !is_setup) ? Fp::one() : Fp::zero();
}

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS>
int launch_ec_double_constraint_tracegen_kernels(
    Fp *d_trace,
    size_t trace_height,
    const uint8_t *d_records,
    size_t num_records,
    size_t record_stride,
    const uint8_t *d_prime,
    uint32_t prime_limb_count,
    const uint8_t *d_a,
    uint32_t a_limb_count,
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
    uint32_t range_checker_num_bins
) {
    auto [constraint_grid, constraint_block] = kernel_launch_params(trace_height, 128);

    const uint32_t q_counts[4] = {q0_limbs, q1_limbs, q2_limbs, q3_limbs};
    const uint32_t c_counts[4] = {c0_limbs, c1_limbs, c2_limbs, c3_limbs};
    const size_t vars_col = 1 + 2 * NUM_LIMBS;
    const size_t lambda_num_var_col = vars_col;
    const size_t lambda_col = vars_col + (num_variables == 4 ? NUM_LIMBS : 0);
    const size_t x3_col = lambda_col + NUM_LIMBS;
    const size_t y3_col = x3_col + NUM_LIMBS;

    size_t quotient_col = 1 + 2 * NUM_LIMBS + num_variables * NUM_LIMBS;
    size_t carry_col = quotient_col;
    for (uint32_t q_count : q_counts) {
        carry_col += q_count;
    }
    size_t final_flag_col = carry_col;
    for (uint32_t c_count : c_counts) {
        final_flag_col += c_count;
    }

    int ret = 0;
    uint32_t block = 0;
    if (num_variables == 4) {
        ec_double_lambda_num_constraint_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
            <<<constraint_grid, constraint_block>>>(
                d_trace,
                trace_height,
                d_records,
                num_records,
                record_stride,
                d_prime,
                prime_limb_count,
                d_a,
                a_limb_count,
                lambda_num_var_col,
                quotient_col,
                carry_col,
                q_counts[block],
                c_counts[block],
                d_range_checker,
                range_checker_num_bins
            );
        ret = CHECK_KERNEL();
        if (ret) {
            return ret;
        }
        quotient_col += q_counts[block];
        carry_col += c_counts[block];
        block++;
    }

    ec_double_lambda_den_constraint_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
        <<<constraint_grid, constraint_block>>>(
            d_trace,
            trace_height,
            d_records,
            num_records,
            record_stride,
            d_prime,
            prime_limb_count,
            d_a,
            a_limb_count,
            lambda_num_var_col,
            lambda_col,
            quotient_col,
            carry_col,
            q_counts[block],
            c_counts[block],
            num_variables == 4,
            setup_opcode,
            d_range_checker,
            range_checker_num_bins
        );
    ret = CHECK_KERNEL();
    if (ret) {
        return ret;
    }
    quotient_col += q_counts[block];
    carry_col += c_counts[block];
    block++;

    ec_double_x3_constraint_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
        <<<constraint_grid, constraint_block>>>(
            d_trace,
            trace_height,
            d_records,
            num_records,
            record_stride,
            d_prime,
            prime_limb_count,
            lambda_col,
            x3_col,
            quotient_col,
            carry_col,
            q_counts[block],
            c_counts[block],
            d_range_checker,
            range_checker_num_bins
        );
    ret = CHECK_KERNEL();
    if (ret) {
        return ret;
    }
    quotient_col += q_counts[block];
    carry_col += c_counts[block];
    block++;

    ec_double_y3_constraint_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
        <<<constraint_grid, constraint_block>>>(
            d_trace,
            trace_height,
            d_records,
            num_records,
            record_stride,
            d_prime,
            prime_limb_count,
            lambda_col,
            x3_col,
            y3_col,
            quotient_col,
            carry_col,
            final_flag_col,
            q_counts[block],
            c_counts[block],
            setup_opcode,
            d_range_checker,
            range_checker_num_bins
        );
    ret = CHECK_KERNEL();
    if (ret) {
        return ret;
    }

    return 0;
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

    if (prime_limb_count <= 32) {
        constexpr size_t BLOCKS = 16;
        constexpr size_t BLOCK_SIZE = 4;
        constexpr size_t NUM_LIMBS = 32;
        const size_t core_width = 1 + 2 * NUM_LIMBS + num_variables * NUM_LIMBS + q0_limbs +
                                  q1_limbs + q2_limbs + q3_limbs + c0_limbs + c1_limbs + c2_limbs +
                                  c3_limbs + 1;
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

        ret = launch_ec_double_constraint_tracegen_kernels<BLOCKS, BLOCK_SIZE, NUM_LIMBS>(
            d_trace,
            trace_height,
            d_records,
            num_records,
            record_stride,
            d_prime,
            prime_limb_count,
            d_a,
            a_limb_count,
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
        if (ret) {
            return ret;
        }
    } else {
        constexpr size_t BLOCKS = 24;
        constexpr size_t BLOCK_SIZE = 4;
        constexpr size_t NUM_LIMBS = 48;
        const size_t core_width = 1 + 2 * NUM_LIMBS + num_variables * NUM_LIMBS + q0_limbs +
                                  q1_limbs + q2_limbs + q3_limbs + c0_limbs + c1_limbs + c2_limbs +
                                  c3_limbs + 1;

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

        ret = launch_ec_double_constraint_tracegen_kernels<BLOCKS, BLOCK_SIZE, NUM_LIMBS>(
            d_trace,
            trace_height,
            d_records,
            num_records,
            record_stride,
            d_prime,
            prime_limb_count,
            d_a,
            a_limb_count,
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
        if (ret) {
            return ret;
        }
    }

    return CHECK_KERNEL();
}
