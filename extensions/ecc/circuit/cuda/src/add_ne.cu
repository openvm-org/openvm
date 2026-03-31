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

template <size_t NUM_LIMBS> struct EcAddNeCoreRecord {
    uint8_t opcode;
    uint8_t input_limbs[4 * NUM_LIMBS];
};

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS> struct EcAddNeRecord {
    Rv32VecHeapAdapterRecord<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE> adapter;
    EcAddNeCoreRecord<NUM_LIMBS> core;
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

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS>
__global__ void ec_add_ne_adapter_tracegen_kernel(
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
        sizeof(Rv32VecHeapAdapterCols<uint8_t, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>);
    if (idx >= num_records) {
        row.fill_zero(0, ADAPTER_WIDTH);
        return;
    }

    const auto *record = reinterpret_cast<const EcAddNeRecord<BLOCKS, BLOCK_SIZE, NUM_LIMBS> *>(
        records + idx * record_stride
    );
    Rv32VecHeapAdapter<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE> adapter(
        pointer_max_bits, range_checker, bitwise_lookup, timestamp_max_bits
    );
    adapter.fill_trace_row(row, record->adapter);
}

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS>
__global__ void ec_add_ne_compute_tracegen_kernel(
    Fp *trace,
    size_t height,
    size_t core_width,
    const uint8_t *records,
    size_t num_records,
    size_t record_stride,
    const uint8_t *prime_limbs,
    uint32_t prime_limb_count,
    const uint8_t *barrett_mu,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height) {
        return;
    }

    constexpr size_t ADAPTER_WIDTH =
        sizeof(Rv32VecHeapAdapterCols<uint8_t, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>);
    RowSlice row(trace + idx, height);
    RowSlice core_row = row.slice_from(ADAPTER_WIDTH);
    VariableRangeChecker range_checker(range_checker_ptr, range_checker_num_bins);

    if (idx >= num_records) {
        core_row.fill_zero(0, core_width);
        return;
    }

    const auto *record = reinterpret_cast<const EcAddNeRecord<BLOCKS, BLOCK_SIZE, NUM_LIMBS> *>(
        records + idx * record_stride
    );

    BigUintGpu prime(prime_limbs, prime_limb_count, 8);
    prime.normalize();

    const uint8_t *input_limbs = record->core.input_limbs;
    BigUintGpu x1(input_limbs + 0 * NUM_LIMBS, NUM_LIMBS, 8);
    BigUintGpu y1(input_limbs + 1 * NUM_LIMBS, NUM_LIMBS, 8);
    BigUintGpu x2(input_limbs + 2 * NUM_LIMBS, NUM_LIMBS, 8);
    BigUintGpu y2(input_limbs + 3 * NUM_LIMBS, NUM_LIMBS, 8);
    x1.normalize();
    y1.normalize();
    x2.normalize();
    y2.normalize();
    x1 = x1.rem(prime);
    y1 = y1.rem(prime);
    x2 = x2.rem(prime);
    y2 = y2.rem(prime);

    BigUintGpu tmp0 = y2.mod_sub(y1, prime);
    BigUintGpu tmp1 = x2.mod_sub(x1, prime);
    BigUintGpu lambda = tmp0.mod_div(tmp1, prime, barrett_mu);

    tmp0 = lambda.mul(lambda);
    tmp0 = tmp0.mod_reduce(prime, barrett_mu);
    BigUintGpu x3 = tmp0.mod_sub(x1, prime);
    x3 = x3.mod_sub(x2, prime);
    tmp0 = x1.mod_sub(x3, prime);
    BigUintGpu y3 = lambda.mul(tmp0);
    y3 = y3.mod_reduce(prime, barrett_mu);
    y3 = y3.mod_sub(y1, prime);

    size_t col = 0;
    core_row[col++] = Fp::one();
    for (size_t i = 0; i < 4 * NUM_LIMBS; i++) {
        core_row[col++] = Fp(input_limbs[i]);
    }

    write_biguint<NUM_LIMBS>(core_row, col, lambda);
    write_biguint<NUM_LIMBS>(core_row, col, x3);
    write_biguint<NUM_LIMBS>(core_row, col, y3);

    for (size_t i = 0; i < 3 * NUM_LIMBS; i++) {
        range_checker.add_count(core_row[1 + 4 * NUM_LIMBS + i].asUInt32(), 8);
    }
}

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS>
__global__ void ec_add_ne_constraint_tracegen_kernel(
    Fp *trace,
    size_t height,
    size_t core_width,
    const uint8_t *records,
    size_t num_records,
    size_t record_stride,
    const uint8_t *prime_limbs,
    uint32_t prime_limb_count,
    const uint8_t *barrett_mu,
    uint32_t q0_limbs,
    uint32_t q1_limbs,
    uint32_t q2_limbs,
    uint32_t c0_limbs,
    uint32_t c1_limbs,
    uint32_t c2_limbs,
    uint32_t setup_opcode,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height) {
        return;
    }

    constexpr size_t ADAPTER_WIDTH =
        sizeof(Rv32VecHeapAdapterCols<uint8_t, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>);
    RowSlice row(trace + idx, height);
    RowSlice core_row = row.slice_from(ADAPTER_WIDTH);
    VariableRangeChecker range_checker(range_checker_ptr, range_checker_num_bins);

    if (idx >= num_records) {
        core_row.fill_zero(0, core_width);
        return;
    }

    const auto *record = reinterpret_cast<const EcAddNeRecord<BLOCKS, BLOCK_SIZE, NUM_LIMBS> *>(
        records + idx * record_stride
    );
    const bool is_setup = record->core.opcode == setup_opcode;

    BigUintGpu prime(prime_limbs, prime_limb_count, 8);
    prime.normalize();
    OverflowInt prime_overflow(prime, prime.num_limbs);

    const uint8_t *input_limbs = record->core.input_limbs;
    BigUintGpu x1(input_limbs + 0 * NUM_LIMBS, NUM_LIMBS, 8);
    BigUintGpu y1(input_limbs + 1 * NUM_LIMBS, NUM_LIMBS, 8);
    BigUintGpu x2(input_limbs + 2 * NUM_LIMBS, NUM_LIMBS, 8);
    BigUintGpu y2(input_limbs + 3 * NUM_LIMBS, NUM_LIMBS, 8);
    x1.normalize();
    y1.normalize();
    x2.normalize();
    y2.normalize();

    const size_t vars_col = 1 + 4 * NUM_LIMBS;
    BigUintGpu lambda = read_biguint<NUM_LIMBS>(core_row, vars_col + 0 * NUM_LIMBS);
    BigUintGpu x3 = read_biguint<NUM_LIMBS>(core_row, vars_col + 1 * NUM_LIMBS);
    BigUintGpu y3 = read_biguint<NUM_LIMBS>(core_row, vars_col + 2 * NUM_LIMBS);
    BigIntGpu x1_big(x1);
    BigIntGpu y1_big(y1);
    BigIntGpu x2_big(x2);
    BigIntGpu y2_big(y2);
    BigIntGpu lambda_big(lambda);
    BigIntGpu x3_big(x3);
    BigIntGpu y3_big(y3);
    BigIntGpu lambda_den_big(x2);
    lambda_den_big -= x1_big;
    BigIntGpu x1_minus_x3_big(x1);
    x1_minus_x3_big -= x3_big;
    BigUintGpu lambda_sq = lambda.mul(lambda);

    const uint32_t q_counts[3] = {q0_limbs, q1_limbs, q2_limbs};
    const uint32_t c_counts[3] = {c0_limbs, c1_limbs, c2_limbs};

    size_t col = 1 + 4 * NUM_LIMBS + 3 * NUM_LIMBS;
    size_t carry_col = col + q0_limbs + q1_limbs + q2_limbs;

    {
        BigIntGpu constraint_big(prime.limb_bits);
        constraint_big = lambda_den_big.mul(lambda_big);
        constraint_big -= y2_big;
        constraint_big += y1_big;
        BigIntGpu quotient(prime.limb_bits);
        quotient = constraint_big.div_biguint(prime);
        quotient.normalize();
        for (uint32_t limb = 0; limb < q_counts[0]; limb++) {
            int32_t q_signed =
                limb < quotient.mag.num_limbs ? (int32_t)quotient.mag.limbs[limb] : 0;
            if (quotient.is_negative) {
                q_signed = -q_signed;
            }
            write_signed_bigint_limb(core_row, col, quotient, limb);
            range_checker.add_count((uint32_t)(q_signed + (1 << 8)), 9);
        }
        OverflowInt constraint_ov(x2, NUM_LIMBS);
        OverflowInt tmp_ov(x1, NUM_LIMBS);
        constraint_ov -= tmp_ov;
        tmp_ov = OverflowInt(lambda, NUM_LIMBS);
        constraint_ov *= tmp_ov;
        tmp_ov = OverflowInt(y2, NUM_LIMBS);
        constraint_ov -= tmp_ov;
        tmp_ov = OverflowInt(y1, NUM_LIMBS);
        constraint_ov += tmp_ov;
        OverflowInt result = constraint_ov;
        OverflowInt q_ov(quotient, q_counts[0]);
        tmp_ov = q_ov;
        tmp_ov *= prime_overflow;
        result -= tmp_ov;
        OverflowInt carries = result.carry_limbs(c_counts[0]);
        uint32_t carry_bits = result.max_overflow_bits - 8;
        uint32_t carry_min_abs = 1u << carry_bits;
        carry_bits++;
        for (uint32_t limb = 0; limb < c_counts[0]; limb++) {
            int32_t carry = carries.limbs[limb];
            core_row[carry_col++] =
                carry >= 0 ? Fp((uint32_t)carry) : (Fp::zero() - Fp((uint32_t)(-carry)));
            range_checker.add_count((uint32_t)(carry + (int32_t)carry_min_abs), carry_bits);
        }
    }

    {
        BigIntGpu constraint_big(lambda_sq);
        constraint_big -= x1_big;
        constraint_big -= x2_big;
        constraint_big -= x3_big;
        BigIntGpu quotient(prime.limb_bits);
        quotient = constraint_big.div_biguint(prime);
        quotient.normalize();
        for (uint32_t limb = 0; limb < q_counts[1]; limb++) {
            int32_t q_signed =
                limb < quotient.mag.num_limbs ? (int32_t)quotient.mag.limbs[limb] : 0;
            if (quotient.is_negative) {
                q_signed = -q_signed;
            }
            write_signed_bigint_limb(core_row, col, quotient, limb);
            range_checker.add_count((uint32_t)(q_signed + (1 << 8)), 9);
        }
        OverflowInt constraint_ov(lambda, NUM_LIMBS);
        OverflowInt tmp_ov(lambda, NUM_LIMBS);
        constraint_ov *= tmp_ov;
        tmp_ov = OverflowInt(x1, NUM_LIMBS);
        constraint_ov -= tmp_ov;
        tmp_ov = OverflowInt(x2, NUM_LIMBS);
        constraint_ov -= tmp_ov;
        tmp_ov = OverflowInt(x3, NUM_LIMBS);
        constraint_ov -= tmp_ov;
        OverflowInt result = constraint_ov;
        OverflowInt q_ov(quotient, q_counts[1]);
        tmp_ov = q_ov;
        tmp_ov *= prime_overflow;
        result -= tmp_ov;
        OverflowInt carries = result.carry_limbs(c_counts[1]);
        uint32_t carry_bits = result.max_overflow_bits - 8;
        uint32_t carry_min_abs = 1u << carry_bits;
        carry_bits++;
        for (uint32_t limb = 0; limb < c_counts[1]; limb++) {
            int32_t carry = carries.limbs[limb];
            core_row[carry_col++] =
                carry >= 0 ? Fp((uint32_t)carry) : (Fp::zero() - Fp((uint32_t)(-carry)));
            range_checker.add_count((uint32_t)(carry + (int32_t)carry_min_abs), carry_bits);
        }
    }

    {
        BigIntGpu constraint_big(prime.limb_bits);
        constraint_big = x1_minus_x3_big.mul(lambda_big);
        constraint_big -= y1_big;
        constraint_big -= y3_big;
        BigIntGpu quotient(prime.limb_bits);
        quotient = constraint_big.div_biguint(prime);
        quotient.normalize();
        for (uint32_t limb = 0; limb < q_counts[2]; limb++) {
            int32_t q_signed =
                limb < quotient.mag.num_limbs ? (int32_t)quotient.mag.limbs[limb] : 0;
            if (quotient.is_negative) {
                q_signed = -q_signed;
            }
            write_signed_bigint_limb(core_row, col, quotient, limb);
            range_checker.add_count((uint32_t)(q_signed + (1 << 8)), 9);
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
        OverflowInt q_ov(quotient, q_counts[2]);
        tmp_ov = q_ov;
        tmp_ov *= prime_overflow;
        result -= tmp_ov;
        OverflowInt carries = result.carry_limbs(c_counts[2]);
        uint32_t carry_bits = result.max_overflow_bits - 8;
        uint32_t carry_min_abs = 1u << carry_bits;
        carry_bits++;
        for (uint32_t limb = 0; limb < c_counts[2]; limb++) {
            int32_t carry = carries.limbs[limb];
            core_row[carry_col++] =
                carry >= 0 ? Fp((uint32_t)carry) : (Fp::zero() - Fp((uint32_t)(-carry)));
            range_checker.add_count((uint32_t)(carry + (int32_t)carry_min_abs), carry_bits);
        }
    }

    core_row[carry_col++] = is_setup ? Fp::zero() : Fp::one();
    while (carry_col < core_width) {
        core_row[carry_col++] = Fp::zero();
    }
}

extern "C" int launch_ec_add_ne_tracegen(
    Fp *d_trace,
    size_t trace_height,
    const uint8_t *d_records,
    size_t num_records,
    size_t record_stride,
    const uint8_t *d_prime,
    uint32_t prime_limb_count,
    const uint8_t *d_barrett_mu,
    uint32_t q0_limbs,
    uint32_t q1_limbs,
    uint32_t q2_limbs,
    uint32_t c0_limbs,
    uint32_t c1_limbs,
    uint32_t c2_limbs,
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
        const size_t core_width = 1 + 4 * NUM_LIMBS + 3 * NUM_LIMBS + q0_limbs + q1_limbs +
                                  q2_limbs + c0_limbs + c1_limbs + c2_limbs + 1;

        ec_add_ne_adapter_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS><<<main_grid, main_block>>>(
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

        ec_add_ne_compute_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS><<<main_grid, main_block>>>(
            d_trace,
            trace_height,
            core_width,
            d_records,
            num_records,
            record_stride,
            d_prime,
            prime_limb_count,
            d_barrett_mu,
            d_range_checker,
            range_checker_num_bins
        );
        ret = CHECK_KERNEL();
        if (ret) {
            return ret;
        }

        ec_add_ne_constraint_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
            <<<constraint_grid, constraint_block>>>(
                d_trace,
                trace_height,
                core_width,
                d_records,
                num_records,
                record_stride,
                d_prime,
                prime_limb_count,
                d_barrett_mu,
                q0_limbs,
                q1_limbs,
                q2_limbs,
                c0_limbs,
                c1_limbs,
                c2_limbs,
                setup_opcode,
                d_range_checker,
                range_checker_num_bins
            );
    } else {
        constexpr size_t BLOCKS = 24;
        constexpr size_t BLOCK_SIZE = 4;
        constexpr size_t NUM_LIMBS = 48;
        const size_t core_width = 1 + 4 * NUM_LIMBS + 3 * NUM_LIMBS + q0_limbs + q1_limbs +
                                  q2_limbs + c0_limbs + c1_limbs + c2_limbs + 1;

        ec_add_ne_adapter_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS><<<main_grid, main_block>>>(
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

        ec_add_ne_compute_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS><<<main_grid, main_block>>>(
            d_trace,
            trace_height,
            core_width,
            d_records,
            num_records,
            record_stride,
            d_prime,
            prime_limb_count,
            d_barrett_mu,
            d_range_checker,
            range_checker_num_bins
        );
        ret = CHECK_KERNEL();
        if (ret) {
            return ret;
        }

        ec_add_ne_constraint_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
            <<<constraint_grid, constraint_block>>>(
                d_trace,
                trace_height,
                core_width,
                d_records,
                num_records,
                record_stride,
                d_prime,
                prime_limb_count,
                d_barrett_mu,
                q0_limbs,
                q1_limbs,
                q2_limbs,
                c0_limbs,
                c1_limbs,
                c2_limbs,
                setup_opcode,
                d_range_checker,
                range_checker_num_bins
            );
    }

    return CHECK_KERNEL();
}
