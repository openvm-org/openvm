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

template <size_t NUM_LIMBS> struct Fp2MulDivCoreRecord {
    uint8_t opcode;
    uint8_t input_limbs[4 * NUM_LIMBS];
};

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS> struct Fp2MulDivRecord {
    Rv32VecHeapAdapterRecord<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE> adapter;
    Fp2MulDivCoreRecord<NUM_LIMBS> core;
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

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS>
__global__ void fp2_muldiv_adapter_tracegen_kernel(
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
    if (idx >= height)
        return;

    RowSlice row(trace + idx, height);
    VariableRangeChecker range_checker(range_checker_ptr, range_checker_num_bins);
    BitwiseOperationLookup bitwise_lookup(bitwise_lookup_ptr, bitwise_num_bits);

    constexpr size_t ADAPTER_WIDTH =
        sizeof(Rv32VecHeapAdapterCols<uint8_t, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>);
    if (idx >= num_records) {
        row.fill_zero(0, ADAPTER_WIDTH);
        return;
    }

    const auto *record = reinterpret_cast<const Fp2MulDivRecord<BLOCKS, BLOCK_SIZE, NUM_LIMBS> *>(
        records + idx * record_stride
    );
    Rv32VecHeapAdapter<2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE> adapter(
        pointer_max_bits, range_checker, bitwise_lookup, timestamp_max_bits
    );
    adapter.fill_trace_row(row, record->adapter);
}

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS>
__global__ void fp2_muldiv_compute_tracegen_kernel(
    Fp *trace,
    size_t height,
    size_t core_width,
    const uint8_t *records,
    size_t num_records,
    size_t record_stride,
    const uint8_t *prime_limbs,
    uint32_t prime_limb_count,
    const uint8_t *barrett_mu,
    uint32_t mul_opcode,
    uint32_t div_opcode,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height)
        return;

    constexpr size_t ADAPTER_WIDTH =
        sizeof(Rv32VecHeapAdapterCols<uint8_t, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>);
    RowSlice row(trace + idx, height);
    RowSlice core_row = row.slice_from(ADAPTER_WIDTH);
    VariableRangeChecker range_checker(range_checker_ptr, range_checker_num_bins);

    if (idx >= num_records) {
        core_row.fill_zero(0, core_width);
        return;
    }

    const auto *record = reinterpret_cast<const Fp2MulDivRecord<BLOCKS, BLOCK_SIZE, NUM_LIMBS> *>(
        records + idx * record_stride
    );
    const bool is_mul = record->core.opcode == mul_opcode;
    const bool is_div = record->core.opcode == div_opcode;

    BigUintGpu prime(prime_limbs, prime_limb_count, 8);
    prime.normalize();

    const uint8_t *input_limbs = record->core.input_limbs;
    BigUintGpu x0(input_limbs + 0 * NUM_LIMBS, NUM_LIMBS, 8);
    BigUintGpu x1(input_limbs + 1 * NUM_LIMBS, NUM_LIMBS, 8);
    BigUintGpu y0(input_limbs + 2 * NUM_LIMBS, NUM_LIMBS, 8);
    BigUintGpu y1(input_limbs + 3 * NUM_LIMBS, NUM_LIMBS, 8);
    x0.normalize();
    x1.normalize();
    y0.normalize();
    y1.normalize();
    x0 = x0.rem(prime);
    x1 = x1.rem(prime);
    y0 = y0.rem(prime);
    y1 = y1.rem(prime);

    BigUintGpu z0 = x0;
    BigUintGpu z1 = x1;
    if (is_mul) {
        BigUintGpu t0 = x0.mul(y0).mod_reduce(prime, barrett_mu);
        BigUintGpu t1 = x1.mul(y1).mod_reduce(prime, barrett_mu);
        z0 = t0.mod_sub(t1, prime);
        t0 = x1.mul(y0).mod_reduce(prime, barrett_mu);
        t1 = x0.mul(y1).mod_reduce(prime, barrett_mu);
        t0.add_in_place(t1);
        z1 = t0.mod_reduce(prime, barrett_mu);
    } else if (is_div) {
        BigUintGpu denom = y0.mul(y0).mod_reduce(prime, barrett_mu);
        BigUintGpu t1 = y1.mul(y1).mod_reduce(prime, barrett_mu);
        denom.add_in_place(t1);
        denom = denom.mod_reduce(prime, barrett_mu);

        BigUintGpu num0 = x0.mul(y0).mod_reduce(prime, barrett_mu);
        t1 = x1.mul(y1).mod_reduce(prime, barrett_mu);
        num0.add_in_place(t1);
        num0 = num0.mod_reduce(prime, barrett_mu);

        BigUintGpu num1 = x1.mul(y0).mod_reduce(prime, barrett_mu);
        t1 = x0.mul(y1).mod_reduce(prime, barrett_mu);
        num1 = num1.mod_sub(t1, prime);

        z0 = num0.mod_div(denom, prime, barrett_mu);
        z1 = num1.mod_div(denom, prime, barrett_mu);
    }

    size_t col = 0;
    core_row[col++] = Fp::one();
    for (size_t i = 0; i < 4 * NUM_LIMBS; i++) {
        core_row[col++] = Fp(input_limbs[i]);
    }

    write_biguint<NUM_LIMBS>(core_row, col, z0);
    write_biguint<NUM_LIMBS>(core_row, col, z1);
    for (size_t i = 0; i < 2 * NUM_LIMBS; i++) {
        range_checker.add_count(core_row[1 + 4 * NUM_LIMBS + i].asUInt32(), 8);
    }
}

template <size_t BLOCKS, size_t BLOCK_SIZE, size_t NUM_LIMBS>
__global__ void fp2_muldiv_constraint_tracegen_kernel(
    Fp *trace,
    size_t height,
    size_t core_width,
    const uint8_t *records,
    size_t num_records,
    size_t record_stride,
    const uint8_t *prime_limbs,
    uint32_t prime_limb_count,
    uint32_t q0_limbs,
    uint32_t q1_limbs,
    uint32_t c0_limbs,
    uint32_t c1_limbs,
    uint32_t mul_opcode,
    uint32_t div_opcode,
    uint32_t *range_checker_ptr,
    uint32_t range_checker_num_bins
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height)
        return;

    constexpr size_t ADAPTER_WIDTH =
        sizeof(Rv32VecHeapAdapterCols<uint8_t, 2, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>);
    RowSlice row(trace + idx, height);
    RowSlice core_row = row.slice_from(ADAPTER_WIDTH);
    VariableRangeChecker range_checker(range_checker_ptr, range_checker_num_bins);

    if (idx >= num_records)
        return;

    const auto *record = reinterpret_cast<const Fp2MulDivRecord<BLOCKS, BLOCK_SIZE, NUM_LIMBS> *>(
        records + idx * record_stride
    );
    const bool is_mul = record->core.opcode == mul_opcode;
    const bool is_div = record->core.opcode == div_opcode;

    BigUintGpu prime(prime_limbs, prime_limb_count, 8);
    prime.normalize();
    OverflowInt prime_overflow(prime, prime.num_limbs);

    const uint8_t *input_limbs = record->core.input_limbs;
    BigUintGpu x0(input_limbs + 0 * NUM_LIMBS, NUM_LIMBS, 8);
    BigUintGpu x1(input_limbs + 1 * NUM_LIMBS, NUM_LIMBS, 8);
    BigUintGpu y0(input_limbs + 2 * NUM_LIMBS, NUM_LIMBS, 8);
    BigUintGpu y1(input_limbs + 3 * NUM_LIMBS, NUM_LIMBS, 8);
    x0.normalize();
    x1.normalize();
    y0.normalize();
    y1.normalize();
    const size_t vars_col = 1 + 4 * NUM_LIMBS;
    BigUintGpu z0 = read_biguint<NUM_LIMBS>(core_row, vars_col + 0 * NUM_LIMBS);
    BigUintGpu z1 = read_biguint<NUM_LIMBS>(core_row, vars_col + 1 * NUM_LIMBS);

    const BigUintGpu &l0 = is_mul ? x0 : z0;
    const BigUintGpu &l1 = is_mul ? x1 : z1;
    const BigUintGpu &r0 = is_mul ? z0 : x0;
    const BigUintGpu &r1 = is_mul ? z1 : x1;

    const uint32_t q_counts[2] = {q0_limbs, q1_limbs};
    const uint32_t c_counts[2] = {c0_limbs, c1_limbs};
    size_t col = vars_col + 2 * NUM_LIMBS;
    size_t carry_col = col + q0_limbs + q1_limbs;

    {
        BigIntGpu constraint_big(l0);
        BigIntGpu tmp_big(y0);
        constraint_big *= tmp_big;
        BigIntGpu l1y1(l1);
        tmp_big = BigIntGpu(y1);
        l1y1 *= tmp_big;
        constraint_big -= l1y1;
        tmp_big = BigIntGpu(r0);
        constraint_big -= tmp_big;

        BigIntGpu quotient = constraint_big.div_biguint(prime);
        int64_t q_signed_limbs[MAX_LIMBS];
        for (uint32_t i = 0; i < MAX_LIMBS; i++) {
            q_signed_limbs[i] = 0;
        }
        for (uint32_t limb = 0; limb < q_counts[0]; limb++) {
            int32_t q_signed =
                limb < quotient.mag.num_limbs ? (int32_t)quotient.mag.limbs[limb] : 0;
            if (quotient.is_negative) {
                q_signed = -q_signed;
            }
            q_signed_limbs[limb] = q_signed;
            write_signed_bigint_limb(core_row, col, quotient, limb);
        }
        for (uint32_t limb = 0; limb < q_counts[0]; limb++) {
            int32_t q_signed = (int32_t)q_signed_limbs[limb];
            range_checker.add_count((uint32_t)(q_signed + (1 << 8)), 9);
        }

        OverflowInt constraint_ov(l0, NUM_LIMBS);
        OverflowInt tmp_ov(y0, NUM_LIMBS);
        constraint_ov *= tmp_ov;
        OverflowInt l1y1_ov(l1, NUM_LIMBS);
        tmp_ov = OverflowInt(y1, NUM_LIMBS);
        l1y1_ov *= tmp_ov;
        constraint_ov -= l1y1_ov;
        tmp_ov = OverflowInt(r0, NUM_LIMBS);
        constraint_ov -= tmp_ov;
        OverflowInt result = constraint_ov;
        OverflowInt q_overflow(q_signed_limbs, q_counts[0], 8);
        tmp_ov = q_overflow * prime_overflow;
        result -= tmp_ov;
        uint32_t carry_bits = result.max_overflow_bits - 8;
        uint32_t carry_min_abs = 1u << carry_bits;
        carry_bits++;
        OverflowInt carries = result.carry_limbs(c_counts[0]);
        for (uint32_t limb = 0; limb < c_counts[0]; limb++) {
            int32_t carry = carries.limbs[limb];
            core_row[carry_col++] =
                carry >= 0 ? Fp((uint32_t)carry) : (Fp::zero() - Fp((uint32_t)(-carry)));
            range_checker.add_count((uint32_t)(carry + (int32_t)carry_min_abs), carry_bits);
        }
    }

    {
        BigIntGpu constraint_big(l0);
        BigIntGpu tmp_big(y1);
        constraint_big *= tmp_big;
        BigIntGpu l1y0(l1);
        tmp_big = BigIntGpu(y0);
        l1y0 *= tmp_big;
        constraint_big += l1y0;
        tmp_big = BigIntGpu(r1);
        constraint_big -= tmp_big;

        BigIntGpu quotient = constraint_big.div_biguint(prime);
        int64_t q_signed_limbs[MAX_LIMBS];
        for (uint32_t i = 0; i < MAX_LIMBS; i++) {
            q_signed_limbs[i] = 0;
        }
        for (uint32_t limb = 0; limb < q_counts[1]; limb++) {
            int32_t q_signed =
                limb < quotient.mag.num_limbs ? (int32_t)quotient.mag.limbs[limb] : 0;
            if (quotient.is_negative) {
                q_signed = -q_signed;
            }
            q_signed_limbs[limb] = q_signed;
            write_signed_bigint_limb(core_row, col, quotient, limb);
        }
        for (uint32_t limb = 0; limb < q_counts[1]; limb++) {
            int32_t q_signed = (int32_t)q_signed_limbs[limb];
            range_checker.add_count((uint32_t)(q_signed + (1 << 8)), 9);
        }

        OverflowInt constraint_ov(l0, NUM_LIMBS);
        OverflowInt tmp_ov(y1, NUM_LIMBS);
        constraint_ov *= tmp_ov;
        OverflowInt l1y0_ov(l1, NUM_LIMBS);
        tmp_ov = OverflowInt(y0, NUM_LIMBS);
        l1y0_ov *= tmp_ov;
        constraint_ov += l1y0_ov;
        tmp_ov = OverflowInt(r1, NUM_LIMBS);
        constraint_ov -= tmp_ov;
        OverflowInt result = constraint_ov;
        OverflowInt q_overflow(q_signed_limbs, q_counts[1], 8);
        tmp_ov = q_overflow * prime_overflow;
        result -= tmp_ov;
        uint32_t carry_bits = result.max_overflow_bits - 8;
        uint32_t carry_min_abs = 1u << carry_bits;
        carry_bits++;
        OverflowInt carries = result.carry_limbs(c_counts[1]);
        for (uint32_t limb = 0; limb < c_counts[1]; limb++) {
            int32_t carry = carries.limbs[limb];
            core_row[carry_col++] =
                carry >= 0 ? Fp((uint32_t)carry) : (Fp::zero() - Fp((uint32_t)(-carry)));
            range_checker.add_count((uint32_t)(carry + (int32_t)carry_min_abs), carry_bits);
        }
    }

    core_row[carry_col++] = is_mul ? Fp::one() : Fp::zero();
    core_row[carry_col++] = is_div ? Fp::one() : Fp::zero();
    while (carry_col < core_width) {
        core_row[carry_col++] = Fp::zero();
    }
}

extern "C" int launch_fp2_muldiv_tracegen(
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
    uint32_t c0_limbs,
    uint32_t c1_limbs,
    uint32_t mul_opcode,
    uint32_t div_opcode,
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
        const size_t core_width =
            1 + 4 * NUM_LIMBS + 2 * NUM_LIMBS + q0_limbs + q1_limbs + c0_limbs + c1_limbs + 2;

        fp2_muldiv_adapter_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
            <<<main_grid, main_block>>>(
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
        if (ret)
            return ret;

        fp2_muldiv_compute_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
            <<<main_grid, main_block>>>(
                d_trace,
                trace_height,
                core_width,
                d_records,
                num_records,
                record_stride,
                d_prime,
                prime_limb_count,
                d_barrett_mu,
                mul_opcode,
                div_opcode,
                d_range_checker,
                range_checker_num_bins
            );
        ret = CHECK_KERNEL();
        if (ret)
            return ret;

        fp2_muldiv_constraint_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
            <<<constraint_grid, constraint_block>>>(
                d_trace,
                trace_height,
                core_width,
                d_records,
                num_records,
                record_stride,
                d_prime,
                prime_limb_count,
                q0_limbs,
                q1_limbs,
                c0_limbs,
                c1_limbs,
                mul_opcode,
                div_opcode,
                d_range_checker,
                range_checker_num_bins
            );
    } else {
        constexpr size_t BLOCKS = 24;
        constexpr size_t BLOCK_SIZE = 4;
        constexpr size_t NUM_LIMBS = 48;
        const size_t core_width =
            1 + 4 * NUM_LIMBS + 2 * NUM_LIMBS + q0_limbs + q1_limbs + c0_limbs + c1_limbs + 2;

        fp2_muldiv_adapter_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
            <<<main_grid, main_block>>>(
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
        if (ret)
            return ret;

        fp2_muldiv_compute_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
            <<<main_grid, main_block>>>(
                d_trace,
                trace_height,
                core_width,
                d_records,
                num_records,
                record_stride,
                d_prime,
                prime_limb_count,
                d_barrett_mu,
                mul_opcode,
                div_opcode,
                d_range_checker,
                range_checker_num_bins
            );
        ret = CHECK_KERNEL();
        if (ret)
            return ret;

        fp2_muldiv_constraint_tracegen_kernel<BLOCKS, BLOCK_SIZE, NUM_LIMBS>
            <<<constraint_grid, constraint_block>>>(
                d_trace,
                trace_height,
                core_width,
                d_records,
                num_records,
                record_stride,
                d_prime,
                prime_limb_count,
                q0_limbs,
                q1_limbs,
                c0_limbs,
                c1_limbs,
                mul_opcode,
                div_opcode,
                d_range_checker,
                range_checker_num_bins
            );
    }

    return CHECK_KERNEL();
}
